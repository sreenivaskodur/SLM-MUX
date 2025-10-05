# call_model_together.py

import os
import requests
import logging
import time
import random
import threading
import contextlib
from typing import List, Dict, Any, Optional, Tuple
try:
    # OpenAI-compatible client (used for AWS HF endpoint)
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # Fallback if not installed; we'll log an error when used

def unify_model_name(model_name: str) -> str:
    """
    Simple example:
    - Replace underscores '_' in the incoming model_name with slashes '/'.
    - Remove possible replication marker suffixes (e.g., "::rep1") to prevent invalid model names from being sent to the API.
    """
    # Remove suffixes like "model::rep1"
    base_name = model_name.split("::", 1)[0]
    return base_name.replace('_', '/')

logger = logging.getLogger(__name__)

def _is_gemma_27b_it(model_name: str) -> bool:
    """Return True if model appears to be Gemma 2 27B IT (incl. forked names)."""
    name = (model_name or "").lower()
    return "gemma-2-27b-it" in name

def _prepare_messages_for_aws(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    AWS HF TGI-compatible endpoints may not support the 'system' role.
    Merge all system prompts into the first user message (or create one), and
    drop system roles. Preserve the rest order. Convert unknown roles to 'user'.
    """
    if not isinstance(messages, list):
        return []

    system_texts: List[str] = []
    non_system: List[Dict[str, str]] = []
    for msg in messages:
        role = (msg or {}).get("role", "")
        content = (msg or {}).get("content", "")
        if role == "system":
            if content:
                system_texts.append(str(content))
        else:
            # Keep message but normalize role if unexpected
            if role not in ("user", "assistant"):
                non_system.append({"role": "user", "content": str(content)})
            else:
                non_system.append({"role": role, "content": str(content)})

    if system_texts:
        combined_system = "\n\n".join(system_texts).strip()
        # Find first user message among non_system
        first_user_index = None
        for i, m in enumerate(non_system):
            if m.get("role") == "user":
                first_user_index = i
                break
        if first_user_index is not None:
            first_user = non_system[first_user_index]
            combined_content = (combined_system + "\n\n" + (first_user.get("content") or "")).strip()
            new_first = {"role": "user", "content": combined_content}
            rest = [m for idx, m in enumerate(non_system) if idx != first_user_index]
            return [new_first] + rest
        else:
            # No user message exists; create one with system text at the start
            return [{"role": "user", "content": combined_system}] + non_system

    return non_system

# Simple global QPS throttle for AWS branch (best-effort within a single process)
_aws_rate_lock = threading.Lock()
_aws_last_request_ts = 0.0

def _throttle_aws_qps():
    global _aws_last_request_ts
    try:
        max_qps_env = os.environ.get("AWS_HF_MAX_QPS", "")
        max_qps = float(max_qps_env) if max_qps_env else 0.0
    except Exception:
        max_qps = 0.0
    if max_qps and max_qps > 0.0:
        min_interval = 1.0 / max_qps
        with _aws_rate_lock:
            now = time.monotonic()
            wait = (_aws_last_request_ts + min_interval) - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            _aws_last_request_ts = now

_global_req_lock = threading.Lock()
_global_req_semaphore: Optional[threading.Semaphore] = None
_global_req_limit_cached: int = 0
_global_inflight_counter = 0

def _get_env_global_limit() -> int:
    try:
        val = int(os.environ.get("GLOBAL_MAX_INFLIGHT_REQUESTS", "0") or "0")
        return max(0, val)
    except Exception:
        return 0

def _ensure_global_semaphore(limit: int) -> Optional[threading.Semaphore]:
    global _global_req_semaphore, _global_req_limit_cached
    if limit <= 0:
        return None
    with _global_req_lock:
        if _global_req_semaphore is None or _global_req_limit_cached != limit:
            _global_req_semaphore = threading.Semaphore(limit)
            _global_req_limit_cached = limit
    return _global_req_semaphore

@contextlib.contextmanager
def _acquire_global_slot():
    """
    Acquire a process-wide slot to cap total in-flight HTTP requests.
    Controlled via env GLOBAL_MAX_INFLIGHT_REQUESTS. If 0/absent, no-op.
    """
    global _global_inflight_counter
    limit = _get_env_global_limit()
    sema = _ensure_global_semaphore(limit)
    if sema is None:
        yield
        return
    acquired = False
    try:
        sema.acquire()
        acquired = True
        with _global_req_lock:
            _global_inflight_counter += 1
            cur = _global_inflight_counter
        logger.debug(f"[GlobalLimiter] acquired; in_flight={cur}/{limit}")
        yield
    finally:
        if acquired:
            with _global_req_lock:
                _global_inflight_counter -= 1
                cur = _global_inflight_counter
            try:
                sema.release()
            finally:
                logger.debug(f"[GlobalLimiter] released; in_flight={cur}/{limit}")

def _should_wait_longer_on_aws_error(exc: Exception) -> bool:
    """
    Heuristically determine if the AWS HF endpoint likely returned a 5xx
    server error, in which case we should sleep ~1 minute to allow the
    backend to recover/restart.
    """
    # Try common attributes first (OpenAI client style / requests style)
    try:
        status = getattr(exc, "status_code", None)
        if status is None:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None)
        if isinstance(status, int) and 500 <= status <= 599:
            return True
    except Exception:
        pass

    # Fallback: inspect exception text
    text = str(exc).lower()
    server_tokens = [
        " 500", " 501", " 502", " 503", " 504", " 505",
        "http 5", "status 5", "server error", "internal server error",
    ]
    return any(tok in text for tok in server_tokens)

def call_model_together(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retry_count: int = 5,
    timeout: Optional[Tuple[int, int]] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call the Together API individually and return a data structure aligned with llm_utils.py:
      {
        "content": str,       # Model output
        "reasoning": str,     # Together does not return reasoning for now, so it can be left empty
        "token_usage": {
          "prompt_tokens": 0,
          "completion_tokens": 0,
          "total_tokens": int
        }
      }
    """
    # Keep the original incoming name for logging; then unify it for the actual Together call
    raw_model_name = model_name

    # --- Special routing: Gemma 2 27B IT goes to AWS HF OpenAI-compatible endpoint ---
    if _is_gemma_27b_it(raw_model_name):
        if OpenAI is None:
            logger.error(
                f"[AWS HF API Error | model={raw_model_name}] openai package not installed, please pip install openai first"
            )
            return {
                "content": "",
                "reasoning": "",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        base_url = os.environ.get(
            "AWS_HF_BASE_URL",
            "https://n45kflpp1962557t.us-east-2.aws.endpoints.huggingface.cloud/v1/",
        )
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error(
                f"[AWS HF API Error | model={raw_model_name}] HF_TOKEN is not set!"
            )
            return {
                "content": "",
                "reasoning": "",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        client = OpenAI(base_url=base_url, api_key=hf_token)

        usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_error_msg = ""
        # Consistent default timeout setting with the Together branch (openai library can control timeout at the client level; using API default here)
        for attempt in range(retry_count):
            should_wait_longer = False
            try:
                # Use non-streaming to maintain consistency with the existing return structure
                aws_messages = _prepare_messages_for_aws(messages)
                kwargs: Dict[str, Any] = {
                    "model": "tgi",
                    "messages": aws_messages,
                    "temperature": float(temperature),
                    "stream": False,
                }
                if isinstance(max_tokens, int) and max_tokens > 0:
                    kwargs["max_tokens"] = int(max_tokens)
                # Some OpenAI-compatible backends may accept these params
                if isinstance(top_p, float):
                    kwargs["top_p"] = float(top_p)
                if isinstance(top_k, int):
                    kwargs["top_k"] = int(top_k)
                if isinstance(seed, int):
                    kwargs["seed"] = int(seed)

                _throttle_aws_qps()
                with _acquire_global_slot():
                    data = client.chat.completions.create(**kwargs)
                content = ""
                if getattr(data, "choices", None):
                    choice0 = data.choices[0]
                    # openai>=1.0: message.content
                    message_obj = getattr(choice0, "message", None)
                    if message_obj is not None:
                        content = (getattr(message_obj, "content", None) or "").strip()

                usage_obj = getattr(data, "usage", None)
                if usage_obj is not None:
                    try:
                        usage_dict["total_tokens"] = int(getattr(usage_obj, "total_tokens", 0) or 0)
                    except Exception:
                        usage_dict["total_tokens"] = 0

                return {
                    "content": content,
                    "reasoning": "",
                    "token_usage": usage_dict,
                }
            except Exception as e:
                last_error_msg = str(e)
                logger.error(
                    f"[AWS HF API Error | model={raw_model_name}, attempt {attempt+1}/{retry_count}]: {e}"
                )
                # Record whether to wait for a long time to avoid referencing e outside of except
                try:
                    should_wait_longer = _should_wait_longer_on_aws_error(e)
                except Exception:
                    should_wait_longer = False
            # Backoff wait: if it's a server 5xx error, prioritize waiting for about 60s to allow the backend to restart
            if attempt < retry_count - 1:
                if should_wait_longer:
                    backoff_seconds = 60.0
                else:
                    backoff_seconds = min(60, 2 ** attempt) + random.random()
                logger.info(
                    f"[AWS HF API | model={raw_model_name}] Retrying in {backoff_seconds:.1f}s..."
                )
                time.sleep(backoff_seconds)

        logger.error(
            f"AWS HF API failed for model={raw_model_name} after {retry_count} attempts. Last error: {last_error_msg}"
        )
        return {
            "content": "",
            "reasoning": "",
            "token_usage": usage_dict,
        }

    # --- Default: Together API path for all other models ---
    model_name = unify_model_name(model_name)
    endpoint = "https://api.together.xyz/v1/chat/completions"
    api_key = os.environ.get("TOGETHER_API_KEY")

    if not api_key:
        logger.error(f"[Together API Error | model={raw_model_name}] TOGETHER_API_KEY is not set!")
        return {
            "content": "",
            "reasoning": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    # Only set max_tokens when it is explicitly passed to avoid 400 errors due to exceeding model limits
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    # Only pass optional sampling parameters when provided
    if isinstance(top_p, float):
        payload["top_p"] = float(top_p)
    if isinstance(top_k, int):
        payload["top_k"] = int(top_k)
    if isinstance(seed, int):
        payload["seed"] = int(seed)

    content = ""
    usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Default timeout: connect 10s, read 300s, to avoid threads hanging indefinitely
    if timeout is None:
        timeout = (10, 300)

    last_error_msg = ""
    for attempt in range(retry_count):
        should_wait_longer = False
        try:
            with _acquire_global_slot():
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # Get model output
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"].strip()

            # Get total_tokens from usage (other counts are set to 0)
            total_tokens = data.get("usage", {}).get("total_tokens", 0)
            usage_dict["total_tokens"] = total_tokens

            return {
                "content": content,
                "reasoning": "",
                "token_usage": usage_dict
            }
        except requests.exceptions.HTTPError as e:
            # Print more detailed response information to help locate the specific cause of a 400 error
            response_text = ""
            if e.response is not None:
                try:
                    response_text = e.response.text
                except Exception:
                    response_text = ""
            last_error_msg = str(e)
            logger.error(
                f"[Together API Error | model={raw_model_name}, attempt {attempt+1}/{retry_count}]: {e}"
            )
            if response_text:
                logger.error(
                    f"[Together API Error | model={raw_model_name}] Response body: {response_text}"
                )
            try:
                should_wait_longer = _should_wait_longer_on_aws_error(e)
            except Exception:
                should_wait_longer = False
        except requests.exceptions.RequestException as e:
            last_error_msg = str(e)
            logger.error(
                f"[Together API Error | model={raw_model_name}, attempt {attempt+1}/{retry_count}]: {e}"
            )
            try:
                should_wait_longer = _should_wait_longer_on_aws_error(e)
            except Exception:
                should_wait_longer = False
        except ValueError as e:
            last_error_msg = str(e)
            logger.error(
                f"[Together API Error | model={raw_model_name}, attempt {attempt+1}/{retry_count}]: {e}"
            )
            try:
                should_wait_longer = _should_wait_longer_on_aws_error(e)
            except Exception:
                should_wait_longer = False
        # Error backoff wait
        if attempt < retry_count - 1:
            if should_wait_longer:
                backoff_seconds = 60.0
            else:
                backoff_seconds = min(60, 2 ** attempt) + random.random()
            logger.info(
                f"[Together API | model={raw_model_name}] Retrying in {backoff_seconds:.1f}s..."
            )
            time.sleep(backoff_seconds)

    logger.error(
        f"Together API failed for model={raw_model_name} after {retry_count} attempts. Last error: {last_error_msg}"
    )
    return {
        "content": "",
        "reasoning": "",
        "token_usage": usage_dict,
    }
