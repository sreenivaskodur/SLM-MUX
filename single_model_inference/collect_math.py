#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
import time
import logging
import argparse
import random
import re
import hashlib
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG = int(os.environ.get("DEBUG", "0"))

# ===================== Inline Prompt Templates (Editable) =====================
# Edit these three strings when doing prompt tuning. They will also be saved to JSON.
SYSTEM_PROMPT = (
    # "You are a PhD student in math. "
    "Solve the following problem step by step. "
    "Review what you have done and make sure you have not made any mistakes. "
    "Be careful with intervals and plus or minus signs. Those parts are very easy to make mistakes. "
    "Provide the final answer enclosed in LaTeX \\boxed{...}."
)

# Will be formatted with {problem}
USER_PROMPT_TEMPLATE = (
    "Problem:\n{problem}\n"
)

# Appended to the user message to enforce answer format compatible with extractor
FORMAT_PROMPT = (
    "\nPlease provide your final answer in the form: \\boxed{{...}}"
)

def _build_messages_from_templates(problem_text: str) -> List[Dict[str, str]]:
    user_content = USER_PROMPT_TEMPLATE.format(problem=problem_text) + FORMAT_PROMPT
    messages: List[Dict[str, str]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    return messages

# ===================== LLM Verification (GPT-4o) =====================
EQUAL_VERIFY_PROMPT_TEMPLATE = (
    """
You are a mathematics teacher grading a student's answer on a test.

The standard (correct) answer is: {reference}

The student wrote: {extracted}

As a fair but rigorous math teacher:
- Would you mark the student's answer as CORRECT?
- The student's notation may differ from yours (e.g., using 3 vs 3^\\circ for angles, or 1/sqrt(3) vs sqrt(3)/3)
- If the student's answer is mathematically equivalent to the standard answer, it should be marked correct
- Minor notational differences are acceptable as long as the mathematical meaning is preserved
- The student's answer must represent exactly the same value or function as the standard answer

Respond ONLY with "Yes" (I would mark it correct) or "No" (I would mark it incorrect).
Answer:
"""
).strip()


def _build_equal_verify_prompt(reference: str, extracted: str) -> str:
    return EQUAL_VERIFY_PROMPT_TEMPLATE.format(reference=reference, extracted=extracted)


def verify_with_llm_gpt4o(reference: str, extracted: str, judge_model: str = "gpt-4o", temperature: float = 0.0) -> Dict[str, str]:
    """Call OpenAI GPT-4o to judge equivalence; returns {"decision": "Yes"|"No"|"", "raw": full_text}."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; falling back to rule-based verification.")
        return {"decision": "", "raw": ""}

    endpoint = "https://api.openai.com/v1/chat/completions"
    prompt = _build_equal_verify_prompt(reference, extracted)

    for sleep_time in [1, 2, 4]:
        try:
            res = requests.post(
                endpoint,
                json={
                    "model": judge_model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            data = res.json()
            if "error" in data:
                logger.error({"judge_error": data.get("error")})
                time.sleep(sleep_time)
                continue
            text = data["choices"][0]["message"]["content"].strip()
            decision = text.strip().strip('"').split()[0] if text else ""
            return {"decision": decision, "raw": text}
        except Exception as e:
            logger.error(f"OpenAI judge error: {e}")
            time.sleep(sleep_time)

    return {"decision": "", "raw": ""}

# =============【Answer Extraction and Normalization Logic for MATH Dataset】============
def extract_boxed_text(output: str):
    start_tag = r'\boxed{'
    start_idx = output.find(start_tag)
    if start_idx == -1:
        return None

    idx = start_idx + len(start_tag)
    depth = 1
    out_chars = []

    while idx < len(output) and depth > 0:
        c = output[idx]
        if c == '{':
            depth += 1
            out_chars.append(c)
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
            else:
                out_chars.append(c)
        else:
            out_chars.append(c)
        idx += 1

    if depth != 0:
        return None

    return ''.join(out_chars).strip()


def extract_answer_math(output: str) -> str:
    raw_in_box = extract_boxed_text(output)
    if raw_in_box is not None:
        normalized = normalize_answer(raw_in_box)
        return normalized
    return ""


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = answer.strip()

    # 如果最外层是 \text{...}，去掉它
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
    if m is not None:
        answer = m.group("text").strip()

    return _strip_string(answer)


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string
    if "=" in string:
        parts = string.split("=")
        if len(parts) == 2 and len(parts[0]) <= 2:
            string = parts[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _remove_right_units(string: str) -> str:
    if "\\text{" in string:
        splits = string.split("\\text{")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split.startswith("{"):
            a = split[0]
            new_substr = f"\\sqrt{{{a}}}{split[1:]}"
            new_string += new_substr
        else:
            new_string += "\\sqrt" + split
    return new_string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for sub in substrs[1:]:
            new_str += "\\frac"
            if sub.startswith("{"):
                new_str += sub
            else:
                if len(sub) >= 2:
                    a, b = sub[0], sub[1]
                    if b != "{":
                        if len(sub) > 2:
                            new_str += f"{{{a}}}{{{b}}}{sub[2:]}"
                        else:
                            new_str += f"{{{a}}}{{{b}}}"
                    else:
                        new_str += f"{{{a}}}{sub[1:]}"
                else:
                    new_str += sub
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if string.count("/") == 1:
        a, b = string.split("/")
        try:
            a_int = int(a)
            b_int = int(b)
            return f"\\frac{{{a_int}}}{{{b_int}}}"
        except:
            return string
    return string


# =============【Together API Call Function with Usage Tracking】============
def generate_together(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
):
    """
    Function to generate a response using the Together API.
    Returns a dict: {"content": str, "token_usage": int}
    """
    endpoint = "https://api.together.xyz/v1/chat/completions"
    output_text = ""
    token_usage = 0

    for sleep_time in [1, 1, 1, 2, 4, 8, 16]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages to model: {model}. Last user content: {messages[-1]['content'][:50]}..."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )

            data = res.json()
            if "error" in data:
                logger.error(data)
                return {"content": "", "token_usage": 0}

            output_text = data["choices"][0]["message"]["content"]
            # If the Together API response includes usage information, it can be extracted here.
            # A common structure is data["usage"]["total_tokens"], depending on the actual API format:
            usage_info = data.get("usage", {})
            token_usage = usage_info.get("total_tokens", 0)

            break

        except Exception as e:
            logger.error(f"Error calling {model}: {e}")
            if DEBUG:
                logger.debug(f"Msgs: {messages}")
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    return {"content": output_text.strip(), "token_usage": token_usage}


def create_chat_prompt(system_msg: str, user_msg: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect MATH_500 answers from multiple Together models')
    parser.add_argument('--model_list', type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo",
                        help='Comma-separated list of Together model names')
    parser.add_argument('--sample_size', type=int, default=-1, help='Number of questions to sample from MATH_500 dataset')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='Max tokens in each generation')
    parser.add_argument('--num_threads', type=int, default=50, help='Concurrent threads for processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_path', type=str, default='Datasets/MATH_500.json', help='Path to the MATH_500 dataset JSON file')
    parser.add_argument('--output_dir', type=str, default='./MATH_500_response', help='Directory to save benchmark results')
    parser.add_argument('--verify_mode', type=str, choices=['llm', 'rule'], default='llm',
                        help='Answer verification mode: llm (GPT-4o judge) or rule (string equality)')
    parser.add_argument('--judge_model', type=str, default='gpt-4o',
                        help='OpenAI judge model to use when verify_mode=llm')
    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(args.seed)

    # Prepare output directories with prompt hash, similar to GPQA script
    os.makedirs(args.output_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_hash_src = (SYSTEM_PROMPT or "") + (USER_PROMPT_TEMPLATE or "") + (FORMAT_PROMPT or "")
    prompt_hash = hashlib.sha256(prompt_hash_src.encode('utf-8')).hexdigest()[:8]
    run_dir = os.path.join(args.output_dir, f"MATH_500_{run_timestamp}_p{prompt_hash}")
    os.makedirs(run_dir, exist_ok=True)

    model_list = [m.strip() for m in args.model_list.split(",")]

    # Load dataset from provided path
    try:
        math_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        data_list = list(math_dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return
    if 0 < args.sample_size < len(data_list):
        data_list = random.sample(data_list, args.sample_size)

    # Write run metadata for reproducibility and prompt tracking
    run_meta = {
        "timestamp": run_timestamp,
        "dataset_path": args.dataset_path,
        "model_list": model_list,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "sample_size_requested": args.sample_size,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "verification": {
            "mode": args.verify_mode,
            "judge_model": args.judge_model if args.verify_mode == 'llm' else None,
        },
        "prompts": {
            "system": SYSTEM_PROMPT,
            "user_template": USER_PROMPT_TEMPLATE,
            "format": FORMAT_PROMPT,
        },
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    for model in model_list:
        logger.info(f"Processing model: {model}")
        results_for_this_model = []
        total_token_usage = 0

        def process_one(item):
            problem_text = item["problem"]
            reference_answer = normalize_answer(item.get("answer", ""))

            messages = _build_messages_from_templates(problem_text)
            resp = generate_together(
                model=model,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature
            )

            raw_output = resp["content"]
            usage_tokens = resp["token_usage"]
            if not raw_output:
                return {
                    "model": model,
                    "problem": problem_text,
                    "reference_answer": reference_answer,
                    "model_response": "",
                    "extracted_answer": "",
                    "is_correct": False,
                    "token_usage": {"total_tokens": usage_tokens},
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": USER_PROMPT_TEMPLATE.format(problem=problem_text) + FORMAT_PROMPT,
                    "format_prompt": FORMAT_PROMPT,
                    "verification": {
                        "mode": args.verify_mode,
                        "judge_model": args.judge_model if args.verify_mode == 'llm' else None,
                        "judge_text": None,
                    }
                }

            extracted_answer = extract_answer_math(raw_output)
            is_correct = (extracted_answer == reference_answer)

            judge_text = None
            if args.verify_mode == 'llm' and extracted_answer and reference_answer:
                judge = verify_with_llm_gpt4o(reference_answer, extracted_answer, judge_model=args.judge_model)
                decision = judge.get("decision", "").lower()
                judge_text = judge.get("raw", "")
                if decision.startswith("yes"):
                    is_correct = True
                elif decision.startswith("no"):
                    is_correct = False
                else:
                    # undecided from judge, keep rule-based result
                    pass

            return {
                "model": model,
                "problem": problem_text,
                "reference_answer": reference_answer,
                "model_response": raw_output,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "token_usage": {"total_tokens": usage_tokens},
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT_TEMPLATE.format(problem=problem_text) + FORMAT_PROMPT,
                "format_prompt": FORMAT_PROMPT,
                "verification": {
                    "mode": args.verify_mode,
                    "judge_model": args.judge_model if args.verify_mode == 'llm' else None,
                    "judge_text": judge_text,
                }
            }

        all_futures = []
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            for item in data_list:
                future = executor.submit(process_one, item)
                all_futures.append(future)

            for f in tqdm(as_completed(all_futures), total=len(all_futures),
                          desc=f"Collecting responses for {model}", unit="q"):
                results_for_this_model.append(f.result())

        correct_count = sum(1 for r in results_for_this_model if r["is_correct"])
        total_count = len(results_for_this_model)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Aggregate total token usage
        total_token_usage = sum(r["token_usage"].get("total_tokens", 0) for r in results_for_this_model)

        # Save to this run's directory; filename without timestamp for easier comparison
        out_file = f"{run_dir}/{model.replace('/', '_')}.json"

        output_data = {
            "model": model,
            "timestamp": run_timestamp,
            "prompts": {
                "system": SYSTEM_PROMPT,
                "user_template": USER_PROMPT_TEMPLATE,
                "format": FORMAT_PROMPT,
            },
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "sample_size": total_count,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "token_usage": {"total_tokens": total_token_usage},
            "verification": {
                "mode": args.verify_mode,
                "judge_model": args.judge_model if args.verify_mode == 'llm' else None,
            },
            "responses": results_for_this_model
        }

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(
            f"[*] Model={model}  Accuracy={accuracy:.2%} "
            f"({correct_count}/{total_count}), "
            f"Tokens: Total={total_token_usage}, saved to: {out_file}"
        )

    logger.info("All models processed. Finished.")


if __name__ == "__main__":
    main()
