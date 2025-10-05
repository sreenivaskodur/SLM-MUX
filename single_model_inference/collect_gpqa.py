# collect_GPQA_unify.py

import os
import json
import logging
import argparse
import random
import hashlib
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from datasets import load_dataset

# === Keep GPQA utility functions ===
from utils.GPQA_utils import (
    extract_answer_gpqa,
)

# === Change to import the call function for Together API ===
from utils.together_utils import call_model_together

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG = int(os.environ.get("DEBUG", "0"))

# ===================== Inline Prompt Templates (Editable) =====================
# Edit these three strings when doing prompt tuning. They will also be saved to JSON.
SYSTEM_PROMPT = (
    "You are a PhD student in science. "
    "Reason step by step through the following question and provide the best answer. "
    "Review what you have done and make sure you have not made any mistakes. "
    "Give your single-letter choice as '##Answer: X##'."
)

# Will be formatted with {question} and {choices}
USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Choices:\n{choices}\n"
)

# Appended to the user message to enforce answer format compatible with extractor
FORMAT_PROMPT = (
    "\nPlease provide your final single-letter answer in the format: ##Answer: X##."
)

def _build_messages_from_templates(question_text: str, choices: List[str]) -> List[Dict[str, str]]:
    choices_text = "\n".join(choices)
    user_content = USER_PROMPT_TEMPLATE.format(question=question_text, choices=choices_text) + FORMAT_PROMPT
    messages: List[Dict[str, str]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    return messages

def parse_arguments():
    parser = argparse.ArgumentParser(description='Unified GPQA benchmark for multiple LLM providers (using Together API).')
    parser.add_argument('--model_list', type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo",
                        help='Comma-separated list of Together model names')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='Number of questions to sample from GPQA dataset. -1 for all.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Nucleus sampling p; default 1.0 for greedy-like behavior')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Top-k sampling; default 1 for greedy decoding')
    parser.add_argument('--decode_seed', type=int, default=42,
                        help='Fixed decode seed if backend supports deterministic sampling')
    parser.add_argument('--max_new_tokens', type=int, default=-1,
                        help='Max tokens in each generation. -1 means default/undefined.')
    parser.add_argument('--num_threads', type=int, default=50,
                        help='Concurrent threads for processing (default 1 for determinism)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dataset_path', type=str, default="Datasets/shuffled_gpqa.json",
                        help='Path to the GPQA dataset JSON file')
    parser.add_argument('--output_dir', type=str, default="./GPQA_response",
                        help='Directory to save benchmark results')
    return parser.parse_args()

def main():
    """ 
    Main function: Use Together API to replace the original Google/OpenAI calls,
    run GPQA tests on the specified list of models, and output JSON results.
    """
    args = parse_arguments()
    random.seed(args.seed)

    # Create output directory (parent directory + subdirectory for this run)
    os.makedirs(args.output_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_hash_src = (SYSTEM_PROMPT or "") + (USER_PROMPT_TEMPLATE or "") + (FORMAT_PROMPT or "")
    prompt_hash = hashlib.sha256(prompt_hash_src.encode('utf-8')).hexdigest()[:8]
    run_dir = os.path.join(args.output_dir, f"GPQA_{run_timestamp}_p{prompt_hash}")
    os.makedirs(run_dir, exist_ok=True)

    # Parse model list
    model_list = [m.strip() for m in args.model_list.split(",")]

    # Load GPQA dataset
    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        data_list = list(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return

    # Sampling
    if 0 < args.sample_size < len(data_list):
        data_list = random.sample(data_list, args.sample_size)

    # Write the settings for this run to a metadata file for easy comparison of different prompts
    run_meta = {
        "timestamp": run_timestamp,
        "dataset_path": args.dataset_path,
        "model_list": model_list,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "sample_size_requested": args.sample_size,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "prompts": {
            "system": SYSTEM_PROMPT,
            "user_template": USER_PROMPT_TEMPLATE,
            "format": FORMAT_PROMPT,
        },
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    # Process models one by one (output to a separate folder for this run)
    for model in model_list:
        logger.info(f"Processing model: {model} (via Together API)")

        results_for_this_model = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        def process_one(item):
            """Process a single GPQA question: send request, extract answer, check correctness, record token consumption, etc."""
            question_text = item["question"]
            choices = item["choices"]
            correct_answer = item.get("answer", "")  # Correct answer (single letter)

            # Build Chat Prompt (using editable local templates)
            messages = _build_messages_from_templates(question_text, choices)

            # Call Together model
            resp = call_model_together(
                model_name=model,
                messages=messages,
                max_tokens=args.max_new_tokens if args.max_new_tokens > 0 else None,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.decode_seed,
                # retry_count=... Add retry count here if needed
            )

            raw_output = resp.get("content", "")
            usage_details = resp.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            reasoning_content = resp.get("reasoning", "")

            # Extract model answer
            extracted_answer = extract_answer_gpqa(raw_output, choices)
            is_correct = (extracted_answer == correct_answer)

            # Record the actual system and user prompts used this time (set to empty if the role does not exist)
            system_prompt = ""
            user_prompt = ""
            for m in messages:
                if m.get("role") == "system" and not system_prompt:
                    system_prompt = m.get("content", "")
                elif m.get("role") == "user" and not user_prompt:
                    user_prompt = m.get("content", "")
            # Expose the format snippet separately for subsequent statistics and parameter tuning
            format_prompt = FORMAT_PROMPT

            return {
                "model": model,
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_answer,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "format_prompt": format_prompt,
                "model_response": raw_output,
                "reasoning_content": reasoning_content,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "token_usage": usage_details
            }

        # Multi-threaded concurrency
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(process_one, item) for item in data_list]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Collecting responses for {model}", unit="q"):
                results_for_this_model.append(f.result())

        # Calculate accuracy
        correct_count = sum(1 for r in results_for_this_model if r["is_correct"])
        total_count = len(results_for_this_model)
        accuracy = correct_count / total_count if total_count else 0.0

        # Calculate token usage
        for r in results_for_this_model:
            token_usage = r["token_usage"]
            total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            total_completion_tokens += token_usage.get("completion_tokens", 0)
            total_tokens += token_usage.get("total_tokens", 0)

        # Save results to the subdirectory for this run; the filename no longer includes a timestamp for easy comparison
        out_file = f"{run_dir}/{model.replace('/', '_')}.json"

        output_data = {
            "model": model,
            "timestamp": run_timestamp,
            "provider": "together",
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
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens
            },
            "responses": results_for_this_model
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[*] Model={model} (via Together)  Accuracy={accuracy:.2%} "
              f"({correct_count}/{total_count}), Tokens: Prompt={total_prompt_tokens}, "
              f"Completion={total_completion_tokens}, Total={total_tokens}, saved to: {out_file}")

    logger.info("All models processed. Finished.")


if __name__ == "__main__":
    main()
