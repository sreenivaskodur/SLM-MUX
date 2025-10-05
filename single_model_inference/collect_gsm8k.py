#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_GSM_together_unify.py

Adapted from collect_MATH_together_unify.py to evaluate LLMs on the GSM dataset.
- Uses HuggingFace `gsm` dataset (`main` split by default).
- Prompts the model to return the final answer after reasoning, prefixed with `####`.
- Extracts both reference and model answers by parsing the substring after the last `####`.
- Supports the same CLI arguments as the original script.
"""

import os
import json
import logging
import argparse
import random
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.together_utils import call_model_together  # unchanged dependency
from utils.GSM_utils import extract_answer_gsm, normalize_answer, create_chat_prompt

from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG = int(os.environ.get("DEBUG", "0"))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def generate_together(model: str,
                      messages: List[Dict[str, str]],
                      max_tokens: int = 1024,
                      temperature: float = 0.0):
    """Call Together AI model and return content and usage."""
    result = call_model_together(
        model_name=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return {
        "content": result["content"],
        "token_usage": result["token_usage"]["total_tokens"]
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect GSM answers from multiple Together models")
    parser.add_argument("--model_list", type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo",
                        help="Comma-separated list of Together-compatible model names")
    parser.add_argument("--sample_size", type=int, default=-1,
                        help="Number of questions to sample from GSM (train split)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens for each generation")
    parser.add_argument("--num_threads", type=int, default=50, help="Concurrent threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(args.seed)

    os.makedirs("./GSM_500_response", exist_ok=True)
    model_list = [m.strip() for m in args.model_list.split(",")]



    with open("Datasets/GSM_500.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if 0 < args.sample_size < len(data_list):
        data_list = random.sample(data_list, args.sample_size)

    def build_prompt(question_text: str):
        system_msg = (
            "You are a helpful math tutor. Reason step by step to solve the problem. "
            "After your reasoning, output the final answer on a new line prefixed with '####'. "
            "Example: #### 42"
        )
        user_msg = f"Problem:\n{question_text}\n\nProvide your reasoning, then the final answer."
        return create_chat_prompt(system_msg, user_msg)

    for model in model_list:
        logger.info(f"Processing model: {model}")
        results_for_this_model = []

        def process_one(item):
            question_text = item["question"]
            reference_answer = normalize_answer(extract_answer_gsm(item["answer"]))

            messages = build_prompt(question_text)
            resp = generate_together(
                model=model,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature
            )

            raw_output = resp["content"]
            usage_tokens = resp["token_usage"]

            extracted_answer = normalize_answer(extract_answer_gsm(raw_output))
            is_correct = (extracted_answer == reference_answer)

            return {
                "model": model,
                "question": question_text,
                "reference_answer": reference_answer,
                "model_response": raw_output,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "token_usage": usage_tokens
            }

        futures = []
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            for itm in data_list:
                futures.append(executor.submit(process_one, itm))
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"GSM {model}", unit="q"):
                results_for_this_model.append(fut.result())

        correct = sum(r["is_correct"] for r in results_for_this_model)
        total = len(results_for_this_model)
        accuracy = correct / total if total else 0.0
        total_token_usage = sum(r["token_usage"] for r in results_for_this_model)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_file = f"./GSM_500_response/{model.replace('/', '_')}_{timestamp}.json"
        output_data = {
            "model": model,
            "timestamp": timestamp,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "sample_size": total,
            "accuracy": accuracy,
            "correct_count": correct,
            "total_token_usage": total_token_usage,
            "responses": results_for_this_model
        }
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[*] {model}: Accuracy={accuracy:.2%} ({correct}/{total}), "
              f"Total tokens={total_token_usage}, saved to {out_file}")

    logger.info("Finished processing all models.")


if __name__ == "__main__":
    main()
