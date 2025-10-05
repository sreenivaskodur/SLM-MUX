#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

from tqdm import tqdm

# GPQA-specific tools (for building prompts and answer extraction)
from utils.GPQA_utils import build_prompt_gpqa, extract_answer_gpqa
# Together API calling tools
from utils.together_utils import call_model_together


##############################################################################
# 1. Prompt configuration (original only)
##############################################################################
def build_prompt(model_name: str, question_text: str, choices: list):
    # Always use the original GPQA prompt builder
    return build_prompt_gpqa(question_text, choices)


##############################################################################
# 2. Multiple calls to the same model (self-consistency voting)
##############################################################################
def repeated_calls_for_model(
    model_name: str,
    question_text: str,
    choices: list,
    first_raw_content: str,
    first_extracted: str,
    first_token_usage: dict,
    extra_calls: int,
):
    """
    Run the same model multiple times (total = extra_calls including the 1st one),
    record raw outputs, extracted answers, and token usage, then compute vote stats.

    Returns a dict:
      {
        "model_name": str,
        "calls": [ { round, raw_content, extracted_answer, token_usage }, ... ],
        "vote_counts": { answer: count, ... },
        "best_answer": str,
        "best_count": int,
        "total_usage": { prompt_tokens, completion_tokens, total_tokens }
      }
    """
    calls_info = []

    # Round 1 from outside
    calls_info.append({
        "round": 1,
        "raw_content": first_raw_content,
        "extracted_answer": first_extracted,
        "token_usage": first_token_usage,
    })

    # Subsequent rounds
    for i in range(2, extra_calls + 1):
        messages = build_prompt(model_name, question_text, choices)
        resp = call_model_together(
            model_name=model_name,
            messages=messages,
            temperature=0.3,
        )
        raw_content = resp.get("content", "")
        usage = resp.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        extracted = extract_answer_gpqa(raw_content, choices)

        calls_info.append({
            "round": i,
            "raw_content": raw_content,
            "extracted_answer": extracted,
            "token_usage": usage,
        })

    # Voting statistics
    ans_count = {}
    for c in calls_info:
        ans = c["extracted_answer"].strip().upper()
        ans_count[ans] = ans_count.get(ans, 0) + 1

    best_answer = max(ans_count, key=ans_count.get)
    best_count = ans_count[best_answer]

    # Accumulate token usage
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for c in calls_info:
        tu = c.get("token_usage", {})
        total_usage["prompt_tokens"] += tu.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += tu.get("completion_tokens", 0)
        total_usage["total_tokens"] += tu.get("total_tokens", 0)

    return {
        "model_name": model_name,
        "calls": calls_info,
        "vote_counts": ans_count,
        "best_answer": best_answer,
        "best_count": best_count,
        "total_usage": total_usage,
    }


##############################################################################
# 3. Single-problem processing + tie-breaking
##############################################################################
def find_majority_answer_tiebreak(records, ans_count, model_list):
    """
    Find the most frequent answer in ans_count. If there's a tie, prefer the answer
    proposed by the earliest model in model_list (based on records order).
    """
    if not ans_count:
        return "", 0

    max_count = max(ans_count.values())
    candidates = [ans for ans, ct in ans_count.items() if ct == max_count]
    if len(candidates) == 1:
        return candidates[0], max_count

    # Tie: prefer the model appearing first in model_list whose answer is in candidates
    for model_name in model_list:
        for r in records:
            if r["model"] == model_name:
                ans = r.get("extracted_answer", "").strip().upper()
                if ans in candidates:
                    return ans, max_count
    return candidates[0], max_count


def process_one_problem(question_text: str, choices: list, correct_answer: str, records: list, extra_calls: int, model_list: list):
    """
    Process one GPQA problem:
      - Inspect first-round answers from all models
      - If all agree, adopt it directly
      - Otherwise, perform self-consistency calls per model and choose the model
        with the highest self-consistency (tie-break by model order)
    """
    # Normalize correct answer letter
    correct_letter = (correct_answer or "").strip().upper()

    # First-round answers
    first_answers = [r.get("extracted_answer", "").strip().upper() for r in records]
    n_models = len(first_answers)

    # Count answers and resolve tie by model order
    ans_count = {}
    for ans in first_answers:
        ans_count[ans] = ans_count.get(ans, 0) + 1

    max_answer, max_count = find_majority_answer_tiebreak(records, ans_count, model_list)

    # If everyone agrees in the first pass
    if max_count == n_models:
        final_answer = max_answer
        is_correct = (final_answer == correct_letter and final_answer != "")

        model_records = []
        for r in records:
            model_name = r["model"]
            raw_content = r.get("model_response", "")
            extracted = r.get("extracted_answer", "")
            token_usage = r.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            model_records.append({
                "model_name": model_name,
                "calls": [
                    {
                        "round": 1,
                        "raw_content": raw_content,
                        "extracted_answer": extracted,
                        "token_usage": token_usage,
                    }
                ],
                "vote_counts": {extracted: n_models},
                "best_answer": extracted,
                "best_count": n_models,
                "total_usage": token_usage,
            })

        return {
            "question": question_text,
            "choices": choices,
            "correct_answer": correct_letter,
            "all_agree": True,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "selected_model": "all_agree",
            "reason": "all_models_same_in_first_call",
            "model_records": model_records,
        }

    # Otherwise: self-consistency per model
    model_records = []
    for r in records:
        model_name = r["model"]
        first_raw = r.get("model_response", "")
        first_ext = r.get("extracted_answer", "")
        first_usage = r.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        ret = repeated_calls_for_model(
            model_name=model_name,
            question_text=question_text,
            choices=choices,
            first_raw_content=first_raw,
            first_extracted=first_ext,
            first_token_usage=first_usage,
            extra_calls=extra_calls,
        )
        model_records.append(ret)

    # Pick the model with the highest self-consistency; tie-break via model_list order
    best_model = None
    best_answer = ""
    best_consistency = -1.0

    for mrec in model_records:
        sc = mrec["best_count"] / max(1, extra_calls)
        if sc > best_consistency:
            best_consistency = sc
            best_model = mrec["model_name"]
            best_answer = mrec["best_answer"]
        elif abs(sc - best_consistency) < 1e-9:
            if model_list.index(mrec["model_name"]) < model_list.index(best_model):
                best_model = mrec["model_name"]
                best_answer = mrec["best_answer"]

    is_correct = (best_answer.strip().upper() == correct_letter and best_answer.strip() != "")

    return {
        "question": question_text,
        "choices": choices,
        "correct_answer": correct_letter,
        "all_agree": False,
        "final_answer": best_answer,
        "is_correct": is_correct,
        "selected_model": best_model,
        "reason": f"highest_self_consistency({best_consistency:.2f})",
        "model_records": model_records,
    }


##############################################################################
# 4. First-pass calls for all models (per problem)
##############################################################################
def get_initial_responses(question_text: str, choices: list, correct_answer: str, model_list: list):
    records = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for model_name in model_list:
        messages = build_prompt(model_name, question_text, choices)
        resp = call_model_together(
            model_name=model_name,
            messages=messages,
            temperature=0.3,
        )

        raw = resp.get("content", "")
        extracted = extract_answer_gpqa(raw, choices)
        usage = resp.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        records.append({
            "model": model_name,
            "extracted_answer": extracted,
            "correct_answer": correct_answer,
            "model_response": raw,
            "token_usage": usage,
        })

        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)

    return records, total_usage


def process_problem_wrapper(question_text, choices, correct_answer, model_list, extra_calls):
    initial_records, _ = get_initial_responses(question_text, choices, correct_answer, model_list)

    result = process_one_problem(
        question_text=question_text,
        choices=choices,
        correct_answer=correct_answer,
        records=initial_records,
        extra_calls=extra_calls,
        model_list=model_list,
    )

    return result


##############################################################################
# 5. Main: dataset loading, parallel processing, stats, output
##############################################################################
# Removed JSON prompt mapping support; original prompt mode only


def main(dataset_file, model_list_str, extra_calls, size, num_workers):
    # Parse model list
    if not model_list_str or model_list_str.strip() == "":
        model_list = []
    else:
        model_list = [m.strip() for m in model_list_str.split(",") if m.strip()]

    if not model_list:
        print("Model list is empty. Please provide models via --model_list.")
        return

    # Ensure dataset exists
    if not os.path.isfile(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return

    # Load dataset (json or jsonl). Expect fields: question (str), choices (list[str]), answer (letter)
    problems = []
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            if dataset_file.endswith(".jsonl"):
                for line in f:
                    data = json.loads(line)
                    problems.append({
                        "question": data["question"],
                        "choices": data["choices"],
                        "answer": data.get("answer") or data.get("correct_answer") or data.get("label"),
                    })
            elif dataset_file.endswith(".json"):
                full_data = json.load(f)
                # Might be either a list of items or an object with "data" list
                if isinstance(full_data, dict) and "data" in full_data:
                    iterable = full_data["data"]
                else:
                    iterable = full_data
                for item in iterable:
                    problems.append({
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item.get("answer") or item.get("correct_answer") or item.get("label"),
                    })
            else:
                print(f"Unsupported file format: {dataset_file}. Please use .json or .jsonl")
                return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {dataset_file}: {e}")
        return
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return

    # Random sampling if requested
    if size is not None and size > 0 and len(problems) > size:
        print(f"Randomly sampling {size} problems from the dataset.")
        problems = random.sample(problems, size)

    problem_count = len(problems)
    if problem_count == 0:
        print("No problems to process.")
        return

    results = []
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for p in problems:
            qtext = p["question"]
            choices = p["choices"]
            correct_ans = (p.get("answer") or "").strip().upper()
            fut = executor.submit(
                process_problem_wrapper,
                qtext,
                choices,
                correct_ans,
                model_list,
                extra_calls,
            )
            futures[fut] = qtext

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing GPQA problems"):
            try:
                result = fut.result()
            except Exception as e:
                print(f"[Worker Error] {e}")
                continue

            results.append(result)

            # Sum token usage over all models' total_usage within this problem
            for mrec in result.get("model_records", []):
                usage = mrec.get("total_usage", {})
                total_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_token_usage["total_tokens"] += usage.get("total_tokens", 0)

    # Accuracy
    correct_count = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct_count / problem_count if problem_count else 0.0

    # Output directory and file
    output_dir = "./GPQA_Weak_Compositions"
    os.makedirs(output_dir, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fn = os.path.join(output_dir, f"GPQA_FromScratch_{now_str}.json")

    output_data = {
        "strategy": "GPQA_From_Scratch",
        "timestamp": now_str,
        "problem_count": problem_count,
        "extra_calls": extra_calls,
        "size": size,
        "model_list": model_list,
        "token_usage": total_token_usage,
        "results": results,
        "accuracy": accuracy,
    }

    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Processed {problem_count} problems.")
    print(f"Final Accuracy: {accuracy:.2%} ({correct_count}/{problem_count})")
    print("Output JSON =>", out_fn)
    print(f"Total Token Usage: {total_token_usage['total_tokens']} tokens")
    print("=" * 80)


##############################################################################
# 6. CLI entry
##############################################################################
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    default_dataset = os.path.join(parent_dir, "Datasets/shuffled_gpqa.json")

    parser = argparse.ArgumentParser(description="Run GPQA from scratch by calling Together models with self-consistency.")
    parser.add_argument("--dataset_file", default=default_dataset, help="Path to the GPQA dataset (.json or .jsonl).")
    parser.add_argument(
        "--model_list",
        default="google/gemma-2-27b-it,Mistralai/Mistral-Small-24B-Instruct-2501",
        help="Comma-separated Together model names",
    )
    parser.add_argument("--extra_calls", type=int, default=3, help="Total calls per model for self-consistency (including first).")
    parser.add_argument("--size", type=int, default=-1, help="Max number of problems to process. 0 or negative => all.")
    parser.add_argument("--num_workers", type=int, default=5, help="Parallel workers for problem processing.")
    args = parser.parse_args()

    main(
        dataset_file=args.dataset_file,
        model_list_str=args.model_list,
        extra_calls=args.extra_calls,
        size=args.size,
        num_workers=args.num_workers,
    )


