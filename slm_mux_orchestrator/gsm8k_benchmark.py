#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from tqdm import tqdm

# Project utils
from utils.together_utils import call_model_together
from utils.GSM_utils import extract_answer_gsm, normalize_answer, create_chat_prompt

##############################################################################
# Prompt builder (original style, aligned with collector)
##############################################################################

def build_prompt(question_text: str):
    """Build chat messages for GSM: reason step-by-step and output final answer prefixed with ####.

    Mirrors Collect/collect_GSM_temp05_runs3.py
    """
    system_msg = (
        "You are a helpful math tutor. Reason step by step to solve the problem. "
        "After your reasoning, output the final answer on a new line prefixed with '####'. "
        "Example: #### 42"
    )
    user_msg = f"Problem:\n{question_text}\n\nProvide your reasoning, then the final answer."
    return create_chat_prompt(system_msg, user_msg)


##############################################################################
# Repeated calls (self-consistency per model)
##############################################################################

def repeated_calls_for_model(
    model_name: str,
    question_text: str,
    first_raw_content: str,
    first_extracted: str,
    first_token_usage: dict,
    extra_calls: int,
    temperature: float,
    max_new_tokens: int,
):
    calls_info = []

    calls_info.append({
        "round": 1,
        "raw_content": first_raw_content,
        "extracted_answer": first_extracted,
        "token_usage": first_token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })

    for i in range(2, extra_calls + 1):
        messages = build_prompt(question_text)
        response_dict = call_model_together(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        usage = response_dict.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        raw_content = response_dict.get("content", "")
        extracted = normalize_answer(extract_answer_gsm(raw_content))
        calls_info.append({
            "round": i,
            "raw_content": raw_content,
            "extracted_answer": extracted,
            "token_usage": usage,
        })

    # count votes, preferring non-empty
    ans_count = {}
    for cinfo in calls_info:
        ans = cinfo["extracted_answer"] or ""
        ans_count[ans] = ans_count.get(ans, 0) + 1

    best_answer = ""
    best_count = 0
    if ans_count:
        # Prefer non-empty answers; if none, falls back to empty
        non_empty = {k: v for k, v in ans_count.items() if k.strip() != ""}
        use_counts = non_empty if non_empty else ans_count
        best_answer = max(use_counts, key=use_counts.get)
        best_count = use_counts[best_answer]

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for cinfo in calls_info:
        tu = cinfo["token_usage"] or {}
        total_usage["prompt_tokens"] += int(tu.get("prompt_tokens", 0) or 0)
        total_usage["completion_tokens"] += int(tu.get("completion_tokens", 0) or 0)
        total_usage["total_tokens"] += int(tu.get("total_tokens", 0) or 0)

    return {
        "model_name": model_name,
        "calls": calls_info,
        "vote_counts": ans_count,
        "best_answer": best_answer,
        "best_count": best_count,
        "total_usage": total_usage,
    }


##############################################################################
# Tie-break helper (prefer earlier model order when counts tie)
##############################################################################

def find_majority_answer_tiebreak(records, ans_count, model_list):
    max_count = max(ans_count.values()) if ans_count else 0
    tie_candidates = [ans for ans, ct in ans_count.items() if ct == max_count]
    if len(tie_candidates) == 1:
        return tie_candidates[0], max_count

    for model_name in model_list:
        for r in records:
            if r["model"] == model_name:
                ans = r.get("extracted_answer", "")
                if ans in tie_candidates:
                    return ans, max_count
    return (tie_candidates[0] if tie_candidates else ""), max_count


##############################################################################
# Initial calls and per-question processing
##############################################################################

def get_initial_responses(question_text: str, reference_answer: str, model_list: list, temperature: float, max_new_tokens: int):
    records = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for model_name in model_list:
        messages = build_prompt(question_text)
        response_dict = call_model_together(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        raw_content = response_dict.get("content", "")
        extracted = normalize_answer(extract_answer_gsm(raw_content))
        usage = response_dict.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        records.append({
            "model": model_name,
            "extracted_answer": extracted,
            "reference_answer": reference_answer,
            "model_response": raw_content,
            "token_usage": usage,
        })

        total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        total_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

    return records, total_usage


def process_one_question(question_text: str, reference_answer: str, records: list, extra_calls: int, model_list: list, temperature: float, max_new_tokens: int, sc_after_unanimous: bool):
    reference_norm = normalize_answer(reference_answer)

    first_answers = [r.get("extracted_answer", "") for r in records]
    n_models = len(first_answers)

    ans_count = {}
    for ans in first_answers:
        ans_count[ans] = ans_count.get(ans, 0) + 1

    max_answer, max_count = find_majority_answer_tiebreak(records, ans_count, model_list)
    all_agree_flag = (max_count == n_models)

    if all_agree_flag and not sc_after_unanimous:
        final_answer = max_answer
        correct_flag = (normalize_answer(final_answer) == reference_norm and final_answer.strip() != "")

        model_records = []
        for r in records:
            model_name = r["model"]
            raw_content = r.get("model_response", "")
            extracted = r.get("extracted_answer", "")
            token_usage = r.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            model_records.append({
                "model_name": model_name,
                "calls": [{
                    "round": 1,
                    "raw_content": raw_content,
                    "extracted_answer": extracted,
                    "token_usage": token_usage,
                }],
                "vote_counts": {extracted: n_models},
                "best_answer": extracted,
                "best_count": n_models,
                "total_usage": token_usage,
            })

        return {
            "question": question_text,
            "reference_answer": reference_answer,
            "all_agree": True,
            "final_answer": final_answer,
            "is_correct": correct_flag,
            "selected_model": "all_agree",
            "reason": "all_models_same_in_first_call",
            "model_records": model_records,
        }

    model_records = []
    for r in records:
        model_name = r["model"]
        first_raw_content = r.get("model_response", "")
        first_extracted = r.get("extracted_answer", "")
        first_token_usage = r.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        ret = repeated_calls_for_model(
            model_name=model_name,
            question_text=question_text,
            first_raw_content=first_raw_content,
            first_extracted=first_extracted,
            first_token_usage=first_token_usage,
            extra_calls=extra_calls,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        model_records.append(ret)

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

    correct_flag = (normalize_answer(best_answer) == reference_norm and best_answer.strip() != "")

    return {
        "question": question_text,
        "reference_answer": reference_answer,
        "all_agree": all_agree_flag,
        "final_answer": best_answer,
        "is_correct": correct_flag,
        "selected_model": best_model,
        "reason": f"highest_self_consistency({best_consistency:.2f})",
        "model_records": model_records,
    }


##############################################################################
# Dataset loading helpers
##############################################################################

def load_gsm_dataset(dataset_file: str):
    """Load GSM dataset from JSON (array) or JSONL (objects per line).

    Expected keys: question, answer
    """
    items = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        if dataset_file.endswith(".jsonl"):
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = obj.get("question")
                a = obj.get("answer")
                if not q or a is None:
                    continue
                items.append({
                    "question": q,
                    "reference_answer": normalize_answer(extract_answer_gsm(a)),
                })
        elif dataset_file.endswith(".json"):
            data = json.load(f)
            for obj in data:
                q = obj.get("question")
                a = obj.get("answer")
                if not q or a is None:
                    continue
                items.append({
                    "question": q,
                    "reference_answer": normalize_answer(extract_answer_gsm(a)),
                })
        else:
            raise ValueError("Unsupported file format. Use .json or .jsonl")
    return items


##############################################################################
# Wrapper for parallel execution
##############################################################################

def process_question_wrapper(qtext, ref_ans, model_list, extra_calls, temperature, max_new_tokens, sc_after_unanimous):
    initial_records, _initial_usage = get_initial_responses(qtext, ref_ans, model_list, temperature, max_new_tokens)
    result = process_one_question(
        question_text=qtext,
        reference_answer=ref_ans,
        records=initial_records,
        extra_calls=extra_calls,
        model_list=model_list,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        sc_after_unanimous=sc_after_unanimous,
    )
    return result


##############################################################################
# Main
##############################################################################

def main(dataset_file, model_list_str, extra_calls, size, num_workers, temperature, max_new_tokens, sc_after_unanimous):
    if not model_list_str or model_list_str.strip() == "":
        model_list = []
    else:
        model_list = [m.strip() for m in model_list_str.split(",") if m.strip()]

    if not model_list:
        print("Model list is empty. Please provide models via --model_list.")
        return

    if not os.path.isfile(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return

    try:
        dataset_items = load_gsm_dataset(dataset_file)
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return

    if size is not None and size > 0 and len(dataset_items) > size:
        print(f"Randomly sampling {size} questions from the dataset.")
        dataset_items = random.sample(dataset_items, size)

    question_count = len(dataset_items)
    if question_count == 0:
        print("No questions to process.")
        return

    results = []
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for item in dataset_items:
            qtext = item["question"]
            ref_ans = item.get("reference_answer", "")
            fut = executor.submit(process_question_wrapper, qtext, ref_ans, model_list, extra_calls, temperature, max_new_tokens, sc_after_unanimous)
            futures[fut] = qtext

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing GSM questions"):
            try:
                result = fut.result()
            except Exception as e:
                print(f"[Worker Error] {e}")
                continue
            results.append(result)

            for mrec in result.get("model_records", []):
                usage = mrec.get("total_usage", {})
                total_token_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                total_token_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                total_token_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

    correct_count = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct_count / question_count if question_count else 0.0

    output_dir = "./EXP_Weak_Compositions"
    os.makedirs(output_dir, exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fn = os.path.join(output_dir, f"GSM_FromScratch_{now_str}.json")

    output_data = {
        "strategy": "Hierarchy_Consistency_From_Scratch_GSM",
        "timestamp": now_str,
        "question_count": question_count,
        "extra_calls": extra_calls,
        "size": size,
        "model_list": model_list,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "token_usage": total_token_usage,
        "results": results,
        "accuracy": accuracy,
    }

    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Processed {question_count} questions.")
    print(f"Final Accuracy: {accuracy:.2%} ({correct_count}/{question_count})")
    print("Output JSON =>", out_fn)
    print(f"Total Token Usage: {total_token_usage['total_tokens']} tokens")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # Auto-run equivalence checker using the generated output JSON
    # This will update the same file in place with equivalence-based metrics
    # Requires OPENAI_API_KEY to be set
    # ----------------------------------------------------------------------
    try:
        if os.getenv("OPENAI_API_KEY"):
            import importlib.util
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            eq_file = os.path.join(root_dir, "Equal_form", "check_equal_form_all_GSM.py")
            if os.path.isfile(eq_file):
                print("Running equivalence check (Equal_form/check_equal_form_all_GSM.py)...")
                spec = importlib.util.spec_from_file_location("check_equal_form_all_GSM", eq_file)
                eq_mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(eq_mod)
                asyncio.run(eq_mod.process_all_results(out_fn, out_fn))
                print("Equivalence check completed and file updated.")
            else:
                print(f"Equivalence checker not found at {eq_file}; skipping.")
        else:
            print("OPENAI_API_KEY not set; skipping equivalence check.")
    except Exception as e:
        print(f"Equivalence check failed: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    default_dataset = os.path.join(parent_dir, "Datasets/GSM_500.jsonl")

    parser = argparse.ArgumentParser(description="Run Hierarchy Consistency for GSM from scratch by calling models.")
    parser.add_argument("--dataset_file", default=default_dataset, help="Path to the GSM dataset file (.json or .jsonl)")
    parser.add_argument("--model_list", default="Mistralai/Mistral-Small-24B-Instruct-2501,Qwen/Qwen2.5-7B-Instruct-Turbo", help="Comma-separated model names")
    parser.add_argument("--extra_calls", type=int, default=3, help="Total calls per model for self-consistency (including the first)")
    parser.add_argument("--size", type=int, default=-1, help="Max number of questions to process. 0 or negative => all")
    parser.add_argument("--num_workers", type=int, default=10, help="Parallel workers for question processing")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default 0.3)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens (collector default)")
    parser.add_argument("--sc_after_unanimous", action="store_true", help="If set, continue self-consistency even when first round is unanimous (default)")
    parser.add_argument("--no-sc_after_unanimous", dest="sc_after_unanimous", action="store_false", help="If set, early-exit when first round is unanimous.")
    parser.set_defaults(sc_after_unanimous=True)
    args = parser.parse_args()

    main(
        dataset_file=args.dataset_file,
        model_list_str=args.model_list,
        extra_calls=args.extra_calls,
        size=args.size,
        num_workers=args.num_workers,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        sc_after_unanimous=args.sc_after_unanimous,
    )
