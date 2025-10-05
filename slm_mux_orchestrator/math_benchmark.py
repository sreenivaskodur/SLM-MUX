#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import math
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

##############################################################################
# Removed Google/Gemini related imports and initialization logic, using Together API instead.
##############################################################################

# Assume these two functions are available: extract_answer_math, normalize_answer
# from utils.math_utils import extract_answer_math, normalize_answer

# This is just an example, you need to actually import it in your project:
from utils.math_utils import extract_answer_math, normalize_answer

##############################################################################
# Call Together API
##############################################################################
from utils.together_utils import call_model_together

##############################################################################
# 1. Prompt Generation Related
##############################################################################

def create_chat_prompt(system_msg: str, user_msg: str):
    """
    Returns a dialog structure like:
    [
      {"role": "system", "content": system_msg},
      {"role": "user",   "content": user_msg},
    ]
    which can be directly passed to call_model_together().
    """
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

def replace_empty_with_random(ans: str) -> str:
    """If extracted answer is empty, replace with a random integer in [-1_000_000, -10_000]."""
    if not isinstance(ans, str):
        ans = "" if ans is None else str(ans)
    if ans.strip() == "":
        return str(random.randint(-1_000_000, -10_000))
    return ans

def build_prompt(problem_text: str):
    """Return a chat prompt."""
    system_msg = (
        "You are a helpful math assistant. "
        "Solve the following problem step by step. "
        "Please provide your final answer enclosed in LaTeX \\\\boxed{...}."
    )
    user_msg = f"Problem:\\n{problem_text}\\n\\nPlease provide the final answer in the form: \\\\boxed{{...}}"
    return create_chat_prompt(system_msg, user_msg)

##############################################################################
# 2. Repeated calls for the same model, recording detailed information
##############################################################################
def repeated_calls_for_model(
    model_name: str,
    problem_text: str,
    first_raw_content: str,
    first_extracted: str,
    first_token_usage: dict, # New
    first_prompt: list,
    extra_calls: int,
    temperature: float,
):
    """
    Repeatedly call the same model, recording raw_content + extracted_answer + token_usage for each call.
    extra_calls indicates the total number of calls, including the first one (which already exists externally).
    
    Example of the returned structure:
    {
      "model_name": "xxx",
      "calls": [
        {
          "round": 1,
          "raw_content": "...",
          "extracted_answer": "...",
          "token_usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
        },
        ...
      ],
      "vote_counts": {"ans1": 2, "ans2": 1, ...},
      "best_answer": "ans1",
      "best_count": 2,
      "total_usage": {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    }
    """
    calls_info = []

    # Round 1 data comes from outside
    # Ensure first extracted answer is non-empty by replacement
    first_extracted = replace_empty_with_random(first_extracted)

    calls_info.append({
        "round": 1,
        "raw_content": first_raw_content,
        "extracted_answer": first_extracted,
        "token_usage": first_token_usage,
        "prompt": first_prompt
    })

    # Subsequent (extra_calls-1) rounds
    for i in range(2, extra_calls + 1):
        messages = build_prompt(problem_text)
        response_dict = call_model_together(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
        )
        usage = response_dict.get("token_usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
        raw_content = response_dict.get("content", "")
        extracted = extract_answer_math(raw_content)
        extracted = replace_empty_with_random(extracted)
        calls_info.append({
            "round": i,
            "raw_content": raw_content,
            "extracted_answer": extracted,
            "token_usage": usage,
            "prompt": messages
        })

    # Count votes
    ans_count = {}
    for cinfo in calls_info:
        ans = cinfo["extracted_answer"].strip()
        ans_count[ans] = ans_count.get(ans, 0) + 1

    # Find the most frequent answer
    # Note: This is voting on multiple answers from the same model, there is no tie-breaking between models, so model_list information is not needed here.
    best_answer = max(ans_count, key=ans_count.get)
    best_count = ans_count[best_answer]

    # Accumulate token usage
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    for cinfo in calls_info:
        tu = cinfo["token_usage"]
        total_usage["prompt_tokens"] += tu.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += tu.get("completion_tokens", 0)
        total_usage["total_tokens"] += tu.get("total_tokens", 0)

    return {
        "model_name": model_name,
        "calls": calls_info,
        "vote_counts": ans_count,
        "best_answer": best_answer,
        "best_count": best_count,
        "total_usage": total_usage
    }

##############################################################################
# 3. Process a single problem + Tie-Breaking
##############################################################################

def find_majority_answer_tiebreak(records, ans_count, model_list):
    """
    Find the most frequent answer in the first-round ans_count (answer -> count).
    If there is more than one answer with the highest frequency (a tie), the answer from the model that appears first in the model_list is prioritized.
    """
    max_count = max(ans_count.values()) if ans_count else 0
    # Find all answers with a count equal to max_count
    tie_candidates = [ans for ans, ct in ans_count.items() if ct == max_count]
    if len(tie_candidates) == 1:
        # No tie
        return tie_candidates[0], max_count

    # A tie exists, need to decide priority based on model_list order
    # Idea: In the order of model_list, find the first model that gives one of the tie_candidate answers,
    # and return that answer.
    for model_name in model_list:
        # Find the extracted_answer corresponding to this model
        for r in records:
            if r["model"] == model_name:
                ans = r["extracted_answer"]
                if ans in tie_candidates:
                    return ans, max_count
    # If not found, just return tie_candidates[0]
    # This case should ideally not happen if tie_candidates are from records.
    # But as a fallback:
    if tie_candidates:
        return tie_candidates[0], max_count
    return "", max_count # Should not be reached if ans_count is not empty


def process_one_problem(problem_text: str, records: list, extra_calls: int, model_list: list, sc_after_unanimous: bool, temperature: float):
    """
    Process one problem: including multiple calls for all models, recording details, and selecting the final answer.
    
    records: First-round (already completed) answers for this problem from all models in the external JSON, structured as:
    [
      {
        "model": "xxx",
        "extracted_answer": "...",
        "reference_answer": "...",
        "model_response": "..."
      },
      ...
    ]
    model_list: User-specified list of models, used for tie-breaking.
    """
    # First, get the reference_answer (it's sufficient to get it from the first record, assuming they are all the same)
    reference_answer = records[0].get("reference_answer", "A")
    reference_norm = normalize_answer(extract_answer_math(reference_answer))

    # Collect first-round answers
    first_answers = [r.get("extracted_answer", "") for r in records]
    n_models = len(first_answers)

    # Calculate the frequency of each answer in the first round
    ans_count = {}
    for ans in first_answers:
        ans_count[ans] = ans_count.get(ans, 0) + 1

    # Find the most frequent answer + tie-breaking
    max_answer, max_count = find_majority_answer_tiebreak(records, ans_count, model_list)
    
    # Check if all agree
    all_agree_flag = (max_count == n_models)

    # If all agree in the first round and the user chose to finish early upon agreement, take the early exit branch
    if all_agree_flag and not sc_after_unanimous:
        final_answer = max_answer
        correct_flag = (
            normalize_answer(final_answer) == reference_norm
            and final_answer.strip() != ""
        )

        # Only record one round of data
        model_records = []
        for r in records:
            model_name = r["model"]
            raw_content = r.get("model_response", "")
            extracted = r.get("extracted_answer", "")
            token_usage = r.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            prompt = r.get("prompt", [])

            model_records.append({
                "model_name": model_name,
                "calls": [
                    {
                        "round": 1,
                        "raw_content": raw_content,
                        "extracted_answer": extracted,
                        "token_usage": token_usage,
                        "prompt": prompt
                    }
                ],
                "vote_counts": {extracted: n_models},
                "best_answer": extracted,
                "best_count": n_models,
                "total_usage": token_usage
            })

        return {
            "problem": problem_text,
            "reference_answer": reference_answer,
            "all_agree": True,
            "final_answer": final_answer,
            "is_correct": correct_flag,
            "selected_model": "all_agree",
            "reason": "all_models_same_in_first_call",
            "model_records": model_records
        }

    # Perform multiple calls (self-consistency), continue sampling even if the first round is unanimous
    model_records = []
    for r in records:
        model_name = r["model"]
        first_raw_content = r.get("model_response", "")
        first_extracted = r.get("extracted_answer", "")
        first_token_usage = r.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        first_prompt = r.get("prompt", [])

        ret = repeated_calls_for_model(
            model_name=model_name,
            problem_text=problem_text,
            first_raw_content=first_raw_content,
            first_extracted=first_extracted,
            first_token_usage=first_token_usage,
            first_prompt=first_prompt,
            extra_calls=extra_calls,
            temperature=temperature,
        )
        model_records.append(ret)

    # After multiple calls, select the model answer with the highest self-consistency.
    # Add Tie-break: If multiple models have the same sc, prioritize the one that appears earlier in model_list.
    best_model = None
    best_answer = ""
    best_consistency = -1.0

    for mrec in model_records:
        sc = mrec["best_count"] / extra_calls
        if sc > best_consistency:
            best_consistency = sc
            best_model = mrec["model_name"]
            best_answer = mrec["best_answer"]
        elif abs(sc - best_consistency) < 1e-9:
            # Tie occurred: see who comes first in model_list
            if model_list.index(mrec["model_name"]) < model_list.index(best_model):
                best_model = mrec["model_name"]
                best_answer = mrec["best_answer"]

    correct_flag = (
        normalize_answer(best_answer) == reference_norm
        and best_answer.strip() != ""
    )

    return {
        "problem": problem_text,
        "reference_answer": reference_answer,
        "all_agree": all_agree_flag,
        "final_answer": best_answer,
        "is_correct": correct_flag,
        "selected_model": best_model,
        "reason": f"highest_self_consistency({best_consistency:.2f})",
        "model_records": model_records
    }

##############################################################################
# 4. Main process + Accuracy statistics
##############################################################################
def get_initial_responses(problem_text: str, reference_answer: str, model_list: list, temperature: float):
    """
    For a given problem, call all models in model_list to get initial responses.
    """
    records = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for model_name in model_list:
        messages = build_prompt(problem_text)
        response_dict = call_model_together(
            model_name=model_name,
            messages=messages,
            temperature=temperature
        )
        
        raw_content = response_dict.get("content", "")
        extracted = extract_answer_math(raw_content)
        extracted = replace_empty_with_random(extracted)
        usage = response_dict.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        records.append({
            "model": model_name,
            "extracted_answer": extracted,
            "reference_answer": reference_answer,
            "model_response": raw_content,
            "token_usage": usage, # Include token usage for the first call
            "prompt": messages
        })
        
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        
    return records, total_usage

def process_problem_wrapper(ptext, ref_ans, model_list, extra_calls, sc_after_unanimous, temperature):
    """
    Wrapper function for parallel execution.
    Gets initial responses and then calls the main processing function.
    """
    initial_records, initial_usage = get_initial_responses(ptext, ref_ans, model_list, temperature)
    
    result = process_one_problem(
        problem_text=ptext, 
        records=initial_records, 
        extra_calls=extra_calls, 
        model_list=model_list,
        sc_after_unanimous=sc_after_unanimous,
        temperature=temperature,
    )
    
    # Add initial token usage to the total usage reported in the result
    if "model_records" in result:
        total_p_tokens = initial_usage['prompt_tokens']
        total_c_tokens = initial_usage['completion_tokens']
        
        # If self-consistency was triggered, add those tokens as well
        if not result.get("all_agree") and not result.get("reason", "").startswith(">=2/3"):
            for mrec in result["model_records"]:
                # The first call's prompt tokens are already in initial_usage.
                # The prompt tokens for subsequent calls in self-consistency are what we need to add.
                # However, repeated_calls_for_model recalculates the total usage for that model's calls.
                # Let's adjust the logic in process_one_problem to be cleaner.
                # For now, let's just sum up what's there.
                # This part will be tricky. Let's re-evaluate how tokens are summed.
                pass # Token logic will be fixed in `process_one_problem`

    # For now, we'll let process_one_problem handle all token summations.
    # We pass the initial records with their usage, and it should do the rest.
    return result


def main(dataset_file, model_list_str, extra_calls, size, num_workers, sc_after_unanimous, temperature):
    """
    Core logic:
    1) Read problems from the dataset file
    2) For each problem, call all models to get the first-round answers
    3) Process problems in parallel
    4) Calculate the final overall accuracy
    5) Output the results in JSON format
    """
    # Determine which model list to use
    if not model_list_str or model_list_str.strip() == "":
        model_list = []
    else:
        model_list = [m.strip() for m in model_list_str.split(",") if m.strip()]

    if not model_list:
        print("Model list is empty. Please provide models via --model_list or a valid --summary_file.")
        return

    # Ensure dataset_file exists
    if not os.path.isfile(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return

    # Read dataset
    problems = []
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            # Check file extension to decide how to load
            if dataset_file.endswith(".jsonl"):
                for line in f:
                    data = json.loads(line)
                    problems.append({
                        "problem": data["problem"],
                        "reference_answer": data.get("solution") or data.get("reference_answer")
                    })
            elif dataset_file.endswith(".json"):
                # Load the entire JSON file which is expected to be a list of objects
                full_data = json.load(f)
                for item in full_data:
                    problems.append({
                        "problem": item["problem"],
                        "reference_answer": item.get("solution") or item.get("reference_answer")
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


    # Select problems to process (size items), changed to random sampling
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
            ptext = p["problem"]
            ref_ans = p["reference_answer"]
            fut = executor.submit(process_problem_wrapper, ptext, ref_ans, model_list, extra_calls, sc_after_unanimous, temperature)
            futures[fut] = ptext

        # Use tqdm to show progress
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
            try:
                result = fut.result()
            except Exception as e:
                print(f"[Worker Error] {e}")
                continue
            results.append(result)

            # Accumulate tokens
            for mrec in result.get("model_records", []):
                usage = mrec.get("total_usage", {})
                total_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_token_usage["total_tokens"] += usage.get("total_tokens", 0)


    # Calculate final accuracy
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = correct_count / problem_count if problem_count else 0.0

    # Modify save directory
    output_dir = "./EXP_Weak_Compositions"
    os.makedirs(output_dir, exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fn = os.path.join(output_dir, f"MATH_FromScratch_{now_str}.json")

    output_data = {
        "strategy": "Hierarchy_Consistency_From_Scratch",
        "timestamp": now_str,
        "problem_count": problem_count,
        "extra_calls": extra_calls,
        "temperature": temperature,
        "size": size,
        "model_list": model_list,
        "token_usage": total_token_usage,
        "results": results,
        "accuracy": accuracy
    }

    # Write file
    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print information
    print("=" * 80)
    print(f"Processed {problem_count} problems.")
    print(f"Final Accuracy: {accuracy:.2%} ({correct_count}/{problem_count})")
    print("Output JSON =>", out_fn)
    print(f"Total Token Usage: {total_token_usage['total_tokens']} tokens")
    print("=" * 80)

    # Automatically call the equivalence check script, using the JSON generated this time as both input and output
    try:
        import sys
        import subprocess
        script_dir_local = os.path.dirname(os.path.abspath(__file__))
        parent_dir_local = os.path.dirname(script_dir_local)
        checker_path = os.path.join(parent_dir_local, "Equal_form", "check_equal_form_all.py")
        if os.path.isfile(checker_path):
            print("Running equivalence checker on the output JSON...")
            subprocess.run([sys.executable, checker_path, "-i", out_fn, "-o", out_fn], check=True)
        else:
            print(f"Equivalence checker not found at {checker_path}. Skipping equivalence validation.")
    except Exception as e:
        print(f"Failed to run equivalence checker: {e}")


##############################################################################
# 5. Command-line entry point
##############################################################################
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    # Assumed new dataset file path
    default_dataset = os.path.join(parent_dir, "Datasets/MATH_500.json")
    default_summary = os.path.join(parent_dir, "prompt_tuning_results/MATH_multi_krep/2025-09-03_15-35-48/summary.json")

    parser = argparse.ArgumentParser(description="Run Hierarchy Consistency from scratch by calling models.")
    parser.add_argument("--dataset_file", default=default_dataset,
                        help="Path to the .jsonl dataset file.")
    parser.add_argument("--model_list", default="qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-Small-24B-Instruct-2501", 
                        help="Comma-separated model names, e.g. 'model1,model2'")

    parser.add_argument("--extra_calls", type=int, default=5,
                        help="Total calls per model for self-consistency (if needed).")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature passed to Together API calls.")
    parser.add_argument("--size", type=int, default=-1,
                        help="Max number of problems to process. 0 or negative => unlimited.")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Parallel workers for problem processing.")
    parser.add_argument("--sc_after_unanimous", action="store_true",
                        help="If set, continue self-consistency even when first round is unanimous (default).")
    parser.add_argument("--no-sc_after_unanimous", dest="sc_after_unanimous", action="store_false",
                        help="If set, early-exit when first round is unanimous.")
    parser.set_defaults(sc_after_unanimous=True)
    args = parser.parse_args()

    main(
        dataset_file=args.dataset_file,
        model_list_str=args.model_list,
        extra_calls=args.extra_calls,
        size=args.size,
        num_workers=args.num_workers,
        sc_after_unanimous=args.sc_after_unanimous,
        temperature=args.temperature,
    )
