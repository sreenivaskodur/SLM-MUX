import openai
import json
import os
import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor

# For MATH_FromScratch*.json: Use GPT-4o to judge equivalence for each question (check all questions)

API_KEY = os.getenv("OPENAI_API_KEY")


def init_openai_client(api_key: str):
    """
    Initializes and returns a new OpenAI client.
    """
    client = openai.OpenAI(api_key=api_key)
    return client


client = init_openai_client(API_KEY)


def check_equivalence(reference: str, extracted: str) -> str:
    """
    Calls GPT-4o to determine if the two are mathematically equivalent.
    Returns only 'Yes' or 'No'.
    """
    prompt = f"""
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
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced mathematics teacher evaluating student responses. "
                    "Your task is to determine if a student's answer is mathematically equivalent to "
                    "the standard answer. You should mark an answer as correct if it represents "
                    "the same mathematical value or function, even if the notation differs. "
                    "Mathematical equivalence is what matters, not identical notation. "
                    "Answer only with 'Yes' or 'No'."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=5,
    )

    return response.choices[0].message.content.strip()


async def check_equivalence_async(reference: str, extracted: str) -> str:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, check_equivalence, reference, extracted)


async def process_all_results(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    total_count = len(results)
    correct_count = 0

    # Create a task for each problem (requires checking every problem)
    tasks = []
    valid_indices = []
    skipped_missing_reference = 0
    skipped_missing_final = 0
    skipped_missing_both = 0
    counted_missing_final = 0
    for idx, item in enumerate(results):
        reference_answer = item.get("reference_answer", "")
        final_answer = item.get("final_answer", "")
        has_ref = bool(str(reference_answer).strip())
        has_final = bool(str(final_answer).strip())
        if has_ref and has_final:
            valid_indices.append(idx)
            tasks.append(check_equivalence_async(reference_answer, final_answer))
        elif has_ref and not has_final:
            # final is empty/whitespace: counted as checked and incorrect
            item["is_correct"] = False
            counted_missing_final += 1
        else:
            if not has_ref and not has_final:
                skipped_missing_both += 1
            elif not has_ref:
                skipped_missing_reference += 1
            else:
                skipped_missing_final += 1

    checks = await asyncio.gather(*tasks) if tasks else []

    # Backfill the judgment results and statistics
    for idx, verdict in zip(valid_indices, checks):
        is_yes = (verdict or "").strip().lower() == "yes"
        results[idx]["is_correct"] = bool(is_yes)
        if is_yes:
            correct_count += 1

    # If there are questions that cannot be judged due to missing fields, keep their original is_correct status unchanged and do not include them in correct_count

    # Update statistics (calculated only based on successfully verified items)
    checked_count = len(valid_indices) + counted_missing_final
    skipped_count = total_count - checked_count
    data["checked_count"] = checked_count
    data["skipped_count"] = skipped_count
    data["skipped_missing_reference"] = skipped_missing_reference
    data["skipped_missing_final"] = 0  # missing/whitespace final is already counted in checked and judged as incorrect
    data["skipped_missing_both"] = skipped_missing_both
    data["counted_missing_final_as_incorrect"] = counted_missing_final
    data["correct_count"] = correct_count
    data["accuracy"] = (correct_count / checked_count) if checked_count > 0 else 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(
        f"Checked {checked_count}/{total_count} (including {counted_missing_final} blank finals as incorrect). "
        f"Skipped {skipped_count} (missing reference/both). Correct: {correct_count}/{checked_count} => "
        f"Accuracy: {data['accuracy']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check mathematical equivalence for all problems in a JSON file using GPT-4o."
    )
    parser.add_argument(
        "-i", "--input",
        default="EXP_Weak_Compositions/g2/20250908_151703/MATH_FromScratch_20250908_163546.json",
        help="Path to the input JSON file. Defaults to EXP_Weak_Compositions/MATH_FromScratch_20250813_141531.json"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output JSON file. If not provided, the input file will be overwritten."
    )
    args = parser.parse_args()

    output_file = args.output if args.output else args.input

    asyncio.run(process_all_results(args.input, output_file))


