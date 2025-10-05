import re
from typing import List, Dict

def build_prompt_gpqa_1(question_text: str, choices: List[str]) -> List[Dict[str, str]]:
    """
    Builds a GPQA-style chat prompt.
    Requires the model to provide a single-letter answer in the format '##Answer: X##'.
    """
    system_msg = (
        ""
    )
    user_msg = "Question:\n" + question_text + "\n\nChoices:\n"
    for choice in choices:
        user_msg += choice + "\n"
    user_msg += '''State your choice in the form of ##Answer: X##.
'''

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_prompt_gpqa(question_text: str, choices: List[str]) -> List[Dict[str, str]]:
    """
    Builds a GPQA-style chat prompt.
    Requires the model to provide a single-letter answer in the format '##Answer: X##'.
    """
    system_msg = (
        "You are a helpful assistant. "
        "Reason step by step through the following question and provide the best answer. "
        "Finally, give your single-letter choice as '##Answer: X##'."
    )
    user_msg = "Question:\n" + question_text + "\n\nChoices:\n"
    for choice in choices:
        user_msg += choice + "\n"
    user_msg += "\nPlease provide your final single-letter answer in the format: ##Answer: X##."

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# def extract_answer_gpqa(model_output: str, valid_choices: List[str]) -> str:
#     """
#     Extracts an answer in the format '##Answer: X##' (where X is a single letter) from the model output.
#     Returns an empty string if no match is found or if the answer is not within the valid choices.
#     """
#     match = re.search(r'##Answer:\s*([A-Z])', model_output)
#     if match:
#         letter = match.group(1)
#         # Check if the letter is among the valid_choices (comparing only the first letter).
#         if any(letter == c[:1].upper() for c in valid_choices):
#             return letter
#     return ""


# def extract_answer_gpqa(model_output: str, valid_choices: List[str]) -> str:
#     """
#     Checks the last 10 characters of the model output for combinations like ' A', ' B', ' C', or ' D'.
#     If a letter is in valid_choices, returns the letter (uppercase); otherwise, returns an empty string.
#     """
#     # 1. Take the last 10 characters of the output (or all of it if it's shorter than 10).
#     tail_text = model_output
#    
#     # 2. Search for patterns like ' A', ' B', ' C', etc.; case-insensitive.
#     #    Here (?:[A-Za-z]) is used to capture a single letter, which will be extracted with group(1).
#     pattern = r'\s([A-Za-z])'
#     matches = re.findall(pattern, tail_text)
#    
#     if not matches:
#         return ""
#    
#     # 3. If multiple matches are found, prioritize the last one (index -1).
#     letter_found = matches[-1].upper()  # Get the last match and convert to uppercase.
#    
#     # 4. Verify if this letter is in the set of first letters of valid_choices.
#     #    (e.g., valid_choices can be ["A", "B", "C", "D"] or ["Answer A", "Answer B"])
#     valid_letters = {c[:1].upper() for c in valid_choices}
#     if letter_found in valid_letters:
#         return letter_found
#     else:
#         return ""

def extract_answer_gpqa(text: str, valid_choices: List[str]) -> str:
    """
    A more robust answer extractor for GPQA multiple-choice questions.
    It tries several common formats in order of priority, returning the uppercase letter upon the first match, otherwise returns an empty string.
    """
    valid = {c[:1].upper() for c in valid_choices}

    # 1) "Answer: X" - most reliable
    m = re.search(r'answer[^A-Za-z0-9]*([A-D])', text, flags=re.I)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    # 2) Wrapped in double hashes like "## ... X ##"
    m = re.search(r'##\s*([A-D])\s*##', text)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    # 3) A single letter on its own line
    for line in text.splitlines():
        s = line.strip().upper()
        if s in valid and len(s) == 1:
            return s

    # 4) Parentheses or at the end like "...(X)" or " X)" (takes the last one in reverse order)
    for m in reversed(re.findall(r'\(([A-D])\)', text)):
        ans = m.upper()
        if ans in valid:
            return ans

    # 5) Fallback to the original "space + letter" strategy (still takes the last one)
    for m in reversed(re.findall(r'\s([A-D])', text)):
        ans = m.upper()
        if ans in valid:
            return ans

    m = re.search(r"Final Answer: \(([A-Z])\)", text)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans
    return None
