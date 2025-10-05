import re

def extract_answer_gsm(text: str) -> str:
    """Extract the answer following the GSM convention `#### <answer>`."""
    if text is None:
        return ""
    # Take the last occurrence of #### to be safe
    matches = list(re.finditer(r"####\s*([-]?\d+(?:\.\d+)?)", text))
    if matches:
        return matches[-1].group(1)
    # Fallback: last standalone number
    nums = re.findall(r"[-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def normalize_answer(ans: str) -> str:
    """Normalize answer by stripping spaces, commas, and leading zeros."""
    if ans is None:
        return ""
    ans = ans.strip()
    if ans.startswith("####"):
        ans = ans[4:].strip()
    ans = ans.replace(",", "")
    # Remove leading zeros (but keep single zero)
    if re.fullmatch(r"0+", ans):
        return "0"
    ans = re.sub(r"^0+", "", ans)
    return ans


def create_chat_prompt(system_msg: str, user_msg: str):
    """
    Returns a chat structure in the form:
    [
      {"role": "system", "content": system_msg},
      {"role": "user",   "content": user_msg},
    ]
    """
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
