# utils/math_utils.py

import re

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
    """
    Extract the content within \\boxed{...} from the response text and normalize it.
    """
    raw_in_box = extract_boxed_text(output)
    if raw_in_box is not None:
        normalized = normalize_answer(raw_in_box)
        return normalized
    return ""


def normalize_answer(answer: str) -> str:
    """
    Clean up various LaTeX, spaces, and special characters that may appear in the MATH questions.
    """
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
