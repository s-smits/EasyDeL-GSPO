"""Math utilities for extracting and processing mathematical expressions."""


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \boxed{...} expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove the \\boxed{...} wrapper from a string."""
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left):]
        return s

    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]

    return s


