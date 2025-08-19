from typing import List
try:
    # Math-Verify: robust evaluator for math expressions
    from math_verify import parse, verify  # type: ignore
except Exception as _e:  # pragma: no cover
    parse = None  # type: ignore
    verify = None  # type: ignore


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def _last_boxed_only_string(string: str) -> str | None:
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


def _strip_string(string: str) -> str:
    s = string.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    # Remove units: split at "\\text{ " and keep left part if present
    if "\\text{ " in s:
        parts = s.split("\\text{ ")
        if len(parts) == 2:
            s = parts[0]
    s = s.replace("\\%", "").replace("\%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    # remove spaces
    s = s.replace(" ", "")
    # normalize simple a/b forms that are pure integers
    if s.count("/") == 1:
        a, b = s.split("/")
        try:
            ai, bi = int(a), int(b)
            if s == f"{ai}/{bi}":
                s = f"\\frac{{{ai}}}{{{bi}}}"
        except Exception:
            pass
    if s == "0.5":
        s = "\\frac{1}{2}"
    return s


def _is_equiv(str1: str | None, str2: str | None) -> bool:
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def format_reward(completions: List[list[dict]], prompts=None, batch=None, **kwargs) -> List[float]:
    """Structural reward for MATH: enforce exactly one \boxed{...} inside <answer>.

    Returns 1.0 if there is exactly one <answer>...</answer> block and within it exactly one \boxed{...}, else 0.0.
    """
    out: List[float] = []
    for comp in completions:
        text = comp[0]["content"] if comp and comp[0] else ""
        # Must contain one think and one answer block
        ok_blocks = (text.count("<think>") == 1 and text.count("</think>") == 1 and text.count("<answer>") == 1 and text.count("</answer>") == 1)
        if not ok_blocks:
            out.append(0.0)
            continue
        # Extract answer body and check boxed count
        try:
            ans = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
        except Exception:
            out.append(0.0)
            continue
        has_one_box = (ans.count("\\boxed{") + ans.count("\\boxed ") == 1)
        out.append(1.0 if has_one_box else 0.0)
    weight = float(kwargs.get("format_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in out]


def answer_reward(prompts, completions: List[list[dict]], batch, **kwargs) -> List[float]:
    """Math verification using Math-Verify.

    - Extract model answer primarily from inside <answer>...</answer>, else fallback to entire text.
    - Parse both gold and answer with Math-Verify (defaults include Latex and Expr extraction).
    - Score 1.0 if verify(gold, answer) else 0.0. If math-verify is unavailable, fallback to \boxed-based equivalence.
    """
    # Prefer normalized ground truth if provided by preprocessing
    gts = batch.get("solution_normalized", batch.get("solution", []))
    if isinstance(gts, str):
        gts = [gts]
    # Replicate to match B*R if needed
    repeat = (len(completions) // len(gts)) if gts else 1
    gts = gts * max(1, repeat)

    use_mv = callable(parse) and callable(verify)  # type: ignore

    out: List[float] = []
    for comp, gt in zip(completions, gts, strict=False):
        text = comp[0]["content"] if comp and comp[0] else ""
        # Prefer content inside <answer> block
        if "<answer>" in text and "</answer>" in text:
            try:
                ans_text = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
            except Exception:
                ans_text = text
        else:
            ans_text = text

        if use_mv:
            try:
                gold_parsed = parse(gt)  # default uses both Latex and Expr extraction
                ans_parsed = parse(ans_text)
                out.append(1.0 if verify(gold_parsed, ans_parsed) else 0.0)
                continue
            except Exception:
                # Fall through to boxed-based heuristic
                pass

        # Fallback heuristic: compare last \boxed{...} with normalized GT
        boxed = _last_boxed_only_string(ans_text)
        if boxed is None:
            # as a last resort, try whole response
            boxed = _last_boxed_only_string(text)
        if boxed is None:
            out.append(0.0)
            continue
        ans = _remove_boxed(boxed)
        out.append(1.0 if _is_equiv(ans, gt) else 0.0)
    weight = float(kwargs.get("answer_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in out]


__all__ = [
    "format_reward",
    "answer_reward",
]


