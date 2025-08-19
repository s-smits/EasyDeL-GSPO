import re
from typing import List


def _answer_check(solution_str: str, ground_truth: str) -> bool:
    """Return True if the last number in solution_str equals ground_truth.

    Mirrors VERL's gsm8k reward: extract last number (int/float) and compare as string.
    """
    numbers = re.findall(r"-?\d+\.?\d*", solution_str)
    if not numbers:
        return False
    return numbers[-1] == ground_truth


def _extract_answer_from_xml(solution_str: str) -> str | None:
    """Extract content inside <answer>...</answer>. Returns None if absent."""
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    return m.group(1).strip() if m else None


def format_reward(completions: List[list[dict]], prompts=None, batch=None, **kwargs) -> List[float]:
    """Simple structural reward encouraging <think> and <answer> tags.

    completions: [[{"role": "assistant", "content": text}], ...]
    Returns a float per completion in [0, 1].
    """
    out: List[float] = []
    for comp in completions:
        text = comp[0]["content"] if comp and comp[0] else ""
        has_think = (text.count("<think>") == 1 and text.count("</think>") == 1)
        has_answer = (text.count("<answer>") == 1 and text.count("</answer>") == 1)
        out.append(1.0 if (has_think and has_answer) else 0.0)
    # Allow weighting from kwargs for logging consistency
    weight = float(kwargs.get("format_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in out]


def answer_reward(prompts, completions: List[list[dict]], batch, **kwargs) -> List[float]:
    """Exact/robust answer reward for GSM8K.

    - If <answer>...</answer> present, use its content; else use full text.
    - Compare last number with ground truth from decoded answer_ids.
    """
    # Ground truths are provided directly as strings in batch["answer"]
    gt_list: List[str] = batch.get("answer", [])
    if isinstance(gt_list, str):
        gt_list = [gt_list]
    # Replicate to match completions length (B * R)
    repeat = (len(completions) // len(gt_list)) if gt_list else 1
    gt_list = gt_list * max(1, repeat)

    rewards: List[float] = []
    for comp, gt in zip(completions, gt_list, strict=False):
        text = comp[0]["content"] if comp and comp[0] else ""
        # Fallback: if no <answer>, use entire response's last number
        ans = _extract_answer_from_xml(text) or text
        rewards.append(1.0 if _answer_check(ans, gt) else 0.0)
    # Optional weighting for logging consistency
    weight = float(kwargs.get("answer_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in rewards]


__all__ = [
    "format_reward",
    "answer_reward",
]


