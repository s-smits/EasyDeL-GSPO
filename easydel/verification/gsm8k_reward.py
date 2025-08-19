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
    gt_raw = batch.get("answer", [])
    # Normalize container to Python list of strings
    if isinstance(gt_raw, list):
        gts = gt_raw
    else:
        try:
            import numpy as np  # type: ignore
            if isinstance(gt_raw, np.ndarray):
                gts = gt_raw.tolist()
            else:
                gts = [gt_raw] if gt_raw is not None else []
        except Exception:
            gts = [gt_raw] if gt_raw is not None and not isinstance(gt_raw, list) else []
    # Replicate to match completions length (B * R)
    target = len(completions)
    if len(gts) == 0:
        gt_list: List[str] = [""] * target
    elif len(gts) == target:
        gt_list = gts
    elif target % len(gts) == 0:
        factor = target // len(gts)
        gt_list = [x for x in gts for _ in range(factor)]
    else:
        times = (target + len(gts) - 1) // len(gts)
        gt_list = (gts * times)[:target]

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


