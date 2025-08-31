from __future__ import annotations

from typing import Any, List
import os

try:  # Optional JAX for multihost-safe aggregation
    import jax  # type: ignore
except Exception:  # pragma: no cover
    jax = None  # type: ignore
import re


def extract_text(completion: Any) -> str:
    """Extract plain text from a completion item.

    Supports formats like [[{"role": "assistant", "content": text}], ...] or raw strings.
    """
    if isinstance(completion, list) and completion:
        c0 = completion[0]
        if isinstance(c0, dict) and "content" in c0:
            return c0["content"]
        if isinstance(c0, str):
            return c0
        return str(c0)
    if isinstance(completion, str):
        return completion
    return ""


def normalize_to_list_str(obj: Any) -> List[str]:
    """Normalize array-like or scalar-like to a Python list[str].

    Handles numpy arrays, jax arrays, tuples, scalars, None, etc.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        base = obj
    elif isinstance(obj, str):
        base = [obj]
    else:
        try:
            # numpy / jax arrays
            if hasattr(obj, "tolist"):
                base = obj.tolist()
            else:
                base = list(obj)  # type: ignore[arg-type]
        except Exception:
            base = [obj]

    out: List[str] = []
    for x in base:
        if x is None:
            out.append("")
        elif isinstance(x, str):
            out.append(x)
        else:
            try:
                out.append(str(x))
            except Exception:
                out.append("")
    return out


def replicate_to_length(items: List[str], target_len: int, interleaved: bool = False) -> List[str]:
    """Replicate items to exactly target_len elements.
    Defaults to contiguous prompt-major blocks; interleaved kept for backward compat.
    """
    if target_len <= 0:
        return []
    if not items:
        return [""] * target_len
    n = len(items)
    if n == target_len:
        return items
    factor, remainder = divmod(target_len, n)
    if interleaved:
        # [a,b,c] -> [a,b,c, a,b,c, ...] then tail
        return items * factor + items[:remainder]
    # Contiguous blocks: [a * factor, b * factor, ...] then tail per-item
    result: List[str] = [x for x in items for _ in range(factor)]
    if remainder:
        result.extend(items[:remainder])
    return result


def is_main_process() -> bool:
    """Return True if this is process 0, or if JAX is unavailable."""
    if jax is None:
        return True
    try:
        return int(jax.process_index()) == 0
    except Exception:
        return True


def safe_global_sum(value: Any) -> Any:
    """Best-effort global sum across processes using JAX multihost utils.

    Falls back to returning the local value on error or single-process setups.
    """
    # Allow disabling via env to avoid TPU host collectives if unstable
    if os.getenv("EASYDEL_DISABLE_GLOBAL_AGG", "0").lower() in {"1", "true", "yes"}:
        return value
    if jax is None:
        return value
    try:
        pc = int(jax.process_count())
    except Exception:
        pc = 1
    if pc <= 1:
        return value
    try:
        gathered = jax.experimental.multihost_utils.process_allgather(value)  # type: ignore[attr-defined]
        try:
            # jnp/np arrays
            return gathered.sum()
        except Exception:
            # Python sequence fallback
            try:
                return sum(gathered)
            except Exception:
                return value
    except Exception:
        return value


def extract_answer_from_xml(solution_str: str) -> str | None:
    """Extract content inside <answer>...</answer>. Returns None if absent.

    Shared across GSM8K and Math rewards.
    """
    try:
        m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        return m.group(1).strip() if m else None
    except Exception:
        return None


