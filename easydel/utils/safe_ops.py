from __future__ import annotations

import typing as tp


def safe_call(
    desc: str,
    fn: tp.Callable[..., tp.Any],
    *args,
    default: tp.Any = None,
    swallow: bool = True,
    **kwargs,
) -> tp.Any:
    """Run a callable safely and return a default on failure.

    - Prints a concise note on failure; returns `default` if `swallow=True`.
    - Returns function output when successful; if the output is None, returns `default`.
    """
    try:
        out = fn(*args, **kwargs)
        return default if out is None else out
    except Exception as e:  # pragma: no cover - best-effort guard
        try:
            # Delay importing JAX only in environments where it exists
            import jax  # type: ignore

            if jax.process_index() == 0:
                print(f"SAFE_CALL_FAIL: {desc}: {e}")
        except Exception:
            # If jax is unavailable or printing fails, keep silent
            ...
        if not swallow:
            raise
        return default


