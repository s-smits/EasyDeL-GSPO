from __future__ import annotations

import numbers
import typing as tp

import jax
import numpy as np


def _is_scalar_like(x: tp.Any) -> bool:
    try:
        # numpy / jax arrays
        size = int(np.size(x))  # np.size handles jax.Array shape metadata without fetching
        return size == 1
    except Exception:
        return isinstance(x, numbers.Number)


def safe_device_get_local(x) -> tp.Any | None:
    """Best-effort local-only device_get for jax.Array.

    Returns host value for fully-addressable arrays or the first addressable shard's data.
    Returns None if local retrieval is not possible without a cross-host collective.
    """
    if not isinstance(x, jax.Array):
        return x

    # Case 1: fully addressable (replicated or single-host) — safe to device_get
    try:
        if hasattr(x, "is_fully_addressable") and x.is_fully_addressable:  # type: ignore[attr-defined]
            return jax.device_get(x)
    except Exception:
        pass

    # Case 2: only fetch local shard data
    try:
        shards = getattr(x, "addressable_shards", None)
        if shards and len(shards) > 0:
            return jax.device_get(shards[0].data)
    except Exception:
        pass

    # Case 3: no safe local access
    return None


def safe_to_float(x: tp.Any, default: float = 0.0) -> tp.Any:
    """Convert scalar-like values to float without cross-host collectives.

    - Python numbers -> float
    - NumPy scalars -> float
    - jax.Array/np.ndarray with size==1 -> local-only fetch to float
    - Non-scalar arrays/structures -> returned unchanged
    - On failure to safely fetch scalar -> return default
    """
    # Fast path for plain numbers
    if isinstance(x, numbers.Number):
        return float(x)

    # NumPy scalar
    if isinstance(x, np.generic):
        try:
            return float(x.item())
        except Exception:
            return default

    # Arrays: only attempt conversion if truly scalar-like
    if _is_scalar_like(x):
        if isinstance(x, jax.Array):
            host_val = safe_device_get_local(x)
            if host_val is None:
                return default
            try:
                arr = np.asarray(host_val).reshape(-1)
                return float(arr[0]) if arr.size > 0 else default
            except Exception:
                return default
        try:
            # numpy.ndarray or other scalar-likes
            arr = np.asarray(x).reshape(-1)
            return float(arr[0]) if arr.size > 0 else default
        except Exception:
            return default

    # Non-scalar: leave as-is for histogram/structured logging paths
    return x

