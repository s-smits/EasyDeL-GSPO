from __future__ import annotations

import typing as tp

import jax
import numpy as np


def safe_to_float(x: tp.Any, default: float = 0.0) -> float:
    """
    Best-effort, multi-host-safe conversion of a value to a Python float.

    - Fast path handles native scalars and numpy scalars.
    - For JAX arrays, avoid cross-host collectives:
      1) Prefer fully addressable fetch
      2) Fall back to first local shard only
    - On failure, return the provided default.
    """
    # Native Python scalars
    if isinstance(x, (float, int)):
        return float(x)

    # NumPy scalar/array with single element
    if isinstance(x, np.generic):
        try:
            return float(x.item())
        except Exception:
            return float(default)

    # JAX array handling
    if isinstance(x, jax.Array):
        try:
            # 1) Fully addressable (replicated/single-device)
            if getattr(x, "is_fully_addressable", False):
                arr = jax.device_get(x)
                if np.size(arr) == 1:
                    return float(np.asarray(arr).reshape(()))

            # 2) Local shard only (no cross-host collective)
            shards = getattr(x, "addressable_shards", [])
            if shards and len(shards) > 0:
                local_data = jax.device_get(shards[0].data)
                if np.size(local_data) == 1:
                    return float(np.asarray(local_data).reshape(()))
        except Exception:
            return float(default)

    # Generic fallback
    try:
        return float(x)
    except Exception:
        return float(default)

from __future__ import annotations

import typing as tp

try:  # pragma: no cover - optional at import time
    import jax  # type: ignore
    from jax import numpy as jnp  # type: ignore
except Exception:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def safe_to_float(x: tp.Any, default: float = 0.0) -> float:
    """Best-effort conversion of a value (including JAX arrays) to a Python float.

    Avoids cross-host device fetches when arrays are not fully addressable. When the
    value cannot be safely converted, returns a provided default.
    """
    # Fast paths for native types
    try:
        if isinstance(x, (float, int)):
            return float(x)
    except Exception:
        pass

    # Numpy scalars
    try:
        if np is not None and isinstance(x, np.generic):  # type: ignore[arg-type]
            return float(x.item())
    except Exception:
        pass

    # JAX arrays (scalar or small arrays)
    if jax is not None:
        try:
            if isinstance(x, jax.Array):  # type: ignore[attr-defined]
                # Prefer fully addressable fetch
                try:
                    if getattr(x, "is_fully_addressable", False):
                        arr = jax.device_get(x)
                        # Collapse to scalar if possible
                        try:
                            arr = np.array(arr)  # type: ignore
                            if arr.size == 1:
                                return float(arr.reshape(()).item())
                        except Exception:
                            pass
                except Exception:
                    pass

                # Try local shard only (do not trigger cross-host operations)
                try:
                    shards = getattr(x, "addressable_shards", None)
                    if shards and len(shards) > 0:
                        local = jax.device_get(shards[0].data)
                        try:
                            local = np.array(local)  # type: ignore
                            if local.size == 1:
                                return float(local.reshape(()).item())
                        except Exception:
                            pass
                except Exception:
                    pass

                # Last resort: attempt best-effort numpy conversion without device_get
                try:
                    arr = np.array(x)  # type: ignore
                    if arr.size >= 1:
                        return float(arr.reshape(-1)[0])
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: try generic numpy conversion
    try:
        if np is not None:
            arr = np.array(x)  # type: ignore
            if arr.size == 1:
                return float(arr.reshape(()).item())
            elif arr.size >= 1:
                return float(arr.reshape(-1)[0])
    except Exception:
        pass

    return float(default)


def safe_item(x: tp.Any, default: tp.Any = 0.0) -> tp.Any:
    """Return a native scalar from 0-d array-like, otherwise return the original value.

    Avoids cross-host device fetches and falls back to the provided default if necessary.
    """
    try:
        # Native numeric types
        if isinstance(x, (float, int)):
            return x
    except Exception:
        pass

    # NumPy scalar
    try:
        if np is not None and isinstance(x, np.generic):  # type: ignore[arg-type]
            return x.item()
    except Exception:
        pass

    # JAX scalar
    if jax is not None:
        try:
            if isinstance(x, jax.Array) and getattr(x, "ndim", 1) == 0:  # type: ignore[attr-defined]
                # Fully addressable path
                try:
                    if getattr(x, "is_fully_addressable", False):
                        return jax.device_get(x)
                except Exception:
                    pass
                # Local shard path
                try:
                    shards = getattr(x, "addressable_shards", None)
                    if shards and len(shards) > 0:
                        return jax.device_get(shards[0].data)
                except Exception:
                    pass
                # Best-effort numpy conversion
                try:
                    return np.array(x).reshape(()).item()  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

    # Generic array-like
    try:
        if np is not None:
            arr = np.array(x)  # type: ignore
            if arr.size == 1:
                return arr.reshape(()).item()
    except Exception:
        pass

    return default

