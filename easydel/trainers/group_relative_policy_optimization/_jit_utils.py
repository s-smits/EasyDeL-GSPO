from __future__ import annotations

import inspect
from typing import Tuple

from easydel.utils.compiling_utils import ejit


def compile_step_pair(
    step_fn,
    state_shardings,
    empty_sharding,
    static_args_train: tuple,
    static_args_eval: tuple,
):
    """
    JIT-compile training and evaluation step functions with consistent static_argnums.

    - Derives static_argnums starting at positional index 2 (after state, batch)
      based on the provided static args and the step function signature.
    - Returns compiled (train_fn, eval_fn) with `static_argnums_` attached for reference.
    """
    sig = inspect.signature(step_fn)
    max_pos_index = len(sig.parameters) - 1
    end = min(2 + len(static_args_train), max_pos_index + 1)
    static_argnums = tuple(range(2, end))

    train_fn = ejit(
        step_fn,
        in_shardings=(state_shardings, None),
        out_shardings=(state_shardings, empty_sharding),
        donate_argnums=(0,),
        static_argnums=static_argnums,
    )

    eval_fn = ejit(
        step_fn,
        in_shardings=(state_shardings, None),
        out_shardings=empty_sharding,
        static_argnums=static_argnums,
    )

    # Attach for downstream reference (parity with previous implementation)
    train_fn.static_argnums_ = static_argnums
    eval_fn.static_argnums_ = static_argnums

    return train_fn, eval_fn

