# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from dataclasses import dataclass
import jax
from jax.sharding import PartitionSpec
from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AdaptiveMeshPlan:
    """Minimal adaptive mesh plan for common DP×TP setups."""
    dp: int
    fsdp: int
    ep: int
    tp: int
    sp: int
    step_partition_spec: PartitionSpec
    input_partition_spec: PartitionSpec


def plan_adaptive_mesh(
    total_batch_size: int,
    num_return_sequences: int,
    num_devices: int | None = None,
    prefer_data_parallel: bool = True,  # kept for API compatibility; not used
    force_tensor_parallel: int | None = None,
    force_data_parallel: int | None = None,
    mini_batch_size: int | None = None,  # kept for API compatibility; not used
) -> AdaptiveMeshPlan:
    """Single-source planner for common DP×TP cases.

    Rules:
    - If both DP and TP are forced: use them; set FSDP to fill remaining devices.
    - If only TP forced: pick DP to use remaining devices (bounded by batch size).
    - If only DP forced: fill remaining devices with FSDP.
    - Otherwise: simple auto — DP=min(total_batch_size, num_devices), FSDP fills remainder.
    - SP is not used in the simplified planner (set to 1).
    - Input spec avoids TP; step spec uses TP when tp>1.
    """
    if num_devices is None:
        try:
            num_devices = jax.device_count()
        except Exception:
            num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))

    # Compute dp/tp/fsdp in a straightforward way
    if force_tensor_parallel and force_data_parallel:
        tp = int(force_tensor_parallel)
        dp = int(force_data_parallel)
        used = dp * tp
        if used > num_devices:
            raise ValueError(
                f"dp({dp})×tp({tp}) exceeds num_devices({num_devices})"
            )
        fsdp = max(1, num_devices // used)
    elif force_tensor_parallel:
        tp = int(force_tensor_parallel)
        if num_devices % tp != 0:
            raise ValueError(f"num_devices({num_devices}) must be divisible by tp({tp})")
        remaining = num_devices // tp
        # choose dp bounded by batch size and available slots
        dp = min(max(1, total_batch_size), remaining)
        fsdp = max(1, remaining // dp)
    elif force_data_parallel:
        dp = int(force_data_parallel)
        if num_devices % dp != 0:
            raise ValueError(f"num_devices({num_devices}) must be divisible by dp({dp})")
        tp = 1
        fsdp = max(1, num_devices // dp)
    else:
        # auto: prefer DP up to batch size, then FSDP
        dp = min(max(1, total_batch_size), num_devices)
        tp = 1
        fsdp = max(1, num_devices // dp)

    ep = 1
    sp = 1  # simplified planner: avoid SP

    # Step spec: DP (+FSDP when evenly divisible), TP on sequence if used
    step_batch_parts = []
    if dp > 1 and (total_batch_size % dp == 0):
        step_batch_parts.append("dp")
    if fsdp > 1 and (total_batch_size % (dp * fsdp) == 0):
        step_batch_parts.append("fsdp")
    step_batch = None if not step_batch_parts else (step_batch_parts[0] if len(step_batch_parts) == 1 else tuple(step_batch_parts))
    step_seq = "tp" if tp > 1 else None
    step_spec = PartitionSpec(step_batch, step_seq)

    # Input spec: avoid TP; keep only DP (+FSDP) when cleanly divisible
    in_batch_parts = []
    if dp > 1 and (total_batch_size % dp == 0):
        in_batch_parts.append("dp")
    if fsdp > 1 and (total_batch_size % (dp * fsdp) == 0):
        in_batch_parts.append("fsdp")
    in_batch = None if not in_batch_parts else (in_batch_parts[0] if len(in_batch_parts) == 1 else tuple(in_batch_parts))
    in_spec = PartitionSpec(in_batch, None)

    return AdaptiveMeshPlan(
        dp=dp,
        fsdp=fsdp,
        ep=ep,
        tp=tp,
        sp=sp,
        step_partition_spec=step_spec,
        input_partition_spec=in_spec,
    )


def configure_adaptive_mesh_inplace(arguments) -> AdaptiveMeshPlan:
    """Compute plan once and write specs into `arguments`.

    Keeps the system simple and avoids per-site overrides.
    """
    total_batch_size = getattr(arguments, "total_batch_size", 1)
    num_return_sequences = getattr(arguments, "num_return_sequences", 1)
    force_tensor_parallel = getattr(arguments, "force_tensor_parallel", None)
    force_data_parallel = getattr(arguments, "force_data_parallel", None)

    try:
        num_devices = jax.device_count()
    except Exception:
        num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))

    plan = plan_adaptive_mesh(
        total_batch_size=total_batch_size,
        num_return_sequences=num_return_sequences,
        num_devices=num_devices,
        force_tensor_parallel=force_tensor_parallel,
        force_data_parallel=force_data_parallel,
    )

    setattr(arguments, "step_partition_spec", plan.step_partition_spec)
    setattr(arguments, "input_partition_spec", plan.input_partition_spec)
    setattr(arguments, "mesh_dims", (plan.dp, plan.fsdp, plan.ep, plan.tp, plan.sp))
    validate_mesh_config(plan.dp, plan.fsdp, plan.tp, num_devices, total_batch_size)
    return plan


def calculate_optimal_mesh_dims(
    total_batch_size: int,
    num_return_sequences: int,
    num_devices: int = None,
    prefer_data_parallel: bool = True,  # kept for API compatibility; not used
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,  # kept for API compatibility; not used
) -> tuple[int, int, int, int, int]:
    """Simple DP×TP mesh sizing for common cases.

    Returns (dp, fsdp, ep, tp, sp). `sp` is always 1 in this simplified model.
    """
    if num_devices is None:
        try:
            num_devices = jax.device_count()
        except Exception:
            num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))

    if force_tensor_parallel and force_data_parallel:
        tp = int(force_tensor_parallel)
        dp = int(force_data_parallel)
        used = dp * tp
        if used > num_devices:
            raise ValueError(
                f"dp({dp})×tp({tp}) exceeds num_devices({num_devices})"
            )
        fsdp = max(1, num_devices // used)
        return (dp, fsdp, 1, tp, 1)

    if force_tensor_parallel:
        tp = int(force_tensor_parallel)
        if num_devices % tp != 0:
            raise ValueError(f"num_devices({num_devices}) must be divisible by tp({tp})")
        remaining = num_devices // tp
        dp = min(max(1, total_batch_size), remaining)
        fsdp = max(1, remaining // dp)
        return (dp, fsdp, 1, tp, 1)

    if force_data_parallel:
        dp = int(force_data_parallel)
        if num_devices % dp != 0:
            raise ValueError(f"num_devices({num_devices}) must be divisible by dp({dp})")
        tp = 1
        fsdp = max(1, num_devices // dp)
        return (dp, fsdp, 1, tp, 1)

    dp = min(max(1, total_batch_size), num_devices)
    tp = 1
    fsdp = max(1, num_devices // dp)
    return (dp, fsdp, 1, tp, 1)


def get_adaptive_sharding_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
    num_return_sequences: int = 8,
) -> PartitionSpec:
    """
    Get appropriate sharding spec for input tensors based on batch size.
    
    Returns:
        PartitionSpec for input sharding
    """
    if num_devices is None:
        try:
            num_devices = jax.device_count()
        except Exception:
            num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))
    
    # Use the centralized planner
    plan = plan_adaptive_mesh(
        total_batch_size=total_batch_size,
        num_return_sequences=num_return_sequences,
        num_devices=num_devices,
        force_tensor_parallel=force_tensor_parallel,
        force_data_parallel=force_data_parallel,
        mini_batch_size=mini_batch_size,
    )
    return plan.input_partition_spec


def get_adaptive_step_partition_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
    num_return_sequences: int = 8,
) -> PartitionSpec:
    """
    Get appropriate step partition spec for training based on batch size.
    
    Returns:
        PartitionSpec for step partitioning
    """
    if num_devices is None:
        try:
            num_devices = jax.device_count()
        except Exception:
            num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))
    
    # Use the centralized planner
    plan = plan_adaptive_mesh(
        total_batch_size=total_batch_size,
        num_return_sequences=num_return_sequences,
        num_devices=num_devices,
        force_tensor_parallel=force_tensor_parallel,
        force_data_parallel=force_data_parallel,
        mini_batch_size=mini_batch_size,
    )
    return plan.step_partition_spec


def validate_mesh_config(
    dp: int, fsdp: int, tp: int, num_devices: int, total_batch_size: int
) -> bool:
    """Lightweight validation for common cases."""
    total = dp * fsdp * tp
    if total != max(1, num_devices):
        logger.warning(
            f"Mesh dims dp({dp})×fsdp({fsdp})×tp({tp})={total} != num_devices({num_devices})."
        )
    if dp > max(1, total_batch_size):
        logger.warning(
            f"dp({dp}) exceeds batch size({total_batch_size}); this may be inefficient."
        )
    return True