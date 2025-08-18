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
import math
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
    # Derived metadata to aid downstream configuration (optional)
    total_workers: int = 0
    data_parallel_workers: int = 0
    per_process_rollouts_capacity: int = 0
    global_rollouts_capacity: int = 0


def _largest_divisor_not_exceeding(n: int, limit: int) -> int:
    """Return the largest positive divisor of n that is <= limit.

    Falls back to 1 if no larger divisor found (always exists since 1 divides n).
    """
    if n <= 0:
        return 1
    if limit <= 1:
        return 1
    limit = min(limit, n)
    # Iterate downwards; n is typically small-ish (<= 4096 devices), this is fine
    for candidate in range(limit, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def plan_adaptive_mesh(
    total_batch_size: int,
    num_return_sequences: int,
    num_devices: int | None = None,
    prefer_data_parallel: bool = True,  # kept for API compatibility; not used
    force_tensor_parallel: int | None = None,
    force_data_parallel: int | None = None,
    mini_batch_size: int | None = None,  # kept for API compatibility; not used
    rollouts_per_step: int | None = None,
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

    # Note: rollouts_per_step is passed through but not used in mesh planning
    # The trainer will handle deriving num_return_sequences after mesh is configured

    # Compute dp/tp/fsdp in a robust, general way:
    # - Snap forced values to the nearest feasible divisors when possible
    # - Ensure dp * fsdp * tp == num_devices
    # - Keep dp <= total_batch_size
    desired_tp = int(force_tensor_parallel) if force_tensor_parallel else 1
    desired_tp = max(1, desired_tp)
    if num_devices % desired_tp != 0:
        # Snap TP down to largest divisor of num_devices not exceeding desired_tp
        snapped_tp = _largest_divisor_not_exceeding(num_devices, desired_tp)
        if snapped_tp != desired_tp:
            logger.warning(
                f"Snapping tp from {desired_tp} to feasible {snapped_tp} for num_devices={num_devices}"
            )
        desired_tp = snapped_tp
    tp = desired_tp

    remaining_after_tp = max(1, num_devices // tp)

    if force_data_parallel:
        desired_dp = max(1, int(force_data_parallel))
        # Respect batch cap and remaining slots after TP
        desired_dp = min(desired_dp, max(1, total_batch_size), remaining_after_tp)
        # Snap DP to a divisor of remaining_after_tp
        dp = _largest_divisor_not_exceeding(remaining_after_tp, desired_dp)
        if dp != desired_dp:
            logger.warning(
                f"Snapping dp from {desired_dp} to feasible {dp} for num_devices={num_devices}, tp={tp}"
            )
        fsdp = max(1, remaining_after_tp // dp)
    else:
        # Auto DP: prefer as large as possible up to batch size while dividing remaining_after_tp
        # If rollouts_per_step is provided, try to meet the target by increasing DP (within limits)
        if rollouts_per_step and rollouts_per_step > 0:
            denom = max(1, total_batch_size * max(1, num_return_sequences))
            dp_required = (int(rollouts_per_step) + denom - 1) // denom  # ceil division
            dp_target = min(max(1, dp_required), min(max(1, total_batch_size), remaining_after_tp))
        else:
            dp_target = min(max(1, total_batch_size), remaining_after_tp)
        dp = _largest_divisor_not_exceeding(remaining_after_tp, dp_target)
        fsdp = max(1, remaining_after_tp // dp)

    ep = 1
    sp = 1  # simplified planner: avoid SP

    # Step spec: DP on batch; TP on sequence if used. Do NOT shard batch over FSDP.
    step_batch_parts = []
    if dp > 1 and (total_batch_size % dp == 0):
        step_batch_parts.append("dp")
    step_batch = None if not step_batch_parts else (step_batch_parts[0] if len(step_batch_parts) == 1 else tuple(step_batch_parts))
    step_seq = "tp" if tp > 1 else None
    step_spec = PartitionSpec(step_batch, step_seq)

    # Input spec: avoid TP; keep only DP on batch. Replicate across FSDP.
    in_batch_parts = []
    if dp > 1 and (total_batch_size % dp == 0):
        in_batch_parts.append("dp")
    in_batch = None if not in_batch_parts else (in_batch_parts[0] if len(in_batch_parts) == 1 else tuple(in_batch_parts))
    in_spec = PartitionSpec(in_batch, None)

    # Derived metadata
    total_workers = int(dp) * int(fsdp) * int(tp)
    data_parallel_workers = int(dp) * int(fsdp)
    per_process_rollouts_capacity = int(total_batch_size) * max(1, int(num_return_sequences))
    global_rollouts_capacity = data_parallel_workers * per_process_rollouts_capacity

    return AdaptiveMeshPlan(
        dp=dp,
        fsdp=fsdp,
        ep=ep,
        tp=tp,
        sp=sp,
        step_partition_spec=step_spec,
        input_partition_spec=in_spec,
        total_workers=total_workers,
        data_parallel_workers=data_parallel_workers,
        per_process_rollouts_capacity=per_process_rollouts_capacity,
        global_rollouts_capacity=global_rollouts_capacity,
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
    """General DP×TP×FSDP sizing.

    Returns (dp, fsdp, ep, tp, sp). `sp` is always 1 in this simplified model.
    This delegates to `plan_adaptive_mesh` to keep behavior consistent.
    """
    if num_devices is None:
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
        mini_batch_size=mini_batch_size,
    )
    return (plan.dp, plan.fsdp, plan.ep, plan.tp, plan.sp)


def get_adaptive_sharding_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
    num_return_sequences: int = 8,
    rollouts_per_step: int | None = None,
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
        rollouts_per_step=rollouts_per_step,
    )
    return plan.input_partition_spec


def get_adaptive_step_partition_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
    num_return_sequences: int = 8,
    rollouts_per_step: int | None = None,
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
        rollouts_per_step=rollouts_per_step,
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