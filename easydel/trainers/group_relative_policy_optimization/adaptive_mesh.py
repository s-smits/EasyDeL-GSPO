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
            print(f"DEBUG: adaptive_mesh.num_devices={num_devices}")
        except Exception as e:
            print(f"DEBUG: adaptive_mesh.num_devices fallback due to {e}")
            num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))

    # Note: rollouts_per_step is passed through but not used in mesh planning
    # The trainer will handle deriving num_return_sequences after mesh is configured

    # Compute dp/tp/fsdp in a robust, general way:
    # - Snap forced values to the nearest feasible divisors when possible
    # - Ensure dp * fsdp * tp == num_devices
    # - Keep dp <= total_batch_size
    try:
        desired_tp = int(force_tensor_parallel) if force_tensor_parallel else 1
        desired_tp = max(1, desired_tp)
        if num_devices % desired_tp != 0:
            snapped_tp = _largest_divisor_not_exceeding(num_devices, desired_tp)
            if snapped_tp != desired_tp:
                logger.warning(
                    f"Snapping tp from {desired_tp} to feasible {snapped_tp} for num_devices={num_devices}"
                )
            desired_tp = snapped_tp
        tp = desired_tp
        print(f"DEBUG: adaptive_mesh.tp={tp}")
    except Exception as e:
        print(f"DEBUG: adaptive_mesh.tp failed: {e}")
        tp = 1

    remaining_after_tp = max(1, num_devices // tp)
    print(f"DEBUG: adaptive_mesh.remaining_after_tp={remaining_after_tp}")

    if force_data_parallel:
        try:
            desired_dp = max(1, int(force_data_parallel))
            desired_dp = min(desired_dp, remaining_after_tp)
            dp = _largest_divisor_not_exceeding(remaining_after_tp, desired_dp)
            if dp != desired_dp:
                logger.warning(
                    f"Snapping dp from {desired_dp} to feasible {dp} for num_devices={num_devices}, tp={tp}"
                )
            fsdp = max(1, remaining_after_tp // dp)
            print(f"DEBUG: adaptive_mesh.forced dp={dp}, fsdp={fsdp}")
        except Exception as e:
            print(f"DEBUG: adaptive_mesh.dp failed: {e}")
            dp = 1
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
    try:
        step_spec = PartitionSpec(step_batch, step_seq)
        print(f"DEBUG: adaptive_mesh.step_spec={step_spec}")
    except Exception as e:
        print(f"DEBUG: adaptive_mesh.step_spec failed: {e}")
        step_spec = PartitionSpec(None, None)

    # Input spec: avoid TP; keep only DP on batch. Replicate across FSDP.
    in_batch_parts = []
    if dp > 1 and (total_batch_size % dp == 0):
        in_batch_parts.append("dp")
    in_batch = None if not in_batch_parts else (in_batch_parts[0] if len(in_batch_parts) == 1 else tuple(in_batch_parts))
    try:
        in_spec = PartitionSpec(in_batch, None)
        print(f"DEBUG: adaptive_mesh.input_spec={in_spec}")
    except Exception as e:
        print(f"DEBUG: adaptive_mesh.input_spec failed: {e}")
        in_spec = PartitionSpec(None, None)

    # Warn if DP does not divide total_batch_size (layout will replicate batch over DP in that case)
    if dp > 1 and (total_batch_size % dp != 0):
        logger.warning(
            f"total_batch_size ({total_batch_size}) is not divisible by dp ({dp}); "
            f"batch will be replicated over DP for step/input specs. Consider using total_batch_size=k*dp for best efficiency."
        )

    # Derived metadata
    total_workers = int(dp) * int(fsdp) * int(tp)
    data_parallel_workers = int(dp) * int(fsdp)
    print(f"DEBUG: adaptive_mesh.final dp={dp}, fsdp={fsdp}, tp={tp}, total_workers={total_workers}")
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
    rollouts_per_step = getattr(arguments, "rollouts_per_step", None)

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
        rollouts_per_step=rollouts_per_step,
    )

    # Update arguments with computed mesh configuration
    setattr(arguments, "step_partition_spec", plan.step_partition_spec)
    setattr(arguments, "input_partition_spec", plan.input_partition_spec)
    setattr(arguments, "mesh_dims", (plan.dp, plan.fsdp, plan.ep, plan.tp, plan.sp))
    
    # Update sharding axis dims to match mesh dims if present
    if hasattr(arguments, "sharding_axis_dims"):
        setattr(arguments, "sharding_axis_dims", (plan.dp, plan.fsdp, plan.ep, plan.tp, plan.sp))
    
    validate_mesh_config(plan.dp, plan.fsdp, plan.tp, num_devices, total_batch_size)

    # If the user forced DP larger than process_count but we don't implement intra-process DP sub-shards here,
    # cap DP to process_count and recompute plan to avoid duplicated prompts.
    try:
        procs = jax.process_count()
    except Exception:
        procs = 1
    if force_data_parallel and int(plan.dp) > int(procs):
        logger.warning(
            f"Requested DP={force_data_parallel} exceeds process_count={procs}; reducing DP to {procs}."
        )
        plan = plan_adaptive_mesh(
            total_batch_size=total_batch_size,
            num_return_sequences=num_return_sequences,
            num_devices=num_devices,
            force_tensor_parallel=force_tensor_parallel,
            force_data_parallel=procs,
            rollouts_per_step=rollouts_per_step,
        )
        # Update arguments with recomputed specs
        setattr(arguments, "step_partition_spec", plan.step_partition_spec)
        setattr(arguments, "input_partition_spec", plan.input_partition_spec)
        setattr(arguments, "mesh_dims", (plan.dp, plan.fsdp, plan.ep, plan.tp, plan.sp))
        if hasattr(arguments, "sharding_axis_dims"):
            setattr(arguments, "sharding_axis_dims", (plan.dp, plan.fsdp, plan.ep, plan.tp, plan.sp))

    # Dataset sharding configuration
    # With TP, all TP replicas must see the same data
    # Only DP workers should have different data shards
    try:
        proc_count = jax.process_count()
        
        # Handle single-process multi-device setups (e.g., TPU pods)
        if proc_count == 1 and plan.dp > 1:
            # In single-process mode, true per-DP sharding at the dataloader is not possible.
            # Use a single shard and warn the user. Recommend multi-process launch for true sharding.
            arguments.grain_shard_count = 1
            arguments.grain_shard_index = 0
            try:
                setattr(arguments, "single_process_dp_mode", True)
            except Exception:
                ...
            if jax.process_index() == 0:
                logger.warning(
                    f"Single-process multi-device detected (proc_count=1, mesh_dp={plan.dp}). "
                    f"Dataset sharding is disabled (shard 0/1). For true per-DP sharding, use a multi-process launch (e.g., mpirun -n {plan.dp})."
                )
        elif plan.tp > 1:
            # Multi-process with TP
            num_dp_groups = max(1, int(proc_count) // int(plan.tp))
            dp_group_id = int(jax.process_index()) // int(plan.tp)
            arguments.grain_shard_count = int(num_dp_groups)
            arguments.grain_shard_index = int(dp_group_id)
            if jax.process_index() == 0:
                logger.info(
                    f"Dataset sharding with TP={plan.tp}: {num_dp_groups} data shards across DP groups. "
                    f"Processes {list(range(0, int(plan.tp)))} will see shard 0, "
                    f"processes {list(range(int(plan.tp), 2*int(plan.tp)))} will see shard 1, etc."
                )
                if num_dp_groups > 1:
                    logger.info(
                        f"Note: With {num_dp_groups} DP groups, ensure dataset size is divisible by "
                        f"batch_size * num_dp_groups for optimal sharding efficiency."
                    )
        else:
            # No TP, standard sharding across all processes
            arguments.grain_shard_count = int(proc_count)
            arguments.grain_shard_index = int(jax.process_index())

        if jax.process_index() == 0:
            logger.info(
                f"Dataset configuration: shard {arguments.grain_shard_index}/{arguments.grain_shard_count} "
                f"(process {jax.process_index()}/{proc_count}, mesh_dp={plan.dp}, mesh_tp={plan.tp})"
            )
        # Clamp invalid shard settings as a failsafe
        if arguments.grain_shard_count is None or arguments.grain_shard_count <= 0:
            arguments.grain_shard_count = 1
        if arguments.grain_shard_index is None or arguments.grain_shard_index < 0:
            arguments.grain_shard_index = 0
    except Exception as e:
        logger.warning(f"Failed to configure dataset sharding: {e}")
        # Fallback to safe defaults
        arguments.grain_shard_count = 1
        arguments.grain_shard_index = 0
    return plan


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