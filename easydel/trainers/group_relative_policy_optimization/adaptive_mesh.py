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

import jax
from jax.sharding import PartitionSpec
from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


def calculate_optimal_mesh_dims(
    total_batch_size: int,
    num_return_sequences: int,
    num_devices: int = None,
    prefer_data_parallel: bool = True,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
) -> tuple[int, int, int, int, int]:
    """
    Calculate optimal mesh dimensions for GRPO/GSPO training based on batch size and rollouts.
    
    Args:
        total_batch_size: Number of prompts in the batch
        num_return_sequences: Number of rollouts per prompt
        num_devices: Number of available devices (defaults to JAX device count)
        prefer_data_parallel: Whether to prioritize data parallelism over model parallelism
        force_tensor_parallel: Force specific TP dimension
        force_data_parallel: Force specific DP dimension
        mini_batch_size: Minimum batch size per model instance (used with TP)
        
    Returns:
        Tuple of (dp, fsdp, ep, tp, sp) dimensions
    """
    if num_devices is None:
        num_devices = jax.device_count()
    num_workers = jax.process_count()
    # On TPU v4, devices commonly come as multiples of 8 per worker (megacore).
    # If TPU v4-32 is expected but only 16 devices are visible due to 2 chips per worker,
    # still compute logical dp/tp using total devices across workers.
    try:
        devices_per_process = len([d for d in jax.devices() if d.process_index == jax.process_index()])
    except Exception:
        devices_per_process = None
    
    # Detect TPU configuration
    tpu_type = _detect_tpu_type()
    
    logger.info(
        f"Calculating mesh for batch_size={total_batch_size}, "
        f"rollouts={num_return_sequences}, devices={num_devices}, "
        f"processes={num_workers}, tpu_type={tpu_type}, "
        f"force_tp={force_tensor_parallel}, force_dp={force_data_parallel}, "
        f"mini_batch={mini_batch_size}"
    )
    
    # Strategy 1: Both TP and DP are forced
    if force_tensor_parallel is not None and force_data_parallel is not None:
        tp = force_tensor_parallel
        dp = force_data_parallel
        
        # Validate configuration
        total_used = dp * tp
        if total_used > num_devices:
            raise ValueError(
                f"force_data_parallel({dp}) × force_tensor_parallel({tp}) = {total_used} "
                f"exceeds available devices({num_devices})"
            )
        elif total_used < num_devices:
            # Use remaining devices for FSDP
            fsdp = num_devices // total_used
            logger.warning(
                f"Using {total_used}/{num_devices} devices. "
                f"Remaining {num_devices - total_used} devices will be used for FSDP={fsdp}"
            )
        else:
            fsdp = 1
        
        logger.info(
            f"Using forced configuration: dp={dp}, fsdp={fsdp}, tp={tp} "
            f"(total: {dp * fsdp * tp} devices)"
        )
        return (dp, fsdp, 1, tp, 1)
    
    # Strategy 2: Only TP is forced
    if force_tensor_parallel is not None:
        tp = force_tensor_parallel
        if num_devices % tp != 0:
            raise ValueError(f"num_devices ({num_devices}) must be divisible by tp ({tp})")
        
        # Calculate DP based on remaining devices
        remaining_devices = num_devices // tp
        
        if force_data_parallel is not None:
            dp = force_data_parallel
            if dp > remaining_devices:
                raise ValueError(
                    f"force_data_parallel({dp}) exceeds available slots({remaining_devices}) "
                    f"after tensor_parallel({tp})"
                )
            fsdp = remaining_devices // dp if remaining_devices % dp == 0 else 1
        else:
            # Auto-calculate DP
            if total_batch_size >= remaining_devices:
                # Use all remaining for DP
                dp = remaining_devices
                fsdp = 1
            elif mini_batch_size is not None:
                # Calculate based on mini-batch requirements
                models_needed = max(1, total_batch_size // mini_batch_size)
                dp = min(models_needed, remaining_devices)
                fsdp = remaining_devices // dp if remaining_devices % dp == 0 else 1
            else:
                # Default: balance between DP and FSDP
                dp = min(total_batch_size, remaining_devices)
                fsdp = remaining_devices // dp if remaining_devices % dp == 0 else 1
        
        logger.info(
            f"Using tensor parallel strategy: dp={dp}, fsdp={fsdp}, tp={tp} "
            f"(total: {dp * fsdp * tp} devices)"
        )
        return (dp, fsdp, 1, tp, 1)
    
    # Strategy 3: Only DP is forced
    if force_data_parallel is not None:
        dp = force_data_parallel
        if num_devices % dp != 0:
            raise ValueError(f"num_devices ({num_devices}) must be divisible by dp ({dp})")
        
        remaining_devices = num_devices // dp
        
        # Decide between FSDP and TP for remaining devices
        if remaining_devices > 1:
            # For TPU v4, prefer TP within megacores
            if tpu_type == "v4" and remaining_devices in [2, 4, 8]:
                tp = remaining_devices
                fsdp = 1
            else:
                # Default to FSDP
                fsdp = remaining_devices
                tp = 1
        else:
            fsdp = 1
            tp = 1
        
        logger.info(
            f"Using data parallel strategy: dp={dp}, fsdp={fsdp}, tp={tp} "
            f"(total: {dp * fsdp * tp} devices)"
        )
        return (dp, fsdp, 1, tp, 1)
    
    # Strategy 4: Auto-calculate optimal configuration
    return _auto_calculate_mesh_dims(
        total_batch_size, num_return_sequences, num_devices, 
        prefer_data_parallel, mini_batch_size, tpu_type
    )


def _detect_tpu_type() -> str:
    """Detect TPU version and configuration."""
    try:
        device = jax.devices()[0]
        device_kind = str(getattr(device, 'device_kind', '')).lower()
        
        if 'v5' in device_kind:
            return 'v5'
        elif 'v4' in device_kind:
            return 'v4'
        elif 'v3' in device_kind:
            return 'v3'
        elif 'v2' in device_kind:
            return 'v2'
        else:
            return 'unknown'
    except:
        return 'unknown'


def _auto_calculate_mesh_dims(
    total_batch_size: int,
    num_return_sequences: int,
    num_devices: int,
    prefer_data_parallel: bool,
    mini_batch_size: int,
    tpu_type: str,
) -> tuple[int, int, int, int, int]:
    """Auto-calculate optimal mesh dimensions based on hardware and workload."""
    
    num_workers = jax.process_count()
    
    # Special handling for common TPU configurations
    if tpu_type == "v4" and ((num_devices == 32 and num_workers == 4) or (num_devices == 16 and num_workers == 4)):
        # TPU v4-32: 4 workers × 8 chips
        return _optimize_for_tpu_v4_32(total_batch_size, num_return_sequences, mini_batch_size)
    elif tpu_type == "v4" and num_devices == 8:
        # TPU v4-8: 1 worker × 8 chips
        return _optimize_for_tpu_v4_8(total_batch_size, num_return_sequences, mini_batch_size)
    elif tpu_type == "v5" and num_devices == 8:
        # TPU v5e-8 or v5p-8
        return _optimize_for_tpu_v5_8(total_batch_size, num_return_sequences, mini_batch_size)
    
    # Generic optimization
    if total_batch_size >= num_devices:
        # Pure data parallelism
        return (num_devices, 1, 1, 1, 1)
    elif total_batch_size == 1:
        # Pure model parallelism
        return (1, num_devices, 1, 1, 1)
    else:
        # Hybrid approach
        dp = total_batch_size
        remaining = num_devices // dp
        
        if prefer_data_parallel or remaining == 1:
            fsdp = remaining
            tp = 1
        else:
            # Consider tensor parallelism for small batches
            if remaining in [2, 4, 8] and tpu_type in ["v4", "v5"]:
                tp = min(remaining, 4)  # Cap TP at 4 for efficiency
                fsdp = remaining // tp
            else:
                fsdp = remaining
                tp = 1
        
        return (dp, fsdp, 1, tp, 1)


def _optimize_for_tpu_v4_32(
    total_batch_size: int,
    num_return_sequences: int,
    mini_batch_size: int,
) -> tuple[int, int, int, int, int]:
    """Optimize mesh for TPU v4-32 (4 workers × 8 chips)."""
    
    # TPU v4-32 optimal configurations based on empirical testing
    if total_batch_size >= 32:
        # Full data parallelism
        return (32, 1, 1, 1, 1)
    elif total_batch_size >= 16:
        # DP with some FSDP
        dp = total_batch_size
        fsdp = 32 // dp
        return (dp, fsdp, 1, 1, 1)
    elif total_batch_size >= 8:
        # Balanced DP and FSDP
        dp = total_batch_size
        fsdp = 32 // dp
        return (dp, fsdp, 1, 1, 1)
    elif total_batch_size >= 4:
        # DP=4 (one per worker), rest for model parallelism
        dp = 4
        remaining = 8  # 32 / 4
        
        # Use TP=4 for efficient cross-chip communication within worker
        if mini_batch_size and total_batch_size // mini_batch_size >= 4:
            return (dp, 2, 1, 4, 1)  # 4×2×4 = 32
        else:
            return (dp, 8, 1, 1, 1)  # 4×8 = 32
    elif total_batch_size == 2:
        # 2-way DP, significant model parallelism
        return (2, 4, 1, 4, 1)  # 2×4×4 = 32
    else:
        # Single batch, maximum model parallelism
        # Option 1: Pure FSDP
        # return (1, 32, 1, 1, 1)
        # Option 2: FSDP + TP for better memory usage
        return (1, 8, 1, 4, 1)  # 1×8×4 = 32


def _optimize_for_tpu_v4_8(
    total_batch_size: int,
    num_return_sequences: int,
    mini_batch_size: int,
) -> tuple[int, int, int, int, int]:
    """Optimize mesh for TPU v4-8 (1 worker × 8 chips)."""
    
    if total_batch_size >= 8:
        return (8, 1, 1, 1, 1)
    elif total_batch_size >= 4:
        dp = total_batch_size
        fsdp = 8 // dp
        return (dp, fsdp, 1, 1, 1)
    elif total_batch_size == 2:
        # Consider TP for better memory distribution
        return (2, 2, 1, 2, 1)  # 2×2×2 = 8
    else:
        # Single batch
        return (1, 4, 1, 2, 1)  # 1×4×2 = 8


def _optimize_for_tpu_v5_8(
    total_batch_size: int,
    num_return_sequences: int,
    mini_batch_size: int,
) -> tuple[int, int, int, int, int]:
    """Optimize mesh for TPU v5e-8 or v5p-8."""
    
    # TPU v5 has better inter-chip bandwidth
    if total_batch_size >= 8:
        return (8, 1, 1, 1, 1)
    elif total_batch_size >= 4:
        dp = total_batch_size
        fsdp = 8 // dp
        return (dp, fsdp, 1, 1, 1)
    elif total_batch_size == 2:
        # v5 handles TP well
        return (2, 1, 1, 4, 1)  # 2×1×4 = 8
    else:
        return (1, 2, 1, 4, 1)  # 1×2×4 = 8


def get_adaptive_sharding_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
) -> PartitionSpec:
    """
    Get appropriate sharding spec for input tensors based on batch size.
    
    Returns:
        PartitionSpec for input sharding
    """
    if num_devices is None:
        num_devices = jax.device_count()
    
    # Get mesh dimensions
    dp, fsdp, _, tp, _ = calculate_optimal_mesh_dims(
        total_batch_size, 8, num_devices,
        force_tensor_parallel=force_tensor_parallel,
        force_data_parallel=force_data_parallel,
        mini_batch_size=mini_batch_size
    )
    
    # Build partition spec based on actual mesh configuration
    batch_spec = []
    seq_spec = []
    
    # Batch dimension sharding
    if dp > 1:
        batch_spec.append('dp')
    if fsdp > 1 and total_batch_size % (dp * fsdp) == 0:
        batch_spec.append('fsdp')
    
    # Sequence dimension sharding
    if tp > 1:
        seq_spec.append('tp')
    
    # Create final spec
    if not batch_spec:
        batch_spec = None
    elif len(batch_spec) == 1:
        batch_spec = batch_spec[0]
    else:
        batch_spec = tuple(batch_spec)
    
    if not seq_spec:
        seq_spec = None
    elif len(seq_spec) == 1:
        seq_spec = seq_spec[0]
    
    return PartitionSpec(batch_spec, seq_spec)


def get_adaptive_step_partition_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    force_data_parallel: int = None,
    mini_batch_size: int = None,
) -> PartitionSpec:
    """
    Get appropriate step partition spec for training based on batch size.
    
    Returns:
        PartitionSpec for step partitioning
    """
    if num_devices is None:
        num_devices = jax.device_count()
    
    # Get mesh dimensions
    dp, fsdp, _, tp, sp = calculate_optimal_mesh_dims(
        total_batch_size, 8, num_devices,
        force_tensor_parallel=force_tensor_parallel,
        force_data_parallel=force_data_parallel,
        mini_batch_size=mini_batch_size
    )
    
    # Build partition spec
    batch_spec = []
    seq_spec = []
    
    # Batch dimension - only shard if evenly divisible
    if dp > 1 and total_batch_size % dp == 0:
        batch_spec.append('dp')
    if fsdp > 1 and total_batch_size % (dp * fsdp) == 0:
        batch_spec.append('fsdp')
    
    # Sequence dimension
    if tp > 1:
        seq_spec.append('tp')
    if sp > 1:
        seq_spec.append('sp')
    
    # Create final spec
    if not batch_spec:
        batch_spec = None
    elif len(batch_spec) == 1:
        batch_spec = batch_spec[0]
    else:
        batch_spec = tuple(batch_spec)
    
    if not seq_spec:
        seq_spec = 'sp' if sp > 1 else None
    elif len(seq_spec) == 1:
        seq_spec = seq_spec[0]
    else:
        seq_spec = tuple(seq_spec)
    
    return PartitionSpec(batch_spec, seq_spec)


def validate_mesh_config(
    dp: int, fsdp: int, tp: int, num_devices: int, total_batch_size: int
) -> bool:
    """Validate that mesh configuration is correct."""
    total = dp * fsdp * tp
    if total != num_devices:
        logger.error(
            f"Mesh validation failed: dp({dp}) × fsdp({fsdp}) × tp({tp}) = {total} "
            f"!= num_devices({num_devices})"
        )
        return False
    
    if dp > total_batch_size and total_batch_size > 0:
        logger.warning(
            f"Data parallel dim ({dp}) exceeds batch size ({total_batch_size}). "
            f"This may cause inefficiency."
        )
    
    return True


# Usage example for your specific case (TPU v4-32)
if __name__ == "__main__":
    # Your configuration
    config = {
        "total_batch_size": 1,
        "num_return_sequences": 1,
        "force_tensor_parallel": 1,
        "force_data_parallel": 1,
    }
    
    # Calculate mesh
    dp, fsdp, ep, tp, sp = calculate_optimal_mesh_dims(**config)
    print(f"Mesh dimensions: dp={dp}, fsdp={fsdp}, ep={ep}, tp={tp}, sp={sp}")
    print(f"Total devices used: {dp * fsdp * tp}")
    
    # Validate
    validate_mesh_config(dp, fsdp, tp, 32, config["total_batch_size"])