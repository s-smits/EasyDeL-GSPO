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
    mini_batch_size: int = None,
) -> tuple[int, int, int, int, int]:
    """
    Calculate optimal mesh dimensions for GRPO/GSPO training based on batch size and rollouts.
    
    Args:
        total_batch_size: Number of prompts in the batch
        num_return_sequences: Number of rollouts per prompt
        num_devices: Number of available devices (defaults to JAX device count)
        prefer_data_parallel: Whether to prioritize data parallelism over model parallelism
        force_tensor_parallel: Force specific TP dimension (enables sub-batch processing)
        mini_batch_size: Minimum batch size per model instance (used with TP)
        
    Returns:
        Tuple of (dp, fsdp, ep, tp, sp) dimensions
        
    Examples:
        - batch=1, rollouts=8, 4 TPUs → (1, 4, 1, 1, 1) - model sharding only
        - batch=1, rollouts=8, 4 TPUs, tp=2 → (1, 2, 1, 2, 1) - 2 models on 2 TPUs each
        - batch=2, rollouts=8, 4 TPUs → (2, 2, 1, 1, 1) - hybrid data + model sharding
        - batch=4, rollouts=8, 4 TPUs → (4, 1, 1, 1, 1) - data parallel only
        
    Multi-worker examples:
        - batch=8, 16 TPUs, 4 workers → optimize for cross-worker communication
    """
    if num_devices is None:
        num_devices = jax.device_count()
    num_workers = jax.process_count()
    
    # Calculate total compute requirements
    total_rollouts = total_batch_size * num_return_sequences
    
    devices_per_process = num_devices // num_workers if num_workers > 1 else num_devices
    
    # TPU v4 detection and adjustment
    is_tpu_v4 = False
    try:
        # Check if we're on TPU v4 by examining device properties
        device = jax.devices()[0]
        if hasattr(device, 'device_kind') and 'v4' in str(device.device_kind).lower():
            is_tpu_v4 = True
    except:
        pass
    
    # For TPU v4, each "device" is actually a megacore (2 TensorCores)
    # but batch sizes should be calculated based on actual memory capacity
    effective_devices = num_devices
    if is_tpu_v4 and num_workers > 1:
        # TPU v4-32: 16 megacores, but each has 32GB HBM (like 2 v3 cores)
        # For multi-worker training, account for the actual topology
        logger.info(f"Detected TPU v4 configuration with {num_devices} megacores")
        
    logger.info(
        f"Calculating mesh for batch_size={total_batch_size}, "
        f"rollouts={num_return_sequences}, devices={num_devices}, "
        f"processes={num_workers}, devices_per_process={devices_per_process}, "
        f"force_tp={force_tensor_parallel}, mini_batch={mini_batch_size}, "
        f"is_tpu_v4={is_tpu_v4}"
    )
    
    # Strategy 1: Force tensor parallelism (multi-worker aware)
    if force_tensor_parallel is not None:
        tp = force_tensor_parallel
        if num_devices % tp != 0:
            raise ValueError(f"num_devices ({num_devices}) must be divisible by tp ({tp})")
        
        num_model_slots = num_devices // tp  # How many models we can fit
        
        # For multi-worker setups, prefer spreading DP across workers
        if num_workers > 1 and total_batch_size >= num_workers:
            # Distribute data parallel across workers first
            dp_per_worker = max(1, total_batch_size // num_workers)
            dp = min(dp_per_worker * num_workers, num_model_slots)
            
            # Critical fix: Ensure we use all available devices
            devices_used = dp * tp
            if devices_used < num_devices:
                # Use remaining devices for FSDP
                remaining_devices = num_devices // devices_used
                if remaining_devices > 1:
                    fsdp = remaining_devices
                    logger.info(
                        f"Using remaining {remaining_devices} devices for FSDP: "
                        f"dp={dp}, fsdp={fsdp}, tp={tp} "
                        f"(total: {dp * fsdp * tp} devices)"
                    )
                else:
                    fsdp = 1
            else:
                fsdp = 1
        elif mini_batch_size is not None:
            # Calculate models needed based on mini_batch_size
            models_needed = max(1, total_batch_size // mini_batch_size)
            if models_needed > num_model_slots:
                logger.warning(
                    f"Need {models_needed} models for mini_batch_size={mini_batch_size}, "
                    f"but only have {num_model_slots} model slots. Using all available slots."
                )
            dp = min(models_needed, num_model_slots)
            
            # Ensure we use all available devices
            devices_used = dp * tp
            if devices_used < num_devices:
                remaining_devices = num_devices // devices_used
                fsdp = remaining_devices if remaining_devices > 1 else 1
            else:
                fsdp = 1
        else:
            # Default: use all available model slots for maximum parallelism
            dp = num_model_slots
            
            # Ensure we use all available devices
            devices_used = dp * tp
            if devices_used < num_devices:
                remaining_devices = num_devices // devices_used
                fsdp = remaining_devices if remaining_devices > 1 else 1
            else:
                fsdp = 1
        
        logger.info(
            f"Using tensor parallel strategy: dp={dp}, fsdp={fsdp}, tp={tp} "
            f"({dp} models, each using {fsdp * tp} TPUs: {fsdp} FSDP × {tp} TP) across {num_workers} workers"
        )
        
        # TPU v4 specific recommendations
        if is_tpu_v4 and num_workers == 4 and tp == 2:
            # Aim for 1–2 prompts per worker; never recommend 0
            recommended_per_worker = max(1, min(2, total_batch_size // num_workers))
            recommended_total_batch = recommended_per_worker * num_workers
            if total_batch_size != recommended_total_batch:
                logger.warning(
                    f"TPU v4-32 (4x2x2 topology) recommendation: "
                    f"Use batch_size={recommended_total_batch} "
                    f"({recommended_per_worker} per worker) for optimal memory usage. "
                    f"Current: {total_batch_size}"
                )
        return (dp, fsdp, 1, tp, 1)
    
    # Strategy 2: When batch_size >= num_devices, use pure data parallelism
    if total_batch_size >= num_devices:
        dp = num_devices
        fsdp = 1
        logger.info(f"Using data parallel strategy: dp={dp}, fsdp={fsdp}")
        return (dp, fsdp, 1, 1, 1)
    
    # Strategy 3: When batch_size < num_devices, use hybrid approach
    if prefer_data_parallel and total_batch_size > 1:
        # Find largest divisor of num_devices that's <= total_batch_size
        dp = total_batch_size
        remaining_devices = num_devices // dp
        fsdp = remaining_devices if remaining_devices > 0 else 1
        logger.info(f"Using hybrid strategy: dp={dp}, fsdp={fsdp}")
        return (dp, fsdp, 1, 1, 1)
    
    # Strategy 4: batch_size=1, use pure model parallelism
    dp = 1
    fsdp = num_devices
    logger.info(f"Using model parallel strategy: dp={dp}, fsdp={fsdp}")
    return (dp, fsdp, 1, 1, 1)


def get_adaptive_sharding_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    mini_batch_size: int = None,
) -> PartitionSpec:
    """
    Get appropriate sharding spec for input tensors based on batch size.
    
    Args:
        total_batch_size: Number of prompts in the batch
        num_devices: Number of available devices
        force_tensor_parallel: Force specific TP dimension
        mini_batch_size: Minimum batch size per model instance
        
    Returns:
        PartitionSpec for input sharding
    """
    if num_devices is None:
        num_devices = jax.device_count()
    
    # Get mesh dimensions to determine sharding strategy
    dp, fsdp, _, tp, _ = calculate_optimal_mesh_dims(
        total_batch_size, 8, num_devices, 
        force_tensor_parallel=force_tensor_parallel,
        mini_batch_size=mini_batch_size
    )
    
    # With tensor parallelism, always include tp in sequence dimension
    if tp > 1:
        if total_batch_size == 1 or (mini_batch_size and total_batch_size <= mini_batch_size):
            # Sub-batch processing: no batch sharding
            return PartitionSpec(None, 'tp')
        elif dp == 1:
            # Only FSDP sharding (no data parallelism)
            return PartitionSpec('fsdp', 'tp')
        elif fsdp == 1:
            # Only DP sharding (no FSDP)
            return PartitionSpec('dp', 'tp')
        else:
            # Hybrid DP + FSDP: only shard if batch_size is divisible by dp*fsdp
            if total_batch_size % (dp * fsdp) == 0:
                return PartitionSpec(('dp', 'fsdp'), 'tp')
            else:
                # Fallback to DP only if not evenly divisible
                return PartitionSpec('dp', 'tp')
    
    # Original logic for non-TP cases
    if total_batch_size == 1:
        return PartitionSpec(None, 'tp')
    elif total_batch_size >= num_devices:
        return PartitionSpec('dp', 'tp')
    else:
        return PartitionSpec(('dp', 'fsdp'), 'tp')


def get_adaptive_step_partition_spec(
    total_batch_size: int,
    num_devices: int = None,
    force_tensor_parallel: int = None,
    mini_batch_size: int = None,
) -> PartitionSpec:
    """
    Get appropriate step partition spec for training based on batch size.
    
    Args:
        total_batch_size: Number of prompts in the batch  
        num_devices: Number of available devices
        force_tensor_parallel: Force specific TP dimension
        mini_batch_size: Minimum batch size per model instance
        
    Returns:
        PartitionSpec for step partitioning
    """
    if num_devices is None:
        num_devices = jax.device_count()
    
    # Calculate mesh dimensions
    dp, fsdp, _, tp, sp = calculate_optimal_mesh_dims(
        total_batch_size, 8, num_devices,
        force_tensor_parallel=force_tensor_parallel,
        mini_batch_size=mini_batch_size
    )
    
    # Create appropriate partition spec based on mesh dimensions
    if tp > 1:
        # With tensor parallelism, include tp in sequence dimension
        if dp == 1:
            # Only FSDP. Shard batch dim across fsdp **only** if it divides
            # the batch size; otherwise, do not shard the batch dim.
            if total_batch_size % fsdp == 0:
                return PartitionSpec('fsdp', ('tp', 'sp'))
            else:
                return PartitionSpec(None, ('tp', 'sp'))
        elif fsdp == 1:
            # Only DP: multiple models, each spans multiple TPUs
            return PartitionSpec('dp', ('tp', 'sp'))
        else:
            # Hybrid DP + FSDP: only shard batch dim if divisible by dp*fsdp
            if total_batch_size % (dp * fsdp) == 0:
                return PartitionSpec(('dp', 'fsdp'), ('tp', 'sp'))
            elif total_batch_size % dp == 0:
                # Fallback to DP only when not evenly divisible by dp*fsdp
                return PartitionSpec('dp', ('tp', 'sp'))
            else:
                # Last resort: don't shard batch dimension
                return PartitionSpec(None, ('tp', 'sp'))
    
    # Original logic for non-TP cases
    if dp == 1:
        # Shard batch dim across fsdp only when divisible
        if total_batch_size % fsdp == 0:
            return PartitionSpec('fsdp', 'sp')
        else:
            return PartitionSpec(None, 'sp')
    elif fsdp == 1:
        return PartitionSpec('dp', 'sp')
    else:
        # Hybrid DP + FSDP for non-TP: same divisibility check
        if total_batch_size % (dp * fsdp) == 0:
            return PartitionSpec(('dp', 'fsdp'), 'sp')
        elif total_batch_size % dp == 0:
            return PartitionSpec('dp', 'sp')
        else:
            return PartitionSpec(None, 'sp')