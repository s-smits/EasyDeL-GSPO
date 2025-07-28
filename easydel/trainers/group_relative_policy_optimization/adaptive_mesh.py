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
    
    logger.info(
        f"Calculating mesh for batch_size={total_batch_size}, "
        f"rollouts={num_return_sequences}, devices={num_devices}, "
        f"processes={num_workers}, devices_per_process={devices_per_process}, "
        f"force_tp={force_tensor_parallel}, mini_batch={mini_batch_size}"
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
        elif mini_batch_size is not None:
            # Calculate models needed based on mini_batch_size
            models_needed = max(1, total_batch_size // mini_batch_size)
            if models_needed > num_model_slots:
                logger.warning(
                    f"Need {models_needed} models for mini_batch_size={mini_batch_size}, "
                    f"but only have {num_model_slots} model slots. Using all available slots."
                )
            dp = min(models_needed, num_model_slots)
        else:
            # Default: use all available model slots for maximum parallelism
            dp = num_model_slots
        
        # FSDP is always 1 with tensor parallelism (each model uses TP for internal sharding)
        fsdp = 1
        
        logger.info(
            f"Using tensor parallel strategy: dp={dp}, fsdp={fsdp}, tp={tp} "
            f"({dp} models on {tp} TPUs each) across {num_workers} workers"
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
            # Only FSDP sharding
            return PartitionSpec('fsdp', 'tp')
        else:
            # Hybrid DP + FSDP
            return PartitionSpec(('dp', 'fsdp'), 'tp')
    
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
            # Only FSDP: each model spans multiple TPUs
            return PartitionSpec('fsdp', ('tp', 'sp'))
        elif fsdp == 1:
            # Only DP: multiple models, each spans multiple TPUs
            return PartitionSpec('dp', ('tp', 'sp'))
        else:
            # Hybrid: multiple models with FSDP across TPUs within each model
            return PartitionSpec(('dp', 'fsdp'), ('tp', 'sp'))
    
    # Original logic for non-TP cases
    if dp == 1:
        return PartitionSpec('fsdp', 'sp')
    elif fsdp == 1:
        return PartitionSpec('dp', 'sp')
    else:
        return PartitionSpec(('dp', 'fsdp'), 'sp')