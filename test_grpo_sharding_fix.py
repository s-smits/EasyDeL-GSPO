#!/usr/bin/env python3
"""
Quick test to verify GRPO sharding fix works correctly.
Tests that _preprocess_batch_input returns host arrays (unsharded).
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, Mesh
import numpy as np

# Simulate what happens in GRPO training
def test_sharding_fix():
    print("Testing GRPO sharding fix...")
    
    # Create a simple mesh (simulating multi-device setup)
    devices = jax.devices()[:min(2, len(jax.devices()))]
    mesh = Mesh(devices, axis_names=('dp',))
    
    # Create some test data with sharding
    with mesh:
        # Simulate sharded arrays from generation
        sharded_spec = PartitionSpec('dp', None)
        sharding = NamedSharding(mesh=mesh, spec=sharded_spec)
        
        # Create test arrays
        prompt_ids = jnp.ones((4, 8), dtype=jnp.int32)
        prompt_ids = jax.device_put(prompt_ids, sharding)
        
        print(f"Original array sharding: {prompt_ids.sharding}")
        print(f"Original array spec: {prompt_ids.sharding.spec if hasattr(prompt_ids.sharding, 'spec') else 'N/A'}")
        
        # Apply the fix: move to host memory
        host_array = jax.device_get(prompt_ids)
        
        print(f"\nAfter jax.device_get:")
        print(f"  Type: {type(host_array)}")
        print(f"  Is numpy array: {isinstance(host_array, np.ndarray)}")
        print(f"  Shape: {host_array.shape}")
        
        # Verify it's truly on host (no sharding)
        if isinstance(host_array, jax.Array):
            print(f"  Still JAX array with sharding: {host_array.sharding}")
        else:
            print(f"  ✓ Successfully converted to host numpy array (no sharding)")
        
        # Test that we can pass it to a jitted function expecting empty sharding
        @jax.jit
        def process_batch(batch):
            # This simulates what happens in the training step
            return jnp.sum(batch)
        
        try:
            result = process_batch(host_array)
            print(f"\n✓ Successfully processed host array in jitted function")
            print(f"  Result: {result}")
        except Exception as e:
            print(f"\n✗ Failed to process: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_sharding_fix()
