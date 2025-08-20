#!/usr/bin/env python3
"""
Test script to verify the mesh configuration fix for GRPO trainer.
"""
import jax
import jax.numpy as jnp
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from unittest.mock import MagicMock, patch
import sys

def test_mesh_configuration():
    """Test that the mesh configuration fix works correctly."""
    print("Testing mesh configuration fix...")

    # Mock the necessary components
    config = MagicMock()
    config._name_or_path = "test-model"
    config.mesh = jax.sharding.Mesh(jax.devices(), ("dp",))

    model = MagicMock()
    model.config = config
    model.to_state.return_value = MagicMock()
    model.to_state.return_value.model = model
    model.to_state.return_value.shardings = MagicMock()

    # Create GRPO config with forced parallelism
    grpo_config = GRPOConfig(
        force_data_parallel=8,
        force_tensor_parallel=4,
        total_batch_size=2,
        num_return_sequences=4,
        max_prompt_length=512,
        max_completion_length=512,
    )

    # Mock the adaptive mesh configuration
    with patch('easydel.trainers.group_relative_policy_optimization.adaptive_mesh.configure_adaptive_mesh_inplace') as mock_configure:
        # Mock the returned mesh plan
        mock_plan = MagicMock()
        mock_plan.dp = 8
        mock_plan.fsdp = 1
        mock_plan.ep = 1
        mock_plan.tp = 4
        mock_plan.sp = 1
        mock_configure.return_value = mock_plan

        # Set mesh_dims on the config
        grpo_config.mesh_dims = (8, 1, 1, 4, 1)

        # Create trainer - this should not raise an error now
        try:
            trainer = GRPOTrainer(
                arguments=grpo_config,
                model=model,
                reward_funcs=[MagicMock()],
                processing_class=MagicMock(),
            )
            print("✓ GRPOTrainer created successfully without mesh mismatch error")
            return True
        except ValueError as e:
            if "Mesh DP=" in str(e) and "doesn't match requested DP=" in str(e):
                print(f"✗ Mesh configuration error still occurs: {e}")
                return False
            else:
                # Some other ValueError, re-raise it
                raise e
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False

if __name__ == "__main__":
    success = test_mesh_configuration()
    sys.exit(0 if success else 1)
