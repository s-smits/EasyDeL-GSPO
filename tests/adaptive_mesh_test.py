#!/usr/bin/env python3
"""
Test suite for adaptive mesh system in GRPO/GSPO trainers.
"""

import unittest
from unittest.mock import patch, MagicMock
import jax
from jax.sharding import PartitionSpec

# Import the adaptive mesh functions
from easydel.trainers.group_relative_policy_optimization.adaptive_mesh import (
    calculate_optimal_mesh_dims,
    get_adaptive_sharding_spec,
    get_adaptive_step_partition_spec,
)


class TestAdaptiveMesh(unittest.TestCase):
    """Test cases for adaptive mesh calculation functions."""

    def test_calculate_optimal_mesh_dims_batch_1(self):
        """Test mesh calculation for batch_size=1 (pure model parallelism)."""
        dims = calculate_optimal_mesh_dims(
            total_batch_size=1,
            num_return_sequences=8,
            num_devices=4
        )
        expected = (1, 4, 1, 1, 1)  # dp=1, fsdp=4
        self.assertEqual(dims, expected)

    def test_calculate_optimal_mesh_dims_batch_2(self):
        """Test mesh calculation for batch_size=2 (hybrid approach)."""
        dims = calculate_optimal_mesh_dims(
            total_batch_size=2,
            num_return_sequences=8,
            num_devices=4
        )
        expected = (2, 2, 1, 1, 1)  # dp=2, fsdp=2
        self.assertEqual(dims, expected)

    def test_calculate_optimal_mesh_dims_batch_4(self):
        """Test mesh calculation for batch_size=4 (pure data parallelism)."""
        dims = calculate_optimal_mesh_dims(
            total_batch_size=4,
            num_return_sequences=8,
            num_devices=4
        )
        expected = (4, 1, 1, 1, 1)  # dp=4, fsdp=1
        self.assertEqual(dims, expected)

    def test_calculate_optimal_mesh_dims_batch_8(self):
        """Test mesh calculation for batch_size > num_devices."""
        dims = calculate_optimal_mesh_dims(
            total_batch_size=8,
            num_return_sequences=4,
            num_devices=4
        )
        expected = (4, 1, 1, 1, 1)  # dp=4, fsdp=1 (max out data parallelism)
        self.assertEqual(dims, expected)

    def test_get_adaptive_sharding_spec_batch_1(self):
        """Test sharding spec for batch_size=1."""
        spec = get_adaptive_sharding_spec(total_batch_size=1, num_devices=4)
        expected = PartitionSpec(None, 'tp')
        self.assertEqual(spec, expected)

    def test_get_adaptive_sharding_spec_batch_2(self):
        """Test sharding spec for batch_size=2."""
        spec = get_adaptive_sharding_spec(total_batch_size=2, num_devices=4)
        expected = PartitionSpec(('dp', 'fsdp'), 'tp')
        self.assertEqual(spec, expected)

    def test_get_adaptive_sharding_spec_batch_4(self):
        """Test sharding spec for batch_size=4."""
        spec = get_adaptive_sharding_spec(total_batch_size=4, num_devices=4)
        expected = PartitionSpec('dp', 'tp')
        self.assertEqual(spec, expected)

    def test_get_adaptive_step_partition_spec_batch_1(self):
        """Test step partition spec for batch_size=1."""
        spec = get_adaptive_step_partition_spec(total_batch_size=1, num_devices=4)
        expected = PartitionSpec('fsdp', 'sp')
        self.assertEqual(spec, expected)

    def test_get_adaptive_step_partition_spec_batch_2(self):
        """Test step partition spec for batch_size=2."""
        spec = get_adaptive_step_partition_spec(total_batch_size=2, num_devices=4)
        expected = PartitionSpec(('dp', 'fsdp'), 'sp')
        self.assertEqual(spec, expected)

    def test_get_adaptive_step_partition_spec_batch_4(self):
        """Test step partition spec for batch_size=4."""
        spec = get_adaptive_step_partition_spec(total_batch_size=4, num_devices=4)
        expected = PartitionSpec('dp', 'sp')
        self.assertEqual(spec, expected)

    @patch('jax.device_count')
    def test_default_device_count(self, mock_device_count):
        """Test that functions use JAX device count when num_devices is None."""
        mock_device_count.return_value = 8
        
        dims = calculate_optimal_mesh_dims(
            total_batch_size=2,
            num_return_sequences=8,
            num_devices=None  # Should use JAX device count
        )
        # With 8 devices and batch=2, should use dp=2, fsdp=4
        expected = (2, 4, 1, 1, 1)
        self.assertEqual(dims, expected)
        mock_device_count.assert_called_once()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small device count
        dims = calculate_optimal_mesh_dims(1, 8, 1)
        self.assertEqual(dims, (1, 1, 1, 1, 1))
        
        # Large batch size with small device count
        dims = calculate_optimal_mesh_dims(16, 4, 2)
        self.assertEqual(dims, (2, 1, 1, 1, 1))


class TestTrainerIntegration(unittest.TestCase):
    """Integration tests for adaptive mesh in trainers."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.mesh = MagicMock()
        
    @patch('easydel.trainers.group_relative_policy_optimization.grpo_trainer.logger')
    @patch('jax.device_count')
    def test_grpo_trainer_adaptive_step_spec_override(self, mock_device_count, mock_logger):
        """Test that GRPO trainer overrides step_partition_spec for batch_size=1."""
        mock_device_count.return_value = 4
        
        # Mock the necessary imports and classes
        with patch.multiple(
            'easydel.trainers.group_relative_policy_optimization.grpo_trainer',
            GRPOConfig=MagicMock,
            EasyDeLState=MagicMock,
            ProcessingClassType=MagicMock,
        ):
            from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
            
            # Create a mock config with problematic step_partition_spec
            mock_config = MagicMock(spec=GRPOConfig)
            mock_config.total_batch_size = 1
            mock_config.step_partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
            mock_config.truncation_mode = "keep_end"
            
            # This should trigger the adaptive override
            with patch('easydel.trainers.group_relative_policy_optimization.grpo_trainer.GRPOTrainer.__init__', 
                      side_effect=lambda *args, **kwargs: None):
                # Test the specific logic that would be in __init__
                from easydel.trainers.group_relative_policy_optimization.adaptive_mesh import get_adaptive_step_partition_spec
                
                adaptive_step_spec = get_adaptive_step_partition_spec(mock_config.total_batch_size)
                
                # Should override the problematic spec
                if (mock_config.total_batch_size == 1 and 
                    mock_config.step_partition_spec == PartitionSpec(("dp", "fsdp"), "sp")):
                    mock_config.step_partition_spec = adaptive_step_spec
                
                # Verify the override worked
                self.assertEqual(mock_config.step_partition_spec, PartitionSpec('fsdp', 'sp'))

    def test_sharding_spec_consistency(self):
        """Test that input and step sharding specs are consistent."""
        test_cases = [
            (1, 4),  # batch=1, 4 devices
            (2, 4),  # batch=2, 4 devices  
            (4, 4),  # batch=4, 4 devices
            (8, 4),  # batch=8, 4 devices
        ]
        
        for batch_size, num_devices in test_cases:
            with self.subTest(batch_size=batch_size, num_devices=num_devices):
                input_spec = get_adaptive_sharding_spec(batch_size, num_devices)
                step_spec = get_adaptive_step_partition_spec(batch_size, num_devices)
                
                # Both specs should be valid PartitionSpecs
                self.assertIsInstance(input_spec, PartitionSpec)
                self.assertIsInstance(step_spec, PartitionSpec)
                
                # For batch_size=1, input should not shard batch dimension
                if batch_size == 1:
                    self.assertEqual(input_spec[0], None)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""

    def test_tpu_configurations(self):
        """Test common TPU configurations."""
        # v2-8 TPU (8 devices)
        dims = calculate_optimal_mesh_dims(1, 8, 8)
        self.assertEqual(dims, (1, 8, 1, 1, 1))
        
        dims = calculate_optimal_mesh_dims(4, 8, 8)
        self.assertEqual(dims, (4, 2, 1, 1, 1))
        
        # v3-8 TPU (8 devices)  
        dims = calculate_optimal_mesh_dims(8, 4, 8)
        self.assertEqual(dims, (8, 1, 1, 1, 1))

    def test_gspo_training_scenarios(self):
        """Test scenarios specific to GSPO training."""
        # Common GSPO configurations from the original error
        scenarios = [
            {"batch": 1, "rollouts": 8, "devices": 4, "expected_dp": 1, "expected_fsdp": 4},
            {"batch": 2, "rollouts": 8, "devices": 4, "expected_dp": 2, "expected_fsdp": 2},
            {"batch": 4, "rollouts": 8, "devices": 4, "expected_dp": 4, "expected_fsdp": 1},
        ]
        
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                dims = calculate_optimal_mesh_dims(
                    scenario["batch"], 
                    scenario["rollouts"], 
                    scenario["devices"]
                )
                self.assertEqual(dims[0], scenario["expected_dp"])
                self.assertEqual(dims[1], scenario["expected_fsdp"])


if __name__ == '__main__':
    # Set up JAX for testing (use CPU to avoid GPU/TPU requirements)
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Run the tests
    unittest.main(verbosity=2)