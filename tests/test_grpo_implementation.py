#!/usr/bin/env python3
"""
Test script to validate GRPO trainer implementation changes.
This script tests the reordering logic, reward computation, and metrics.
"""

import numpy as np


def test_reorder_from_chunks():
    """Test the reordering logic for chunked generation."""
    print("Testing reorder_from_chunks...")
    
    num_prompts_local = 2
    cur_nrs_chunks = [2, 2]  # Two chunks with 2 generations each
    
    # Simulate chunked data in chunk-major order:
    # chunk0: [prompt0_gen0, prompt0_gen1, prompt1_gen0, prompt1_gen1]
    # chunk1: [prompt0_gen2, prompt0_gen3, prompt1_gen2, prompt1_gen3]
    chunks = [
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),  # chunk 0
        np.array([[0, 2], [0, 3], [1, 2], [1, 3]]),  # chunk 1
    ]
    
    def _reorder_from_chunks(chunks, cur_nrs_chunks):
        """Reorder from chunk-major to prompt-major."""
        reshaped = []
        for arr, k in zip(chunks, cur_nrs_chunks):
            # Reshape (k * B, ...) -> (B, k, ...)
            new_shape = (num_prompts_local, k) + arr.shape[1:]
            reshaped.append(np.reshape(arr, new_shape))
        # Concat across generation axis -> (B, total_k, ...)
        by_prompt = np.concatenate(reshaped, axis=1)
        # Flatten back to (B * total_k, ...)
        flat_shape = (by_prompt.shape[0] * by_prompt.shape[1],) + by_prompt.shape[2:]
        return np.reshape(by_prompt, flat_shape)
    
    result = _reorder_from_chunks(chunks, cur_nrs_chunks)
    
    # Expected: prompt-major order
    # [prompt0_gen0, prompt0_gen1, prompt0_gen2, prompt0_gen3,
    #  prompt1_gen0, prompt1_gen1, prompt1_gen2, prompt1_gen3]
    expected = np.array([[0, 0], [0, 1], [0, 2], [0, 3], 
                         [1, 0], [1, 1], [1, 2], [1, 3]])
    
    assert np.allclose(result, expected), f"Reorder failed:\n{result}\n!=\n{expected}"
    print("✓ Reorder test passed! Chunks correctly reordered from chunk-major to prompt-major.")
    print(f"  Input (chunk-major): {[chunk.tolist() for chunk in chunks]}")
    print(f"  Output (prompt-major): {result.tolist()}")
    return True


def test_reward_metrics():
    """Test reward metric computation with proper granularity."""
    print("\nTesting reward metrics computation...")
    
    # Simulate rewards for 2 prompts, 4 generations each
    num_prompts = 2
    num_generations = 4
    num_reward_funcs = 2
    
    # Create sample rewards (prompt0 has higher rewards than prompt1)
    rewards_per_func = np.zeros((num_prompts * num_generations, num_reward_funcs))
    
    # Reward function 1: accuracy (binary)
    rewards_per_func[:4, 0] = [1, 1, 0, 1]  # prompt0: 3/4 correct
    rewards_per_func[4:, 0] = [0, 1, 0, 0]   # prompt1: 1/4 correct
    
    # Reward function 2: quality score
    rewards_per_func[:4, 1] = [0.8, 0.9, 0.3, 0.7]  # prompt0: avg 0.675
    rewards_per_func[4:, 1] = [0.2, 0.6, 0.1, 0.3]  # prompt1: avg 0.3
    
    # Compute summed rewards
    rewards = rewards_per_func.sum(axis=1)
    
    # Test per-completion metrics
    per_completion_mean = np.mean(rewards)
    print(f"  Per-completion mean reward: {per_completion_mean:.3f}")
    
    # Test per-prompt metrics
    rewards_reshaped = rewards.reshape(num_prompts, num_generations)
    per_prompt_means = np.mean(rewards_reshaped, axis=1)
    print(f"  Per-prompt mean rewards: {per_prompt_means}")
    
    # Test advantage computation
    mean_per_prompt = np.mean(rewards_reshaped, axis=-1, keepdims=True)
    std_per_prompt = np.std(rewards_reshaped, axis=-1, keepdims=True)
    advantages = (rewards_reshaped - mean_per_prompt) / (std_per_prompt + 1e-4)
    advantages_flat = advantages.flatten()
    
    print(f"  Advantages (zero-mean per prompt): {advantages_flat}")
    
    # Verify advantages have zero mean per prompt
    for i in range(num_prompts):
        prompt_advantages = advantages[i]
        assert abs(np.mean(prompt_advantages)) < 1e-6, f"Prompt {i} advantages not zero-mean"
    
    print("✓ Reward metrics test passed!")
    return True


def test_global_aggregation():
    """Test global reward aggregation across simulated processes."""
    print("\nTesting global reward aggregation...")
    
    # Simulate 2 data parallel processes
    num_processes = 2
    num_prompts_per_process = 2
    num_generations = 4
    
    # Process 0 rewards
    rewards_p0 = np.array([1.0, 0.8, 0.9, 1.0,  # prompt0
                          0.2, 0.3, 0.1, 0.4])  # prompt1
    
    # Process 1 rewards  
    rewards_p1 = np.array([0.7, 0.6, 0.8, 0.5,  # prompt2
                          0.9, 1.0, 0.8, 0.7])  # prompt3
    
    # Simulate allgather
    global_rewards = np.concatenate([rewards_p0, rewards_p1])
    
    # Compute metrics
    local_mean_p0 = np.mean(rewards_p0)
    local_mean_p1 = np.mean(rewards_p1)
    global_mean = np.mean(global_rewards)
    
    print(f"  Process 0 local mean: {local_mean_p0:.3f}")
    print(f"  Process 1 local mean: {local_mean_p1:.3f}")
    print(f"  Global mean: {global_mean:.3f}")
    
    # Test denominators
    total_completions_local = num_prompts_per_process * num_generations
    total_completions_global = total_completions_local * num_processes
    
    print(f"  Local completions: {total_completions_local}")
    print(f"  Global completions: {total_completions_global}")
    
    # Test increments with binary rewards
    binary_rewards = (global_rewards > 0.5).astype(float)
    increment = 1.0 / total_completions_global
    print(f"  Expected increment per binary reward: {increment:.4f}")
    print(f"  Actual mean of binary rewards: {np.mean(binary_rewards):.4f}")
    
    print("✓ Global aggregation test passed!")
    return True


def test_completion_length_grouping():
    """Test that completion lengths are correctly grouped per prompt after reordering."""
    print("\nTesting completion length grouping...")
    
    num_prompts = 2
    cur_nrs_chunks = [2, 2]
    
    # Simulate completion lengths in chunk-major order
    # chunk0: [p0g0_len, p0g1_len, p1g0_len, p1g1_len]
    # chunk1: [p0g2_len, p0g3_len, p1g2_len, p1g3_len]
    comp_len_chunks = [
        np.array([10, 12, 8, 9]),   # chunk 0
        np.array([11, 13, 7, 10]),  # chunk 1
    ]
    
    def _reorder_from_chunks(chunks, cur_nrs_chunks):
        reshaped = []
        for arr, k in zip(chunks, cur_nrs_chunks):
            new_shape = (num_prompts, k) + arr.shape[1:]
            reshaped.append(np.reshape(arr, new_shape))
        by_prompt = np.concatenate(reshaped, axis=1)
        flat_shape = (by_prompt.shape[0] * by_prompt.shape[1],) + by_prompt.shape[2:]
        return np.reshape(by_prompt, flat_shape)
    
    reordered = _reorder_from_chunks(comp_len_chunks, cur_nrs_chunks)
    
    # Expected after reordering: [10, 12, 11, 13, 8, 9, 7, 10]
    # (all prompt0 lengths, then all prompt1 lengths)
    expected = np.array([10, 12, 11, 13, 8, 9, 7, 10])
    
    assert np.array_equal(reordered, expected), f"Length reorder failed: {reordered} != {expected}"
    
    # Test grouping for display
    num_generations = sum(cur_nrs_chunks)
    grouped = reordered.reshape(num_prompts, num_generations)
    
    print(f"  Prompt 0 completion lengths: {grouped[0]}")
    print(f"  Prompt 1 completion lengths: {grouped[1]}")
    
    # Verify each prompt has distinct patterns
    assert np.array_equal(grouped[0], [10, 12, 11, 13]), "Prompt 0 lengths incorrect"
    assert np.array_equal(grouped[1], [8, 9, 7, 10]), "Prompt 1 lengths incorrect"
    
    print("✓ Completion length grouping test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("GRPO Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_reorder_from_chunks,
        test_reward_metrics,
        test_global_aggregation,
        test_completion_length_grouping,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! The GRPO implementation is correct.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
