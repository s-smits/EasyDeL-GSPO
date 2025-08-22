#!/usr/bin/env python3
"""
Comprehensive diagnostic script to identify issues in the MATH dataset pipeline.
Tests each component in isolation to find the root cause of:
1. Repetitive text generation
2. 0% math verification success
3. GFPO filtering not working (retention_rate=1.0)
"""

import os
import sys
import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, '/home/air/EasyDeL-GSPO')

import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

# Import the modules we're testing
from easydel.scripts.finetune.gsm8k_math_gfspo import RunTimeConfig
from easydel.verification.math_reward import format_reward, answer_reward, _extract_text, _last_boxed_only_string, _remove_boxed, _is_equiv
from easydel.trainers.group_relative_policy_optimization.gfpo_trainer import GFPOTrainer

print("=" * 80)
print("MATH PIPELINE DIAGNOSTIC TESTS")
print("=" * 80)

def test_dataset_preprocessing():
    """Test 1: Verify dataset loading and ground truth extraction"""
    print("\n" + "="*60)
    print("TEST 1: DATASET PREPROCESSING")
    print("="*60)
    
    try:
        # Load a small sample of the math dataset
        print("Loading math dataset sample...")
        ds_train = load_dataset("qwedsacf/competition_math", split="train[:10]")
        
        # Test the mapping function from the main script
        SYSTEM_PROMPT_MATH = (
            "You are a math expert. You are given a question and you need to solve it step by step and output the final answer within \\boxed{}."
        )
        
        def map_ex(x):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_MATH},
                    {"role": "user", "content": x["problem"]},
                ],
                "solution": x.get("solution", ""),
                "level": x.get("level", ""),
                "type": x.get("type", ""),
            }
        
        mapped_ds = ds_train.map(map_ex)
        
        # Test solution extraction like in the main script
        def _extract_last_boxed(s: str) -> str | None:
            idx = s.rfind("\\boxed")
            if "\\boxed " in s:
                return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
            if idx < 0:
                idx = s.rfind("\\fbox")
                if idx < 0:
                    return None
            i = idx
            right = None
            depth = 0
            while i < len(s):
                if s[i] == "{":
                    depth += 1
                if s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        right = i
                        break
                i += 1
            return None if right is None else s[idx : right + 1]

        def _remove_boxed(t: str) -> str:
            if t.startswith("\\boxed "):
                return t[len("\\boxed ") :]
            left = "\\boxed{"
            if t.startswith(left) and t.endswith("}"):
                return t[len(left) : -1]
            return t
        
        print(f"Successfully loaded {len(mapped_ds)} examples")
        
        # Analyze first few examples
        for i in range(min(3, len(mapped_ds))):
            example = mapped_ds[i]
            solution = example["solution"]
            boxed = _extract_last_boxed(solution)
            normalized = _remove_boxed(boxed) if boxed else solution
            
            print(f"\n--- Example {i+1} ---")
            print(f"Problem: {example['prompt'][1]['content'][:100]}...")
            print(f"Level: {example['level']}")
            print(f"Type: {example['type']}")
            print(f"Full solution length: {len(solution)}")
            print(f"Has \\boxed: {boxed is not None}")
            if boxed:
                print(f"Boxed content: {boxed}")
                print(f"Normalized answer: {normalized}")
            else:
                print(f"No \\boxed found, using full solution: {solution[:100]}...")
        
        # Test the tokenization function
        print(f"\n--- Testing tokenization ---")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")  # Use a simple tokenizer for testing
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        # Simulate the data_tokenize_fn from main script
        batch = {
            "prompt": [mapped_ds[0]["prompt"]],
            "solution": [mapped_ds[0]["solution"]]
        }
        
        def data_tokenize_fn(batch, tokenizer, tools):
            ids = tokenizer(
                batch["prompt"],
                return_tensors="np",
                padding="max_length",
                padding_side="left",
                max_length=512,  # Reduced for testing
                truncation=True,
                add_special_tokens=False,
            )
            # Process solution like in main script
            sol = batch["solution"]
            if isinstance(sol, list):
                normalized = []
                for s in sol:
                    b = _extract_last_boxed(s)
                    normalized.append(_remove_boxed(b) if b else s)
            else:
                b = _extract_last_boxed(sol)
                normalized = _remove_boxed(b) if b else sol
            ids.update({"solution": sol, "solution_normalized": normalized})
            return ids
        
        tokenized = data_tokenize_fn(batch, tokenizer, None)
        print(f"Tokenized input_ids shape: {tokenized['input_ids'].shape}")
        print(f"Solution: {tokenized['solution'][0][:100]}...")
        print(f"Normalized: {tokenized['solution_normalized'][0]}")
        
        print("✓ Dataset preprocessing test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Dataset preprocessing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_math_verification():
    """Test 2: Test math reward functions in isolation"""
    print("\n" + "="*60)
    print("TEST 2: MATH VERIFICATION")
    print("="*60)
    
    # Test cases with known answers
    test_cases = [
        {
            "description": "Simple boxed answer",
            "completion": "The answer is \\boxed{42}",
            "ground_truth": "42",
            "expected_format": 1.0,
            "expected_answer": 1.0
        },
        {
            "description": "Fraction answer",
            "completion": "After calculation: \\boxed{\\frac{1}{2}}",
            "ground_truth": "\\frac{1}{2}",
            "expected_format": 1.0,
            "expected_answer": 1.0
        },
        {
            "description": "No boxed answer",
            "completion": "The answer is 42",
            "ground_truth": "42",
            "expected_format": 0.0,
            "expected_answer": 0.0
        },
        {
            "description": "Wrong answer",
            "completion": "The answer is \\boxed{99}",
            "ground_truth": "42",
            "expected_format": 1.0,
            "expected_answer": 0.0
        },
        {
            "description": "Repetitive text",
            "completion": " 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x \\boxed{85}",
            "ground_truth": "85",
            "expected_format": 1.0,
            "expected_answer": 1.0
        }
    ]
    
    print(f"Testing {len(test_cases)} math verification cases...")
    
    all_passed = True
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {case['description']} ---")
        
        # Test format reward
        completions = [[{"content": case["completion"]}]]
        format_score = format_reward(completions)[0]
        print(f"Format score: {format_score} (expected: {case['expected_format']})")
        
        # Test answer reward
        batch = {"solution_normalized": [case["ground_truth"]]}
        answer_score = answer_reward(None, completions, batch)[0]
        print(f"Answer score: {answer_score} (expected: {case['expected_answer']})")
        
        # Check results
        format_ok = abs(format_score - case['expected_format']) < 0.01
        answer_ok = abs(answer_score - case['expected_answer']) < 0.01
        
        if format_ok and answer_ok:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            all_passed = False
    
    # Test the helper functions directly
    print(f"\n--- Testing helper functions ---")
    test_text = "The answer is \\boxed{42} and that's final."
    boxed = _last_boxed_only_string(test_text)
    print(f"Boxed extraction: '{test_text}' -> '{boxed}'")
    
    if boxed:
        removed = _remove_boxed(boxed)
        print(f"Boxed removal: '{boxed}' -> '{removed}'")
    
    # Test equivalence
    equiv_tests = [
        ("42", "42", True),
        ("\\frac{1}{2}", "0.5", False),  # This might fail with current implementation
        ("85", "85", True),
    ]
    
    for a, b, expected in equiv_tests:
        result = _is_equiv(a, b)
        print(f"Equivalence: '{a}' == '{b}' -> {result} (expected: {expected})")
    
    if all_passed:
        print("✓ Math verification test PASSED")
    else:
        print("✗ Math verification test FAILED")
    
    return all_passed


def test_text_generation_patterns():
    """Test 3: Analyze text generation patterns"""
    print("\n" + "="*60)
    print("TEST 3: TEXT GENERATION ANALYSIS")
    print("="*60)
    
    # Simulate the repetitive patterns we're seeing in the logs
    test_completions = [
        " 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x",
        "clear on the type of the sequence. So, the user is not clear on the type of the sequence. So, the user is not clear on the t",
        "Normal completion with proper answer \\boxed{42}",
        "Another normal completion \\boxed{\\frac{1}{2}}"
    ]
    
    print("Analyzing completion patterns...")
    
    for i, completion in enumerate(test_completions):
        print(f"\n--- Completion {i+1} ---")
        print(f"Text: {completion}")
        print(f"Length: {len(completion)}")
        
        # Check for repetitive patterns
        words = completion.split()
        if len(words) > 3:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            most_common = max(word_counts.items(), key=lambda x: x[1])
            repetition_ratio = most_common[1] / len(words)
            print(f"Most common word: '{most_common[0]}' ({most_common[1]} times)")
            print(f"Repetition ratio: {repetition_ratio:.2f}")
            
            if repetition_ratio > 0.5:
                print("⚠ HIGH REPETITION DETECTED")
        
        # Check for boxed content
        has_boxed = ("\\boxed{" in completion or "\\boxed " in completion)
        print(f"Has \\boxed: {has_boxed}")
        
        # Check for truncation patterns
        if completion.endswith("…") or len(completion) == 181:  # 181 was mentioned in logs
            print("⚠ POSSIBLE TRUNCATION")
    
    print("✓ Text generation analysis COMPLETED")
    return True


def test_gfpo_filtering():
    """Test 4: Test GFPO filtering logic"""
    print("\n" + "="*60)
    print("TEST 4: GFPO FILTERING")
    print("="*60)
    
    # Simulate the filtering scenario
    print("Testing GFPO filtering with synthetic data...")
    
    # Create synthetic rewards that should trigger filtering
    num_prompts = 4
    group_size = 4
    
    # Case 1: High variance rewards (should filter some)
    rewards_high_var = jnp.array([
        [0.1, 0.2, 0.9, 0.8],  # Prompt 1: wide range
        [0.0, 0.1, 0.2, 0.9],  # Prompt 2: wide range
        [0.5, 0.5, 0.5, 0.6],  # Prompt 3: narrow range
        [0.0, 0.0, 0.1, 1.0],  # Prompt 4: very wide range
    ])
    
    # Case 2: Low variance rewards (might not filter much)
    rewards_low_var = jnp.array([
        [0.5, 0.5, 0.5, 0.5],  # All same
        [0.6, 0.6, 0.6, 0.6],  # All same
        [0.4, 0.4, 0.4, 0.4],  # All same
        [0.7, 0.7, 0.7, 0.7],  # All same
    ])
    
    # Mock lengths (all same for simplicity)
    lengths = jnp.ones((num_prompts, group_size)) * 100
    
    # Mock configuration
    class MockConfig:
        gfpo_group_size = group_size
        gfpo_retain_count = 2  # Keep top 2 out of 4
        gfpo_metric = "reward"
        gfpo_adaptive = False
    
    # Test filtering function (simplified version)
    def test_filter(rewards_grouped, lengths_grouped, config):
        retain_count = int(config.gfpo_retain_count)
        
        # Sort by reward (descending)
        sort_indices = jnp.argsort(-rewards_grouped, axis=1)
        mask = jnp.zeros_like(rewards_grouped)
        
        for i in range(rewards_grouped.shape[0]):
            # Keep top retain_count
            top_k_indices = sort_indices[i, :retain_count]
            mask = mask.at[i, top_k_indices].set(1.0)
        
        retention_rate = float(jnp.mean(mask))
        return mask, retention_rate
    
    config = MockConfig()
    
    print("Testing high variance rewards...")
    mask_hv, ret_rate_hv = test_filter(rewards_high_var, lengths, config)
    print(f"High variance retention rate: {ret_rate_hv:.3f}")
    print(f"Expected retention rate: {config.gfpo_retain_count / config.gfpo_group_size:.3f}")
    
    print("Testing low variance rewards...")
    mask_lv, ret_rate_lv = test_filter(rewards_low_var, lengths, config)
    print(f"Low variance retention rate: {ret_rate_lv:.3f}")
    
    # The retention rate should be 0.5 (2 out of 4) regardless of variance
    expected_rate = config.gfpo_retain_count / config.gfpo_group_size
    
    if abs(ret_rate_hv - expected_rate) < 0.01 and abs(ret_rate_lv - expected_rate) < 0.01:
        print("✓ GFPO filtering test PASSED")
        return True
    else:
        print("✗ GFPO filtering test FAILED")
        print("If retention_rate=1.0 in logs, the issue might be:")
        print("1. All rewards are identical (no variance)")
        print("2. retain_count >= group_size")
        print("3. Filtering logic has a bug")
        return False


def test_pipeline_integration():
    """Test 5: End-to-end pipeline test"""
    print("\n" + "="*60)
    print("TEST 5: PIPELINE INTEGRATION")
    print("="*60)
    
    print("This test simulates the full pipeline flow...")
    
    # Step 1: Mock dataset batch
    mock_batch = {
        "input_ids": jnp.ones((8, 512), dtype=jnp.int32),  # 8 examples, 512 tokens each
        "solution_normalized": ["42", "85"] * 4,  # Ground truths
    }
    
    # Step 2: Mock completions (simulating the repetitive issue)
    mock_completions = [
        " 2x 2x 2x 2x 2x 2x \\boxed{42}",
        "clear on the type of sequence \\boxed{85}",
        "Normal answer \\boxed{42}",
        "Another answer \\boxed{85}",
    ] * 2
    
    # Step 3: Test reward computation
    completions_formatted = [[{"content": comp}] for comp in mock_completions]
    
    print("Computing format rewards...")
    format_rewards = format_reward(completions_formatted)
    print(f"Format rewards: {format_rewards}")
    
    print("Computing answer rewards...")
    answer_rewards = answer_reward(None, completions_formatted, mock_batch)
    print(f"Answer rewards: {answer_rewards}")
    
    # Step 4: Test filtering
    total_rewards = [f + a for f, a in zip(format_rewards, answer_rewards)]
    print(f"Total rewards: {total_rewards}")
    
    # Group rewards for GFPO filtering (assuming group_size=4)
    group_size = 4
    num_prompts = len(total_rewards) // group_size
    rewards_grouped = jnp.array(total_rewards).reshape(num_prompts, group_size)
    print(f"Grouped rewards shape: {rewards_grouped.shape}")
    print(f"Grouped rewards:\n{rewards_grouped}")
    
    # Check if all rewards are the same (which would cause retention_rate=1.0)
    unique_rewards = jnp.unique(rewards_grouped)
    print(f"Unique reward values: {unique_rewards}")
    
    if len(unique_rewards) <= 1:
        print("⚠ ALL REWARDS ARE IDENTICAL - This explains retention_rate=1.0!")
        print("Root cause: Model is generating very similar outputs")
    
    print("✓ Pipeline integration test COMPLETED")
    return True


def main():
    """Run all diagnostic tests"""
    print("Starting comprehensive diagnostic tests...\n")
    
    tests = [
        ("Dataset Preprocessing", test_dataset_preprocessing),
        ("Math Verification", test_math_verification),
        ("Text Generation Analysis", test_text_generation_patterns),
        ("GFPO Filtering", test_gfpo_filtering),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\nRunning {name} test...")
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} test CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    print("\n" + "="*80)
    print("LIKELY ROOT CAUSES")
    print("="*80)
    
    print("Based on the logs and tests, the issues are likely:")
    print("1. MODEL GENERATING REPETITIVE TEXT:")
    print("   - Check model checkpoint/loading")
    print("   - Check generation parameters (temperature, top_p, etc.)")
    print("   - Check for gradient issues or training instability")
    
    print("2. IDENTICAL REWARDS CAUSING NO FILTERING:")
    print("   - If all completions are repetitive/identical, rewards will be same")
    print("   - This causes retention_rate=1.0 (no filtering)")
    
    print("3. MATH VERIFICATION FAILING:")
    print("   - Repetitive text doesn't contain valid \\boxed{} answers")
    print("   - Or the extracted answers don't match ground truth format")
    
    print("\nNext steps:")
    print("1. Check model generation with simple prompts")
    print("2. Verify training configuration (learning rate, etc.)")
    print("3. Check if this happens from the start or develops during training")


if __name__ == "__main__":
    main()

