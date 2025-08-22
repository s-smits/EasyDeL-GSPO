#!/usr/bin/env python3
"""
Simplified diagnostic script that doesn't require JAX.
Focuses on identifying core issues in text processing and dataset handling.
"""

import os
import sys
import re
import json
from typing import List, Dict, Any

print("=" * 80)
print("SIMPLIFIED MATH PIPELINE DIAGNOSTIC")
print("=" * 80)

def test_text_patterns():
    """Test the specific patterns we're seeing in the logs"""
    print("\n" + "="*60)
    print("TEST 1: ANALYZING PROBLEMATIC TEXT PATTERNS")
    print("="*60)
    
    # These are the exact patterns from the user's logs
    problematic_texts = [
        " 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x…",
        " clear on the type of the sequence. So, the user is not clear on the type of the sequence. So, the user is not clear on the t…"
    ]
    
    print("Analyzing the problematic text patterns from logs:")
    
    for i, text in enumerate(problematic_texts):
        print(f"\n--- Pattern {i+1} ---")
        print(f"Text preview: {text[:100]}...")
        print(f"Full length: {len(text)}")
        
        # Check for repetitive patterns
        words = text.split()
        if len(words) > 1:
            # Count word frequencies
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Find most repeated word
            if word_counts:
                most_common = max(word_counts.items(), key=lambda x: x[1])
                total_words = len(words)
                repetition_ratio = most_common[1] / total_words if total_words > 0 else 0
                
                print(f"Most repeated word: '{most_common[0]}' appears {most_common[1]} times")
                print(f"Total words: {total_words}")
                print(f"Repetition ratio: {repetition_ratio:.2f}")
                
                if repetition_ratio > 0.3:  # More than 30% repetition
                    print("🚨 SEVERE REPETITION DETECTED!")
                    print("This suggests the model is stuck in a repetitive loop")
        
        # Check for mathematical content
        has_math = any(marker in text for marker in ["\\boxed", "\\frac", "$", "="])
        print(f"Contains math notation: {has_math}")
        
        # Check for truncation
        if text.endswith("…") or text.endswith("..."):
            print("⚠ Text appears truncated")
    
    print("\n💡 DIAGNOSIS:")
    print("The repetitive patterns suggest:")
    print("1. Model is generating repetitive tokens (possible training issue)")
    print("2. Model may be stuck due to poor generation parameters")
    print("3. Possible tokenization/encoding issue")


def test_math_extraction():
    """Test the math answer extraction without external dependencies"""
    print("\n" + "="*60)
    print("TEST 2: MATH ANSWER EXTRACTION")
    print("="*60)
    
    # Replicate the extraction functions from math_reward.py
    def _last_boxed_only_string(string: str) -> str | None:
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        return None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    def _remove_boxed(s: str) -> str:
        if s.startswith("\\boxed "):
            return s[len("\\boxed ") :]
        left = "\\boxed{"
        if s.startswith(left) and s.endswith("}"):
            return s[len(left) : -1]
        return s

    def _strip_string(string: str) -> str:
        s = string.replace("\n", "")
        s = s.replace("\\!", "")
        s = s.replace("\\\\", "\\")
        s = s.replace("tfrac", "frac").replace("dfrac", "frac")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("^{\\circ}", "").replace("^\\circ", "")
        s = s.replace("\\$", "")
        s = s.replace("\\%", "").replace("%", "")
        s = s.replace(" .", " 0.").replace("{.", "{0.")
        if s and s[0] == ".":
            s = "0" + s
        if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
            s = s.split("=")[1]
        s = s.replace(" ", "")
        return s

    def _is_equiv(str1: str | None, str2: str | None) -> bool:
        if str1 is None or str2 is None:
            return False
        try:
            return _strip_string(str1) == _strip_string(str2)
        except Exception:
            return str1 == str2
    
    # Test cases including the problematic ones
    test_cases = [
        {
            "name": "Normal boxed answer",
            "text": "The solution is \\boxed{85}",
            "expected_gt": "85",
            "should_extract": True
        },
        {
            "name": "Repetitive text with answer",
            "text": " 2x 2x 2x 2x 2x 2x 2x 2x \\boxed{85}",
            "expected_gt": "85", 
            "should_extract": True
        },
        {
            "name": "Repetitive text without answer",
            "text": " 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x 2x",
            "expected_gt": "85",
            "should_extract": False
        },
        {
            "name": "Truncated repetitive text",
            "text": " clear on the type of the sequence. So, the user is not clear on the type of the sequence. So, the user is not clear on the t",
            "expected_gt": "0.5",
            "should_extract": False
        },
        {
            "name": "Fraction answer",
            "text": "The answer is \\boxed{\\frac{1}{2}}",
            "expected_gt": "\\frac{1}{2}",
            "should_extract": True
        }
    ]
    
    print(f"Testing {len(test_cases)} extraction scenarios...")
    
    extraction_success = 0
    equivalence_success = 0
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Case {i+1}: {case['name']} ---")
        print(f"Text: {case['text']}")
        print(f"Expected GT: {case['expected_gt']}")
        
        # Test extraction
        boxed = _last_boxed_only_string(case['text'])
        if boxed:
            extracted = _remove_boxed(boxed)
            print(f"Extracted: '{extracted}'")
            
            if case['should_extract']:
                extraction_success += 1
                print("✓ Extraction successful")
                
                # Test equivalence
                is_correct = _is_equiv(extracted, case['expected_gt'])
                print(f"Equivalent to GT: {is_correct}")
                if is_correct:
                    equivalence_success += 1
                    print("✓ Answer correct")
                else:
                    print("✗ Answer incorrect")
            else:
                print("⚠ Extracted answer but shouldn't have")
        else:
            print("No \\boxed{} found")
            if not case['should_extract']:
                extraction_success += 1
                print("✓ Correctly found no answer")
            else:
                print("✗ Should have found answer")
    
    print(f"\n📊 RESULTS:")
    print(f"Extraction accuracy: {extraction_success}/{len(test_cases)} ({extraction_success/len(test_cases)*100:.1f}%)")
    print(f"Answer accuracy: {equivalence_success}/{len(test_cases)} ({equivalence_success/len(test_cases)*100:.1f}%)")
    
    return extraction_success, equivalence_success


def analyze_tokenization_issues():
    """Analyze potential tokenization problems"""
    print("\n" + "="*60)
    print("TEST 3: TOKENIZATION ANALYSIS")
    print("="*60)
    
    # From the logs: pred_len=181 appears twice - this is suspicious
    suspicious_length = 181
    
    print(f"The logs show prediction length of {suspicious_length} appearing multiple times.")
    print("This suggests:")
    
    print(f"\n🔍 HYPOTHESIS 1: Fixed truncation at {suspicious_length} tokens")
    print("- Model generates longer text but gets truncated")
    print("- Truncation happens mid-repetition, creating weird patterns")
    
    print(f"\n🔍 HYPOTHESIS 2: Model generates exactly {suspicious_length} tokens")
    print("- Generation parameters force this length")
    print("- Model fills remaining tokens with repetition")
    
    print(f"\n🔍 HYPOTHESIS 3: Tokenization encoding issue")
    print("- Special tokens or encoding causing repetitive patterns")
    print("- Padding/attention mask issues")
    
    # Simulate what 181 tokens of " 2x" would look like
    simulated_repetitive = " 2x" * (suspicious_length // 3)  # Roughly 181 tokens
    print(f"\nSimulated {len(simulated_repetitive.split())} token repetition:")
    print(f"Text: {simulated_repetitive[:100]}...")
    print(f"Length: {len(simulated_repetitive)} chars, {len(simulated_repetitive.split())} words")


def analyze_reward_computation():
    """Analyze why all rewards might be identical"""
    print("\n" + "="*60)
    print("TEST 4: REWARD COMPUTATION ANALYSIS")
    print("="*60)
    
    # Simulate the reward computation with repetitive texts
    def simple_format_reward(text):
        """Simplified format reward: 1.0 if has \\boxed, else 0.0"""
        return 1.0 if ("\\boxed{" in text or "\\boxed " in text) else 0.0
    
    def simple_answer_reward(text, ground_truth):
        """Simplified answer reward: 1.0 if extracted answer matches GT"""
        # Simple boxed extraction
        if "\\boxed{" in text:
            start = text.find("\\boxed{") + 7
            end = text.find("}", start)
            if end > start:
                extracted = text[start:end]
                return 1.0 if extracted == ground_truth else 0.0
        return 0.0
    
    # Test scenarios
    scenarios = [
        {
            "name": "All repetitive, no answers",
            "completions": [" 2x 2x 2x 2x 2x"] * 4,
            "ground_truths": ["85"] * 4
        },
        {
            "name": "All repetitive, same wrong answer", 
            "completions": [" 2x 2x 2x \\boxed{42}"] * 4,
            "ground_truths": ["85"] * 4
        },
        {
            "name": "All have correct answer",
            "completions": [" 2x 2x 2x \\boxed{85}"] * 4,
            "ground_truths": ["85"] * 4
        },
        {
            "name": "Mixed answers (should have different rewards)",
            "completions": [
                " 2x 2x \\boxed{85}",
                " 2x 2x \\boxed{42}", 
                " 2x 2x 2x",
                " 2x 2x \\boxed{85}"
            ],
            "ground_truths": ["85"] * 4
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        format_rewards = [simple_format_reward(comp) for comp in scenario['completions']]
        answer_rewards = [
            simple_answer_reward(comp, gt) 
            for comp, gt in zip(scenario['completions'], scenario['ground_truths'])
        ]
        total_rewards = [f + a for f, a in zip(format_rewards, answer_rewards)]
        
        print(f"Completions: {len(scenario['completions'])}")
        print(f"Format rewards: {format_rewards}")
        print(f"Answer rewards: {answer_rewards}")  
        print(f"Total rewards: {total_rewards}")
        
        unique_total = set(total_rewards)
        print(f"Unique total rewards: {unique_total}")
        
        if len(unique_total) <= 1:
            print("🚨 ALL REWARDS IDENTICAL - explains retention_rate=1.0!")
        else:
            print("✓ Rewards have variance - filtering should work")


def main():
    """Run simplified diagnostic tests"""
    print("Running simplified diagnostics (no JAX required)...")
    
    try:
        test_text_patterns()
        extraction_success, answer_success = test_math_extraction()
        analyze_tokenization_issues()
        analyze_reward_computation()
        
        print("\n" + "="*80)
        print("SIMPLIFIED DIAGNOSTIC SUMMARY")
        print("="*80)
        
        print("\n🔍 KEY FINDINGS:")
        
        print(f"\n1. TEXT PATTERN ANALYSIS:")
        print("   - Severe repetitive patterns detected")
        print("   - Model appears stuck in repetitive loops")
        print("   - Likely generation/training issue")
        
        print(f"\n2. MATH EXTRACTION:")
        print(f"   - Can extract answers when \\boxed{{}} is present")
        print(f"   - Problem: repetitive text often lacks \\boxed{{}}")
        
        print(f"\n3. TOKENIZATION:")
        print("   - Suspicious fixed length (181 tokens)")
        print("   - Suggests truncation or generation limits")
        
        print(f"\n4. REWARD COMPUTATION:")
        print("   - If all completions are repetitive without answers: all rewards = 0")
        print("   - If all completions have same answer: all rewards identical")
        print("   - This explains retention_rate=1.0 (no filtering)")
        
        print(f"\n🎯 ROOT CAUSE HYPOTHESIS:")
        print("The core issue is MODEL GENERATION:")
        print("1. Model generates repetitive text instead of proper solutions")
        print("2. This leads to identical/poor rewards across all completions")
        print("3. No reward variance = no GFPO filtering = retention_rate=1.0")
        print("4. No valid answers = 0% accuracy")
        
        print(f"\n🛠 RECOMMENDED FIXES:")
        print("1. Check model checkpoint - may be corrupted/undertrained")
        print("2. Adjust generation parameters (temperature, top_p, max_length)")
        print("3. Check training configuration and learning rate")
        print("4. Verify model isn't overfitting or in a degenerate state")
        print("5. Test model inference on simple prompts outside training loop")
        
    except Exception as e:
        print(f"Error in diagnostics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
