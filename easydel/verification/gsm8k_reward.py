import re
import logging
from typing import List
try:  # Optional JAX for proc-0 gating of logs
    import jax
except Exception:  # pragma: no cover
    jax = None  # type: ignore

try:
    # Math-Verify: robust evaluator for numeric expressions
    from math_verify import (
        parse, 
        verify, 
        ExprExtractionConfig,
    )  # type: ignore
except Exception as _e:  # pragma: no cover
    parse = None  # type: ignore
    verify = None  # type: ignore
    ExprExtractionConfig = None  # type: ignore

logger = logging.getLogger(__name__)


def _extract_text(comp) -> str:
    if isinstance(comp, list) and comp:
        c0 = comp[0]
        if isinstance(c0, dict) and "content" in c0:
            return c0["content"]
        if isinstance(c0, str):
            return c0
        return str(c0)
    if isinstance(comp, str):
        return comp
    return ""

def _normalize_number_text(text: str) -> str:
    """Normalize numeric text by stripping commas/currency/percent and whitespace."""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    return text.replace(",", "").replace("$", "").replace("%", "").strip()

def _answer_check(solution_str: str, ground_truth: str) -> bool:
    """Return True if the last number in solution_str equals ground_truth.

    Mirrors VERL's gsm8k reward: extract last number (int/float) and compare as string.
    """
    pred_norm = _normalize_number_text(solution_str)
    gt_norm = _normalize_number_text(ground_truth)
    numbers = re.findall(r"-?\d+\.?\d*", pred_norm)
    if not numbers:
        return False
    return numbers[-1] == gt_norm


def _math_verify_numeric_check(solution_str: str, ground_truth: str, **kwargs) -> tuple[bool, str]:
    """Enhanced numeric verification using Math-Verify.
    
    Returns:
        tuple[bool, str]: (is_correct, method_used)
    """
    if not (parse and verify and ExprExtractionConfig):
        return False, "math_verify_unavailable"
    
    try:
        # Configure for numeric expressions following Math-Verify patterns
        extraction_config = [ExprExtractionConfig(try_extract_without_anchor=True)]
        
        # Parse both gold and prediction
        gold_parsed = parse(
            ground_truth,
            extraction_config=extraction_config,
            raise_on_error=False
        )
        pred_parsed = parse(
            solution_str,
            extraction_config=extraction_config,
            raise_on_error=False
        )
        
        if not (gold_parsed and pred_parsed):
            return False, "math_verify_parse_failed"
        
        # Use Math-Verify's verify with GSM8K-appropriate settings
        result = verify(
            gold=gold_parsed[0],
            target=pred_parsed[0],
            float_rounding=kwargs.get("float_rounding", 6),
            numeric_precision=kwargs.get("numeric_precision", 15),
            strict=kwargs.get("strict", True),
            timeout_seconds=kwargs.get("timeout_seconds", 3),  # Shorter timeout for numbers
            raise_on_error=False
        )
        
        return result, "math_verify_success"
        
    except Exception as e:
        logger.debug(f"Math-Verify numeric check failed: {e}")
        return False, f"math_verify_error: {str(e)[:50]}"


def _extract_answer_from_xml(solution_str: str) -> str | None:
    """Extract content inside <answer>...</answer>. Returns None if absent."""
    m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    return m.group(1).strip() if m else None


def format_reward(completions: List[list[dict]], prompts=None, batch=None, **kwargs) -> List[float]:
    """Simple structural reward encouraging <think> and <answer> tags.

    completions: [[{"role": "assistant", "content": text}], ...]
    Returns a float per completion in [0, 1].
    """
    out: List[float] = []
    for comp in completions:
        text = _extract_text(comp)
        has_think = (text.count("<think>") == 1 and text.count("</think>") == 1)
        has_answer = (text.count("<answer>") == 1 and text.count("</answer>") == 1)
        out.append(1.0 if (has_think and has_answer) else 0.0)
    # Allow weighting from kwargs for logging consistency
    weight = float(kwargs.get("format_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in out]


def answer_reward(prompts, completions: List[list[dict]], batch, **kwargs) -> List[float]:
    """GSM8K verification following Math-Verify's structured approach.

    - Extract answer from <answer> tags or full text
    - Use numeric comparison for final answer  
    - Provide detailed logging following Math-Verify patterns
    - Store verification details for debugging
    """

    # Ground truths are provided directly as strings in batch["answer"]
    gt_raw = batch.get("answer", [])
    # Normalize container to Python list of strings
    if isinstance(gt_raw, list):
        gts = gt_raw
    else:
        try:
            import numpy as np  # type: ignore
            if isinstance(gt_raw, np.ndarray):
                gts = gt_raw.tolist()
            else:
                gts = [gt_raw] if gt_raw is not None else []
        except Exception:
            gts = [gt_raw] if gt_raw is not None and not isinstance(gt_raw, list) else []
    # Replicate to match completions length (B * R)
    target = len(completions)
    if len(gts) == 0:
        gt_list: List[str] = [""] * target
    elif len(gts) == target:
        gt_list = gts
    elif target % len(gts) == 0:
        factor = target // len(gts)
        gt_list = [x for x in gts for _ in range(factor)]
    else:
        times = (target + len(gts) - 1) // len(gts)
        gt_list = (gts * times)[:target]

    # Gate logs to process 0 only to avoid cross-host spam
    try:
        is_proc0 = (jax is None) or (jax.process_index() == 0)
    except Exception:
        is_proc0 = True

    rewards: List[float] = []
    verification_details = []

    for i, (comp, gt) in enumerate(zip(completions, gt_list, strict=False)):
        text = _extract_text(comp)

        detail = {
            "completion_idx": i,
            "full_text": text[-200:] if len(text) > 200 else text,
            "ground_truth": gt,
            "verification_method": "unknown",
            "parsed_successfully": False,
            "extracted_answer": None,
            "last_20_tokens": None,
            "extracted_numbers": [],
            "score": 0.0,
            "fallback_used": False
        }

        # Extract last 20 tokens for analysis
        try:
            tokens = re.findall(r'\S+|\s+', text)
            detail["last_20_tokens"] = tokens[-20:] if len(tokens) >= 20 else tokens
        except Exception:
            detail["last_20_tokens"] = ["TOKENIZATION_FAILED"]

        # Extract answer from XML tags or use full text
        ans = _extract_answer_from_xml(text) or text
        detail["extracted_answer"] = ans

        # Try Math-Verify first (more robust than regex)
        score = 0.0
        is_correct, method = _math_verify_numeric_check(ans, gt, **kwargs)
        detail["verification_method"] = method
        
        if is_correct:
            detail["score"] = 1.0
            detail["parsed_successfully"] = True
            if is_proc0:
                logger.info(f"✓ GSM8K Math-Verify SUCCESS [idx={i}] - Method: {method}")
                logger.info(f"  Ground truth: {gt}")
                logger.info(f"  Extracted answer: {ans}")
                logger.info(f"  Last 20 tokens: {detail['last_20_tokens']}")
        else:
            # Fallback to regex-based approach
            detail["fallback_used"] = True
            norm_ans = _normalize_number_text(ans)
            numbers = re.findall(r"-?\d+\.?\d*", norm_ans)
            detail["extracted_numbers"] = numbers

            if not numbers:
                detail["verification_method"] = "no_numbers_found"
                # Enhanced debugging for no numbers found - this is likely the main issue
                if is_proc0:
                    logger.warning(f"✗ NO NUMBERS FOUND [idx={i}]")
                    logger.warning(f"  Full original text: '{text}'")
                    logger.warning(f"  Extracted answer: '{ans}'")
                    logger.warning(f"  Ground truth: '{gt}'")
                    logger.warning(f"  Text length: {len(text)} chars")
                # Try more patterns to see if there are any digits at all
                all_digits = re.findall(r'\d+', norm_ans)
                all_numbers_expanded = re.findall(r'-?\d*\.?\d+', norm_ans)
                if is_proc0:
                    logger.warning(f"  Any digits found: {all_digits}")
                    logger.warning(f"  Numbers with expanded pattern: {all_numbers_expanded}")
                    logger.warning(f"  Math-Verify method: {method}")
            else:
                last_number = numbers[-1]
                detail["parsed_successfully"] = True
                detail["verification_method"] = "regex_fallback"

                # Check if the answer matches ground truth using regex
                is_correct = _answer_check(norm_ans, gt)
                detail["score"] = 1.0 if is_correct else 0.0

                if is_proc0:
                    if is_correct:
                        logger.info(f"✓ GSM8K REGEX FALLBACK SUCCESS [idx={i}]")
                        logger.info(f"  Extracted numbers: {numbers}")
                        logger.info(f"  Last number: {last_number}")
                        logger.info(f"  Ground truth: {gt}")
                    else:
                        logger.warning(f"✗ GSM8K REGEX MISMATCH [idx={i}]")
                        logger.warning(f"  Math-Verify method: {method}")
                        logger.warning(f"  Regex extracted: '{last_number}' (type: {type(last_number)})")
                        logger.warning(f"  Ground truth: '{gt}' (type: {type(gt)})")
                        logger.warning(f"  All extracted numbers: {numbers}")
                        logger.warning(f"  Full extracted answer: '{ans}'")
                        logger.warning(f"  Full original text: '{text}'")

        rewards.append(detail["score"])
        verification_details.append(detail)

    # Enhanced summary statistics following Math-Verify patterns (proc0 only)
    successful_verifications = sum(1 for d in verification_details if d["score"] > 0.0)
    math_verify_successes = sum(1 for d in verification_details if d["verification_method"] == "math_verify_success")
    regex_fallback_successes = sum(1 for d in verification_details if d["verification_method"] == "regex_fallback" and d["score"] > 0.0)
    no_numbers_count = sum(1 for d in verification_details if d["verification_method"] == "no_numbers_found")

    if is_proc0:
        # Derive per-prompt pass@k locally when prompts are provided
        try:
            total_comps = len(verification_details)
            if prompts and isinstance(prompts, list) and len(prompts) == total_comps:
                try:
                    unique_prompts = []
                    seen = set()
                    for p in prompts:
                        if p not in seen:
                            seen.add(p)
                            unique_prompts.append(p)
                    B = len(unique_prompts)
                    R = max(1, total_comps // max(1, B))
                except Exception:
                    B = total_comps
                    R = 1
            else:
                B = total_comps
                R = 1

            # Pass@k across prompts
            scores = [1.0 if d.get("score", 0.0) > 0.0 else 0.0 for d in verification_details]
            pass_cnt = 0
            if B > 0 and R > 0 and B * R == total_comps:
                for i in range(B):
                    grp = scores[i * R : (i + 1) * R]
                    pass_cnt += 1 if any(s > 0 for s in grp) else 0
            else:
                # Fallback: treat each completion as a prompt (degenerate)
                pass_cnt = sum(scores)
                B = total_comps
                R = 1

            logger.info("GSM8K VERIFICATION SUMMARY (local):")
            logger.info(f"  Local prompts: {B}")
            logger.info(f"  Local completions: {total_comps} ({R} per prompt)")
            logger.info(f"  Successful completions: {successful_verifications}")
            logger.info(f"  Math-Verify successes: {math_verify_successes}")
            logger.info(f"  Regex fallback successes: {regex_fallback_successes}")
            logger.info(f"  No numbers found: {no_numbers_count}")
            logger.info(f"  Pass@{R} (prompts): {pass_cnt}/{B} ({(pass_cnt/max(1,B)):.2%})")

            # Global metrics are reported by the trainer; avoid cross-host collectives here.
        except Exception:
            # Fallback to minimal summary
            logger.info("GSM8K VERIFICATION SUMMARY (local):")
            logger.info(f"  Local completions: {len(verification_details)}")
            logger.info(f"  Successful completions: {successful_verifications} ({successful_verifications/len(verification_details):.2%})")

    # Store verification details in kwargs for potential external access
    if "verification_details" in kwargs:
        kwargs["verification_details"].extend(verification_details)

    # Optional weighting for logging consistency
    weight = float(kwargs.get("answer_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in rewards]


def debug_model_outputs(completions: List[list[dict]], batch, max_examples: int = 5) -> None:
    """Debug function to analyze model outputs and identify issues with GSM8K verification.

    This function helps diagnose why verification is failing by analyzing:
    - Model output format and content
    - Answer extraction from <answer> tags
    - Number extraction patterns
    - Common issues that prevent successful verification

    Args:
        completions: List of model completions from the training batch
        batch: Batch data containing ground truths
        max_examples: Maximum number of examples to debug (default: 5)
    """
    logger.info("=" * 80)
    logger.info("MODEL OUTPUT DEBUG ANALYSIS")
    logger.info("=" * 80)

    # Ground truths are provided directly as strings in batch["answer"]
    gt_raw = batch.get("answer", [])
    if isinstance(gt_raw, list):
        gts = gt_raw
    else:
        try:
            import numpy as np
            if isinstance(gt_raw, np.ndarray):
                gts = gt_raw.tolist()
            else:
                gts = [gt_raw] if gt_raw is not None else []
        except Exception:
            gts = [gt_raw] if gt_raw is not None and not isinstance(gt_raw, list) else []

    for i, (comp, gt) in enumerate(zip(completions[:max_examples], gts[:max_examples])):
        # Handle different completion formats
        if isinstance(comp, list) and len(comp) > 0:
            if isinstance(comp[0], dict) and "content" in comp[0]:
                text = comp[0]["content"]
            elif isinstance(comp[0], str):
                text = comp[0]
            else:
                text = str(comp[0])
        else:
            text = str(comp)

        logger.info(f"\n--- DEBUG EXAMPLE {i+1} ---")
        logger.info(f"Ground truth: '{gt}' (type: {type(gt)})")
        logger.info(f"Full text length: {len(text)} chars")
        logger.info(f"Full text: '{text}'")

        # Check for answer tags
        if "<answer>" in text and "</answer>" in text:
            try:
                ans = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
                logger.info(f"Answer tag content: '{ans}'")
            except Exception as e:
                logger.warning(f"Answer tag extraction failed: {e}")
                ans = text
        else:
            ans = text
            logger.warning("No <answer> tags found - using full text")

        # Analyze numbers in different contexts
        logger.info("NUMBER ANALYSIS:")
        numbers_in_ans = re.findall(r"-?\d+\.?\d*", ans)
        numbers_in_full = re.findall(r"-?\d+\.?\d*", text)
        all_digits_ans = re.findall(r'\d+', ans)
        all_digits_full = re.findall(r'\d+', text)

        logger.info(f"  Numbers in extracted answer: {numbers_in_ans}")
        logger.info(f"  Numbers in full text: {numbers_in_full}")
        logger.info(f"  All digits in extracted answer: {all_digits_ans}")
        logger.info(f"  All digits in full text: {all_digits_full}")

        # Check for common issues
        if not text.strip():
            logger.error("  ISSUE: Empty text!")
        elif not ans.strip():
            logger.error("  ISSUE: Empty answer after extraction!")
        elif not numbers_in_ans and not all_digits_ans:
            logger.warning("  POTENTIAL ISSUE: No numbers found in answer")
        else:
            logger.info("  OK: Numbers found")

    logger.info("\n" + "=" * 80)
    logger.info("DEBUG ANALYSIS COMPLETE")
    logger.info("=" * 80)


def test_verification_with_sample_data():
    """Test the verification system with sample data to identify issues."""
    # Sample data that mimics what might be causing the verification failures
    # Format: [[{"role": "assistant", "content": text}], ...]
    sample_completions = [
        [{"role": "assistant", "content": "The answer is 42"}],
        [{"role": "assistant", "content": "<think>Let me calculate this step by step.</think><answer>15</answer>"}],
        [{"role": "assistant", "content": "The result is 25.5"}],
        [{"role": "assistant", "content": "<answer>7</answer>"}],
        [{"role": "assistant", "content": "I think it's 10"}],
    ]

    sample_batch = {
        "answer": ["42", "15", "25.5", "7", "10"]
    }

    print("Testing GSM8K verification with sample data...")
    print("This will help identify if the issue is with the model outputs or verification logic.")
    print()

    # Test with debug mode
    debug_model_outputs(sample_completions, sample_batch, max_examples=5)

    # Test verification - answer_reward(prompts, completions, batch)
    scores = answer_reward(None, sample_completions, sample_batch)

    print(f"\nVerification Results:")
    print(f"Scores: {scores}")
    print(f"Successful verifications: {sum(1 for s in scores if s > 0)}/{len(scores)}")

    return scores


__all__ = [
    "format_reward",
    "answer_reward",
    "debug_model_outputs",
    "test_verification_with_sample_data",
]


