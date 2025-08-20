import re
import logging
from typing import List

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


def _answer_check(solution_str: str, ground_truth: str) -> bool:
    """Return True if the last number in solution_str equals ground_truth.

    Mirrors VERL's gsm8k reward: extract last number (int/float) and compare as string.
    """
    numbers = re.findall(r"-?\d+\.?\d*", solution_str)
    if not numbers:
        return False
    return numbers[-1] == ground_truth


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
        text = comp[0]["content"] if comp and comp[0] else ""
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

    rewards: List[float] = []
    verification_details = []

    for i, (comp, gt) in enumerate(zip(completions, gt_list, strict=False)):
        text = comp[0]["content"] if comp and comp[0] else ""

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
            logger.info(f"✓ GSM8K Math-Verify SUCCESS [idx={i}] - Method: {method}")
            logger.info(f"  Ground truth: {gt}")
            logger.info(f"  Extracted answer: {ans}")
            logger.info(f"  Last 20 tokens: {detail['last_20_tokens']}")
        else:
            # Fallback to regex-based approach
            detail["fallback_used"] = True
            numbers = re.findall(r"-?\d+\.?\d*", ans)
            detail["extracted_numbers"] = numbers

            if not numbers:
                detail["verification_method"] = "no_numbers_found"
                logger.debug(f"✗ NO NUMBERS FOUND [idx={i}] - Text: {ans[-100:]}")
            else:
                last_number = numbers[-1]
                detail["parsed_successfully"] = True
                detail["verification_method"] = "regex_fallback"

                # Check if the answer matches ground truth using regex
                is_correct = _answer_check(ans, gt)
                detail["score"] = 1.0 if is_correct else 0.0

                if is_correct:
                    logger.info(f"✓ GSM8K REGEX FALLBACK SUCCESS [idx={i}]")
                    logger.info(f"  Extracted numbers: {numbers}")
                    logger.info(f"  Last number: {last_number}")
                    logger.info(f"  Ground truth: {gt}")
                else:
                    logger.debug(f"✗ GSM8K ALL METHODS FAILED [idx={i}]")
                    logger.debug(f"  Math-Verify method: {method}")
                    logger.debug(f"  Regex got: '{last_number}', Expected: '{gt}'")
                    logger.debug(f"  All extracted numbers: {numbers}")

        rewards.append(detail["score"])
        verification_details.append(detail)

    # Enhanced summary statistics following Math-Verify patterns
    successful_verifications = sum(1 for d in verification_details if d["score"] > 0.0)
    math_verify_successes = sum(1 for d in verification_details if d["verification_method"] == "math_verify_success")
    regex_fallback_successes = sum(1 for d in verification_details if d["verification_method"] == "regex_fallback" and d["score"] > 0.0)
    no_numbers_count = sum(1 for d in verification_details if d["verification_method"] == "no_numbers_found")

    logger.info(f"GSM8K VERIFICATION SUMMARY:")
    logger.info(f"  Total completions: {len(verification_details)}")
    logger.info(f"  Successful verifications: {successful_verifications}")
    logger.info(f"  Math-Verify successes: {math_verify_successes}")
    logger.info(f"  Regex fallback successes: {regex_fallback_successes}")
    logger.info(f"  No numbers found: {no_numbers_count}")
    logger.info(f"  Success rate: {successful_verifications/len(verification_details):.2%}")
    if successful_verifications > 0:
        math_verify_rate = math_verify_successes / successful_verifications
        logger.info(f"  Math-Verify usage: {math_verify_rate:.1%} of successful verifications")

    # Store verification details in kwargs for potential external access
    if "verification_details" in kwargs:
        kwargs["verification_details"].extend(verification_details)

    # Optional weighting for logging consistency
    weight = float(kwargs.get("answer_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in rewards]


__all__ = [
    "format_reward",
    "answer_reward",
]


