from typing import List
import logging
import time

try:
    # Math-Verify: robust evaluator for math expressions with full feature set
    from math_verify import (
        parse, 
        verify, 
        LatexExtractionConfig, 
        ExprExtractionConfig,
        StringExtractionConfig,
        math_metric
    )  # type: ignore
except Exception as _e:  # pragma: no cover
    parse = None  # type: ignore
    verify = None  # type: ignore
    LatexExtractionConfig = None  # type: ignore
    ExprExtractionConfig = None  # type: ignore
    StringExtractionConfig = None  # type: ignore
    math_metric = None  # type: ignore

logger = logging.getLogger(__name__)
try:
    # Shared reward utils
    from .reward_utils import (
        normalize_to_list_str,
        replicate_to_length,
        is_main_process,
        safe_global_sum,
        extract_answer_from_xml,
    )
except Exception:  # pragma: no cover
    normalize_to_list_str = None  # type: ignore
    replicate_to_length = None  # type: ignore
    is_main_process = lambda: True  # type: ignore
    safe_global_sum = lambda x: x  # type: ignore


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

# Removed math_metric cache and indirection to adhere strictly to parse+verify + boxed fallback.

# Removed task-specific extraction/verification param helpers to keep evaluation strict and simple.


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


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


def _strip_string(string: str) -> str:
    s = string.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    # Remove units: split at "\\text{ " and keep left part if present
    if "\\text{ " in s:
        parts = s.split("\\text{ ")
        if len(parts) == 2:
            s = parts[0]
    s = s.replace("\\%", "").replace("\%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    # remove spaces
    s = s.replace(" ", "")
    # normalize simple a/b forms that are pure integers
    if s.count("/") == 1:
        a, b = s.split("/")
        try:
            ai, bi = int(a), int(b)
            if s == f"{ai}/{bi}":
                s = f"\\frac{{{ai}}}{{{bi}}}"
        except Exception:
            pass
    if s == "0.5":
        s = "\\frac{1}{2}"
    return s


def _is_equiv(str1: str | None, str2: str | None) -> bool:
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def format_reward(completions: List[list[dict]], prompts=None, batch=None, **kwargs) -> List[float]:
    """Structural reward for MATH: enforce exactly one \boxed{...} inside <answer>.

    Returns 1.0 if there is exactly one <answer>...</answer> block and within it exactly one \boxed{...}, else 0.0.
    """
    out: List[float] = []
    for comp in completions:
        text = _extract_text(comp)
        # Must contain one think and one answer block
        ok_blocks = (text.count("<think>") == 1 and text.count("</think>") == 1 and text.count("<answer>") == 1 and text.count("</answer>") == 1)
        if not ok_blocks:
            out.append(0.0)
            continue
        # Extract answer body and check boxed count
        try:
            ans = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
        except Exception:
            out.append(0.0)
            continue
        has_one_box = (ans.count("\\boxed{") + ans.count("\\boxed ") == 1)
        out.append(1.0 if has_one_box else 0.0)

    # Allow weighting for consistency with other reward modules
    weight = float(kwargs.get("format_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in out]


def answer_reward(prompts, completions: List[list[dict]], batch, **kwargs) -> List[float]:
    """Math verification using Math-Verify's structured approach.

    Follows Math-Verify patterns:
    - Uses their parse() function with proper extraction configs
    - Uses their verify() function with configurable parameters  
    - Provides detailed logging following their error handling approach
    - Falls back to boxed extraction when Math-Verify is unavailable
    """
    # Get verification details storage for debugging
    verification_details = kwargs.get("verification_details", [])
    
    # Prefer normalized ground truth if provided by preprocessing
    gts = batch.get("solution_normalized", batch.get("solution", []))
    if callable(normalize_to_list_str):
        gts = normalize_to_list_str(gts)  # type: ignore
    else:
        # Fallback normalization
        if gts is None:
            gts = []
        if isinstance(gts, str):
            gts = [gts]
        elif not isinstance(gts, list):
            try:
                gts = list(gts)
            except Exception:
                gts = [str(gts)]
        gts = ["" if x is None else (x if isinstance(x, str) else str(x)) for x in gts]
    # Replicate to match B*R if needed
    c_len = len(completions)
    if callable(replicate_to_length):
        gts = replicate_to_length(gts, c_len)  # type: ignore
    else:
        g_len = len(gts)
        if g_len == 0:
            gts = [""] * c_len
        elif c_len % g_len == 0:
            factor = c_len // g_len
            gts = [x for x in gts for _ in range(factor)]
        else:
            times = (c_len + g_len - 1) // g_len
            gts = (gts * times)[:c_len]

    # Configure Math-Verify extraction (fixed, strict) and parameters
    use_mv = callable(parse) and callable(verify)  # type: ignore
    problem_type = kwargs.get("problem_type", "math")
    gold_extraction_config = [ExprExtractionConfig()] if use_mv else None
    pred_extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()] if use_mv else None
    # Strict defaults; allow kwargs to override explicitly if passed by caller
    kwargs.setdefault("float_rounding", 6)
    kwargs.setdefault("numeric_precision", 15)
    kwargs.setdefault("strict", True)
    kwargs.setdefault("allow_set_relation_comp", False)
    kwargs.setdefault("timeout_seconds", 5)

    out: List[float] = []
    
    for i, (comp, gt) in enumerate(zip(completions, gts, strict=False)):
        text = _extract_text(comp)
        
        # Store verification details for debugging (following Math-Verify's approach)
        detail = {
            "completion_idx": i,
            "ground_truth": gt,
            "full_text": text[-200:] if len(text) > 200 else text,  # Last 200 chars
            "verification_method": "unknown",
            "parser_used": "none",
            "score": 0.0,
            "error_message": "",
            "extracted_answer": None,
            "problem_type": problem_type,
            "extraction_configs_used": {
                "gold": str(gold_extraction_config) if gold_extraction_config else None,
                "pred": str(pred_extraction_config) if pred_extraction_config else None
            }
        }
        
        # Prefer content inside <answer> block
        try:
            _ans = extract_answer_from_xml(text)
            if _ans is not None:
                ans_text = _ans
            else:
                ans_text = text
        except Exception:
            ans_text = text
        detail["extracted_answer"] = ans_text

        if use_mv:
            try:
                # Parse using Math-Verify's structured approach
                gold_parsed = parse(gt, extraction_config=gold_extraction_config, raise_on_error=False)
                ans_parsed = parse(ans_text, extraction_config=pred_extraction_config, raise_on_error=False)
                
                # Track parser information following Math-Verify patterns
                if gold_parsed and ans_parsed:
                    # Determine which parser was successful (following Math-Verify's approach)
                    detail["parser_used"] = "math_verify"
                    if any("latex" in str(type(p)).lower() for p in ans_parsed):
                        detail["parser_used"] += "_latex"
                    elif any("expr" in str(type(p)).lower() for p in ans_parsed):
                        detail["parser_used"] += "_expr"
                    
                    # Prefer math_metric when available (simpler, robust), fallback to verify loop
                    # Try all parsed candidates from the end (often contains the final answer)
                    is_correct = False
                    try_candidates = list(reversed(ans_parsed))
                    for cand in try_candidates:
                        is_correct = verify(
                            gold=gold_parsed[0] if gold_parsed else gt,
                            target=cand,
                            float_rounding=kwargs.get("float_rounding", 6),
                            numeric_precision=kwargs.get("numeric_precision", 15),
                            strict=kwargs.get("strict", True),
                            allow_set_relation_comp=kwargs.get("allow_set_relation_comp", False),
                            timeout_seconds=kwargs.get("timeout_seconds", 5),
                            raise_on_error=False
                        )
                        if is_correct:
                            break
                    
                    detail["verification_method"] = "math_verify"
                    detail["score"] = 1.0 if is_correct else 0.0
                    
                    # Log following Math-Verify's patterns
                    if is_correct:
                        logger.debug(f"✓ Math-Verify success: {detail['parser_used']}")
                    else:
                        logger.debug(f"✗ Math-Verify mismatch: {detail['parser_used']}")
                    
                    out.append(detail["score"])
                    verification_details.append(detail)
                    continue
                else:
                    detail["error_message"] = "Math-Verify parsing failed"
                    detail["verification_method"] = "math_verify_parse_failed"
                    logger.debug(f"Math-Verify parsing failed for: {ans_text[:50]}...")
                    
            except Exception as e:
                detail["error_message"] = str(e)
                detail["verification_method"] = "math_verify_error"
                logger.debug(f"Math-Verify error: {e}")
                # Fall through to boxed-based heuristic

        # Fallback heuristic: compare last \boxed{...} with normalized GT
        # This follows Math-Verify's fallback patterns
        boxed = _last_boxed_only_string(ans_text)
        if boxed is None:
            # as a last resort, try whole response
            boxed = _last_boxed_only_string(text)
            
        if boxed is None:
            detail["verification_method"] = "no_boxed_found"
            detail["score"] = 0.0
            logger.debug(f"✗ No boxed content found in: {ans_text[:50]}...")
        else:
            ans = _remove_boxed(boxed)
            detail["extracted_answer"] = ans
            detail["verification_method"] = "boxed_fallback"
            detail["parser_used"] = "boxed_extraction"
            
            is_correct = _is_equiv(ans, gt)
            detail["score"] = 1.0 if is_correct else 0.0
            
            if is_correct:
                logger.debug(f"✓ Boxed fallback success: '{ans}' == '{gt}'")
            else:
                logger.debug(f"✗ Boxed fallback failed: '{ans}' != '{gt}'")
                
        out.append(detail["score"])
        verification_details.append(detail)

    # Enhanced logging following Math-Verify's logging patterns
    successful = sum(1 for d in verification_details if d["score"] > 0.0)
    total = len(verification_details)
    if total > 0:
        logger.info(f"Math verification ({problem_type}): {successful}/{total} successful ({successful/total:.1%})")

        # Attempt global aggregation (best-effort, proc0 logs only)
        try:
            sc_global = safe_global_sum(successful)
            tot_global = safe_global_sum(total)
            # Convert to python scalars if possible
            try:
                scg = int(getattr(sc_global, "item", lambda: sc_global)())  # type: ignore
            except Exception:
                try:
                    scg = int(sc_global)
                except Exception:
                    scg = successful
            try:
                tg = int(getattr(tot_global, "item", lambda: tot_global)())  # type: ignore
            except Exception:
                try:
                    tg = int(tot_global)
                except Exception:
                    tg = total
            if is_main_process():
                rate = (float(scg) / max(1.0, float(tg)))
                logger.info(f"  Global (best-effort): {scg}/{tg} successful ({rate:.1%}) [aggregation ok]")
        except Exception as e:
            if is_main_process():
                logger.info(f"  Global aggregation failed in math_reward: {e}")

        # Method breakdown (local)
        methods = {}
        parsers = {}
        for detail in verification_details:
            method = detail["verification_method"]
            parser = detail["parser_used"]
            methods[method] = methods.get(method, 0) + 1
            if parser != "none":
                parsers[parser] = parsers.get(parser, 0) + 1

        logger.info(f"  Problem type: {problem_type}")
        logger.info(
            f"  Extraction configs: gold={len(gold_extraction_config) if gold_extraction_config else 0}, pred={len(pred_extraction_config) if pred_extraction_config else 0}"
        )

        for method, count in methods.items():
            logger.debug(f"  Method {method}: {count}")

        if parsers:
            logger.debug(f"  Parser usage: {dict(parsers)}")

    weight = float(kwargs.get("answer_weight", 1.0)) if kwargs else 1.0
    return [min(1.0, max(0.0, r * weight)) for r in out]


def create_math_verify_demo():
    """Demonstrate Math-Verify's structured verification features.
    
    This function showcases how to use Math-Verify's built-in classes and functions
    following their recommended patterns and documentation.
    """
    if not (parse and verify and LatexExtractionConfig and ExprExtractionConfig):
        print("Math-Verify not available - cannot run demo")
        return
    
    print("=" * 80)
    print("MATH-VERIFY STRUCTURED VERIFICATION DEMO")
    print("Following Math-Verify's official patterns and documentation")
    print("=" * 80)
    
    # Demo cases showing Math-Verify's extraction capabilities
    demo_cases = [
        {
            "description": "LaTeX fraction with boxed format",
            "gold": "1/2", 
            "prediction": "The answer is $\\boxed{\\frac{1}{2}}$",
            "expected_config": [LatexExtractionConfig(), ExprExtractionConfig()]
        },
        {
            "description": "Plain expression format",
            "gold": "42",
            "prediction": "The final answer is 42",
            "expected_config": [ExprExtractionConfig()]
        },
        {
            "description": "Complex mathematical expression",
            "gold": "\\sqrt{3}",
            "prediction": "After calculation: $\\sqrt{3}$",
            "expected_config": [LatexExtractionConfig(), ExprExtractionConfig()]
        },
        {
            "description": "Answer with units (should extract number)",
            "gold": "15",
            "prediction": "The result is 15 cm",
            "expected_config": [ExprExtractionConfig()]
        }
    ]
    
    print("\n1. EXTRACTION CONFIGURATION DEMO")
    print("-" * 40)
    print("Math-Verify provides three main ExtractionTarget classes:")
    print("  - LatexExtractionConfig: For LaTeX expressions")
    print("  - ExprExtractionConfig: For plain mathematical expressions") 
    print("  - StringExtractionConfig: For literal strings (A, B, C, D)")
    
    print("\n2. PARSING AND VERIFICATION DEMO")
    print("-" * 40)
    
    for i, case in enumerate(demo_cases):
        print(f"\nCase {i+1}: {case['description']}")
        print(f"Gold: {case['gold']}")
        print(f"Prediction: {case['prediction']}")
        
        try:
            # Parse using Math-Verify's official API
            gold_parsed = parse(case['gold'], extraction_config=[ExprExtractionConfig()])
            pred_parsed = parse(case['prediction'], extraction_config=case['expected_config'])
            
            print(f"Gold parsed: {gold_parsed}")
            print(f"Prediction parsed: {pred_parsed}")
            
            if gold_parsed and pred_parsed:
                # Use Math-Verify's verify function with their recommended parameters
                result = verify(
                    gold=gold_parsed[0],
                    target=pred_parsed[0], 
                    float_rounding=6,  # Math-Verify default
                    numeric_precision=15,  # Math-Verify default
                    strict=True,  # Math-Verify default
                    allow_set_relation_comp=False,  # Math-Verify default
                    timeout_seconds=5,  # Math-Verify default
                    raise_on_error=False  # Math-Verify default
                )
                print(f"✓ Verification result: {result}")
            else:
                print("✗ Parsing failed")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n3. MATH-VERIFY CONFIGURATION OPTIONS")
    print("-" * 40)
    print("Key Math-Verify parameters:")
    print("  - float_rounding: Decimal places for float rounding (default: 6)")
    print("  - numeric_precision: Precision for numeric comparisons (default: 15)")
    print("  - strict: Variable matching mode (default: True)")
    print("  - allow_set_relation_comp: Set-relation comparison (default: False)")
    print("  - timeout_seconds: Timeout for operations (default: 5)")
    print("  - raise_on_error: Error handling mode (default: False)")
    
    print("\n4. EXTRACTION TARGET CONFIGURATION")
    print("-" * 40)
    
    # Demonstrate extraction configuration following Math-Verify patterns
    latex_config = LatexExtractionConfig(
        try_extract_without_anchor=True,
        boxed_match_priority=50  # Math-Verify default
    )
    
    expr_config = ExprExtractionConfig(
        try_extract_without_anchor=True
    )
    
    print(f"LatexExtractionConfig: boxed_match_priority={latex_config.boxed_match_priority}")
    print(f"ExprExtractionConfig: try_extract_without_anchor={expr_config.try_extract_without_anchor}")
    
    print("\n5. MATH METRIC FUNCTION DEMO")
    print("-" * 40)
    
    if math_metric:
        # Use Math-Verify's math_metric function following their documentation
        metric_fn = math_metric(
            gold_extraction_target=[ExprExtractionConfig()],
            pred_extraction_target=[LatexExtractionConfig(), ExprExtractionConfig()],
            precision=6
        )
        
        # Test the metric
        test_golds = ["1/2", "42"] 
        test_preds = ["$\\boxed{\\frac{1}{2}}$", "The answer is 42"]
        
        try:
            score, debug_info = metric_fn(test_golds, test_preds)
            print(f"Math metric score: {score}")
            if debug_info:
                print(f"Debug info: {debug_info}")
        except Exception as e:
            print(f"Math metric error: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE - Following Math-Verify's official patterns")
    print("=" * 80)


__all__ = [
    "format_reward",
    "answer_reward", 
    "create_math_verify_demo",
    "get_extraction_config",
    "get_verification_params",
]


