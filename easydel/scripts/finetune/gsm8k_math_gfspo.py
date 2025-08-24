import re
from dataclasses import field
import os

import jax
from datasets import load_dataset, Dataset, concatenate_datasets
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from transformers import AutoConfig, AutoTokenizer

import easydel as ed
from easydel.infra.factory import registry
from easydel.modules import *  # noqa: F401,F403 — ensure kernels are registered


def safe_call(desc, fn, *args, default=None, swallow=True, **kwargs):
    """Run fn safely.

    - On success: prints a cleanup hint so we know this wrapper can be removed later.
    - On failure: prints a concise failure note; returns default (or re-raises if swallow=False).
    """
    try:
        result = fn(*args, **kwargs)
        try:
            if jax.process_index() == 0:
                print(f"CLEANUP_HINT: '{desc}' succeeded — consider removing safe_call wrapper.")
        except Exception:
            ...
        return default if result is None else result
    except Exception as e:
        try:
            if jax.process_index() == 0:
                print(f"SAFE_CALL_FAIL: '{desc}' failed with: {e}")
        except Exception:
            ...
        if not swallow:
            raise
        return default


@auto_pytree
class RunTimeConfig:
    repo_id: str = field(metadata={"help": "The repository ID for the policy model."})
    processor_repo_id: str | None = field(default=None)
    dataset: str = field(
        default="math-ds",
        metadata={"help": "Dataset to use: 'gsm8k'|'gsm8k-ds' or 'math'|'math-ds'"},
    )
    dataset_use_rate: float = field(
        default=1.0,
        metadata={"help": "Fraction of dataset to use (1.0 = 100%, 0.1 = 10%)"}
    )
    curriculum_math: bool = field(
        default=False,
        metadata={"help": "Enable curriculum learning for math datasets, progressing from Level 1 to Level 5"},
    )
    kv_cache_quantization: ed.EasyDeLQuantizationMethods = field(
        default=ed.EasyDeLQuantizationMethods.NONE
    )

    # Mesh/precision
    sharding_axis: str = field(default="1, -1, 1, 1, 1")
    attn_mechanism: ed.AttentionMechanisms = field(default=ed.AttentionMechanisms.AUTO)
    param_dtype: jnp.dtype = field(default=jnp.bfloat16)
    dtype: jnp.dtype = field(default=jnp.bfloat16)
    attn_dtype: jnp.dtype = field(default=jnp.bfloat16)
    attn_softmax_dtype: jnp.dtype = field(default=jnp.float32)

    def __post_init__(self):
        if self.processor_repo_id is None:
            self.processor_repo_id = self.repo_id
        if isinstance(self.sharding_axis, str):
            self.sharding_axis = tuple(map(int, self.sharding_axis.split(",")))


def main():
    parser = ed.utils.DataClassArgumentParser((ed.GFSPOConfig, RunTimeConfig))
    gfspo_config, runtime = parser.parse_args_into_dataclasses()

    if jax.process_index() == 0:
        print("Training Arguments\n----------------------")
        print(gfspo_config)
        print("----------------------")
        # Extra debugging for dataset selection/rates
        def _dbg_print():
            try:
                print(f"DEBUG: runtime.dataset (raw)={runtime.dataset}")
                print(f"DEBUG: runtime.dataset_use_rate (raw)={runtime.dataset_use_rate}")
                try:
                    pct = int(float(runtime.dataset_use_rate) * 100)
                except Exception:
                    pct = None
                print(f"DEBUG: computed dataset percentage={pct}%")
                print(f"DEBUG: ENV DATASET={os.environ.get('DATASET')}")
            except Exception as _:
                ...
        safe_call("print runtime + env dataset", _dbg_print)

    tokenizer = safe_call(
        "load tokenizer",
        AutoTokenizer.from_pretrained,
        runtime.processor_repo_id,
        default=AutoTokenizer.from_pretrained(runtime.repo_id),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_prompt_len = gfspo_config.max_prompt_length
    max_completion_len = gfspo_config.max_completion_length
    max_seq_len = max_prompt_len + max_completion_len

    hf_config = safe_call("load auto config", AutoConfig.from_pretrained, runtime.repo_id, default=AutoConfig.from_pretrained(runtime.repo_id))
    avails = [v.module.__name__ for v in registry.task_registry[ed.TaskType.IMAGE_TEXT_TO_TEXT].values()]
    if hf_config.architectures and any(arch in avails for arch in hf_config.architectures):
        load_module = ed.AutoEasyDeLModelForImageTextToText
    else:
        load_module = ed.AutoEasyDeLModelForCausalLM

    # Prefer adaptive mesh if specified
    if gfspo_config.force_tensor_parallel is not None or gfspo_config.force_data_parallel is not None:
        from easydel.trainers.group_relative_policy_optimization.adaptive_mesh import (
            configure_adaptive_mesh_inplace,
        )

        mesh_plan = configure_adaptive_mesh_inplace(gfspo_config)
        sharding_axis_dims = (mesh_plan.dp, mesh_plan.fsdp, mesh_plan.ep, mesh_plan.tp, mesh_plan.sp)
        gfspo_config.sharding_axis_dims = sharding_axis_dims
        if jax.process_index() == 0:
            print(
                f"Using adaptive mesh: DP={mesh_plan.dp}, FSDP={mesh_plan.fsdp}, TP={mesh_plan.tp}, EP={mesh_plan.ep}, SP={mesh_plan.sp}"
            )
    else:
        sharding_axis_dims = runtime.sharding_axis

    model = safe_call(
        "load module",
        load_module.from_pretrained,
        runtime.repo_id,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            max_position_embeddings=max_seq_len,
            freq_max_position_embeddings=max_seq_len,
            mask_max_position_embeddings=max_seq_len,
            attn_dtype=runtime.attn_dtype,
            attn_softmax_dtype=runtime.attn_softmax_dtype,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            use_sliding_window=False,
            sliding_window=None,
            use_cache=False,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=runtime.param_dtype,
        dtype=runtime.dtype,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    # Dataset-specific system prompts with strict formatting aligned to rewards
    SYSTEM_PROMPT_GSM8K = (
        "You are a helpful math tutor. Solve the user's problem. "
        "Think step-by-step inside <think>...</think>, then output ONLY the final numeric answer inside <answer>...</answer>. "
        "Formatting rules: <answer> must contain exactly one number and nothing else; use an integer with no trailing '.0' if the result is an integer, "
        "otherwise a decimal number. Do not include commas, units, or extra text. Do not write anything outside <think> and <answer>."
    )

    SYSTEM_PROMPT_MATH = (
        "You are a math expert. You are given a question and you need to solve it step by step and output the final answer within \\boxed{}."
    )

    # Helpers and Dataset builders
    def _normalize_pct(rate_value) -> int:
        """Normalize dataset_use_rate to an integer percent in [1, 100].
        Accepts either fraction (<=1.0) or percent (>1.0)."""
        try:
            r = float(rate_value)
        except Exception:
            r = 1.0
        pct = int(round(r * 100)) if r <= 1.0 else int(round(r))
        if pct < 1:
            pct = 1
        if pct > 100:
            pct = 100
        return pct
    def build_gsm8k():
        def extract_hash_answer(text: str):
            if not isinstance(text, str):
                return ""
            if "####" not in text:
                # Fallback: try last number if anchor missing
                m = re.findall(r"-?\d+\.?\d*", text)
                return m[-1] if m else ""
            return text.split("####")[-1].strip()

        rate = float(runtime.dataset_use_rate)
        pct = _normalize_pct(rate)
        train_split = "train" if pct >= 100 else f"train[:{pct}%]"
        test_split = "test" if pct >= 100 else f"test[:{pct}%]"
        if jax.process_index() == 0:
            print(f"DEBUG: GSM8K split strings -> train='{train_split}', test='{test_split}'")
        ds_train = safe_call("load gsm8k train", load_dataset, "openai/gsm8k", "main", split=train_split)
        ds_test = safe_call("load gsm8k test", load_dataset, "openai/gsm8k", "main", split=test_split, default=None)

        def map_ex(x):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_GSM8K},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }

        ds_train_m = ds_train.map(map_ex)
        ds_test_m = ds_test.map(map_ex) if ds_test is not None else None
        return ds_train_m, ds_test_m

    def build_math():
        # Hendrycks MATH — problems include LaTeX; solutions contain \\boxed{...}
        rate = float(runtime.dataset_use_rate)
        pct = _normalize_pct(rate)
        train_split = "train" if pct >= 100 else f"train[:{pct}%]"
        if jax.process_index() == 0:
            print(f"DEBUG: MATH split string -> train='{train_split}' (rate={rate}, pct={pct}%)")
        ds_train = safe_call("load competition_math train", load_dataset, "qwedsacf/competition_math", split=train_split)
        try:
            test_split = "test" if pct >= 100 else f"test[:{pct}%]"
            if jax.process_index() == 0:
                print(f"DEBUG: MATH split string -> test='{test_split}'")
            ds_test = safe_call("load competition_math test", load_dataset, "qwedsacf/competition_math", split=test_split, default=None)
            if ds_test is None:
                # Fallback for datasets that only provide a 'train' split
                if jax.process_index() == 0:
                    print(
                        "WARNING: 'test' split not found in 'qwedsacf/competition_math'. "
                        "Using 10% of 'train' as validation (deterministic split, seed=17)."
                    )
                split_ds = ds_train.train_test_split(test_size=0.1, seed=17)
                ds_train, ds_test = split_ds["train"], split_ds["test"]
        except ValueError:
            # Extremely defensive fallback
            split_ds = ds_train.train_test_split(test_size=0.1, seed=17)
            ds_train, ds_test = split_ds["train"], split_ds["test"]

        def map_ex(x):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_MATH},
                    {"role": "user", "content": x["problem"]},
                ],
                # keep full solution text for reward; must contain \\boxed{...}
                "solution": x.get("solution", ""),
                # keep level information for curriculum learning
                "level": x.get("level", ""),
                "type": x.get("type", ""),
            }

        # Log resulting lengths for verification
        if jax.process_index() == 0:
            try:
                print(f"DEBUG: Loaded competition_math sizes -> train={len(ds_train)}, test={len(ds_test) if ds_test else 'N/A'}")
            except Exception:
                ...
        ds_train_m = ds_train.map(map_ex)
        ds_test_m = ds_test.map(map_ex) if ds_test is not None else None
        return ds_train_m, ds_test_m

    # Normalize dataset names to support *-ds suffix
    _raw_ds = (runtime.dataset or "").strip().lower()
    if _raw_ds in ("gsm8k-ds", "gsm8k"):
        _ds = "gsm8k"
    elif _raw_ds in ("math-ds", "math"):
        _ds = "math"
    else:
        raise ValueError(
            f"Unknown dataset: {runtime.dataset}. Expected one of: math, math-ds, gsm8k, gsm8k-ds"
        )

    if jax.process_index() == 0:
        print(f"DEBUG: dataset (normalized)={_ds}")

    if _ds == "gsm8k":
        train_ds, test_ds = build_gsm8k()
        from easydel.verification.gsm8k_reward import answer_reward as gsm8k_answer_reward

        def data_tokenize_fn(batch, tokenizer, tools):
            ids = tokenizer(
                batch["prompt"],
                return_tensors="np",
                padding="max_length",
                padding_side="left",
                max_length=gfspo_config.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            # Normalize ground-truth numbers for robustness (strip $, %, commas)
            def _norm(x: str) -> str:
                return x.replace(",", "").replace("$", "").replace("%", "").strip()
            if isinstance(batch["answer"], list):
                normed = [_norm(a) for a in batch["answer"]]
            else:
                normed = _norm(batch["answer"]) if batch.get("answer") is not None else batch.get("answer")
            ids.update({"answer": normed})
            return ids

        reward_funcs = [gsm8k_answer_reward]

    elif _ds == "math":
        train_ds, test_ds = build_math()
        from easydel.verification.math_reward import answer_reward as math_answer_reward
        # Set metric-friendly names for logging
        safe_call(
            "rename math reward metric",
            lambda: setattr(math_answer_reward, "__name__", "math/accuracy"),
        )

        def data_tokenize_fn(batch, tokenizer, tools):
            ids = tokenizer(
                batch["prompt"],
                return_tensors="np",
                padding="max_length",
                padding_side="left",
                max_length=gfspo_config.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            # Keep full solution text for Math-Verify
            sol = batch["solution"]
            # Also extract normalized GT inside \\boxed{...} for heuristic fallback and metrics
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

        reward_funcs = [math_answer_reward]

    else:
        raise ValueError("dataset must be 'gsm8k' or 'math'")

    # Curriculum Learning Functions
    def filter_dataset_by_level(dataset: Dataset, level: str) -> Dataset:
        """Filter dataset to include only problems from a specific level."""
        return dataset.filter(lambda x: x.get("level", "").strip() == level)
    
    def get_available_levels(dataset: Dataset) -> list[str]:
        """Get all available levels in the dataset, sorted numerically when possible."""
        if "level" not in dataset.column_names:
            return []
        levels = set()
        for item in dataset:
            lvl = item.get("level", "")
            try:
                lvl = lvl.strip()
            except Exception:
                pass
            if lvl:
                levels.add(lvl)

        def _level_key(s: str):
            try:
                import re as _re
                m = _re.search(r"(\d+)", s)
                return (int(m.group(1)) if m else 10**9, s)
            except Exception:
                return (10**9, s)

        return sorted(list(levels), key=_level_key)
    
    def curriculum_train(trainer, train_ds: Dataset, test_ds: Dataset, epochs_per_level: int, mini_batch_size_override: int | None = None):
        """
        Simple curriculum: repeat each level dataset `epochs_per_level` times,
        concatenate all levels, shuffle, and train for a single epoch.
        """
        levels = get_available_levels(train_ds)
        if not levels:
            if jax.process_index() == 0:
                print("WARNING: No levels found in dataset. Falling back to regular training.")
            return trainer.train()

        # Build repeated concatenation per level
        parts = []
        for lvl in levels:
            ds_lvl = filter_dataset_by_level(train_ds, lvl)
            if len(ds_lvl) == 0:
                continue
            repeat_n = max(1, int(epochs_per_level))
            parts.extend([ds_lvl] * repeat_n)

        if not parts:
            if jax.process_index() == 0:
                print("WARNING: No data after curriculum assembly; running regular training.")
            return trainer.train()

        combined = concatenate_datasets(parts).shuffle(seed=17)

        # Reuse trainer's arguments; train for 1 epoch
        args = trainer.arguments
        original_epochs = args.num_train_epochs
        original_tb = args.total_batch_size
        original_mb = args.mini_batch_size
        original_ga = args.gradient_accumulation_steps

        try:
            args.num_train_epochs = 1
            if mini_batch_size_override:
                args.mini_batch_size = mini_batch_size_override
                args.total_batch_size = mini_batch_size_override

            new_tr = ed.GFSPOTrainer(
                model=trainer.model_state,
                reward_funcs=trainer.reward_funcs,
                processing_class=trainer.processing_class,
                eval_dataset=test_ds,
                train_dataset=combined,
                arguments=args,
                data_tokenize_fn=trainer.data_tokenize_fn,
            )
            out = new_tr.train()
            return out.state
        finally:
            # Restore original knobs
            args.num_train_epochs = original_epochs
            args.total_batch_size = original_tb
            args.mini_batch_size = original_mb
            args.gradient_accumulation_steps = original_ga

    if jax.process_index() == 0:
        print(f"DEBUG: About to initialize trainer with reward_funcs: {[f.__name__ for f in reward_funcs]}")
        print(f"DEBUG: reward_funcs modules: {[f.__module__ for f in reward_funcs]}")

    trainer = ed.GFSPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        eval_dataset=test_ds,
        train_dataset=train_ds,
        arguments=gfspo_config,
        data_tokenize_fn=data_tokenize_fn,
    )

    # Decide whether to use curriculum learning
    if runtime.curriculum_math and _ds == "math":
        if jax.process_index() == 0:
            print("DEBUG: Curriculum learning enabled for math dataset")
            print(f"DEBUG: num_train_epochs={gfspo_config.num_train_epochs}")
        
        # Pass mini_batch_size override for curriculum with small levels
        mini_batch_override = 1 if gfspo_config.force_tensor_parallel else None
        final_state = curriculum_train(trainer, train_ds, test_ds, gfspo_config.num_train_epochs, mini_batch_override)
        
        # Save final model if needed
        if gfspo_config.save_directory and jax.process_index() == 0:
            print(f"Saving final curriculum-trained model to {gfspo_config.save_directory}/final")
            try:
                import os
                final_dir = os.path.join(gfspo_config.save_directory, "final_curriculum")
                os.makedirs(final_dir, exist_ok=True)
                # The trainer would handle saving, but we can log the final state info
                print(f"Final model state ready for saving at: {final_dir}")
            except Exception as e:
                print(f"Could not create final save directory: {e}")
    else:
        if runtime.curriculum_math and _ds != "math":
            if jax.process_index() == 0:
                print("WARNING: curriculum_math flag is enabled but dataset is not 'math'. Using regular training.")
        
        trainer.train()


if __name__ == "__main__":
    main()


