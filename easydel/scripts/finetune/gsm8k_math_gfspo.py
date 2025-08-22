import re
from dataclasses import field

import jax
from datasets import load_dataset, Dataset
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from transformers import AutoConfig, AutoTokenizer

import easydel as ed
from easydel.infra.factory import registry
from easydel.modules import *  # noqa: F401,F403 — ensure kernels are registered


@auto_pytree
class RunTimeConfig:
    repo_id: str = field(metadata={"help": "The repository ID for the policy model."})
    processor_repo_id: str | None = field(default=None)
    dataset: str = field(
        default="math-ds",
        metadata={"help": "Dataset to use: 'gsm8k'|'gsm8k-ds' or 'math'|'math-ds'"},
    )
    dataset_use_rate: int = field(default=100)
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
        try:
            import os
            print(f"DEBUG: runtime.dataset (raw)={runtime.dataset}")
            print(f"DEBUG: ENV DATASET={os.environ.get('DATASET')}")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(runtime.processor_repo_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_prompt_len = gfspo_config.max_prompt_length
    max_completion_len = gfspo_config.max_completion_length
    max_seq_len = max_prompt_len + max_completion_len

    hf_config = AutoConfig.from_pretrained(runtime.repo_id)
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

    model = load_module.from_pretrained(
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
        "The answer must contain exactly one number and nothing else; use an integer with no trailing '.0' if the result is an integer, otherwise a decimal number. Do not include commas, units, or extra text."
    )

    SYSTEM_PROMPT_MATH = (
        "You are a competitive math solver. Solve the problem step by step and output the final answer within \\boxed{}."
    )

    # Dataset builders
    def build_gsm8k() -> tuple[Dataset, Dataset]:
        def extract_hash_answer(text: str):
            if not isinstance(text, str):
                return ""
            if "####" not in text:
                # Fallback: try last number if anchor missing
                m = re.findall(r"-?\d+\.?\d*", text)
                return m[-1] if m else ""
            return text.split("####")[-1].strip()

        ds_train = load_dataset("openai/gsm8k", "main", split=f"train[:{runtime.dataset_use_rate}%]")
        ds_test = load_dataset("openai/gsm8k", "main", split=f"test[:{runtime.dataset_use_rate}%]")

        def map_ex(x):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_GSM8K},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }

        return ds_train.map(map_ex), ds_test.map(map_ex)

    def build_math() -> tuple[Dataset, Dataset]:
        # Hendrycks MATH — problems include LaTeX; solutions contain \\boxed{...}
        ds_train = load_dataset("qwedsacf/competition_math", split=f"train[:{runtime.dataset_use_rate}%]")
        try:
            ds_test = load_dataset("qwedsacf/competition_math", split=f"test[:{runtime.dataset_use_rate}%]")
        except ValueError as e:
            # Fallback for datasets that only provide a 'train' split
            if "Unknown split" in str(e) or "test" in str(e):
                if jax.process_index() == 0:
                    print(
                        "WARNING: 'test' split not found in 'qwedsacf/competition_math'. "
                        "Using 10% of 'train' as validation (deterministic split, seed=17)."
                    )
                split_ds = ds_train.train_test_split(test_size=0.1, seed=17)
                ds_train, ds_test = split_ds["train"], split_ds["test"]
            else:
                raise

        def map_ex(x):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_MATH},
                    {"role": "user", "content": x["problem"]},
                ],
                # keep full solution text for reward; must contain \\boxed{...}
                "solution": x.get("solution", ""),
            }

        return ds_train.map(map_ex), ds_test.map(map_ex)

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
        from easydel.verification.gsm8k_reward import format_reward as gsm8k_format_reward
        from easydel.verification.gsm8k_reward import answer_reward as gsm8k_answer_reward
        # Set metric-friendly names for logging
        try:
            gsm8k_format_reward.__name__ = "gsm8k/format_rate"  # type: ignore
            gsm8k_answer_reward.__name__ = "gsm8k/accuracy"  # type: ignore
        except Exception:
            pass

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

        reward_funcs = [gsm8k_format_reward, gsm8k_answer_reward]

    elif _ds == "math":
        train_ds, test_ds = build_math()
        from easydel.verification.math_reward import answer_reward as math_answer_reward
        from easydel.verification.math_reward import format_reward as math_format_reward
        # Set metric-friendly names for logging
        try:
            math_format_reward.__name__ = "math/format_rate"  # type: ignore
            math_answer_reward.__name__ = "math/accuracy"  # type: ignore
        except Exception:
            pass

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

        reward_funcs = [math_format_reward, math_answer_reward]

    else:
        raise ValueError("dataset must be 'gsm8k' or 'math'")

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

    trainer.train()


if __name__ == "__main__":
    main()


