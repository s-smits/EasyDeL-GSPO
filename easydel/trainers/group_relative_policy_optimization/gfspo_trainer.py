# gfspo_trainer.py

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger
from .gspo_trainer import GSPOTrainer
from .gfpo_trainer import GFPOTrainer
from .gfspo_config import GFSPOConfig

logger = get_logger(__name__)


class GFSPOTrainer(GSPOTrainer):
    """
    Combined trainer: applies GFPO filtering and per-subset advantage standardization
    within the GRPO preprocessing pipeline, then uses GSPO's sequence-level importance
    sampling in the step function (via gspo_step).

    Inherits GSPO generation and step plumbing from `GSPOTrainer` and overrides
    preprocessing similarly to `GFPOTrainer`.
    """

    arguments: GFSPOConfig  # type hinting

    def __init__(
        self,
        arguments: GFSPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs,  # same RewardFunc union
        train_dataset=None,
        eval_dataset=None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        assert isinstance(arguments, GFSPOConfig), f"arguments type must be `GFSPOConfig` but got {type(arguments)}"

        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )

        # Ensure local alias to config type
        self.arguments = arguments

        try:
            print(f"DEBUG: Initializing GFSPO trainer - G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, importance_sampling={arguments.importance_sampling_level}")
            logger.info(
                f"Initialized GFSPO trainer: G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}, "
                f"adaptive={arguments.gfpo_adaptive}, importance_sampling_level={arguments.importance_sampling_level}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
        except Exception as e:
            print(f"DEBUG: Failed to log GFSPO trainer initialization: {e}")
            logger.warning(f"Failed to log GFSPO trainer initialization: {e}")

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        # First run the GRPO preprocessing implemented by the parent
        grpo_batch, metrics_dict = super()._preprocess_batch_input(state, batch, is_train)

        # Apply GFPO filtering and recompute advantages over the retained subset
        try:
            num_prompts = int(batch["input_ids"].shape[0])
        except Exception:
            num_prompts = 0
        G = int(self.arguments.gfpo_group_size)

        eps = jnp.float32(self.arguments.advantage_epsilon)
        if "rewards" in grpo_batch:
            rewards_arr = grpo_batch["rewards"]
            try:
                rewards_grouped = rewards_arr.reshape(num_prompts, G)
            except Exception:
                # Fallback: reconstruct from advantages
                advantages = grpo_batch["advantages"].reshape(num_prompts, G)
                mean_per_prompt = jnp.mean(advantages, axis=1, keepdims=True)
                std_per_prompt = jnp.std(advantages, axis=1, keepdims=True)
                std_per_prompt = jnp.maximum(std_per_prompt, eps)
                rewards_grouped = advantages * std_per_prompt + mean_per_prompt
        else:
            advantages = grpo_batch["advantages"].reshape(num_prompts, G)
            mean_per_prompt = jnp.mean(advantages, axis=1, keepdims=True)
            std_per_prompt = jnp.std(advantages, axis=1, keepdims=True)
            std_per_prompt = jnp.maximum(std_per_prompt, eps)
            rewards_grouped = advantages * std_per_prompt + mean_per_prompt

        lengths = jnp.sum(grpo_batch["completion_mask"], axis=-1)
        lengths_grouped = lengths.reshape(num_prompts, G)

        # Reuse GFPO filter helper directly
        mask = GFPOTrainer._filter_mask_per_prompt(self, rewards_grouped, lengths_grouped)
        # Extra debug guardrails to ensure filtering is effective
        try:
            if jax.process_index() == 0:
                try:
                    mask_sum = float(jnp.sum(mask))
                    expected_min = float(num_prompts * min(int(self.arguments.gfpo_retain_count), G))
                    expected_max = float(num_prompts * G)
                except Exception:
                    mask_sum, expected_min, expected_max = -1.0, -1.0, -1.0
                print(
                    f"DEBUG: GFSPO grouping check: B={num_prompts}, G={G}, k={int(getattr(self.arguments,'gfpo_retain_count',-1))}, "
                    f"mask_sum={mask_sum}, expected_min≈{expected_min}, expected_max={expected_max}"
                )
        except Exception:
            ...

        selected_count = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        selected_sum = jnp.sum(rewards_grouped * mask, axis=1, keepdims=True)
        mu_S = selected_sum / selected_count

        diff_sq = jnp.where(mask > 0.0, (rewards_grouped - mu_S) ** 2, 0.0)
        denom = jnp.maximum(selected_count - 1.0, 1.0)
        sigma_S = jnp.sqrt(jnp.sum(diff_sq, axis=1, keepdims=True) / denom)
        sigma_S = jnp.maximum(sigma_S, eps)

        advantages_gfpo = ((rewards_grouped - mu_S) / sigma_S) * mask
        grpo_batch["advantages"] = advantages_gfpo.reshape(-1)

        try:
            retention_rate = float(jnp.mean(mask))
            avg_retained_length = float(
                jnp.sum(lengths * mask.reshape(-1)) / float(jnp.maximum(jnp.sum(mask), 1.0))
            )
            print(f"DEBUG: GFSPO metrics - retention_rate={retention_rate:.3f}, avg_retained_length={avg_retained_length:.1f}")
            metrics_dict["gfpo/retention_rate"] = retention_rate
            metrics_dict["gfpo/avg_retained_length"] = avg_retained_length
        except Exception as e:
            print(f"DEBUG: Failed to compute GFSPO metrics: {e}")
            pass

        return grpo_batch, metrics_dict


def trainer(**kwargs) -> GFSPOTrainer:
    """Convenience factory for building a GFSPOTrainer."""
    return GFSPOTrainer(**kwargs)


