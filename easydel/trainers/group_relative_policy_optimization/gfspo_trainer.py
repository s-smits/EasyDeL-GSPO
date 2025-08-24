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
from .gfspo_config import GFSPOConfig
from .gfpo_utils import GFPOFilterMixin

logger = get_logger(__name__)


class GFSPOTrainer(GFPOFilterMixin, GSPOTrainer):
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
        assert isinstance(
            arguments, GFSPOConfig
        ), f"arguments type must be `GFSPOConfig` but got {type(arguments)}"

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
            print(
                "DEBUG: Initializing GFSPO trainer - "
                f"G={arguments.gfpo_group_size}, "
                f"k={arguments.gfpo_retain_count}, "
                f"importance_sampling={arguments.importance_sampling_level}"
            )
            logger.info(
                f"Initialized GFSPO trainer: G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}, "
                f"adaptive={arguments.gfpo_adaptive}, importance_sampling_level={arguments.importance_sampling_level}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
        except Exception as e:
            print(f"DEBUG: Failed to log GFSPO trainer initialization: {e}")
            logger.warning(f"Failed to log GFSPO trainer initialization: {e}")

    # Remove local copy; rely on GFPOFilterMixin via _gfpo_build_mask_host

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

        # Compute rewards_grouped with robust fallbacks
        try:
            if "rewards" in grpo_batch:
                rewards_arr = grpo_batch["rewards"]
                rewards_grouped = rewards_arr.reshape(num_prompts, G)
            else:
                advantages = grpo_batch["advantages"].reshape(num_prompts, G)
                mean_per_prompt = jnp.mean(advantages, axis=1, keepdims=True)
                std_per_prompt = jnp.std(advantages, axis=1, keepdims=True)
                std_per_prompt = jnp.maximum(std_per_prompt, eps)
                rewards_grouped = advantages * std_per_prompt + mean_per_prompt
        except Exception:
            # Last-resort fallback: reconstruct from advantages or use zeros
            try:
                advantages = grpo_batch["advantages"].reshape(num_prompts, G)
                mean_per_prompt = jnp.mean(advantages, axis=1, keepdims=True)
                std_per_prompt = jnp.std(advantages, axis=1, keepdims=True)
                std_per_prompt = jnp.maximum(std_per_prompt, eps)
                rewards_grouped = advantages * std_per_prompt + mean_per_prompt
            except Exception:
                rewards_grouped = jnp.zeros((num_prompts, G), dtype=jnp.float32)

        # Compute lengths_grouped with fallback
        try:
            lengths = jnp.sum(grpo_batch["completion_mask"], axis=-1)
            lengths_grouped = lengths.reshape(num_prompts, G)
        except Exception:
            lengths_grouped = jnp.ones((num_prompts, G), dtype=jnp.float32)

        # Use shared host-only GFPO filter (simplified: fewer nested fallbacks)
        try:
            mask = self._gfpo_build_mask_host(rewards_grouped, lengths_grouped)
        except Exception as _e:
            if jax.process_index() == 0:
                try:
                    print("DEBUG: GFSPO preprocessing error; using GRPO advantages unchanged:", _e)
                except Exception:
                    ...
            mask = jnp.ones((num_prompts, G), dtype=jnp.float32)
        # Extra debug guardrails (proc0-only, host-only math)
        try:
            if jax.process_index() == 0:
                try:
                    import numpy as _np
                    mh = jax.device_get(mask)
                    mask_sum = float(_np.sum(mh))
                except Exception:
                    mask_sum = -1.0
                expected_min = float(
                    num_prompts
                    * min(int(getattr(self.arguments, "gfpo_retain_count", G)), G)
                )
                expected_max = float(num_prompts * G)
                print(
                    f"DEBUG: GFSPO grouping check: B={num_prompts}, G={G}, "
                    f"k={int(getattr(self.arguments,'gfpo_retain_count',-1))}, "
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
            # Host-only metric compute for stability
            m = self._gfpo_compute_metrics_host(mask, lengths_grouped)
            sel = float(m.get("gfpo/retention_rate", 0.0))
            avg_len = float(m.get("gfpo/avg_retained_length", 0.0))
            if jax.process_index() == 0:
                try:
                    print(f"DEBUG: GFSPO metrics - retention_rate={sel:.3f}, avg_retained_length={avg_len:.1f}")
                except Exception:
                    ...
            metrics_dict["gfpo/retention_rate"] = sel
            metrics_dict["gfpo/avg_retained_length"] = avg_len
        except Exception as e:
            try:
                if jax.process_index() == 0:
                    print(f"DEBUG: Failed to compute GFSPO metrics: {e}")
            except Exception:
                ...

        return grpo_batch, metrics_dict


def trainer(**kwargs) -> GFSPOTrainer:
    """Convenience factory for building a GFSPOTrainer."""
    return GFSPOTrainer(**kwargs)
