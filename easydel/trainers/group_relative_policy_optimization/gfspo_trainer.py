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

    def _filter_mask_per_prompt(
        self,
        rewards_grouped: jnp.ndarray,  # (B, G)
        lengths_grouped: jnp.ndarray,  # (B, G)
    ) -> jnp.ndarray:
        """
        Create a binary mask selecting the top-k responses per prompt according to the
        configured metric. Returns mask with shape (B, G) of floats in {0.0, 1.0}.
        Copied from GFPOTrainer to avoid cross-class method calls.
        """
        bsz, gsize = rewards_grouped.shape

        # Compute scores per configured metric
        if self.arguments.gfpo_metric == "length":
            # Lower is better
            scores = lengths_grouped
            ascending = True
        else:
            # token_efficiency = reward / length; Higher is better
            eps_eff = jnp.float32(getattr(self.arguments, "gfpo_efficiency_epsilon", 1e-8))
            scores = rewards_grouped / jnp.maximum(lengths_grouped, eps_eff)
            ascending = False

        # Determine k per prompt (fixed or adaptive per Algorithm 2)
        if not self.arguments.gfpo_adaptive:
            k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_retain_count), dtype=jnp.int32)
        else:
            # Method: warmup then either rolling history or EMA percentiles
            avg_rewards = jnp.mean(rewards_grouped, axis=1)
            try:
                model_state = getattr(self, "model_state", None)
                if model_state is not None and hasattr(model_state, "step"):
                    state_step = int(jax.device_get(model_state.step))
                else:
                    state_step = 0
            except Exception:
                state_step = 0

            warmup_steps = int(getattr(self.arguments, "gfpo_adaptive_warmup_steps", 10))
            if state_step < warmup_steps:
                k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_adaptive_k_map.get("very_hard", 8)), dtype=jnp.int32)
            else:
                method = getattr(self.arguments, "gfpo_adaptive_method", "rolling")
                if method == "ema":
                    current_percentiles = jnp.percentile(
                        avg_rewards, jnp.array([25.0, 50.0, 75.0], dtype=jnp.float32)
                    )
                    alpha = jnp.float32(getattr(self.arguments, "gfpo_adaptive_ema_alpha", 0.1))
                    if not hasattr(self, "_running_percentiles"):
                        self._running_percentiles = current_percentiles
                    else:
                        self._running_percentiles = (1.0 - alpha) * self._running_percentiles + alpha * current_percentiles
                    q25, q50, q75 = self._running_percentiles

                    km = self.arguments.gfpo_adaptive_k_map
                    k_vh = int(km.get("very_hard", 8))
                    k_h = int(km.get("hard", 8))
                    k_m = int(km.get("medium", 6))
                    k_e = int(km.get("easy", 4))
                    k_per_prompt = jnp.where(
                        avg_rewards < q25,
                        k_vh,
                        jnp.where(
                            avg_rewards < q50,
                            k_h,
                            jnp.where(avg_rewards < q75, k_m, k_e),
                        ),
                    )
                    k_per_prompt = k_per_prompt.astype(jnp.int32)
                else:
                    # rolling history on CPU
                    if not hasattr(self, "_difficulty_buffer"):
                        self._difficulty_buffer = []  # python list on CPU
                    try:
                        self._difficulty_buffer.extend([float(x) for x in jax.device_get(avg_rewards)])
                    except Exception:
                        pass
                    max_hist = int(getattr(self.arguments, "gfpo_adaptive_history_max", 20000))
                    if len(self._difficulty_buffer) > max_hist:
                        self._difficulty_buffer = self._difficulty_buffer[-max_hist:]

                    if len(self._difficulty_buffer) < 40:
                        k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_adaptive_k_map.get("very_hard", 8)), dtype=jnp.int32)
                    else:
                        hist = jnp.asarray(self._difficulty_buffer, dtype=jnp.float32)
                        q25, q50, q75 = jnp.percentile(
                            hist, jnp.array([25.0, 50.0, 75.0], dtype=jnp.float32)
                        )
                        km = self.arguments.gfpo_adaptive_k_map
                        k_vh = int(km.get("very_hard", 8))
                        k_h = int(km.get("hard", 8))
                        k_m = int(km.get("medium", 6))
                        k_e = int(km.get("easy", 4))
                        k_per_prompt = jnp.where(
                            avg_rewards < q25,
                            k_vh,
                            jnp.where(
                                avg_rewards < q50,
                                k_h,
                                jnp.where(avg_rewards < q75, k_m, k_e),
                            ),
                        )
                        k_per_prompt = k_per_prompt.astype(jnp.int32)

        # Clamp k to valid range [1, G-1] to avoid degenerate full retention
        upper = max(1, int(gsize) - 1)
        k_per_prompt = jnp.clip(k_per_prompt, 1, upper).astype(jnp.int32)

        # Build mask per prompt by argsort (vectorized, no Python loop)
        idx_sorted = jnp.argsort(scores, axis=1) if ascending else jnp.argsort(-scores, axis=1)
        arange_g = jnp.arange(gsize)[None, :]
        k_col = k_per_prompt.reshape(-1, 1)
        mask_sorted = (arange_g < k_col).astype(jnp.float32)
        row_idx = jnp.arange(bsz)[:, None]
        mask = jnp.zeros((bsz, gsize), dtype=jnp.float32).at[row_idx, idx_sorted].set(mask_sorted)
        return mask

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

        # Call our own filter method (inherited from GFPOTrainer)
        mask = self._filter_mask_per_prompt(rewards_grouped, lengths_grouped)
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
            total_selected = jnp.maximum(jnp.sum(mask), 1.0)
            avg_retained_length = float(jnp.sum(lengths_grouped * mask) / total_selected)
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


