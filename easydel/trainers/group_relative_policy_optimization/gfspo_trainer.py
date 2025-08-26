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

        # Runtime scalars (avoid recompiles): initialize defaults
        self._epsilon_scale: float | None = None
        self._beta_scale: float = 1.0

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

        # Use shared host-only GFPO filter (soft or hard depending on config)
        try:
            mask = self._gfpo_build_mask_host(rewards_grouped, lengths_grouped)
        except Exception as _e:
            mask = jnp.ones((num_prompts, G), dtype=jnp.float32)

        # Weighted + shrinkage subset stats (stable for small k and soft masks)
        sum_w = jnp.sum(mask, axis=1, keepdims=True)
        sum_w2 = jnp.sum(mask * mask, axis=1, keepdims=True)
        n_eff = (sum_w * sum_w) / jnp.maximum(sum_w2, 1e-6)

        mu_S = jnp.sum(rewards_grouped * mask, axis=1, keepdims=True) / jnp.maximum(sum_w, 1e-6)
        mu_G = jnp.mean(rewards_grouped, axis=1, keepdims=True)

        var_S_num = jnp.sum(mask * (rewards_grouped - mu_S) ** 2, axis=1, keepdims=True)
        var_S = var_S_num / jnp.maximum(n_eff - 1.0, 1.0)
        var_G = jnp.var(rewards_grouped, axis=1, keepdims=True)

        alpha = float(getattr(self.arguments, "gfpo_shrinkage_alpha", 0.5))
        lam = alpha * (1.0 - (self.arguments.gfpo_retain_count / self.arguments.gfpo_group_size))
        lam = jnp.clip(lam, 0.0, 1.0)

        c = float(getattr(self.arguments, "gfpo_sigma_floor_c", 0.25))
        sigma2 = (1.0 - lam) * var_S + lam * var_G + (c * c) / jnp.maximum(n_eff - 1.0, 1.0)
        sigma = jnp.sqrt(jnp.maximum(sigma2, eps))

        center = (1.0 - lam) * mu_S + lam * mu_G
        # Unmasked standardized advantages; selection_weights applied inside step
        advantages_gfpo = (rewards_grouped - center) / sigma
        grpo_batch["advantages"] = advantages_gfpo.reshape(-1)
        grpo_batch["selection_weights"] = mask.reshape(-1)

        # Provide completion lengths for reuse in step (avoid recompute)
        try:
            if "completion_lengths" not in grpo_batch:
                grpo_batch["completion_lengths"] = jnp.sum(grpo_batch["completion_mask"], axis=-1)
        except Exception:
            pass

        try:
            # Host-only metric compute for stability
            m = self._gfpo_compute_metrics_host(mask, lengths_grouped)
            metrics_dict["gfpo/retention_rate"] = float(m.get("gfpo/retention_rate", 0.0))
            metrics_dict["gfpo/avg_retained_length"] = float(m.get("gfpo/avg_retained_length", 0.0))
        except Exception:
            pass

        # Runtime knobs to avoid recompiles in step
        # epsilon_scale: prefer dynamic ESS-based if enabled and available; else fallback to k/G
        try:
            if bool(getattr(self.arguments, "use_ess_epsilon", False)) and (self._epsilon_scale is not None):
                grpo_batch["epsilon_scale"] = jnp.asarray(float(self._epsilon_scale), dtype=jnp.float32)
            else:
                k = float(self.arguments.gfpo_retain_count)
                g = float(self.arguments.gfpo_group_size)
                grpo_batch["epsilon_scale"] = jnp.asarray(k / max(g, 1.0), dtype=jnp.float32)
        except Exception:
            pass

        # beta_scale: multiplicative controller on top of static beta
        try:
            grpo_batch["beta_scale"] = jnp.asarray(float(self._beta_scale), dtype=jnp.float32)
        except Exception:
            pass

        # Keep batch lean: only pass arrays that are consumed by the step

        return grpo_batch, metrics_dict

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics,
        step: int,
    ):
        state, metrics = super().on_step_end(state, metrics, step)

        # Update epsilon_scale using previous-step ESS if enabled
        try:
            if bool(getattr(self.arguments, "use_ess_epsilon", False)):
                ess_mean_norm = metrics.get("ess/mean_norm", None)
                if ess_mean_norm is not None:
                    try:
                        ess_val = float(ess_mean_norm)
                    except Exception:
                        import jax
                        ess_val = float(jax.device_get(ess_mean_norm))
                    power = float(getattr(self.arguments, "epsilon_ess_power", 0.5))
                    scale = max(1e-6, ess_val) ** power
                    smin = float(getattr(self.arguments, "epsilon_min_scale", 0.1))
                    smax = float(getattr(self.arguments, "epsilon_max_scale", 1.0))
                    self._epsilon_scale = float(min(max(scale, smin), smax))
        except Exception:
            pass

        # Update beta_scale via simple proportional controller to hit KL target
        try:
            kl_obs = metrics.get("kl/mean", None)
            if kl_obs is not None:
                try:
                    kl_val = float(kl_obs)
                except Exception:
                    import jax
                    kl_val = float(jax.device_get(kl_obs))
                kl_target = float(getattr(self.arguments, "kl_target", 0.02))
                eta = float(getattr(self.arguments, "beta_update_rate", 0.1))
                # multiplicative update on the scale
                new_scale = float(self._beta_scale) * float(jnp.exp(jnp.asarray(eta * (kl_val - kl_target))))
                smin = float(getattr(self.arguments, "beta_min_scale", 0.1))
                smax = float(getattr(self.arguments, "beta_max_scale", 10.0))
                self._beta_scale = float(min(max(new_scale, smin), smax))
        except Exception:
            pass

        return state, metrics


def trainer(**kwargs) -> GFSPOTrainer:
    """Convenience factory for building a GFSPOTrainer."""
    return GFSPOTrainer(**kwargs)
