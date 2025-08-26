# gfspo_stable_trainer.py

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from .gfspo_trainer import GFSPOTrainer
from .gspo_trainer import GSPOTrainer
from .gfpo_utils import GFPOFilterMixin

logger = get_logger(__name__)


class GFSPOStableTrainer(GFSPOTrainer):
    """
    GFSPO variant that uses weighted, shrinkage-normalized advantages with
    sample-size-aware variance flooring and passes `selection_weights` for
    masking KL + PG in the step function.

    Wraps the existing GFSPO pipeline and only overrides preprocessing.
    """

    def __init__(
        self,
        arguments,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs,
        train_dataset=None,
        eval_dataset=None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
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
        # Initialization log (mirror other trainers)
        try:
            alpha = getattr(arguments, "gfpo_shrinkage_alpha", 0.5)
            c = getattr(arguments, "gfpo_sigma_floor_c", 0.25)
            soft = getattr(arguments, "gfpo_soft_mask", False)
            temp = getattr(arguments, "gfpo_soft_temperature", 0.5)
            iw_norm = getattr(arguments, "importance_weight_normalization", "mean")
            clip_log = getattr(arguments, "clip_in_log_space", True)
            log_eps = getattr(arguments, "log_clip_epsilon", arguments.epsilon)

            print(
                "DEBUG: Initializing GFSPO stable trainer - "
                f"G={arguments.gfpo_group_size}, "
                f"k={arguments.gfpo_retain_count}, "
                f"alpha={alpha}, c={c}, soft_mask={soft}, temp={temp}, "
                f"iw_norm={iw_norm}, clip_log={clip_log}, log_eps={log_eps}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
            logger.info(
                "Initialized GFSPO stable trainer with "
                f"G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, "
                f"alpha={alpha}, sigma_floor_c={c}, soft_mask={soft}, temp={temp}, "
                f"iw_norm={iw_norm}, clip_in_log_space={clip_log}, log_clip_epsilon={log_eps}, "
                f"importance_sampling_level={arguments.importance_sampling_level}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
        except Exception as e:
            try:
                print(f"DEBUG: Failed to log GFSPO stable trainer initialization: {e}")
            except Exception:
                ...
            logger.warning(f"Failed to log GFSPO stable trainer initialization: {e}")

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        # Bypass GFSPO's override to preserve raw GRPO/GSPO preprocessing outputs
        grpo_batch, metrics = GSPOTrainer._preprocess_batch_input(self, state, batch, is_train)

        # Shapes
        try:
            num_prompts = int(batch["input_ids"].shape[0])
        except Exception:
            num_prompts = 0
        G = int(self.arguments.gfpo_group_size)

        # Recover grouped rewards or reconstruct from advantages
        eps = jnp.asarray(getattr(self.arguments, "advantage_epsilon", 1e-8), dtype=jnp.float32)
        rw = grpo_batch.get("rewards", None)
        if rw is not None:
            rewards_grouped = rw.reshape(num_prompts, G)
        else:
            try:
                adv = grpo_batch["advantages"].reshape(num_prompts, G)
                mu = jnp.mean(adv, axis=1, keepdims=True)
                sd = jnp.maximum(jnp.std(adv, axis=1, keepdims=True), eps)
                rewards_grouped = adv * sd + mu
            except Exception:
                rewards_grouped = jnp.zeros((num_prompts, G), dtype=jnp.float32)

        # Grouped lengths (for metrics + possible metric-based filtering)
        try:
            lengths = jnp.sum(grpo_batch["completion_mask"], axis=-1)
            lengths_grouped = lengths.reshape(num_prompts, G)
        except Exception:
            lengths_grouped = jnp.ones((num_prompts, G), dtype=jnp.float32)

        # Build selection weights (mask) on host; allow soft [0,1] but binary is fine
        try:
            mask = self._gfpo_build_mask_host(rewards_grouped, lengths_grouped)  # (B, G) float
        except Exception:
            mask = jnp.ones_like(rewards_grouped, dtype=jnp.float32)

        # Weighted shrinkage stats
        sum_w = jnp.sum(mask, axis=1, keepdims=True)  # (B,1)
        sum_w2 = jnp.sum(mask * mask, axis=1, keepdims=True)
        n_eff = (sum_w * sum_w) / jnp.maximum(sum_w2, 1e-6)

        mu_S = jnp.sum(mask * rewards_grouped, axis=1, keepdims=True) / jnp.maximum(sum_w, 1e-6)
        mu_G = jnp.mean(rewards_grouped, axis=1, keepdims=True)

        var_S_num = jnp.sum(mask * (rewards_grouped - mu_S) ** 2, axis=1, keepdims=True)
        var_S = var_S_num / jnp.maximum(n_eff - 1.0, 1.0)
        var_G = jnp.var(rewards_grouped, axis=1, keepdims=True)

        alpha = float(getattr(self.arguments, "gfpo_shrinkage_alpha", 0.5))
        lam = alpha * (1.0 - (self.arguments.gfpo_retain_count / self.arguments.gfpo_group_size))
        lam = jnp.clip(lam, 0.0, 1.0)

        sigma2 = (1.0 - lam) * var_S + lam * var_G
        c = float(getattr(self.arguments, "gfpo_sigma_floor_c", 0.25))
        sigma2 = sigma2 + (c * c) / jnp.maximum(n_eff - 1.0, 1.0)
        sigma = jnp.sqrt(jnp.maximum(sigma2, jnp.asarray(1e-6, dtype=sigma2.dtype)))

        center = (1.0 - lam) * mu_S + lam * mu_G
        # Weighted, shrunk advantages; do NOT multiply by mask here.
        # Downweighting is applied via selection_weights in the step function.
        A = (rewards_grouped - center) / sigma

        grpo_batch["advantages"] = A.reshape(-1)
        grpo_batch["selection_weights"] = mask.reshape(-1)

        # Provide epsilon_scale and beta_scale as runtime scalars (avoid recompiles)
        try:
            if bool(getattr(self.arguments, "use_ess_epsilon", False)) and getattr(self, "_epsilon_scale", None) is not None:
                grpo_batch["epsilon_scale"] = jnp.asarray(float(self._epsilon_scale), dtype=jnp.float32)
            else:
                k = float(self.arguments.gfpo_retain_count)
                g = float(self.arguments.gfpo_group_size)
                grpo_batch["epsilon_scale"] = jnp.asarray(k / max(g, 1.0), dtype=jnp.float32)
        except Exception:
            pass

        try:
            grpo_batch["beta_scale"] = jnp.asarray(float(getattr(self, "_beta_scale", 1.0)), dtype=jnp.float32)
        except Exception:
            pass

        # Optional metrics (host-only); skip if anything fails
        try:
            metrics.update(self._gfpo_compute_metrics_host(mask, lengths_grouped))
        except Exception:
            pass

        # Keep batch lean: only pass arrays used by the step

        return grpo_batch, metrics

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics,
        step: int,
    ):
        state, metrics = super().on_step_end(state, metrics, step)
        try:
            clipped = float(metrics.get("clipped_fraction", 0.0))
            tau_clip = float(getattr(self.arguments, "gfpo_metric_switch_clip", 0.35))
            cur = getattr(self.arguments, "gfpo_metric", "length")
            if clipped > tau_clip and cur != "length":
                self.arguments.gfpo_metric = "length"
            elif clipped < 0.20 and cur != "token_efficiency":
                self.arguments.gfpo_metric = "token_efficiency"
        except Exception:
            pass
        return state, metrics


def trainer(**kwargs) -> GFSPOStableTrainer:
    return GFSPOStableTrainer(**kwargs)

# Backward-compat alias
GFSPOWShrinkTrainer = GFSPOStableTrainer
