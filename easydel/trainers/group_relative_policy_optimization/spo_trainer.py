# spo_trainer.py

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from ..trainer_protocol import TrainerConfigureFunctionOutput
from .gspo_trainer import GSPOTrainer
from .gfspo_config import GFSPOConfig
from .gfpo_utils import GFPOFilterMixin

logger = get_logger(__name__)


class SPOTrainer(GFPOFilterMixin, GSPOTrainer):
    """
    Unified Sequence Policy Optimization trainer that merges:
    - GSPO: sequence-level importance weighting and step plumbing.
    - GFSPO (stable): GFPO filtering + shrinkage-normalized advantages with variance flooring.

    Behavior is controlled by config. If GFPO fields are present (G, k, etc.),
    stable GFPO preprocessing is applied; otherwise it behaves like plain GSPO.
    """

    # Accept both GSPOConfig and GFSPOConfig at call sites; we annotate with GFSPOConfig for full feature hints
    arguments: GFSPOConfig  # type: ignore[assignment]

    def __init__(
        self,
        arguments: GFSPOConfig,  # type: ignore[override]
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs,  # same RewardFunc union
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
        # Use local alias to keep the richer GFSPOConfig available if passed
        self.arguments = arguments

        # Runtime scalars (avoid recompiles): initialize defaults
        self._epsilon_scale: float | None = None
        self._beta_scale: float = 1.0

        try:
            print(
                "DEBUG: Initializing SPO trainer - "
                f"importance_sampling={arguments.importance_sampling_level}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
            logger.info(
                f"Initialized SPO trainer: importance_sampling_level={arguments.importance_sampling_level}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
        except Exception as e:
            try:
                print(f"DEBUG: Failed to log SPO trainer initialization: {e}")
            except Exception:
                ...
            logger.warning(f"Failed to log SPO trainer initialization: {e}")

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Use GSPO's function setup (generation + GSPO step), unified across trainers.
        """
        return super().configure_functions()

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        # First run GSPO/GRPO base preprocessing (generation, ref logps, base rewards/advantages)
        base_batch, metrics = GSPOTrainer._preprocess_batch_input(self, state, batch, is_train)

        # Detect if GFPO is effectively configured (presence of fields and sensible sizes)
        has_gfpo = all(
            hasattr(self.arguments, name)
            for name in ("gfpo_group_size", "gfpo_retain_count")
        )
        if not has_gfpo:
            return base_batch, metrics

        try:
            num_prompts = int(batch["input_ids"].shape[0])
        except Exception:
            num_prompts = 0

        G = int(getattr(self.arguments, "gfpo_group_size", 0))
        k = int(getattr(self.arguments, "gfpo_retain_count", 0))
        if num_prompts <= 0 or G <= 1 or k < 1:
            # Nothing to filter
            return base_batch, metrics

        eps = jnp.asarray(getattr(self.arguments, "advantage_epsilon", 1e-8), dtype=jnp.float32)

        # Recover grouped rewards or reconstruct from base advantages
        rw = base_batch.get("rewards", None)
        if rw is not None:
            rewards_grouped = rw.reshape(num_prompts, G)
        else:
            try:
                adv = base_batch["advantages"].reshape(num_prompts, G)
                mu = jnp.mean(adv, axis=1, keepdims=True)
                sd = jnp.maximum(jnp.std(adv, axis=1, keepdims=True), eps)
                rewards_grouped = adv * sd + mu
            except Exception:
                rewards_grouped = jnp.zeros((num_prompts, G), dtype=jnp.float32)

        # Grouped lengths (for metrics + possible metric-based filtering)
        try:
            lengths = base_batch.get("completion_lengths")
            if lengths is None:
                lengths = jnp.sum(base_batch["completion_mask"], axis=-1)
            lengths_grouped = lengths.reshape(num_prompts, G)
        except Exception:
            lengths_grouped = jnp.ones((num_prompts, G), dtype=jnp.float32)

        # Build selection weights (mask) on host; allow soft [0,1] but binary is fine
        try:
            mask = self._gfpo_build_mask_host(rewards_grouped, lengths_grouped)  # (B, G) float
        except Exception:
            mask = jnp.ones_like(rewards_grouped, dtype=jnp.float32)

        # Weighted shrinkage stats (stable variant)
        sum_w = jnp.sum(mask, axis=1, keepdims=True)  # (B,1)
        sum_w2 = jnp.sum(mask * mask, axis=1, keepdims=True)
        n_eff = (sum_w * sum_w) / jnp.maximum(sum_w2, 1e-6)

        mu_S = jnp.sum(mask * rewards_grouped, axis=1, keepdims=True) / jnp.maximum(sum_w, 1e-6)
        mu_G = jnp.mean(rewards_grouped, axis=1, keepdims=True)

        var_S_num = jnp.sum(mask * (rewards_grouped - mu_S) ** 2, axis=1, keepdims=True)
        var_S = var_S_num / jnp.maximum(n_eff - 1.0, 1.0)
        var_G = jnp.var(rewards_grouped, axis=1, keepdims=True)

        alpha = float(getattr(self.arguments, "gfpo_shrinkage_alpha", 0.5))
        lam = alpha * (1.0 - (k / max(float(G), 1.0)))
        lam = jnp.clip(lam, 0.0, 1.0)

        sigma2 = (1.0 - lam) * var_S + lam * var_G
        c = float(getattr(self.arguments, "gfpo_sigma_floor_c", 0.25))
        sigma2 = sigma2 + (c * c) / jnp.maximum(n_eff - 1.0, 1.0)
        sigma = jnp.sqrt(jnp.maximum(sigma2, jnp.asarray(1e-6, dtype=sigma2.dtype)))

        center = (1.0 - lam) * mu_S + lam * mu_G
        # Weighted, shrunk advantages; do NOT multiply by mask here (mask applied in step via selection_weights)
        A = (rewards_grouped - center) / sigma

        # Write back unified outputs used by gspo_step
        base_batch["advantages"] = A.reshape(-1)
        base_batch["selection_weights"] = mask.reshape(-1)

        # Provide epsilon_scale (ESS or k/G) and beta_scale as runtime scalars or vectors to avoid recompiles
        try:
            _n = int(base_batch["completion_mask"].shape[0])
        except Exception:
            _n = None

        try:
            if bool(getattr(self.arguments, "use_ess_epsilon", False)) and (self._epsilon_scale is not None):
                scale = float(self._epsilon_scale)
            else:
                scale = float(k) / max(float(G), 1.0)
            if _n is not None:
                base_batch["epsilon_scale"] = jnp.full((_n,), scale, dtype=jnp.float32)
            else:
                base_batch["epsilon_scale"] = jnp.asarray(scale, dtype=jnp.float32)
        except Exception:
            pass

        try:
            bscale = float(getattr(self, "_beta_scale", 1.0))
            if _n is not None:
                base_batch["beta_scale"] = jnp.full((_n,), bscale, dtype=jnp.float32)
            else:
                base_batch["beta_scale"] = jnp.asarray(bscale, dtype=jnp.float32)
        except Exception:
            pass

        # Provide completion lengths for reuse in step (avoid recompute)
        try:
            if "completion_lengths" not in base_batch:
                base_batch["completion_lengths"] = jnp.sum(base_batch["completion_mask"], axis=-1)
        except Exception:
            pass

        # Optional metrics (host-only); skip if anything fails
        try:
            metrics.update(self._gfpo_compute_metrics_host(mask, lengths_grouped))
        except Exception:
            pass

        return base_batch, metrics

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

        # Optional: switch GFPO metric based on clipping fraction for stability
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


def trainer(**kwargs) -> SPOTrainer:
    """Convenience factory for building an SPOTrainer."""
    return SPOTrainer(**kwargs)

