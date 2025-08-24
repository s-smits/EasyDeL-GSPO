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
        TPU-stable top‑k mask: compute entirely on host (NumPy) and return as jnp.
        """
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore

        try:
            rg = jax.device_get(rewards_grouped)
            lg = jax.device_get(lengths_grouped)
        except Exception:
            rg, lg = rewards_grouped, lengths_grouped  # type: ignore

        try:
            bsz, gsize = int(rg.shape[0]), int(rg.shape[1])
        except Exception:
            bsz, gsize = int(rewards_grouped.shape[0]), int(rewards_grouped.shape[1])

        # Scores
        ascending = True
        if getattr(self.arguments, "gfpo_metric", "length") == "length":
            scores_h = lg
            ascending = True
        else:
            eps_eff = float(getattr(self.arguments, "gfpo_efficiency_epsilon", 1e-8))
            try:
                scores_h = rg / _np.maximum(lg, eps_eff)  # type: ignore
            except Exception:
                scores_h = rg / (lg + eps_eff)
            ascending = False

        # k-per-prompt
        if not getattr(self.arguments, "gfpo_adaptive", False):
            k_per = [int(self.arguments.gfpo_retain_count)] * bsz
        else:
            try:
                model_state = getattr(self, "model_state", None)
                if model_state is not None and hasattr(model_state, "step"):
                    state_step = int(jax.device_get(model_state.step))
                else:
                    state_step = 0
            except Exception:
                state_step = 0
            warmup = int(getattr(self.arguments, "gfpo_adaptive_warmup_steps", 10))
            if state_step < warmup:
                k0 = int(self.arguments.gfpo_adaptive_k_map.get("very_hard", 8))
                k_per = [k0] * bsz
            else:
                km = self.arguments.gfpo_adaptive_k_map
                k_vh = int(km.get("very_hard", 8))
                k_h = int(km.get("hard", 8))
                k_m = int(km.get("medium", 6))
                k_e = int(km.get("easy", 4))
                try:
                    avg_rewards = _np.mean(rg, axis=1) if _np is not None else rg.mean(axis=1)
                except Exception:
                    avg_rewards = jax.device_get(jnp.mean(rewards_grouped, axis=1))
                method = getattr(self.arguments, "gfpo_adaptive_method", "rolling")
                if method == "ema":
                    try:
                        cur = _np.percentile(avg_rewards, [25.0, 50.0, 75.0])
                    except Exception:
                        cur = jax.device_get(jnp.percentile(jnp.asarray(avg_rewards), jnp.array([25.0, 50.0, 75.0])))
                    alpha = float(getattr(self.arguments, "gfpo_adaptive_ema_alpha", 0.1))
                    if not hasattr(self, "_running_percentiles"):
                        self._running_percentiles = cur
                    else:
                        self._running_percentiles = (1.0 - alpha) * self._running_percentiles + alpha * cur
                    q25, q50, q75 = [float(x) for x in self._running_percentiles]
                else:
                    if not hasattr(self, "_difficulty_buffer"):
                        self._difficulty_buffer = []
                    try:
                        self._difficulty_buffer.extend([float(x) for x in list(avg_rewards)])
                    except Exception:
                        pass
                    max_hist = int(getattr(self.arguments, "gfpo_adaptive_history_max", 20000))
                    if len(self._difficulty_buffer) > max_hist:
                        self._difficulty_buffer = self._difficulty_buffer[-max_hist:]
                    if len(self._difficulty_buffer) < 40:
                        q25 = q50 = q75 = None
                    else:
                        try:
                            q25, q50, q75 = _np.percentile(_np.asarray(self._difficulty_buffer, dtype=_np.float32), [25.0, 50.0, 75.0])
                        except Exception:
                            arr = jnp.asarray(self._difficulty_buffer, dtype=jnp.float32)
                            q25, q50, q75 = [float(x) for x in jax.device_get(jnp.percentile(arr, jnp.array([25.0, 50.0, 75.0])))]
                k_per = []
                for r in (avg_rewards.tolist() if hasattr(avg_rewards, "tolist") else list(avg_rewards)):
                    if q25 is None:
                        k_per.append(k_vh)
                    elif r < q25:
                        k_per.append(k_vh)
                    elif r < q50:
                        k_per.append(k_h)
                    elif r < q75:
                        k_per.append(k_m)
                    else:
                        k_per.append(k_e)

        upper = max(1, int(gsize) - 1)
        k_per = [int(max(1, min(upper, int(x)))) for x in k_per]

        try:
            idx_sorted = _np.argsort(scores_h, axis=1) if ascending else _np.argsort(-scores_h, axis=1)
        except Exception:
            idx_sorted = jax.device_get(jnp.argsort(jnp.asarray(scores_h), axis=1) if ascending else jnp.argsort(-jnp.asarray(scores_h), axis=1))
        mask_h = _np.zeros((bsz, gsize), dtype=_np.float32)
        for i in range(bsz):
            ki = int(k_per[i])
            mask_h[i, idx_sorted[i, :ki]] = 1.0

        return jnp.asarray(mask_h, dtype=jnp.float32)

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
        try:
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

            # Call our own filter method (host-stable)
            mask = self._filter_mask_per_prompt(rewards_grouped, lengths_grouped)
        except Exception as _e:
            # Fallback to GRPO advantages unchanged if GFPO filter fails
            if jax.process_index() == 0:
                try:
                    print(f"DEBUG: GFSPO preprocessing error; using GRPO advantages unchanged: {_e}")
                except Exception:
                    pass
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
                expected_min = float(num_prompts * min(int(getattr(self.arguments, 'gfpo_retain_count', G)), G))
                expected_max = float(num_prompts * G)
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
            # Host-only metric compute for stability
            mh = jax.device_get(mask)
            lg_h = jax.device_get(lengths_grouped)
            import numpy as _np
            sel = float(_np.mean(mh))
            tot = float(max(1.0, float(_np.sum(mh))))
            avg_len = float((_np.sum(lg_h * mh)) / tot)
            if jax.process_index() == 0:
                print(f"DEBUG: GFSPO metrics - retention_rate={sel:.3f}, avg_retained_length={avg_len:.1f}")
            metrics_dict["gfpo/retention_rate"] = sel
            metrics_dict["gfpo/avg_retained_length"] = avg_len
        except Exception as e:
            print(f"DEBUG: Failed to compute GFSPO metrics: {e}")
            pass

        return grpo_batch, metrics_dict


def trainer(**kwargs) -> GFSPOTrainer:
    """Convenience factory for building a GFSPOTrainer."""
    return GFSPOTrainer(**kwargs)
