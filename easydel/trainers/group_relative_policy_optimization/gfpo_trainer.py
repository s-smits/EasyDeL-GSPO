# gfpo_trainer.py

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from ..trainer_protocol import TrainerConfigureFunctionOutput
from ._gfpo_fn import gfpo_step
from .gfpo_config import GFPOConfig
from .gfpo_utils import GFPOFilterMixin
from .grpo_trainer import GRPOTrainer, RewardFunc

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = get_logger(__name__)


class GFPOTrainer(GFPOFilterMixin, GRPOTrainer):
    """
    GFPO (Group Filtered Policy Optimization) Trainer.

    Extends GRPO by sampling G responses per prompt, filtering to k responses using a
    metric (length or reward/length), and computing standardized advantages only over
    the retained subset per prompt. Rejected responses receive zero advantage.
    """

    arguments: GFPOConfig  # type hinting

    def __init__(
        self,
        arguments: GFPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        assert arguments is not None, (
            "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
        )
        assert isinstance(arguments, GFPOConfig), f"arguments type must be `GFPOConfig` but got {type(arguments)}"

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
            logger.info(
                f"Initialized GFPO trainer: G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}, adaptive={arguments.gfpo_adaptive}"
            )
        except Exception:
            pass

    def _filter_mask_per_prompt(
        self,
        rewards_grouped: jnp.ndarray,  # (B, G)
        lengths_grouped: jnp.ndarray,  # (B, G)
    ) -> jnp.ndarray:
        """
        Create a binary mask selecting the top-k responses per prompt according to the
        configured metric. Returns mask with shape (B, G) of floats in {0.0, 1.0}.

        TPU-stable: perform the entire mask construction on host (NumPy) to avoid any
        device program launches outside the main pjit step.
        """
        # Pull to host
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore

        try:
            rg = jax.device_get(rewards_grouped)
            lg = jax.device_get(lengths_grouped)
        except Exception:
            # Fallback: if already on host
            rg, lg = rewards_grouped, lengths_grouped  # type: ignore

        # Host shapes
        try:
            bsz, gsize = int(rg.shape[0]), int(rg.shape[1])
        except Exception:
            bsz, gsize = int(rewards_grouped.shape[0]), int(rewards_grouped.shape[1])

        # Compute scores per configured metric on host
        ascending = True
        if getattr(self.arguments, "gfpo_metric", "length") == "length":
            scores_h = lg
            ascending = True
        else:
            eps_eff = float(getattr(self.arguments, "gfpo_efficiency_epsilon", 1e-8))
            try:
                scores_h = rg / _np.maximum(lg, eps_eff)  # type: ignore
            except Exception:
                # Very defensive fallback
                scores_h = rg / (lg + eps_eff)
            ascending = False

        # Determine k per prompt (fixed or adaptive) on host
        if not getattr(self.arguments, "gfpo_adaptive", False):
            k_per = [int(self.arguments.gfpo_retain_count)] * bsz
        else:
            # Warmup
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
                k0 = int(self.arguments.gfpo_adaptive_k_map.get("very_hard", 8))
                k_per = [k0] * bsz
            else:
                # rolling or ema; compute percentiles on host
                try:
                    avg_rewards = _np.mean(rg, axis=1) if _np is not None else rg.mean(axis=1)
                except Exception:
                    # fallback using jnp then bring back
                    avg_rewards = jax.device_get(jnp.mean(rewards_grouped, axis=1))
                method = getattr(self.arguments, "gfpo_adaptive_method", "rolling")
                km = self.arguments.gfpo_adaptive_k_map
                k_vh = int(km.get("very_hard", 8))
                k_h = int(km.get("hard", 8))
                k_m = int(km.get("medium", 6))
                k_e = int(km.get("easy", 4))
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
                    # rolling buffer on CPU
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
                        q25 = q50 = q75 = None  # force very_hard
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

        # Clamp k to [1, G-1]
        upper = max(1, int(gsize) - 1)
        k_per = [int(max(1, min(upper, int(x)))) for x in k_per]

        # Build mask via argsort on host
        try:
            idx_sorted = _np.argsort(scores_h, axis=1) if ascending else _np.argsort(-scores_h, axis=1)
        except Exception:
            # Fallback: use jnp and bring to host
            idx_sorted = jax.device_get(jnp.argsort(scores_h, axis=1) if ascending else jnp.argsort(-jnp.asarray(scores_h), axis=1))
        mask_h = _np.zeros((bsz, gsize), dtype=_np.float32)
        arange_g = _np.arange(gsize)
        for i in range(bsz):
            k_i = int(k_per[i])
            chosen = idx_sorted[i, :k_i]
            mask_h[i, chosen] = 1.0

        # Proc0-only debug, host-only math
        try:
            if jax.process_index() == 0:
                m_sum = float(mask_h.sum())
                exp_sum = float(sum(k_per))
                print(
                    f"DEBUG: GFPO filter params: G={int(gsize)}, k_fixed={int(getattr(self.arguments,'gfpo_retain_count',-1))}, "
                    f"adaptive={bool(getattr(self.arguments,'gfpo_adaptive', False))}"
                )
                print(f"DEBUG: GFPO mask sum={m_sum}, expected_sum≈{exp_sum} (bsz={int(bsz)})")
        except Exception:
            pass

        return jnp.asarray(mask_h, dtype=jnp.float32)

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """
        Override GRPO preprocessing to add GFPO filtering and per-prompt advantage
        computation on the retained subset S only.
        """
        grpo_batch, metrics_dict = super()._preprocess_batch_input(state, batch, is_train)

        # Shapes
        try:
            num_prompts = int(batch["input_ids"].shape[0])
        except Exception:
            num_prompts = 0
        G = int(self.arguments.gfpo_group_size)

        try:
            # Prefer true rewards if provided by parent; fallback to reconstructing from advantages
            eps = jnp.float32(self.arguments.advantage_epsilon)
            if "rewards" in grpo_batch:
                rewards_arr = grpo_batch["rewards"]
                try:
                    rewards_grouped = rewards_arr.reshape(num_prompts, G)
                except Exception:
                    # Fallback to reconstruction
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

            # Compute completion lengths
            lengths = jnp.sum(grpo_batch["completion_mask"], axis=-1)
            lengths_grouped = lengths.reshape(num_prompts, G)

            # Build selection mask per prompt and compute GFPO advantages per prompt
            mask = self._filter_mask_per_prompt(rewards_grouped, lengths_grouped)  # (B, G)

            # Selected subset statistics per prompt (mu_S, sigma_S) computed from selected only
            selected_count = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
            selected_sum = jnp.sum(rewards_grouped * mask, axis=1, keepdims=True)
            mu_S = selected_sum / selected_count

            # Variance with Bessel's correction on selected entries only
            diff_sq = jnp.where(mask > 0.0, (rewards_grouped - mu_S) ** 2, 0.0)
            denom = jnp.maximum(selected_count - 1.0, 1.0)
            sigma_S = jnp.sqrt(jnp.sum(diff_sq, axis=1, keepdims=True) / denom)
            sigma_S = jnp.maximum(sigma_S, eps)

            # A^(m) = ((R - mu_S) / sigma_S) * m
            advantages_gfpo = ((rewards_grouped - mu_S) / sigma_S) * mask
            grpo_batch["advantages"] = advantages_gfpo.reshape(-1)

            # Minimal GFPO metrics (host-only for stability)
            try:
                import numpy as _np
                mh = jax.device_get(mask)
                lg_h = jax.device_get(lengths_grouped)
                retention_rate = float(_np.mean(mh))
                total_selected = float(max(1.0, float(_np.sum(mh))))
                avg_retained_length = float((_np.sum(lg_h * mh)) / total_selected)
                metrics_dict["gfpo/retention_rate"] = retention_rate
                metrics_dict["gfpo/avg_retained_length"] = avg_retained_length
            except Exception:
                pass
        except Exception as _e:
            # Fallback: keep GRPO advantages unchanged on error to avoid divergence
            if jax.process_index() == 0:
                try:
                    print(f"DEBUG: GFPO preprocessing error; falling back to GRPO advantages: {_e}")
                except Exception:
                    pass
            # No change to grpo_batch['advantages']

        return grpo_batch, metrics_dict

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configure training and evaluation step functions using gfpo_step. The loss is
        identical to GRPO; GFPO behavior arises from preprocessing that filters and
        standardizes advantages per retained subset.
        """
        parent = super().configure_functions()

        from easydel.utils.compiling_utils import ejit
        from jax.sharding import NamedSharding, PartitionSpec
        import inspect as _inspect

        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # Reuse static args built by the parent (num_generations, beta, loss_config, ...)
        _sig = _inspect.signature(gfpo_step)
        _max_pos_index = len(_sig.parameters) - 1
        _end = min(2 + len(self._train_shared_fn_static_args), _max_pos_index + 1)
        static_argnames = tuple(range(2, _end))

        sharded_training_step_function = ejit(
            gfpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        sharded_evaluation_step_function = ejit(
            gfpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=parent.mesh,
            checkpoint_manager=parent.checkpoint_manager,
        )
