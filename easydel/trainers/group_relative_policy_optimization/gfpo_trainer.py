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
from .grpo_trainer import GRPOTrainer, RewardFunc

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = get_logger(__name__)


class GFPOTrainer(GRPOTrainer):
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
            print(f"DEBUG: Initializing GFPO trainer - G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}")
            logger.info(
                f"Initialized GFPO trainer: G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}, adaptive={arguments.gfpo_adaptive}"
            )
        except Exception as e:
            print(f"DEBUG: Failed to log GFPO trainer initialization: {e}")
            logger.warning(f"Failed to log GFPO trainer initialization: {e}")

    def _filter_mask_per_prompt(
        self,
        rewards_grouped: jnp.ndarray,  # (B, G)
        lengths_grouped: jnp.ndarray,  # (B, G)
    ) -> jnp.ndarray:
        """
        Create a binary mask selecting the top-k responses per prompt according to the
        configured metric. Returns mask with shape (B, G) of floats in {0.0, 1.0}.
        """
        bsz, gsize = rewards_grouped.shape
        # Compute scores per configured metric
        if self.arguments.gfpo_metric == "length":
            # Lower is better
            scores = lengths_grouped
            ascending = True
        else:
            # token_efficiency = reward / length; Higher is better
            scores = rewards_grouped / jnp.maximum(lengths_grouped, 1e-8)
            ascending = False

        # Determine k per prompt (fixed or adaptive per Algorithm 2)
        if not self.arguments.gfpo_adaptive:
            k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_retain_count))
        else:
            # Compute average reward per prompt and quartiles over current batch
            avg_rewards = jnp.mean(rewards_grouped, axis=1)
            # Use batch-level percentiles as a lightweight approximation of t-digest
            q25, q50, q75 = jnp.percentile(avg_rewards, jnp.array([25.0, 50.0, 75.0]))
            # Assign k as in paper (very hard/hard/medium/easy): 8, 8, 6, 4
            k_very_hard = 8
            k_hard = 8
            k_medium = 6
            k_easy = 4
            k_per_prompt = jnp.where(
                avg_rewards < q25,
                k_very_hard,
                jnp.where(
                    avg_rewards < q50,
                    k_hard,
                    jnp.where(avg_rewards < q75, k_medium, k_easy),
                ),
            )

        # Build mask per prompt by argsort
        mask = jnp.zeros((bsz, gsize), dtype=jnp.float32)
        for i in range(bsz):
            k_i = int(k_per_prompt[i])
            if ascending:
                chosen = jnp.argsort(scores[i])[:k_i]
            else:
                chosen = jnp.argsort(-scores[i])[:k_i]
            mask = mask.at[i, chosen].set(1.0)
        return mask

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

        # Minimal GFPO metrics
        try:
            retention_rate = float(jnp.mean(mask))
            avg_retained_length = float(
                jnp.sum(lengths * mask.reshape(-1)) / float(jnp.maximum(jnp.sum(mask), 1.0))
            )
            print(f"DEBUG: GFPO metrics - retention_rate={retention_rate:.3f}, avg_retained_length={avg_retained_length:.1f}")
            metrics_dict["gfpo/retention_rate"] = retention_rate
            metrics_dict["gfpo/avg_retained_length"] = avg_retained_length
        except Exception as e:
            print(f"DEBUG: Failed to compute GFPO metrics: {e}")
            pass

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


