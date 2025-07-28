# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import typing as tp
from functools import partial

import flax
import flax.nnx
import jax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time, get_logger
from easydel.utils.traversals import deepcopy_model

from ..prompt_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ._fn import get_per_token_logps
from ._dr_grpo_fn import dr_grpo_step, compute_dr_grpo_advantages
from .dr_grpo_config import DRGRPOConfig
from .grpo_trainer import GRPOTrainer, RewardFunc

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = get_logger(__name__)


class DRGRPOTrainer(GRPOTrainer):
    arguments: DRGRPOConfig  # type hinting

    def __init__(
        self,
        arguments: DRGRPOConfig,
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
        assert isinstance(arguments, DRGRPOConfig), f"arguments type must be `DRGRPOConfig` but got {type(arguments)}"
        
        # Initialize the parent GRPO trainer
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
        
        logger.info(
            f"Initialized DR GRPO trainer with constant_normalization={arguments.use_constant_normalization}, "
            f"constant_factor={arguments.constant_normalization_factor}, "
            f"disable_std_scaling={arguments.disable_std_scaling}, "
            f"advantage_whitening={arguments.use_advantage_whitening}"
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions for DR GRPO.
        """
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # Reuse generation function from parent - no need to redefine
        # The parent's generate_function is already properly set up

        # DR GRPO-specific training step static arguments
        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.use_constant_normalization,
            self.arguments.constant_normalization_factor,
            self.arguments.disable_std_scaling,
            self.arguments.advantage_epsilon,
            self.arguments.use_advantage_whitening,
            self.arguments.whitening_epsilon,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        static_argnames = tuple(range(2, 2 + len(self._train_shared_fn_static_args)))

        sharded_training_step_function = ejit(
            dr_grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        # DR GRPO-specific evaluation step static arguments
        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.use_constant_normalization,
            self.arguments.constant_normalization_factor,
            self.arguments.disable_std_scaling,
            self.arguments.advantage_epsilon,
            self.arguments.use_advantage_whitening,
            self.arguments.whitening_epsilon,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        sharded_evaluation_step_function = ejit(
            dr_grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        # Reuse reference model computation from parent
        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

        # Token sharding should match step_partition_spec for properly sharded tokens
        token_sharding = NamedSharding(
            mesh=mesh,
            spec=self.arguments.step_partition_spec
        )
        
        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                token_sharding,
                token_sharding,
            ),
            out_shardings=empty_sharding,
        )

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """
        Enhanced preprocessing with DR GRPO's corrected advantage computation.
        """
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                sequences, prompt_ids, prompt_mask = jax.block_until_ready(
                    self.generate_function(state, prompt_ids, prompt_mask)
                )
            generation_time = generation_time_fn()
            prompt_completion_ids = sequences
            completion_ids = prompt_completion_ids[..., prompt_ids.shape[-1] :]
            completion_mask = self._make_attn_mask(completion_ids)
            ridmask = prompt_mask.repeat(self.num_generations, 0)

            with capture_time() as token_logps_time_fn:
                ref_per_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    prompt_completion_ids,
                    jnp.concatenate([ridmask, completion_mask], -1),
                )
            token_logps_time = token_logps_time_fn()
            prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            rewards_per_func = jnp.zeros(
                (prompt_ids.shape[0] * self.num_generations, len(self.reward_funcs)),
                dtype="f4",
            )
            with capture_time() as rewarding_time_fn:
                for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes, strict=False)
                ):
                    if isinstance(reward_func, EasyDeLState):
                        if is_conversational:
                            messages = [
                                {"messages": p + c}
                                for p, c in zip(prompts * self.num_generations, completions, strict=False)
                            ]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts * self.num_generations, completions, strict=False)]

                        rew = reward_func.apply_fn(
                            reward_func.graphdef,
                            reward_func.graphstate,
                            reward_func.graphother,
                            dict(
                                reward_processing_class(
                                    texts,
                                    return_tensors="jax",
                                    padding="max_length",
                                    padding_side="right",
                                    add_special_tokens=False,
                                    truncation=True,
                                    return_attention_mask=True,
                                    max_length=self.arguments.max_sequence_length,
                                )
                            ),
                        ).logits[:, 0]
                    else:
                        in_prompts = prompts * self.num_generations
                        output_reward_func = reward_func(
                            prompts=in_prompts,
                            completions=completions,
                            max_length=self.arguments.max_sequence_length,
                            batch=batch,
                        )
                        rew = jnp.array(output_reward_func, dtype="f4")
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()
            
            with capture_time() as grouped_comp_time_fn:
                rewards = rewards_per_func.sum(axis=1)
                
                # DR GRPO: Use corrected advantage computation
                advantages = compute_dr_grpo_advantages(
                    rewards=rewards,
                    num_generations=self.num_generations,
                    use_constant_normalization=self.arguments.use_constant_normalization,
                    constant_normalization_factor=self.arguments.constant_normalization_factor,
                    disable_std_scaling=self.arguments.disable_std_scaling,
                    advantage_epsilon=self.arguments.advantage_epsilon,
                    use_advantage_whitening=self.arguments.use_advantage_whitening,
                    whitening_epsilon=self.arguments.whitening_epsilon,
                )
            grouped_comp_time = grouped_comp_time_fn()
            
        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.sum(completion_mask.sum(-1), -1)
        metrics_dict = {
            "rewards": jnp.mean(rewards, -1),
            "completion_length": completion_length,
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            # DR GRPO specific metrics
            "constant_normalization": float(self.arguments.use_constant_normalization),
            "constant_normalization_factor": self.arguments.constant_normalization_factor,
            "disable_std_scaling": float(self.arguments.disable_std_scaling),
            "advantage_whitening": float(self.arguments.use_advantage_whitening),
        }
        for i, reward_func in enumerate(self.reward_funcs):
            _name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            metrics_dict[_name] = jnp.mean(rewards_per_func[:, i])
        if self.log_table is not None:
            cur_step = jax.device_get(state.step)
            decoded_text = self.processing_class.batch_decode(jax.device_get(completion_ids))
            for text in decoded_text:
                self.log_table.add_data(text, generation_time, completion_length, cur_step)
            wandb.log({"generations": self.log_table}, step=cur_step)

        return (
            {
                "prompt_ids": self._all_gather(prompt_ids),
                "prompt_mask": self._all_gather(prompt_mask),
                "completion_ids": self._all_gather(completion_ids),
                "completion_mask": self._all_gather(completion_mask),
                "ref_per_token_logps": self._all_gather(ref_per_token_logps),
                "advantages": advantages,
            },
            metrics_dict,
        ) 