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

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from ..trainer_protocol import TrainerConfigureFunctionOutput
from ._gspo_fn import gspo_step
from .gspo_config import GSPOConfig
from .grpo_trainer import GRPOTrainer, RewardFunc

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = get_logger(__name__)


class GSPOTrainer(GRPOTrainer):
    arguments: GSPOConfig  # type hinting

    def __init__(
        self,
        arguments: GSPOConfig,
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
        assert isinstance(arguments, GSPOConfig), f"arguments type must be `GSPOConfig` but got {type(arguments)}"
        
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
        
        # Override parent's arguments with GSPO-specific config
        self.arguments = arguments
        
        logger.info(
            f"Initialized GSPO trainer with importance_sampling_level={arguments.importance_sampling_level}, "
            f"epsilon={arguments.epsilon}, beta={arguments.beta}"
        )

    def _configure_gspo_generate_function(self):
        """
        Configure the generate function with adaptive sharding based on batch size.
        """
        from transformers import GenerationConfig
        from easydel.utils.compiling_utils import ejit
        from jax.sharding import NamedSharding, PartitionSpec
        from eformer import common_types
        from .adaptive_mesh import get_adaptive_sharding_spec
        import inspect

        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # Use adaptive sharding based on batch size and tensor parallelism
        _sig = inspect.signature(get_adaptive_sharding_spec)
        _kwargs = dict(
            total_batch_size=self.arguments.total_batch_size,
            force_tensor_parallel=self.arguments.force_tensor_parallel,
            mini_batch_size=self.arguments.mini_batch_size,
        )
        if 'force_data_parallel' in _sig.parameters:
            _kwargs['force_data_parallel'] = self.arguments.force_data_parallel
        adaptive_spec = get_adaptive_sharding_spec(**_kwargs)
        input_sharding = NamedSharding(
            mesh=mesh,
            spec=adaptive_spec
        )

        def generate(state: EasyDeLState, input_ids, attention_mask, num_return_sequences: int):
            module = state.model

            with module.mesh:
                input_ids = module.config.partition_manager.shard(
                    input_ids,
                    axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                    mode=common_types.MODE_PREFILL,
                )
                attention_mask = module.config.partition_manager.shard(
                    attention_mask,
                    axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                    mode=common_types.MODE_PREFILL,
                )
                # Generation config with proper EOS token handling
                generation_config = GenerationConfig(
                    top_p=self.arguments.top_p,
                    top_k=self.arguments.top_k,
                    temperature=self.arguments.temperature,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    max_new_tokens=self.arguments.max_completion_length,
                    max_length=self.arguments.max_completion_length + self.arguments.max_prompt_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    use_cache=False,
                )
                
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                ).sequences
                
                return sequences, input_ids, attention_mask

        self.generate_function = ejit(
            generate,
            in_shardings=(self.state_shardings, input_sharding, input_sharding),
            out_shardings=(empty_sharding, input_sharding, input_sharding),
            static_argnums=(3,),
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions for GSPO.
        Fully overrides parent to ensure GSPO-specific generation function is used.
        """
        # Call parent configure_functions to set up shared components
        parent_result = super().configure_functions()
        
        # Override with GSPO-specific generation function (must be after parent call)
        self._configure_gspo_generate_function()
        
        # Now override just the training and evaluation functions to use gspo_step instead of grpo_step
        from easydel.utils.compiling_utils import ejit
        from jax.sharding import NamedSharding, PartitionSpec
        
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # GSPO-specific training step static arguments (add importance_sampling_level and epsilon)
        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.importance_sampling_level,
            self.arguments.epsilon,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        static_argnames = tuple(range(2, 2 + len(self._train_shared_fn_static_args)))

        sharded_training_step_function = ejit(
            gspo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        # GSPO-specific evaluation step static arguments
        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.importance_sampling_level,
            self.arguments.epsilon,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        sharded_evaluation_step_function = ejit(
            gspo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames

        # Return the same structure but with GSPO-specific step functions
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=parent_result.mesh,
            checkpoint_manager=parent_result.checkpoint_manager,
        ) 