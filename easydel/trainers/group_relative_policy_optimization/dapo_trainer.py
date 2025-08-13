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
from ._dapo_fn import dapo_step, check_batch_diversity
from .dapo_config import DAPOConfig
from .grpo_trainer import GRPOTrainer, RewardFunc

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = get_logger(__name__)


class DAPOTrainer(GRPOTrainer):
    arguments: DAPOConfig  # type hinting

    def __init__(
        self,
        arguments: DAPOConfig,
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
        assert isinstance(arguments, DAPOConfig), f"arguments type must be `DAPOConfig` but got {type(arguments)}"
        
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
        
        # DAPO-specific attributes
        self.resample_attempts = 0
        logger.info(
            f"Initialized DAPO trainer with clip_ratio_low={arguments.clip_ratio_low}, "
            f"clip_ratio_high={arguments.clip_ratio_high}, "
            f"dynamic_sampling={arguments.enable_dynamic_sampling}, "
            f"token_level_loss={arguments.use_token_level_loss}"
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions for DAPO.
        """
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        # Reuse generation function from parent - no need to redefine
        # The parent's generate_function is already properly set up

        # DAPO-specific training step static arguments
        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.clip_ratio_low,
            self.arguments.clip_ratio_high,
            self.arguments.use_token_level_loss,
            self.arguments.enable_overlong_reward_shaping,
            self.arguments.overlong_buffer_length,
            self.arguments.overlong_penalty_scale,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        static_argnames = tuple(range(2, 2 + len(self._train_shared_fn_static_args)))

        sharded_training_step_function = ejit(
            dapo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        # DAPO-specific evaluation step static arguments
        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.clip_ratio_low,
            self.arguments.clip_ratio_high,
            self.arguments.use_token_level_loss,
            self.arguments.enable_overlong_reward_shaping,
            self.arguments.overlong_buffer_length,
            self.arguments.overlong_penalty_scale,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        sharded_evaluation_step_function = ejit(
            dapo_step,
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

        # Use empty sharding for flexibility - the actual sharding is handled in preprocessing
        
        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
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
        Enhanced preprocessing with DAPO's dynamic sampling and overlong reward shaping.
        """
        # Dynamic sampling: Retry if batch lacks diversity
        for attempt in range(self.arguments.max_resample_attempts if self.arguments.enable_dynamic_sampling else 1):
            processed_batch, metrics_dict = super()._preprocess_batch_input(state, batch, is_train)
            
            if not self.arguments.enable_dynamic_sampling or is_train is False:
                # Skip dynamic sampling for evaluation or if disabled
                break
                
            # Check batch diversity
            if check_batch_diversity(
                processed_batch, 
                self.num_generations, 
                self.arguments.min_accuracy_variance
            ):
                # Sufficient diversity found
                break
            else:
                logger.info(f"Insufficient batch diversity, resampling attempt {attempt + 1}/{self.arguments.max_resample_attempts}")
                self.resample_attempts += 1
                
                # For resampling, we would need to regenerate completions
                # This is a simplified version - in practice, you might want to 
                # regenerate with different sampling parameters or use a different batch
                if attempt < self.arguments.max_resample_attempts - 1:
                    # Note: In a full implementation, you'd regenerate with higher temperature
                    # For now, just retry with the same parameters
                    processed_batch, metrics_dict = super()._preprocess_batch_input(state, batch, is_train)
        
        # Add DAPO-specific metrics
        metrics_dict.update({
            "resample_attempts": self.resample_attempts,
            "clip_ratio_low": self.arguments.clip_ratio_low,
            "clip_ratio_high": self.arguments.clip_ratio_high,
            "dynamic_sampling_enabled": float(self.arguments.enable_dynamic_sampling),
            "token_level_loss": float(self.arguments.use_token_level_loss),
            "overlong_reward_shaping": float(self.arguments.enable_overlong_reward_shaping),
        })
        
        return processed_batch, metrics_dict 