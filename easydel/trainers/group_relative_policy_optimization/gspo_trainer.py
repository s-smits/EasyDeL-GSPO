# gspo_trainer.py

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from ..trainer_protocol import TrainerConfigureFunctionOutput
from ._gspo_fn import gspo_step
from ._jit_utils import compile_step_pair
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
        
        try:
            print(f"DEBUG: Initializing GSPO trainer - importance_sampling_level={arguments.importance_sampling_level}, epsilon={arguments.epsilon}, beta={arguments.beta}")
            logger.info(
                f"Initialized GSPO trainer with importance_sampling_level={arguments.importance_sampling_level}, "
                f"epsilon={arguments.epsilon}, beta={arguments.beta}"
            )
        except Exception as e:
            print(f"DEBUG: Failed to log GSPO trainer initialization: {e}")
            logger.warning(f"Failed to log GSPO trainer initialization: {e}")

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions for GSPO.
        Uses parent's generation function and infrastructure; only swaps the step
        functions to GSPO-specific ones and extends static args.
        """
        # Call parent configure_functions to set up shared components (including generate_function)
        parent_result = super().configure_functions()
        
        # Now override just the training and evaluation functions to use gspo_step instead of grpo_step
        from jax.sharding import NamedSharding, PartitionSpec
        
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        
        # GSPO-specific training step static arguments (add importance_sampling_level and epsilon)
        # Allow gradient accumulation to be driven by config (incl. microbatch_one_completion)
        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.importance_sampling_level,
            self.arguments.epsilon,
            getattr(self.arguments, "clip_in_log_space", True),
            getattr(self.arguments, "log_clip_epsilon", self.arguments.epsilon),
            getattr(self.arguments, "importance_weight_normalization", "mean"),
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        # GSPO-specific evaluation step static arguments
        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.importance_sampling_level,
            self.arguments.epsilon,
            getattr(self.arguments, "clip_in_log_space", True),
            getattr(self.arguments, "log_clip_epsilon", self.arguments.epsilon),
            getattr(self.arguments, "importance_weight_normalization", "mean"),
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        # Recompile evaluation step with correct eval static args (train/eval may share shape but differ flag)
        sharded_training_step_function, sharded_evaluation_step_function = compile_step_pair(
            gspo_step,
            self.state_shardings,
            empty_sharding,
            self._train_shared_fn_static_args,
            self._eval_shared_fn_static_args,
        )

        # Return the same structure but with GSPO-specific step functions
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=parent_result.mesh,
            checkpoint_manager=parent_result.checkpoint_manager,
        )
