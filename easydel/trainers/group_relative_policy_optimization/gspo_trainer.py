# gspo_trainer.py

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
        from easydel.utils.compiling_utils import ejit
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
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        # Compute static arg indices robustly, capped by function arity
        import inspect as _inspect
        _sig = _inspect.signature(gspo_step)
        _max_pos_index = len(_sig.parameters) - 1  # zero-based last positional index
        _end = min(2 + len(self._train_shared_fn_static_args), _max_pos_index + 1)
        static_argnames = tuple(range(2, _end))

        sharded_training_step_function = ejit(
            gspo_step,
            in_shardings=(self.state_shardings, None),
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
            in_shardings=(self.state_shardings, None),
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