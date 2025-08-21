# gspo_trainer.py

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from eformer.escale import with_sharding_constraint
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
        if 'rollouts_per_step' in _sig.parameters and getattr(self.arguments, 'rollouts_per_step', None):
            _kwargs['rollouts_per_step'] = self.arguments.rollouts_per_step
        adaptive_spec = get_adaptive_sharding_spec(**_kwargs)
        input_sharding = NamedSharding(
            mesh=mesh,
            spec=adaptive_spec
        )
        # step_sharding is not used further; rely on step_partition_spec from arguments

        def generate(state: EasyDeLState, input_ids, attention_mask, num_return_sequences: int, prng_seed: int):
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
                
                # Build PRNG key from provided seed to ensure per-chunk diversity
                import jax
                prng_key = jax.random.PRNGKey(prng_seed)
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    prng_key=prng_key,
                ).sequences
                
                # Re-constrain inputs to the step partition spec for downstream ops
                input_ids = with_sharding_constraint(input_ids, self.arguments.step_partition_spec)
                attention_mask = with_sharding_constraint(attention_mask, self.arguments.step_partition_spec)
                return sequences, input_ids, attention_mask

        self.generate_function = ejit(
            generate,
            in_shardings=(self.state_shardings, input_sharding, input_sharding, empty_sharding),
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
        
        # Derive rollouts_per_step from num_return_sequences if not provided (preferred)
        mesh_shape = getattr(mesh, "shape", {})
        dp_size = mesh_shape.get("dp", 1) if hasattr(mesh_shape, "get") else 1
        total_dp = dp_size
        if getattr(self.arguments, 'rollouts_per_step', None) is None:
            derived_rps = int(total_dp) * int(self.arguments.total_batch_size) * int(self.arguments.num_return_sequences)
            self.arguments.rollouts_per_step = int(derived_rps)
            if self.arguments.is_process_zero:
                per_process = int(self.arguments.total_batch_size) * int(self.arguments.num_return_sequences)
                global_total = int(total_dp) * per_process
                logger.info(
                    f"GSPO Rollout configuration: num_return_sequences={self.arguments.num_return_sequences}, "
                    f"derived rollouts_per_step={self.arguments.rollouts_per_step}, "
                    f"DP={total_dp}, batch_size={self.arguments.total_batch_size}, "
                    f"per_process_rollouts={per_process}, global_rollouts={global_total}"
                )
        
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