# grpo_trainer.py

from __future__ import annotations

import typing as tp
from functools import cached_property, partial
import inspect

import flax
import flax.nnx
import jax
import numpy as np
from eformer import common_types
from eformer.escale import with_sharding_constraint
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer, GenerationConfig, ProcessorMixin

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
from ._fn import get_per_token_logps, grpo_step
from .grpo_config import GRPOConfig

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
logger = get_logger(__name__)
RewardFunc = tp.Union[EasyDeLBaseModule, EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa


def _fileaf(x):
    return isinstance(x, jax.Array)


def delete_tree(pytree):
    return jax.tree_util.tree_map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None,
        pytree,
        is_leaf=_fileaf,
    )


class GRPOTrainer(Trainer):
    arguments: GRPOConfig  # type hinting

    def __init__(
        self,
        arguments: GRPOConfig,
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
        assert isinstance(arguments, GRPOConfig), f"arguments type must be `GRPOConfig` but got {type(arguments)}"
        assert processing_class is not None, "processing_class must be specified to tokenize a DPO dataset."

        # Configure adaptive mesh if parallelism is forced
        from .adaptive_mesh import configure_adaptive_mesh_inplace
        self._mesh_plan = configure_adaptive_mesh_inplace(arguments) if (
            arguments.force_tensor_parallel or arguments.force_data_parallel
        ) else None

        if self._mesh_plan:
            try:
                if jax.process_index() == 0:
                    print(f"DEBUG: Configured mesh: DP={self._mesh_plan.dp}, FSDP={self._mesh_plan.fsdp}, TP={self._mesh_plan.tp}")
                logger.info(
                    f"Configured mesh: DP={self._mesh_plan.dp}, FSDP={self._mesh_plan.fsdp}, TP={self._mesh_plan.tp}; "
                    f"dataset shards={getattr(arguments, 'grain_shard_count', None)} (index={getattr(arguments, 'grain_shard_index', None)})"
                )
            except Exception as e:
                print(f"DEBUG: Failed to log mesh configuration: {e}")
                logger.warning(f"Failed to log mesh configuration: {e}")
        
        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class

        if not isinstance(model, EasyDeLState):
            model = model.to_state()


        self.ref_state = deepcopy_model(model=model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.model.config._name_or_path,
                padding_side="left",
            )
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=model.model.mesh)

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=False)
        ):
            if isinstance(reward_func, EasyDeLBaseModule | EasyDeLState):
                if isinstance(reward_func, EasyDeLBaseModule):
                    reward_func = reward_func.to_state()
                    sharding = reward_func.shardings

                    @ejit(
                        static_argnums=(0,),
                        in_shardings=(
                            sharding.graphstate,
                            sharding.graphother,
                            empty_sharding,
                        ),
                        out_shardings=empty_sharding,
                    )
                    def apply_fn(gd, gs, gt, batch):
                        batch = with_sharding_constraint(
                            arr=batch,
                            sharding=self.arguments.step_partition_spec,
                        )
                        return nn.merge(gd, gs, gt)(**batch)

                    reward_func = reward_func.replace(apply_fn=apply_fn)

                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.model.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token

                reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
                if reward_processing_classes is not None:
                    reward_processing_classes[i] = reward_processing_class
                reward_funcs[i] = reward_func

        self.num_generations = arguments.num_return_sequences
        self.reward_processing_classes = reward_processing_classes
        self.reward_funcs = reward_funcs
        self.arguments = arguments
        self.processing_class = processing_class
        self.train_is_conversational = False
        self.eval_is_conversational = False
        self.data_tokenize_fn = data_tokenize_fn
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                dataset=train_dataset,
                processing_class=processing_class,
                arguments=arguments,
                dataset_name="train",
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(
                dataset=eval_dataset,
                processing_class=processing_class,
                arguments=arguments,
                dataset_name="eval",
            )
        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
        )
        
        # Validation not needed anymore, mesh config is handled in _get_or_create_mesh

        # Initialize WandB artifacts after the run is initialized by BaseTrainer
        # Guard table logging behind a dedicated flag to avoid memory overhead by default
        log_table = None
        if (
            getattr(self.arguments, "log_generations_table", False)
            and self.arguments.use_wandb
            and self.arguments.can_log_metrics
            and wandb is not None
        ):
            try:
                log_table = wandb.Table(columns=["generations", "took", "length", "step"])
            except Exception:
                log_table = None
        else:
            log_table = None
        self.log_table = log_table

    def _get_or_create_mesh(self):
        """Get mesh from arguments or create from adaptive config."""
        if hasattr(self.arguments, 'mesh_dims') and self.arguments.mesh_dims:
            from eformer.escale import create_mesh
            mesh = create_mesh(
                axis_dims=self.arguments.mesh_dims,
                axis_names=("dp", "fsdp", "ep", "tp", "sp"),
                process_is_granule=False,
                should_sort_granules_by_key=True,
                allow_split_physical_axes=True,
            )
            logger.info(f"Created mesh with adaptive dimensions: {self.arguments.mesh_dims}")
            return mesh
        return self.model.mesh

    def _update_model_mesh(self, mesh):
        """Update model and model_state to use new mesh."""
        # Try to update model config mesh if possible
        if hasattr(self.model, 'config'):
            try:
                self.model.config.mesh = mesh
            except (AttributeError, TypeError):
                logger.debug("Model config mesh is read-only, continuing with new mesh for training")

        # Update model_state with new mesh
        if hasattr(self.model_state, 'model') and hasattr(self.model_state.model, 'config'):
            try:
                self.model_state = self.model_state.replace(
                    model=self.model_state.model.replace(config=self.model_state.model.config.replace(mesh=mesh))
                )
            except (AttributeError, TypeError):
                logger.debug("Could not update model_state config mesh, using new mesh for training only")

    @cached_property
    def pad_token_id(self):
        if isinstance(self.processing_class, ProcessorMixin):
            pad_token_id = self.processing_class.tokenizer.pad_token_id
        else:
            pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is not None:
            return pad_token_id
        else:
            return self.eos_token_id[0]

    @cached_property
    def eos_token_id(self) -> list[int]:
        eos_ids = []
        # 1) Start with EOS from the primary processing class/tokenizer
        if isinstance(self.processing_class, ProcessorMixin):
            tokenizer = self.processing_class.tokenizer
            proc_eos_token_id = tokenizer.eos_token_id
        else:
            tokenizer = self.processing_class
            proc_eos_token_id = getattr(self.processing_class, "eos_token_id", None)

        if isinstance(proc_eos_token_id, int):
            proc_eos_token_id = [proc_eos_token_id]
        if isinstance(proc_eos_token_id, (list, tuple)):
            eos_ids.extend([t for t in proc_eos_token_id if t is not None])

        # 2) Include common Qwen end tokens if the tokenizer knows them
        special_tokens = [
            "<|im_end|>",
            "<|endoftext|>",
        ]
        convert_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if callable(convert_fn):
            for tok in special_tokens:
                try:
                    tid = convert_fn(tok)
                except Exception:
                    tid = None
                if tid is not None and (unk_id is None or tid != unk_id):
                    eos_ids.append(tid)

        # 3) Merge any EOS ids present in model.generation_config (if available)
        if hasattr(self.model, "generation_config"):
            conf_eos = self.model.generation_config.eos_token_id
            if isinstance(conf_eos, int):
                conf_eos = [conf_eos]
            if isinstance(conf_eos, (list, tuple)):
                eos_ids.extend([t for t in conf_eos if t is not None])

        # Return unique list with deterministic ordering
        return sorted(set(eos_ids))

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: ProcessingClassType,
        arguments: GRPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        map_kwargs = {"writer_batch_size": 10}
        from datasets import Dataset

        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = arguments.dataset_num_proc

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
        if dataset_name == "train":
            self.train_is_conversational = is_conversational(dataset[0])
        else:
            self.eval_is_conversational = is_conversational(dataset[0])

        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
        if not self.arguments.skip_apply_chat_template:
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": arguments.tools},
                **map_kwargs,
            )

        # Shard dataset per JAX process to balance difficulty and avoid excessive duplicates
        try:
            import jax as _jax  # local import to avoid top-level overhead
            if isinstance(dataset, Dataset):
                # Deterministic shuffle before sharding to interleave difficulty evenly
                try:
                    _seed = int(getattr(arguments, "seed", 17))
                except Exception:
                    _seed = 17
                dataset = dataset.shuffle(seed=_seed)
                shard_count = getattr(arguments, "grain_shard_count", None)
                shard_index = getattr(arguments, "grain_shard_index", None)
                if shard_count is None:
                    shard_count = max(1, int(_jax.process_count()))
                if shard_index is None:
                    shard_index = max(0, int(_jax.process_index()))
                # Only shard when more than one process
                if int(shard_count) > 1:
                    try:
                        dataset = dataset.shard(num_shards=int(shard_count), index=int(shard_index), contiguous=False)
                        if _jax.process_index() == 0:
                            print(f"DEBUG: Applied dataset sharding for {dataset_name}: index={int(shard_index)} num_shards={int(shard_count)}")
                            logger.info(
                                f"Applied dataset sharding for {dataset_name}: index={int(shard_index)} num_shards={int(shard_count)}"
                            )
                    except Exception as e:
                        print(f"DEBUG: Dataset sharding failed for {dataset_name}: {e}")
                        logger.warning(f"Dataset sharding failed for {dataset_name}: {e}")
        except Exception as _e:
            # Best-effort: continue without sharding if anything goes wrong
            if jax.process_index() == 0:
                logger.debug(f"Skipping dataset sharding for {dataset_name}: {_e}")

        def _tokenize(example):
            return processing_class(
                example["prompt"],
                return_tensors="np",
                padding="max_length",
                padding_side="left",
                max_length=arguments.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=True,
            )

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"tokenizing {dataset_name} dataset"
        if self.data_tokenize_fn is not None:
            dataset = dataset.map(
                self.data_tokenize_fn,
                batched=True,
                fn_kwargs={
                    "tokenizer": processing_class,
                    "tools": arguments.tools,
                },
                **map_kwargs,
            )
        else:
            dataset = dataset.map(
                _tokenize,
                batched=True,
                **map_kwargs,
            )
        return dataset

    @property
    def step_sharding(self):
        return NamedSharding(
            mesh=self.model.mesh,
            spec=self.arguments.step_partition_spec,
        )
    
    def get_batch_shardings(self):
        """
        Create proper sharding specifications for each tensor in the batch.
        Handles mixed-rank tensor collections properly.
        """
        mesh = self.model.mesh
        base_spec = self.arguments.step_partition_spec
        
        # If no sharding needed
        if base_spec is None or base_spec == PartitionSpec():
            empty_spec = PartitionSpec()
            return {
                "prompt_ids": NamedSharding(mesh=mesh, spec=empty_spec),
                "prompt_mask": NamedSharding(mesh=mesh, spec=empty_spec),
                "completion_ids": NamedSharding(mesh=mesh, spec=empty_spec),
                "completion_mask": NamedSharding(mesh=mesh, spec=empty_spec),
                "ref_per_token_logps": NamedSharding(mesh=mesh, spec=empty_spec),
                "advantages": NamedSharding(mesh=mesh, spec=empty_spec),
            }
        
        # Extract batch dimension from base spec
        batch_dim = base_spec[0] if len(base_spec) > 0 else None
        seq_dim = base_spec[1] if len(base_spec) > 1 else None
        
        # For 1D tensors (advantages), only use batch dimension
        # If batch_dim is a tuple like ('dp', 'fsdp'), use just the first element
        if isinstance(batch_dim, tuple) and len(batch_dim) > 0:
            batch_dim_1d = batch_dim[0]
        else:
            batch_dim_1d = batch_dim
        
        spec_1d = PartitionSpec(batch_dim_1d) if batch_dim_1d else PartitionSpec()
        spec_2d = base_spec
        
        return {
            "prompt_ids": NamedSharding(mesh=mesh, spec=spec_2d),
            "prompt_mask": NamedSharding(mesh=mesh, spec=spec_2d),
            "completion_ids": NamedSharding(mesh=mesh, spec=spec_2d),
            "completion_mask": NamedSharding(mesh=mesh, spec=spec_2d),
            "ref_per_token_logps": NamedSharding(mesh=mesh, spec=spec_2d),
            "advantages": NamedSharding(mesh=mesh, spec=spec_1d),
        }

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """
        # Get or create mesh with adaptive configuration
        mesh = self._get_or_create_mesh()
        self._update_model_mesh(mesh)

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        
        # Derive rollouts_per_step from num_return_sequences if not provided
        mesh_shape = getattr(mesh, "shape", {})
        dp_size = mesh_shape.get("dp", 1) if hasattr(mesh_shape, "get") else 1
        # Effective DP is bounded by the actual number of JAX processes participating
        try:
            proc_count = jax.process_count()
        except Exception:
            proc_count = 1
        effective_dp = max(1, min(int(dp_size), int(proc_count)))
        total_dp = dp_size
        if getattr(self.arguments, 'rollouts_per_step', None) is None:
            derived_rps = int(total_dp) * int(self.arguments.total_batch_size) * int(self.arguments.num_return_sequences)
            self.arguments.rollouts_per_step = int(derived_rps)
            try:
                if jax.process_index() == 0:
                    per_process = int(self.arguments.total_batch_size) * int(self.arguments.num_return_sequences)
                    # Use effective dp for expected global actually running this process configuration
                    global_total = int(effective_dp) * per_process
                    print(
                        f"DEBUG: Rollout config - num_return_sequences={self.arguments.num_return_sequences}, global_total={global_total}"
                    )
                    logger.info(
                        f"Rollout configuration: num_return_sequences={self.arguments.num_return_sequences}, "
                        f"expected_global_rollouts_per_step={global_total} (effective DP={effective_dp}, mesh DP={int(dp_size)}), "
                        f"per_process_rollouts={per_process}, "
                        f"batch_size={self.arguments.total_batch_size}"
                    )
            except Exception as e:
                print(f"DEBUG: Failed to log rollout configuration: {e}")
                logger.warning(f"Failed to log rollout configuration: {e}")

        # Use adaptive sharding based on batch size and tensor parallelism
        from .adaptive_mesh import get_adaptive_sharding_spec
        _shard_sig = inspect.signature(get_adaptive_sharding_spec)
        _shard_kwargs = dict(
            total_batch_size=self.arguments.total_batch_size,
            force_tensor_parallel=self.arguments.force_tensor_parallel,
            mini_batch_size=self.arguments.mini_batch_size,
        )
        if 'force_data_parallel' in _shard_sig.parameters:
            _shard_kwargs['force_data_parallel'] = self.arguments.force_data_parallel
        if 'rollouts_per_step' in _shard_sig.parameters and getattr(self.arguments, 'rollouts_per_step', None):
            _shard_kwargs['rollouts_per_step'] = self.arguments.rollouts_per_step
        adaptive_spec = get_adaptive_sharding_spec(**_shard_kwargs)
        input_sharding = NamedSharding(
            mesh=mesh,
            spec=adaptive_spec
        )
        # Store input spec for debugging/inspection
        self.input_partition_spec = adaptive_spec
        step_sharding = NamedSharding(
            mesh=mesh,
            spec=self.arguments.step_partition_spec,
        )
        
        @ejit(
            in_shardings=(self.state_shardings, input_sharding, input_sharding, empty_sharding),
            out_shardings=(empty_sharding, input_sharding, input_sharding),
            static_argnums=(3,),
        )
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
                # Proper generation config that relies on natural EOS stopping
                generation_config = GenerationConfig(
                    top_p=self.arguments.top_p,
                    top_k=self.arguments.top_k,
                    temperature=self.arguments.temperature,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,  # EasyDeL will stop naturally when these tokens are generated
                    max_new_tokens=self.arguments.max_completion_length,
                    max_length=self.arguments.max_completion_length + self.arguments.max_prompt_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    use_cache=False,
                )
                
                # Build PRNG key with per-batch folding to decorrelate identical prompts across TP/DP
                def _hash_u32(ids: jnp.ndarray) -> jnp.uint32:
                    h = jnp.uint32(2166136261)
                    def body(hh, x):
                        hh = jnp.uint32((hh ^ jnp.uint32(x)) * jnp.uint32(16777619))
                        return hh, None
                    hh, _ = jax.lax.scan(body, h, ids.astype(jnp.uint32))
                    return hh

                base_key = jax.random.PRNGKey(prng_seed)
                try:
                    prompt_slice = input_ids[:, : self.arguments.max_prompt_length]
                    ph = jax.vmap(_hash_u32)(prompt_slice)
                    prng_key = jax.random.fold_in(base_key, int(jnp.bitwise_xor.reduce(ph.astype(jnp.uint32))))
                except Exception:
                    prng_key = base_key

                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    prng_key=prng_key,
                ).sequences
                # Return inputs re-constrained to the input sharding spec to allow repeated calls
                input_ids = with_sharding_constraint(input_ids, adaptive_spec)
                attention_mask = with_sharding_constraint(attention_mask, adaptive_spec)
                return sequences, input_ids, attention_mask

        self.generate_function = generate

        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
        )

        # Derive static arg indices robustly based on grpo_step signature
        import inspect as _inspect
        _sig = _inspect.signature(grpo_step)
        _max_pos_index = len(_sig.parameters) - 1  # zero-based last positional index
        _end = min(2 + len(self._train_shared_fn_static_args), _max_pos_index + 1)
        static_argnames = tuple(range(2, _end))
        
        sharded_training_step_function = ejit(
            grpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
        )

        sharded_evaluation_step_function = ejit(
            grpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                # Ensure token arrays conform to the step partitioning spec before compute
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

        # Allow input sharding of token ids and masks to pass through (we re-constrain inside the fn)
        # This avoids mismatches like: pjit expects replicated but arg is sharded as ('dp','tp')

        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                None,
                None,
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

    def _make_attn_mask(self, arr):
        is_eos = jnp.isin(arr, jnp.asarray(self.eos_token_id).reshape(-1))
        return (
            (jnp.arange(is_eos.shape[1])[None, :].repeat(is_eos.shape[0], axis=0))
            <= jnp.where(
                is_eos.any(axis=1),
                jnp.argmax(is_eos.astype(jnp.int32), axis=1),
                jnp.full((is_eos.shape[0],), is_eos.shape[1] - 1),  # Fix off-by-one
            )[:, None]
        ).astype(jnp.int32)


    # _gather_unique_rows method removed to avoid TPU collective issues

    def _ensure_unique_prompts(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Ensure batch has unique prompts without changing the batch size.

        Duplicated prompts are replaced in-place by cycling over the first
        occurrences so that the leading dimension remains unchanged. This
        preserves divisibility requirements for DP sharding.
        """
        # Decode prompts for uniqueness check
        prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
        
        original_size = len(prompts)
        seen_prompts: set[str] = set()
        first_occurrence_indices: list[int] = []

        # Track unique first occurrences
        for i, prompt in enumerate(prompts):
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                first_occurrence_indices.append(i)
        
        # Fast path: nothing to change
        if len(first_occurrence_indices) == original_size:
            return batch
        
        # Build a replacement index list that keeps shape == original_size
        # Keep first occurrences, replace duplicates by cycling over unique ones
        if not first_occurrence_indices:
            # Extremely unlikely, but guard: keep batch unchanged
            return batch

        # Cycle helper
        def _cycled_indices(count: int) -> list[int]:
            base = first_occurrence_indices
            times = (count + len(base) - 1) // len(base)
            return (base * times)[:count]

        replacement_indices: list[int] = []
        used = set()
        for i, prompt in enumerate(prompts):
            if prompt in used:
                replacement_indices.append(i)  # placeholder, will be replaced
            else:
                used.add(prompt)
                replacement_indices.append(i)

        # Identify duplicate positions (beyond first occurrences)
        duplicate_positions = []
        seen_prompts.clear()
        for i, prompt in enumerate(prompts):
            if prompt in seen_prompts:
                duplicate_positions.append(i)
            else:
                seen_prompts.add(prompt)

        fills = _cycled_indices(len(duplicate_positions))

        # Final per-position indices to take from the original batch
        final_indices = list(range(original_size))
        for pos, fill_idx in zip(duplicate_positions, fills, strict=False):
            final_indices[pos] = fill_idx

        if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
            try:
                logger.debug(
                    f"dedup: original={original_size} unique={len(first_occurrence_indices)} "
                    f"replaced={len(duplicate_positions)} head_replacements={final_indices[:8]}"
                )
            except Exception:
                pass
        
        # Rebuild batch in-place using final_indices, preserving leading dimension
        rebuilt_batch: dict[str, tp.Any] = {}
        for key, values in batch.items():
            try:
                vlen = len(values)
            except Exception:
                vlen = None

            if vlen == original_size:
                if isinstance(values, jax.Array):
                    rebuilt_batch[key] = values[final_indices]
                else:
                    try:
                        import numpy as _np  # local import safe
                        if isinstance(values, _np.ndarray):
                            rebuilt_batch[key] = values[final_indices]
                        elif isinstance(values, (list, tuple)):
                            rebuilt_batch[key] = [values[i] for i in final_indices]
                        else:
                            rebuilt_batch[key] = values
                    except Exception:
                        rebuilt_batch[key] = values
            else:
                rebuilt_batch[key] = values
        
        if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
            try:
                pid = rebuilt_batch.get("input_ids", None)
                ans = rebuilt_batch.get("answer", None)
                pid_len = int(pid.shape[0]) if isinstance(pid, jax.Array) else (len(pid) if pid is not None else -1)
                ans_len = (len(ans) if hasattr(ans, "__len__") else -1)
                ans_head = None
                try:
                    if hasattr(ans, "__getitem__"):
                        ans_head = ans[:2]
                except Exception:
                    ans_head = None
                logger.debug(f"dedup: preserved_len={pid_len} answer_len={ans_len} answer_head={ans_head}")
            except Exception:
                pass
        
        return rebuilt_batch

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            # Convert numpy arrays to JAX arrays using the working solution
            try:
                prompt_ids = jnp.asarray(prompt_ids)
                prompt_mask = jnp.asarray(prompt_mask)
            except Exception as e:
                logger.error(f"Failed to convert arrays to JAX on worker {jax.process_index()}: {e}")
                raise RuntimeError(f"Failed to convert numpy arrays to JAX arrays: {e}")


            # Ensure unique prompts if enabled
            if getattr(self.arguments, "ensure_unique_prompts", True):
                batch = self._ensure_unique_prompts(batch)
                # IMPORTANT: re-bind prompt tensors after filtering to keep alignment with batch
                try:
                    prompt_ids = jnp.asarray(batch["input_ids"])  # rebind after dedup
                    prompt_mask = jnp.asarray(batch["attention_mask"])  # rebind after dedup
                except Exception as _e:
                    # If conversion fails, keep previous tensors (better than crashing)
                    pass
                if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                    try:
                        _pid_len = int(prompt_ids.shape[0])
                    except Exception:
                        _pid_len = -1
                    try:
                        _ans_len = len(batch.get("answer", [])) if batch.get("answer", None) is not None else -1
                    except Exception:
                        _ans_len = -1
                    logger.debug(f"preprocess: after-dedup prompts={_pid_len} answers_len={_ans_len}")

            # Chunked generation and reference log-prob computation to reduce peak memory
            rollout_chunk_size = getattr(self.arguments, "rollout_chunk_size", None)
            # Default to generating all num_return_sequences at once when not set
            if rollout_chunk_size is None or rollout_chunk_size <= 0:
                rollout_chunk_size = int(self.num_generations)
            # Clamp lower bound only; allow > num_return_sequences (loop uses min() with remaining)
            rollout_chunk_size = int(max(1, int(rollout_chunk_size)))
            # No TP-based capping; PagedAttention KV caching supports multiple prompts regardless of TP

            sequences_chunks = []
            completion_ids_chunks = []
            completion_mask_chunks = []
            ref_logps_chunks = []
            comp_len_chunks = []
            cur_nrs_chunks = []  # Track chunk sizes to enable correct reordering later
            generation_time = 0.0
            token_logps_time = 0.0

            base_prompt_len = prompt_ids.shape[-1]
            nrs_remaining = int(self.num_generations)
            # Create a simple per-step base seed; fold in process index via the seed formula
            try:
                cur_step_int = int(jax.device_get(state.step))
            except Exception:
                cur_step_int = 0
            chunk_idx = 0
            while nrs_remaining > 0:
                cur_nrs = int(min(rollout_chunk_size, nrs_remaining))
                with capture_time() as generation_time_fn:
                    # Use a simple per-chunk seed; avoids JIT recompiles and ensures diversity across chunks
                    # Ensure seed is always positive and within 32-bit range
                    per_chunk_seed = int((cur_step_int * 131071 + 4099 * abs(jax.process_index()) + chunk_idx) % (2**31 - 1))
                    per_chunk_seed = max(1, per_chunk_seed)
                    seq_chunk, prompt_ids, prompt_mask = jax.block_until_ready(
                        self.generate_function(state, prompt_ids, prompt_mask, cur_nrs, per_chunk_seed)
                    )
                # Debug output removed to prevent host divergence in compiled code
                generation_time += float(generation_time_fn())

                # Avoid explicit cross-host barriers here; rely on pjit collectives only

                # Extract completions for this chunk and build masks
                prompt_completion_ids_chunk = seq_chunk
                completion_ids_chunk = prompt_completion_ids_chunk[..., base_prompt_len:]
                completion_mask_chunk = self._make_attn_mask(completion_ids_chunk)
                ridmask_chunk = prompt_mask.repeat(cur_nrs, 0)

                with capture_time() as token_logps_time_fn:
                    full_mask_chunk = jnp.concatenate([ridmask_chunk, completion_mask_chunk], -1)
                    ref_logps_chunk = self.compute_refmodel_logps(
                        self.ref_state.graphstate,
                        self.ref_state.graphother,
                        prompt_completion_ids_chunk,
                        full_mask_chunk,
                    )
                token_logps_time += float(token_logps_time_fn())

                # Avoid explicit cross-host barriers here; rely on pjit collectives only

                # Accumulate
                sequences_chunks.append(prompt_completion_ids_chunk)
                completion_ids_chunks.append(completion_ids_chunk)
                completion_mask_chunks.append(completion_mask_chunk)
                ref_logps_chunks.append(ref_logps_chunk)
                comp_len_chunks.append(completion_mask_chunk.sum(-1))
                cur_nrs_chunks.append(cur_nrs)

                nrs_remaining -= cur_nrs
                chunk_idx += 1

            # Concatenate accumulated chunks — for memory-opt mode keep concat minimal if only one chunk
            if len(sequences_chunks) == 1:
                prompt_completion_ids = sequences_chunks[0]
                completion_ids = completion_ids_chunks[0]
                completion_mask = completion_mask_chunks[0]
                ref_per_token_logps = ref_logps_chunks[0]
                completion_lengths_per_seq = comp_len_chunks[0]
            else:
                # When generation is chunked across num_return_sequences, the natural concatenation
                # order is [for each chunk c: for each prompt b: for each gen in chunk] along axis 0.
                # Advantage computation expects [for each prompt b: all gens across chunks] ordering.
                # We therefore reorder to (prompt-major, then generation) before downstream use.
                num_prompts_local = prompt_ids.shape[0]

                def _reorder_from_chunks(chunks: list[jax.Array], is_scalar: bool = False):
                    # chunks: list of arrays with shape (cur_nrs_i * B, ...)
                    # Return: array with shape (B * sum(cur_nrs_i), ...), ordered by prompt-major
                    reshaped = []
                    for arr, k in zip(chunks, cur_nrs_chunks, strict=False):
                        # Reshape (k * B, ...) -> (B, k, ...)
                        new_shape = (num_prompts_local, k) + tuple(arr.shape[1:])
                        reshaped.append(jnp.reshape(arr, new_shape))
                    # Concat across generation axis -> (B, total_k, ...)
                    by_prompt = jnp.concatenate(reshaped, axis=1)
                    # Flatten back to (B * total_k, ...)
                    flat_shape = (by_prompt.shape[0] * by_prompt.shape[1],) + tuple(by_prompt.shape[2:])
                    return jnp.reshape(by_prompt, flat_shape)

                prompt_completion_ids = _reorder_from_chunks(sequences_chunks)
                completion_ids = _reorder_from_chunks(completion_ids_chunks)
                completion_mask = _reorder_from_chunks(completion_mask_chunks)
                ref_per_token_logps = _reorder_from_chunks(ref_logps_chunks)
                completion_lengths_per_seq = _reorder_from_chunks(comp_len_chunks)
            # Always initialize prompts to safe placeholders to avoid UnboundLocalError
            try:
                _local_prompt_count = int(batch["input_ids"].shape[0])
            except Exception:
                _local_prompt_count = 0
            prompts = [""] * _local_prompt_count

            if not (getattr(self.arguments, "verify_dataset_sharding", False) and int(jax.device_get(state.step)) == 0):
                prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                try:
                    logger.debug(
                        f"preprocess: prompts_len={len(prompts)} completions_len={len(completions_text)} "
                        f"expected_completions={int(prompt_ids.shape[0]*self.num_generations)}"
                    )
                    # Show small heads for inspection
                    p_head = [p[:64].replace("\n", " ") for p in prompts[:2]] if isinstance(prompts, list) else []
                    a_obj = batch.get("answer", None)
                    if a_obj is not None:
                        try:
                            a_list = a_obj.tolist() if hasattr(a_obj, "tolist") else list(a_obj)
                        except Exception:
                            a_list = []
                    else:
                        a_list = []
                    logger.debug(f"preprocess: prompt_head={p_head} answer_head={a_list[:2]}")
                    # First prompt's first few completions lengths
                    first_r = min(self.num_generations, len(completions_text))
                    comp_lens = [len(completions_text[i]) for i in range(first_r)]
                    logger.debug(f"preprocess: first_prompt_first_{first_r}_completion_lengths={comp_lens}")
                except Exception:
                    pass

            # Print one local example per process each step: prompt, ground truth, extracted prediction
            if getattr(self.arguments, "verbose", True):
                try:
                    example_idx = 0
                    # Extract prompt string
                    example_prompt = ""
                    try:
                        if isinstance(prompts, list) and len(prompts) > 0:
                            example_prompt = prompts[example_idx]
                        else:
                            example_prompt = str(prompts)
                    except Exception:
                        example_prompt = ""

                    # Extract raw completion text for first generation of first prompt
                    example_pred_text = ""
                    try:
                        if isinstance(completions_text, list) and len(completions_text) > example_idx:
                            example_pred_text = completions_text[example_idx]
                        else:
                            example_pred_text = str(completions_text)
                    except Exception:
                        example_pred_text = ""

                    # Determine ground truth value from batch (dataset-dependent)
                    def _get_gt(_batch, idx: int):
                        try:
                            if "solution_normalized" in _batch and _batch["solution_normalized"] is not None:
                                print("DEBUG: Using 'solution_normalized' from batch")
                                v = _batch["solution_normalized"]
                            elif "solution" in _batch and _batch["solution"] is not None:
                                print("DEBUG: Using 'solution' from batch")
                                v = _batch["solution"]
                            elif "answer" in _batch and _batch["answer"] is not None:
                                print("DEBUG: Using 'answer' from batch")
                                v = _batch["answer"]
                            else:
                                print("DEBUG: No ground truth key found in batch")
                                return None
                            if hasattr(v, "__getitem__"):
                                try:
                                    print(f"DEBUG: Attempting to index ground truth with idx={idx}")
                                    return v[idx]
                                except Exception as e:
                                    print(f"DEBUG: Exception indexing ground truth with idx={idx}: {e}, falling back to v[0]")
                                    return v[0]
                            return v
                        except Exception as e:
                            print(f"DEBUG: Exception in _get_gt: {e}")
                            return None

                    example_gt = _get_gt(batch, example_idx)

                    # Extract final value from completion using reward-specific logic (reuse reward modules)
                    example_pred_value = example_pred_text
                    try:
                        # Detect dataset type based on configured reward functions
                        rf_names = [getattr(rf, "__name__", "") for rf in self.reward_funcs]
                        rf_mods = [getattr(rf, "__module__", "") for rf in self.reward_funcs]
                        print(f"DEBUG: Reward function names: {rf_names}")
                        print(f"DEBUG: Reward function modules: {rf_mods}")
                        is_math = any((name.startswith("math/") or mod.endswith("math_reward")) for name, mod in zip(rf_names, rf_mods, strict=False))
                        is_gsm8k = any((name.startswith("gsm8k/") or mod.endswith("gsm8k_reward")) for name, mod in zip(rf_names, rf_mods, strict=False))
                        print(f"DEBUG: Detected dataset type - is_math={is_math}, is_gsm8k={is_gsm8k}")

                        if is_math:
                            try:
                                print("DEBUG: Using math reward extraction")
                                # Just show a preview - Math-Verify handles actual extraction
                                preview = example_pred_text[-200:] if len(example_pred_text) > 200 else example_pred_text
                                preview = preview[:180] + "…" if len(preview) > 180 else preview
                                print(f"DEBUG: Math extraction preview: '{preview}'")
                                example_pred_value = preview
                            except Exception as e:
                                print(f"DEBUG: Math extraction failed: {e}")
                                example_pred_value = example_pred_text
                        elif is_gsm8k:
                            try:
                                print("DEBUG: Using GSM8K reward extraction")
                                import re as _re
                                from easydel.verification.gsm8k_reward import _extract_answer_from_xml as _gx_extract_xml, _normalize_number_text as _gx_norm  # type: ignore
                                _ans = _gx_extract_xml(example_pred_text) or example_pred_text
                                _norm = _gx_norm(_ans)
                                _nums = _re.findall(r"-?\d+\.?\d*", _norm)
                                example_pred_value = _nums[-1] if _nums else _ans
                                print(f"DEBUG: GSM8K extraction result: '{example_pred_value}'")
                            except Exception as e:
                                print(f"DEBUG: GSM8K extraction failed: {e}")
                                pass
                        # else: keep example_pred_value as raw text
                    except Exception as e:
                        print(f"DEBUG: Exception in reward-specific extraction: {e}")
                        pass

                    # Clip long prompt/output for readability
                    def _clip(s: str, n: int = 180) -> str:
                        try:
                            ss = s.replace("\n", " ")
                            return ss if len(ss) <= n else (ss[:n] + "…")
                        except Exception:
                            return str(s)

                    try:
                        print(f"DEBUG: About to log example - prompt_len={len(str(example_prompt))}, gt='{example_gt}', pred_len={len(str(example_pred_value))}")
                        logger.info(
                            f"example/local | prompt={_clip(example_prompt)} | gt={example_gt} | pred={_clip(str(example_pred_value))}"
                        )
                    except Exception as e:
                        print(f"DEBUG: Failed to log example: {e}")
                except Exception as _e:
                    try:
                        logger.debug(f"example/local logging failed: {_e}")
                    except Exception:
                        pass

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            # Calculate completion lengths before rewards (already computed per chunk; keep single computation)
            completion_lengths_per_seq = completion_mask.sum(-1)
            # Global gathering removed to reduce collective overhead

            # Host-side concise summary for completion token lengths
            if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                try:
                    lengths = jax.device_get(completion_lengths_per_seq)
                    mean_v = float(jnp.mean(lengths))
                    std_v = float(jnp.std(lengths))
                    min_v = int(jnp.min(lengths))
                    max_v = int(jnp.max(lengths))
                    logger.info(
                        f"completion_lengths: mean={mean_v:.2f}, std={std_v:.2f}, min={min_v}, max={max_v}"
                    )
                    # Also print a small head of the per-completion token lengths
                    try:
                        head_n = int(min(16, lengths.shape[0]))
                        logger.info(f"completion_lengths_head={lengths[:head_n].tolist()} (n={int(lengths.shape[0])})")
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Could not compute completion length summary: {e}")
                # Brief prompt count
                try:
                    logger.info(f"prompts: count={int(batch['input_ids'].shape[0])}")
                except Exception:
                    pass
            
            # Pre-allocate rewards; when chunk_size==1, we still need the full matrix for grouping
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
                                    # Debug output removed to prevent host divergence
                        output_reward_func = reward_func(
                            prompts=in_prompts,
                            completions=completions,
                            max_length=self.arguments.max_sequence_length,
                            batch=batch,
                            completion_lengths=jax.device_get(completion_lengths_per_seq),
                        )
                        rew = jnp.array(output_reward_func, dtype="f4")
                                    # Debug output removed to prevent host divergence
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()
            
            with capture_time() as grouped_comp_time_fn:
                rewards = rewards_per_func.sum(axis=1)
                # Robust, config-driven advantage normalization per prompt group
                grouped_rewards = rewards.reshape(-1, self.num_generations)
                group_means = jnp.mean(grouped_rewards, axis=-1, keepdims=True)
                group_stds = jnp.std(grouped_rewards, axis=-1, keepdims=True)
                eps = jnp.float32(getattr(self.arguments, "advantage_epsilon", 1e-6))
                # Zero-out groups with very low variance to avoid spurious gradients
                safe_stds = jnp.maximum(group_stds, eps)
                normalized = (grouped_rewards - group_means) / safe_stds
                zero_mask = (group_stds < eps).astype(normalized.dtype)
                normalized = jnp.where(zero_mask > 0, jnp.zeros_like(normalized), normalized)
                advantages = normalized.reshape(-1)
            grouped_comp_time = grouped_comp_time_fn()
            # Compute mean reward per completion locally (no cross-host ops)
            # Optionally compute safe global scalars via allgather of scalars only
            # Compute success metrics (reward > 0) locally and globally
            try:
                successes_local = (rewards > 0).astype(jnp.int32)
                success_count_comp_local = jnp.sum(successes_local)
                total_comp_local = successes_local.shape[0]
                success_rate_comp_local = success_count_comp_local / jnp.maximum(1, total_comp_local)

                # Per-prompt pass@k (at least one success among num_return_sequences)
                per_prompt_success = jnp.max(successes_local.reshape(-1, self.num_generations), axis=1)
                pass_prompt_count_local = jnp.sum(per_prompt_success)
                num_prompts_local = int(prompt_ids.shape[0])
                pass_at_k_local = pass_prompt_count_local / jnp.maximum(1, num_prompts_local)

                # Safe global scalar aggregation (proc-local fallbacks on failure)
                if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                    logger.debug("global aggregation: start")
                try:
                    print(f"DEBUG: Starting global aggregation - process_count={jax.process_count()}")
                    if jax.process_count() > 1:
                        print("DEBUG: Multi-process global aggregation")
                        _sc = jax.experimental.multihost_utils.process_allgather(jnp.array(success_count_comp_local, dtype=jnp.int32))
                        _tc = jax.experimental.multihost_utils.process_allgather(jnp.array(total_comp_local, dtype=jnp.int32))
                        _pp = jax.experimental.multihost_utils.process_allgather(jnp.array(pass_prompt_count_local, dtype=jnp.int32))
                        _np = jax.experimental.multihost_utils.process_allgather(jnp.array(num_prompts_local, dtype=jnp.int32))
                        success_count_comp_global = jnp.sum(_sc)
                        total_comp_global = jnp.sum(_tc)
                        pass_prompt_count_global = jnp.sum(_pp)
                        num_prompts_global = jnp.sum(_np)
                        print(f"DEBUG: Global aggregation successful - total_comp_global={total_comp_global}")
                    else:
                        print("DEBUG: Single-process fallback for global metrics")
                        success_count_comp_global = success_count_comp_local
                        total_comp_global = total_comp_local
                        pass_prompt_count_global = pass_prompt_count_local
                        num_prompts_global = jnp.array(float(num_prompts_local))
                    success_rate_comp_global = success_count_comp_global / jnp.maximum(1, total_comp_global)
                    pass_at_k_global = pass_prompt_count_global / jnp.maximum(1.0, num_prompts_global)
                except Exception as e:
                    print(f"DEBUG: Global aggregation failed: {e}")
                    if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                        logger.debug(f"global aggregation: failed {e}")
                    success_count_comp_global = success_count_comp_local
                    total_comp_global = total_comp_local
                    success_rate_comp_global = success_rate_comp_local
                    pass_prompt_count_global = pass_prompt_count_local
                    num_prompts_global = jnp.array(float(num_prompts_local))
                    pass_at_k_global = pass_at_k_local
                if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                    logger.debug("global aggregation: end")
            except Exception as e:
                if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                    logger.debug(f"metrics aggregation error: {e}")
                # Safe fallbacks
                success_count_comp_local = jnp.array(0)
                total_comp_local = jnp.array(1)
                success_rate_comp_local = jnp.array(0.0)
                pass_prompt_count_local = jnp.array(0)
                pass_at_k_local = jnp.array(0.0)
        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.mean(completion_lengths_per_seq)  # Average for metrics
        # Compute local and global completion length stats for logging/metrics
        try:
            lengths_local_arr = jnp.asarray(completion_lengths_per_seq)
            lengths_local_min = jnp.min(lengths_local_arr)
            lengths_local_max = jnp.max(lengths_local_arr)
            lengths_local_mean = jnp.mean(lengths_local_arr)
        except Exception:
            lengths_local_min = jnp.array(0.0)
            lengths_local_max = jnp.array(0.0)
            lengths_local_mean = jnp.array(0.0)
        
        # Global length aggregation removed; rely on local stats only
        try:
            mesh_shape = getattr(self.model.mesh, "shape", {})
            mesh_dp = mesh_shape.get("dp", 1) if hasattr(mesh_shape, "get") else 1
            tp_size = mesh_shape.get("tp", 1) if hasattr(mesh_shape, "get") else 1
        except Exception:
            mesh_dp = 1
            tp_size = 1
        try:
            proc_count = jax.process_count()
        except Exception:
            proc_count = 1
        dp_size = max(1, min(int(mesh_dp), int(proc_count)))
        # Robust termination ratios: detect EOS presence directly in completions.
        # Works even when pad_token_id == eos_token_id because sequences that ended early are padded with EOS.
        eos_found = jnp.isin(
            completion_ids, jnp.asarray(self.eos_token_id).reshape(-1)
        ).any(axis=1)
        eos_stop_rate = jnp.mean(eos_found.astype(jnp.float32))
        no_eos_maxlen_rate = jnp.float32(1.0) - eos_stop_rate
        
        # Debug output removed to prevent TPU coordination issues
            
        # Ensure all batch tensors share the same leading dimension for downstream minibatching
        # Repeat prompts to match completions if needed
        if prompt_ids.shape[0] * self.num_generations == completion_ids.shape[0]:
            prompt_ids_rep = prompt_ids.repeat(self.num_generations, 0)
            prompt_mask_rep = prompt_mask.repeat(self.num_generations, 0)
        elif prompt_ids.shape[0] == completion_ids.shape[0]:
            prompt_ids_rep = prompt_ids
            prompt_mask_rep = prompt_mask
        else:
            # Fallback: try to infer repeat factor
            repeat_factor = max(1, completion_ids.shape[0] // max(1, prompt_ids.shape[0]))
            prompt_ids_rep = prompt_ids.repeat(repeat_factor, 0)
            prompt_mask_rep = prompt_mask.repeat(repeat_factor, 0)

        # Reward diagnostics with explicit denominators to clarify granularity
        per_completion_mean_reward = jnp.mean(rewards)
        num_prompts_local = int(prompt_ids.shape[0])
        num_completions_local = int(num_prompts_local * self.num_generations)
        # Use effective DP (min(mesh dp, process_count)) for global denominators
        try:
            mesh_shape = getattr(self.model.mesh, "shape", {})
            mesh_dp = mesh_shape.get("dp", 1) if hasattr(mesh_shape, "get") else 1
            tp_size = mesh_shape.get("tp", 1) if hasattr(mesh_shape, "get") else 1
        except Exception:
            mesh_dp = 1
            tp_size = 1
        try:
            proc_count = jax.process_count()
        except Exception:
            proc_count = 1
        dp_size = max(1, min(int(mesh_dp), int(proc_count)))
        # Local metrics plus safe global scalars (if available)
        metrics_dict = {
            "reward/mean_per_completion": per_completion_mean_reward,
            "reward/success_rate_completions": float(success_rate_comp_local),
            "reward/pass_at_k": float(pass_at_k_local),
            # Safe global scalars for display only
            "reward/success_rate_completions_global": float(success_rate_comp_global),
            "reward/pass_at_k_global": float(pass_at_k_global),
            "rollouts/total_global": float(total_comp_global),
            "rollouts/queries_global": float(num_prompts_global),
            "completion_length": completion_length,
            # Completion length stats (local only)
            "rollouts/lengths_min": lengths_local_min,
            "rollouts/lengths_max": lengths_local_max,
            "rollouts/lengths_mean": lengths_local_mean,
            # Rollout accounting (local only)
            "rollouts/completions_per_prompt": float(self.num_generations),
            "rollouts/total_per_process": float(num_completions_local),
            "rollouts/queries_per_process": float(num_prompts_local),
            "rollouts/chunk_size": float(rollout_chunk_size),
            "rollouts/tp_size": float(tp_size or 1),
            "rollouts/dp_size": float(dp_size),
            # Explicit termination diagnostics each step
            "termination/eos_stop_rate": eos_stop_rate,
            "termination/no_eos_max_length_rate": no_eos_maxlen_rate,
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
        }
        # Convert metrics to plain floats early so we can safely log to WandB below
        processed_metrics_dict = {}
        for key, value in metrics_dict.items():
            if hasattr(value, 'item'):
                try:
                    processed_metrics_dict[key] = float(value.item())
                except Exception as e:
                    logger.warning(f"Failed to convert metric '{key}' to float for logging: {e}")
                    processed_metrics_dict[key] = 0.0
            elif isinstance(value, (int, float)):
                processed_metrics_dict[key] = float(value)
            else:
                processed_metrics_dict[key] = value
        def _record_per_reward_metrics(_rewards_per_func):
        for i, reward_func in enumerate(self.reward_funcs):
            _name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            try:
                try:
                    if jax.process_count() > 1 and 'global_rewards_per_func' in locals():
                        global_vals = global_rewards_per_func[:, i]
                        global_mean = jnp.mean(global_vals)
                        global_std = jnp.std(global_vals)
                    else:
                            global_mean = jnp.mean(_rewards_per_func[:, i])
                            global_std = jnp.std(_rewards_per_func[:, i])
                except Exception:
                        global_mean = jnp.mean(_rewards_per_func[:, i])
                        global_std = jnp.std(_rewards_per_func[:, i])
                
                metrics_dict[f"rewards/{_name}/mean"] = global_mean
                metrics_dict[f"rewards/{_name}/std"] = global_std
                metrics_dict[_name] = global_mean
                
                    local_mean = jnp.mean(_rewards_per_func[:, i])
                per_prompt_means = jnp.mean(
                        _rewards_per_func[:, i].reshape(-1, self.num_generations), axis=1
                )
                local_mean_of_prompt_means = jnp.mean(per_prompt_means)
                metrics_dict[f"rewards/{_name}/mean_per_completion_local"] = local_mean
                metrics_dict[f"rewards/{_name}/mean_per_prompt_local"] = local_mean_of_prompt_means
                metrics_dict[f"rewards/{_name}/mean_per_completion_global"] = global_mean
            except Exception:
                    metrics_dict[_name] = jnp.mean(_rewards_per_func[:, i])

        _record_per_reward_metrics(rewards_per_func)
        if self.log_table is not None and jax.process_index() == 0:
            try:
                cur_step = int(jax.device_get(state.step))
            except Exception:
                cur_step = 0

            def _to_local_host(x, name="array"):
                """Safely extract local host data with multiple fallback strategies."""
                if not isinstance(x, jax.Array):
                    return x

                # Try 1: Fully addressable (replicated/single-device)
                try:
                    if hasattr(x, "is_fully_addressable") and x.is_fully_addressable:
                        return jax.device_get(x)
                except Exception:
                    pass

                # Try 2: Get first local shard (partial data is better than crash)
                try:
                    shards = x.addressable_shards
                    if shards and len(shards) > 0:
                        return jax.device_get(shards[0].data)
                except Exception:
                    pass

                # Avoid cross-host collectives here since only process 0 runs this block
                # Last resort: Return empty array (do not allgather)
                return jnp.array([])

            # Extract with fallbacks
            local_completion_ids = _to_local_host(completion_ids, "completion_ids")
            local_comp_lens = _to_local_host(completion_lengths_per_seq, "lengths")

            # Decode with error handling
            try:
                if len(local_completion_ids) > 0:
                    decoded_text = self.processing_class.batch_decode(local_completion_ids)
                else:
                    decoded_text = []
            except Exception:
                decoded_text = []

                            # Handle lengths
            try:
                if isinstance(local_comp_lens, jnp.ndarray) and local_comp_lens.size > 0:
                    individual_lengths = local_comp_lens
                else:
                    individual_lengths = [0] * len(decoded_text)
            except Exception:
                individual_lengths = []

            try:
                n_items = min(len(decoded_text), len(individual_lengths))
                for i in range(n_items):
                    try:
                        length_val = float(individual_lengths[i]) if i < len(individual_lengths) else 0.0
                        self.log_table.add_data(decoded_text[i], generation_time, length_val, cur_step)
                    except Exception:
                        # Add placeholder
                        try:
                            self.log_table.add_data("[decode failed]", generation_time, 0.0, cur_step)
                        except Exception:
                            pass
            except Exception:
                pass

            # Log to WandB
            if self.log_table is not None:
                try:
                    if hasattr(self.log_table, 'data') and len(self.log_table.data) > 0:
                        wandb.log({"generations": self.log_table}, step=cur_step)
                except Exception:
                    # Don't crash training over logging
                    pass

        # Verification report generation removed to reduce overhead

        # Immediately log a small set of lightweight generation/reward metrics every step (process 0 only)
        try:
            if (
                jax.process_index() == 0
                and self.arguments.use_wandb
                and self.arguments.can_log_metrics
                and wandb is not None
                and getattr(self, "wandb_runtime", None) is not None
            ):
                try:
                    cur_step = int(jax.device_get(state.step))
                except Exception:
                    cur_step = 0
                # Log a concise set of metrics immediately every step (local + safe global scalars)
                immediate_keys = [
                    "reward/success_rate_completions",
                    "reward/pass_at_k",
                    "reward/mean_per_completion",
                    "reward/success_rate_completions_global",
                    "reward/pass_at_k_global",
                    "rollouts/completions_per_prompt",
                    "rollouts/total_global",
                    "rollouts/queries_global",
                    "rollouts/lengths_mean",
                    "termination/eos_stop_rate",
                    "generation_time",
                ]
                to_log = {}
                for k in immediate_keys:
                    v = processed_metrics_dict.get(k, None)
                    if isinstance(v, (int, float)):
                        to_log[f"train/{k}"] = float(v)

                # Per-worker metrics removed to avoid TPU collective issues
                # Also log dataset-specific per-reward metrics if provided (e.g., gsm8k/accuracy, math/format_rate)
                try:
                    for rf in self.reward_funcs:
                        _nm = getattr(rf, "__name__", None)
                        if not _nm:
                            continue
                        val = processed_metrics_dict.get(_nm, None)
                        if isinstance(val, (int, float)):
                            to_log[f"train/{_nm}"] = float(val)
                except Exception:
                    pass

                # Periodic verification report logging removed

                if len(to_log) > 0:
                    wandb.log(to_log, step=cur_step)
        except Exception:
            # Safe best-effort logging; never crash the step
            pass
                
        # Ensure all arrays are moved to host memory (unsharded) before returning
        # This is necessary because the training step expects empty_sharding on inputs
        # and arrays from generation/computation may have device sharding that conflicts
        return {
            "prompt_ids": prompt_ids_rep,
            "prompt_mask": prompt_mask_rep,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "rewards": rewards,
            "advantages": advantages,
        }, processed_metrics_dict    
    
    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """hook process to call in start of the step."""

        if (
            self.arguments.sync_ref_model
            and self.ref_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            self.ref_state = self.ref_state.replace(graphstate=deepcopy_model(state.graphstate))
        return state, metrics

    def generate_final_verification_report(self, verification_details_list: list = None):
        """Generate a comprehensive final verification report at the end of training.

        Args:
            verification_details_list: Optional list of verification details from multiple steps.
                                      If None, will use accumulated details from recent steps.
        """
        try:
            from easydel.verification.math_reward import generate_verification_report

            if verification_details_list is None:
                # Try to collect from recent training if available
                verification_details_list = []

            if verification_details_list:
                # Flatten list of lists if needed
                if isinstance(verification_details_list[0], list):
                    all_details = [item for sublist in verification_details_list for item in sublist]
                else:
                    all_details = verification_details_list

                report = generate_verification_report(all_details)

                if jax.process_index() == 0:
                    logger.info("FINAL VERIFICATION PERFORMANCE REPORT:")
                    print(report)

                    # Log to wandb if available
                    if self.arguments.use_wandb and self.arguments.can_log_metrics and wandb is not None:
                        try:
                            wandb.log({"final_verification_report": wandb.Html(f"<pre>{report}</pre>")})
                        except Exception:
                            pass

                return report
            else:
                logger.info("No verification details available for final report")
                return "No verification details available."

        except Exception as e:
            logger.warning(f"Could not generate final verification report: {e}")
            return f"Report generation failed: {e}"
