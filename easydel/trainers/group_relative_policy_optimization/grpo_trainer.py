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
from functools import cached_property, partial
import inspect

import flax
import flax.nnx
import jax
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

        # Apply adaptive mesh configuration before storing arguments
        from .adaptive_mesh import configure_adaptive_mesh_inplace
        
        # Configure mesh and update arguments in-place
        if arguments.force_tensor_parallel is not None or arguments.force_data_parallel is not None:
            mesh_plan = configure_adaptive_mesh_inplace(arguments)
            
            # Ensure dataset sharding aligns with DP only
            try:
                # Dataset should be sharded across DP workers only
                arguments.grain_shard_count = mesh_plan.dp
                arguments.grain_shard_index = jax.process_index() % mesh_plan.dp
                
                if jax.process_index() == 0:
                    logger.info(
                        f"Configured mesh: DP={mesh_plan.dp}, FSDP={mesh_plan.fsdp}, "
                        f"TP={mesh_plan.tp}, dataset shards={mesh_plan.dp}"
                    )
            except Exception as e:
                logger.warning(f"Failed to configure dataset sharding: {e}")
                # Fallback to safe defaults
                arguments.grain_shard_count = 1
                arguments.grain_shard_index = 0
        else:
            # No forced parallelism, use default behavior
            pass
        
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
        log_table = None
        if self.arguments.use_wandb and self.arguments.can_log_metrics and wandb is not None:
            try:
                log_table = wandb.Table(columns=["generations", "took", "length", "step"])
            except Exception:
                log_table = None
        else:
            log_table = None
        self.log_table = log_table

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
        )

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
        mesh = self.model.mesh

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
            if jax.process_index() == 0:
                per_process = int(self.arguments.total_batch_size) * int(self.arguments.num_return_sequences)
                # Use effective dp for expected global actually running this process configuration
                global_total = int(effective_dp) * per_process
                logger.info(
                    f"Rollout configuration: num_return_sequences={self.arguments.num_return_sequences}, "
                    f"expected_global_rollouts_per_step={global_total} (effective DP={effective_dp}, mesh DP={int(dp_size)}), "
                    f"per_process_rollouts={per_process}, "
                    f"batch_size={self.arguments.total_batch_size}"
                )

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
                
                # Build PRNG key from provided seed to ensure per-chunk diversity
                prng_key = jax.random.PRNGKey(prng_seed)
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
            
            # Chunked generation and reference log-prob computation to reduce peak memory
            rollout_chunk_size = getattr(self.arguments, "rollout_chunk_size", None)
            # Default to generating all num_return_sequences at once when not set
            if rollout_chunk_size is None or rollout_chunk_size <= 0:
                rollout_chunk_size = int(self.num_generations)
            # Clamp lower bound only; allow > num_return_sequences (loop uses min() with remaining)
            rollout_chunk_size = int(max(1, int(rollout_chunk_size)))
            # TP does not constrain batch item count for PagedAttention KV caching; no TP ceiling

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
                    per_chunk_seed = int(cur_step_int * 131071 + 4099 * jax.process_index() + chunk_idx)
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
            prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            # Calculate completion lengths before rewards (already computed per chunk; keep single computation)
            completion_lengths_per_seq = completion_mask.sum(-1)
            # Optional: all-gather for logging across processes (disabled by default)
            _global_lengths_for_logging = None
            _global_lengths_shaped = None
            if getattr(self.arguments, "log_global", False):
                try:
                    # Build a de-duplicated local vector by reading one shard (avoids TP replication)
                    shards = getattr(completion_lengths_per_seq, "addressable_shards", None)
                    if shards and len(shards) > 0:
                        _local = jnp.asarray(jax.device_get(shards[0].data).reshape(-1), dtype=jnp.int32)
                    else:
                        _local = jnp.asarray(jax.device_get(completion_lengths_per_seq).reshape(-1), dtype=jnp.int32)
                    # All-gather one vector per process
                    gathered = jax.experimental.multihost_utils.process_allgather(_local)
                    all_lengths_flat = jnp.reshape(gathered, (-1,))
                    _global_lengths_for_logging = all_lengths_flat
                    # Shape into (global_queries, r) for compact per-query display
                    r = max(1, int(self.num_generations))
                    total_completions_global = int(all_lengths_flat.size)
                    total_queries_global = max(1, total_completions_global // r)
                    try:
                        _global_lengths_shaped = jnp.reshape(all_lengths_flat[: total_queries_global * r], (total_queries_global, r))
                    except Exception:
                        _global_lengths_shaped = None
                except Exception:
                    _global_lengths_for_logging = None
                    _global_lengths_shaped = None

            # Host-side diagnostic: print per-rollout completion token lengths (process 0 prints)
            if jax.process_index() == 0:
                try:
                    # Prefer fully addressable arrays
                    if isinstance(completion_lengths_per_seq, jax.Array) and getattr(
                        completion_lengths_per_seq, "is_fully_addressable", False
                    ):
                        _lens = jax.device_get(completion_lengths_per_seq)
                    else:
                        # Fallback to first local shard if available
                        try:
                            _shards = completion_lengths_per_seq.addressable_shards  # type: ignore[attr-defined]
                            if _shards and len(_shards) > 0:
                                _lens = jax.device_get(_shards[0].data)
                            else:
                                _lens = jax.device_get(completion_lengths_per_seq)
                        except Exception:
                            _lens = jax.device_get(completion_lengths_per_seq)
                    _lens = jnp.asarray(_lens).astype(jnp.int32)
                    lengths_list = [int(x) for x in list(_lens)]
                    if lengths_list:
                        # Try to group by num_generations so we print per-query arrays
                        try:
                            num_queries = int(prompt_ids.shape[0])
                        except Exception:
                            num_queries = len(lengths_list) // max(1, int(self.num_generations))
                        r = int(self.num_generations)
                        if r > 0 and len(lengths_list) % r == 0:
                            grouped = [lengths_list[i * r : (i + 1) * r] for i in range(num_queries)]
                            # Clarify that these counts are per-process (local). Global prompts ~= DP * queries
                            try:
                                mesh_shape = getattr(self.model.mesh, "shape", {})
                                dp_sz = mesh_shape.get("dp", 1) if hasattr(mesh_shape, "get") else 1
                            except Exception:
                                dp_sz = 1
                            logger.info(
                                f"per_query_completion_token_lengths (local): queries={num_queries}, rollouts_per_query={r}; "
                                f"min={min(lengths_list)}, max={max(lengths_list)}, mean={sum(lengths_list)/len(lengths_list):.2f}; "
                                f"approx_global_queries={num_queries*int(dp_sz)}"
                            )
                            # Print compactly: one line per query (local view)
                            for qi, arr in enumerate(grouped):
                                logger.info(f"LOCAL query_{qi}_lengths (process {jax.process_index()}): {arr}")
                            
                            # Optional: global logging if explicitly enabled
                            if getattr(self.arguments, "log_global", False):
                                try:
                                    if _global_lengths_for_logging is not None:
                                        all_lengths_flat = _global_lengths_for_logging
                                        # Derive counts from actual gathered data
                                        total_completions_global = int(all_lengths_flat.size)
                                        r = max(1, int(self.num_generations))
                                        total_queries_global = max(1, total_completions_global // r)
                                        logger.info(
                                            f"GLOBAL completion_token_lengths: queries={total_queries_global}, rollouts_per_query={r}, "
                                            f"min={int(jnp.min(all_lengths_flat))}, max={int(jnp.max(all_lengths_flat))}, "
                                            f"mean={float(jnp.mean(all_lengths_flat)):.2f}"
                                        )
                                        # Per-query global breakdown (compact): show first 16 queries
                                        try:
                                            if _global_lengths_shaped is not None:
                                                max_q = min(16, int(_global_lengths_shaped.shape[0]))
                                                for qi in range(max_q):
                                                    arr = [int(x) for x in list(jax.device_get(_global_lengths_shaped[qi]))]
                                                    logger.info(f"GLOBAL query_{qi}_lengths: {arr}")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            
                            # If all per-query arrays are identical, this suggests duplicated prompts or identical RNG streams
                            try:
                                if num_queries > 1:
                                    as_tuples = {tuple(x) for x in grouped}
                                    if len(as_tuples) == 1:
                                        logger.warning(
                                            "All per-query completion length patterns are identical. "
                                            "This often indicates duplicated prompts in the local batch or repeated RNG streams across queries."
                                        )
                            except Exception:
                                pass
                        else:
                            logger.info(
                                f"completion_token_lengths (n={len(lengths_list)}): min={min(lengths_list)}, max={max(lengths_list)}, "
                                f"mean={sum(lengths_list)/len(lengths_list):.2f}; values={lengths_list}"
                            )
                    else:
                        logger.info("completion_token_lengths: []")
                except Exception as e:
                    logger.warning(f"Failed to print completion lengths: {e}")
                # Additionally, show prompt diagnostics to detect accidental duplication
                try:
                    # Only log short prefixes and uniqueness to avoid massive logs
                    prompt_prefixes = [p[:64].replace("\n", " ") for p in prompts]
                    unique_prompt_count = len(set(prompts))
                    logger.info(
                        f"prompt_diagnostics: queries={len(prompts)}, unique={unique_prompt_count}; "
                        f"prefixes={prompt_prefixes}"
                    )
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
            # Optionally aggregate rewards across processes to get true global means per step
            global_rewards_per_func = rewards_per_func
            try:
                if jax.process_count() > 1:
                    gathered = jax.experimental.multihost_utils.process_allgather(rewards_per_func)
                    # Flatten leading gather dimension
                    global_rewards_per_func = jnp.reshape(gathered, (-1, rewards_per_func.shape[-1]))
            except Exception as e:
                # Non-fatal; fall back to local metrics
                logger.warning(f"Global reward aggregation failed; using local metrics. Error: {e}")
            # Also compute global mean for the sum-of-rewards metric
            try:
                global_rewards_sum = jnp.sum(global_rewards_per_func, axis=1)
                global_mean_reward_per_completion = jnp.mean(global_rewards_sum)
            except Exception:
                global_mean_reward_per_completion = jnp.mean(rewards)
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

                # Global completion success rate from gathered rewards
                try:
                    successes_global = (global_rewards_sum > 0).astype(jnp.int32)
                    success_count_comp_global = jnp.sum(successes_global)
                    total_comp_global = successes_global.shape[0]
                    success_rate_comp_global = success_count_comp_global / jnp.maximum(1, total_comp_global)
                except Exception:
                    success_count_comp_global = success_count_comp_local
                    total_comp_global = total_comp_local
                    success_rate_comp_global = success_rate_comp_local

                # Global pass@k via gathering counts across processes
                pass_prompt_count_global = pass_prompt_count_local
                num_prompts_global = jnp.array(float(num_prompts_local))
                try:
                    if jax.process_count() > 1:
                        g_pass = jax.experimental.multihost_utils.process_allgather(pass_prompt_count_local.astype(jnp.float32))
                        g_prompts = jax.experimental.multihost_utils.process_allgather(jnp.array(float(num_prompts_local), dtype=jnp.float32))
                        pass_prompt_count_global = jnp.sum(g_pass)
                        num_prompts_global = jnp.sum(g_prompts)
                except Exception:
                    pass
                pass_at_k_global = pass_prompt_count_global / jnp.maximum(1.0, num_prompts_global)
            except Exception:
                # Safe fallbacks
                success_count_comp_local = jnp.array(0)
                total_comp_local = jnp.array(1)
                success_rate_comp_local = jnp.array(0.0)
                success_count_comp_global = success_count_comp_local
                total_comp_global = total_comp_local
                success_rate_comp_global = success_rate_comp_local
                pass_prompt_count_local = jnp.array(0)
                pass_prompt_count_global = jnp.array(0)
                pass_at_k_local = jnp.array(0.0)
                pass_at_k_global = jnp.array(0.0)
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
        # Default globals to local; override if we can gather
        lengths_global_min = lengths_local_min
        lengths_global_max = lengths_local_max
        lengths_global_mean = lengths_local_mean
        try:
            if jax.process_count() > 1:
                g_lens = jax.experimental.multihost_utils.process_allgather(lengths_local_arr)
                g_lens = jnp.reshape(g_lens, (-1,))
                lengths_global_min = jnp.min(g_lens)
                lengths_global_max = jnp.max(g_lens)
                lengths_global_mean = jnp.mean(g_lens)
        except Exception:
            pass
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
        except Exception:
            mesh_dp = 1
        try:
            proc_count = jax.process_count()
        except Exception:
            proc_count = 1
        dp_size = max(1, min(int(mesh_dp), int(proc_count)))
        metrics_dict = {
            "reward/mean_per_completion": per_completion_mean_reward,
            "reward/denominator/completions_local": float(num_completions_local),
            "reward/denominator/prompts_local": float(num_prompts_local),
            "reward/mean_per_completion_global": global_mean_reward_per_completion,
            "reward/denominator/completions_global": float(num_completions_local * dp_size),
            "reward/denominator/prompts_global": float(num_prompts_local * dp_size),
            # Success metrics (reward > 0)
            "reward/success_count_completions_local": float(success_count_comp_local),
            "reward/success_rate_completions_local": float(success_rate_comp_local),
            "reward/success_count_completions_global": float(success_count_comp_global),
            "reward/success_rate_completions_global": float(success_rate_comp_global),
            # Pass@k over prompts (at least one success among num_return_sequences)
            "reward/success_count_prompts_local": float(pass_prompt_count_local),
            "reward/pass_at_k_local": float(pass_at_k_local),
            "reward/success_count_prompts_global": float(pass_prompt_count_global),
            "reward/pass_at_k_global": float(pass_at_k_global),
            "completion_length": completion_length,
            # Completion length stats (local/global)
            "rollouts/lengths_local_min": lengths_local_min,
            "rollouts/lengths_local_max": lengths_local_max,
            "rollouts/lengths_local_mean": lengths_local_mean,
            "rollouts/lengths_global_min": lengths_global_min,
            "rollouts/lengths_global_max": lengths_global_max,
            "rollouts/lengths_global_mean": lengths_global_mean,
            # Rollout accounting to clarify totals
            "rollouts/completions_per_prompt": float(self.num_generations),
            "rollouts/total_per_process": float(num_completions_local),
            "rollouts/total_global": float(num_completions_local * dp_size),
            "rollouts/queries_per_process": float(num_prompts_local),
            "rollouts/queries_global": float(num_prompts_local * dp_size),
            "rollouts/derived_global_rollouts_per_step": float(getattr(self.arguments, "rollouts_per_step", num_completions_local * dp_size)),
            "rollouts/chunk_size": float(rollout_chunk_size),
            "rollouts/tp_size": float(locals().get('tp_size', 1) or 1),
            "rollouts/auto_one_per_tp": float(1.0 if ((locals().get('tp_size', 1) or 1) > 1 and getattr(self.arguments, "rollout_chunk_size", None) in (None, 0)) else 0.0),
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
        # Add per-reward metrics following TRL's approach for better clarity
        for i, reward_func in enumerate(self.reward_funcs):
            _name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            try:
                # TRL-style per-reward-function metrics with global aggregation
                try:
                    if jax.process_count() > 1:
                        gathered = jax.experimental.multihost_utils.process_allgather(rewards_per_func[:, i])
                        global_vals = jnp.reshape(gathered, (-1,))
                        global_mean = jnp.mean(global_vals)
                        global_std = jnp.std(global_vals)
                    else:
                        global_mean = jnp.mean(rewards_per_func[:, i])
                        global_std = jnp.std(rewards_per_func[:, i])
                except Exception:
                    global_mean = jnp.mean(rewards_per_func[:, i])
                    global_std = jnp.std(rewards_per_func[:, i])
                
                # TRL-compatible reward function metrics
                metrics_dict[f"rewards/{_name}/mean"] = global_mean
                metrics_dict[f"rewards/{_name}/std"] = global_std
                
                # Additional granularity metrics for debugging
                local_mean = jnp.mean(rewards_per_func[:, i])
                per_prompt_means = jnp.mean(
                    rewards_per_func[:, i].reshape(-1, self.num_generations), axis=1
                )
                local_mean_of_prompt_means = jnp.mean(per_prompt_means)
                metrics_dict[f"rewards/{_name}/mean_per_completion_local"] = local_mean
                metrics_dict[f"rewards/{_name}/mean_per_prompt_local"] = local_mean_of_prompt_means
                metrics_dict[f"rewards/{_name}/mean_per_completion_global"] = global_mean
            except Exception:
                # Fallback for backward compatibility
                metrics_dict[_name] = jnp.mean(rewards_per_func[:, i])
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

        # Immediately log a small set of lightweight generation/reward metrics every step (process 0 only)
        try:
            if (
                jax.process_index() == 0
                and self.arguments.use_wandb
                and self.arguments.can_log_metrics
                and wandb is not None
            ):
                try:
                    cur_step = int(jax.device_get(state.step))
                except Exception:
                    cur_step = 0
                immediate_keys = [
                    "reward/success_rate_completions_local",
                    "reward/success_rate_completions_global",
                    "reward/pass_at_k_local",
                    "reward/pass_at_k_global",
                    "reward/mean_per_completion",
                    "reward/mean_per_completion_global",
                    "rollouts/completions_per_prompt",
                    "rollouts/total_per_process",
                    "rollouts/total_global",
                    "rollouts/lengths_local_mean",
                    "rollouts/lengths_global_mean",
                    "rollouts/derived_global_rollouts_per_step",
                    "termination/eos_stop_rate",
                    "generation_time",
                ]
                to_log = {
                    f"train/{k}": float(processed_metrics_dict[k])
                    for k in immediate_keys
                    if k in processed_metrics_dict and isinstance(processed_metrics_dict[k], (int, float))
                }
                if len(to_log) > 0:
                    wandb.log(to_log, step=cur_step)
        except Exception:
            # Safe best-effort logging; never crash the step
            pass

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
                
        # Ensure all arrays are moved to host memory (unsharded) before returning
        # This is necessary because the training step expects empty_sharding on inputs
        # and arrays from generation/computation may have device sharding that conflicts
        return {
            "prompt_ids": prompt_ids_rep,
            "prompt_mask": prompt_mask_rep,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
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
