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
        from .adaptive_mesh import get_adaptive_step_partition_spec, validate_mesh_config, calculate_optimal_mesh_dims
        _step_sig = inspect.signature(get_adaptive_step_partition_spec)
        _step_kwargs = dict(
            total_batch_size=arguments.total_batch_size,
            force_tensor_parallel=arguments.force_tensor_parallel,
            mini_batch_size=arguments.mini_batch_size,
        )
        if 'force_data_parallel' in _step_sig.parameters:
            _step_kwargs['force_data_parallel'] = arguments.force_data_parallel
        adaptive_step_spec = get_adaptive_step_partition_spec(**_step_kwargs)
        
        # Validate mesh configuration for multi-worker training
        if arguments.force_tensor_parallel is not None or arguments.force_data_parallel is not None:
            mesh_kwargs = dict(
                total_batch_size=arguments.total_batch_size,
                num_return_sequences=arguments.num_return_sequences,
                force_tensor_parallel=arguments.force_tensor_parallel,
                force_data_parallel=arguments.force_data_parallel,
                mini_batch_size=arguments.mini_batch_size,
            )
            dp, fsdp, ep, tp, sp = calculate_optimal_mesh_dims(**mesh_kwargs)
            try:
                num_devices = jax.device_count()
            except Exception:
                import os
                num_devices = int(os.getenv("JAX_DEVICE_COUNT", "1"))
            validate_mesh_config(dp, fsdp, tp, num_devices, arguments.total_batch_size)
        
        # Override step_partition_spec if it would cause dimension mismatch or using TP
        should_override = (
            (arguments.total_batch_size == 1 and arguments.step_partition_spec == PartitionSpec(("dp", "fsdp"), "sp")) or
            (arguments.force_tensor_parallel is not None)
        )
        
        if should_override:
            logger.warning(
                f"Overriding step_partition_spec from {arguments.step_partition_spec} to {adaptive_step_spec} "
                f"for batch_size={arguments.total_batch_size}, tp={arguments.force_tensor_parallel}"
            )
            arguments.step_partition_spec = adaptive_step_spec
        
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
            log_table = wandb.Table(columns=["generations", "took", "length", "step"])
            if jax.process_index() == 0:
                print(f"DEBUG: Created WandB table - use_wandb:{self.arguments.use_wandb}, can_log_metrics:{self.arguments.can_log_metrics}, wandb_available:{wandb is not None}")
        else:
            if jax.process_index() == 0:
                print(f"DEBUG: WandB table NOT created - use_wandb:{self.arguments.use_wandb}, can_log_metrics:{self.arguments.can_log_metrics}, wandb_available:{wandb is not None}")
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
        if isinstance(self.processing_class, ProcessorMixin):
            proc_eos_token_id = self.processing_class.tokenizer.eos_token_id
        else:
            proc_eos_token_id = self.processing_class.eos_token_id
        if isinstance(proc_eos_token_id, int):
            proc_eos_token_id = [proc_eos_token_id]
        eos_ids = eos_ids + proc_eos_token_id
        if hasattr(self.model, "generation_config"):
            conf_eos = self.model.generation_config.eos_token_id
            if isinstance(conf_eos, int):
                conf_eos = [conf_eos]
            eos_ids = eos_ids + conf_eos
        return list(set(eos_ids))

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
            in_shardings=(self.state_shardings, input_sharding, input_sharding),
            out_shardings=(empty_sharding, input_sharding, input_sharding),
            static_argnums=(3,),
        )
        def generate(state: EasyDeLState, input_ids, attention_mask, num_return_sequences: int):
            module = state.model

            with module.mesh:
                if jax.process_index() == 0:
                    try:
                        print("DEBUG(gen): in ids", getattr(input_ids, "shape", None), "sharding=", getattr(input_ids, "sharding", None))
                        print("DEBUG(gen): in mask", getattr(attention_mask, "shape", None), "sharding=", getattr(attention_mask, "sharding", None))
                        print("DEBUG(gen): input_spec=", adaptive_spec, "step_spec=", self.arguments.step_partition_spec)
                    except Exception:
                        pass
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
                
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                ).sequences
                if jax.process_index() == 0:
                    try:
                        print("DEBUG(gen): out seq", getattr(sequences, "shape", None), "sharding=", getattr(sequences, "sharding", None))
                        print("DEBUG(gen): post-partition ids", getattr(input_ids, "shape", None), "sharding=", getattr(input_ids, "sharding", None))
                        print("DEBUG(gen): post-partition mask", getattr(attention_mask, "shape", None), "sharding=", getattr(attention_mask, "sharding", None))
                    except Exception:
                        pass
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

        static_argnames = (2, 3, 4, 5, 6, 7, 8)
        
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
            if rollout_chunk_size is None or rollout_chunk_size <= 0:
                rollout_chunk_size = min(2, int(self.num_generations))

            sequences_chunks = []
            completion_ids_chunks = []
            completion_mask_chunks = []
            ref_logps_chunks = []
            comp_len_chunks = []
            generation_time = 0.0
            token_logps_time = 0.0

            base_prompt_len = prompt_ids.shape[-1]
            nrs_remaining = int(self.num_generations)
            while nrs_remaining > 0:
                cur_nrs = int(min(rollout_chunk_size, nrs_remaining))
                with capture_time() as generation_time_fn:
                    seq_chunk, prompt_ids, prompt_mask = jax.block_until_ready(
                        self.generate_function(state, prompt_ids, prompt_mask, cur_nrs)
                    )
                if jax.process_index() == 0:
                    try:
                        print("DEBUG(loop): cur_nrs=", cur_nrs)
                        print("DEBUG(loop): ids", getattr(prompt_ids, "shape", None), "sharding=", getattr(prompt_ids, "sharding", None))
                        print("DEBUG(loop): mask", getattr(prompt_mask, "shape", None), "sharding=", getattr(prompt_mask, "sharding", None))
                    except Exception:
                        pass
                generation_time += float(generation_time_fn())

                # Cross-host barrier to keep hosts in program lockstep after each chunk generation
                try:
                    jax.experimental.multihost_utils.sync_global_devices("after_generate_chunk")
                except Exception:
                    pass

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

                # Barrier after each chunked ref logps computation
                try:
                    jax.experimental.multihost_utils.sync_global_devices("after_ref_logps_chunk")
                except Exception:
                    pass

                # Accumulate
                sequences_chunks.append(prompt_completion_ids_chunk)
                completion_ids_chunks.append(completion_ids_chunk)
                completion_mask_chunks.append(completion_mask_chunk)
                ref_logps_chunks.append(ref_logps_chunk)
                comp_len_chunks.append(completion_mask_chunk.sum(-1))

                nrs_remaining -= cur_nrs

            # Concatenate accumulated chunks
            if jax.process_index() == 0:
                try:
                    print("DEBUG(cat): num seq chunks=", len(sequences_chunks))
                    if len(completion_ids_chunks) > 0:
                        print("DEBUG(cat): shard ids[0] sharding=", getattr(completion_ids_chunks[0], "sharding", None))
                except Exception:
                    pass
            prompt_completion_ids = jnp.concatenate(sequences_chunks, axis=0)
            completion_ids = jnp.concatenate(completion_ids_chunks, axis=0)
            completion_mask = jnp.concatenate(completion_mask_chunks, axis=0)
            ref_per_token_logps = jnp.concatenate(ref_logps_chunks, axis=0)
            completion_lengths_per_seq = jnp.concatenate(comp_len_chunks, axis=0)
            prompts = self.processing_class.batch_decode(batch["input_ids"], skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conversational:
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                completions = completions_text

            # Calculate completion lengths before rewards
            completion_lengths_per_seq = completion_mask.sum(-1)  # Length per sequence
            
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
                        if jax.process_index() == 0 and i == 0:  # Debug first reward function only
                            print(f"DEBUG REWARD FUNCTION:")
                            print(f"  Function: {getattr(reward_func, '__name__', 'unknown')}")
                            print(f"  Prompts count: {len(in_prompts)}")
                            print(f"  Completions type: {type(completions)}")
                            print(f"  First completion type: {type(completions[0]) if completions else 'empty'}")
                            if completions and len(completions) > 0:
                                # Show FULL completion text to see the ending
                                full_text = completions[0][0]["content"] if isinstance(completions[0], list) else completions[0]
                                # print(f"  FULL completion text:\n{full_text}") don't display full completion text
                                print(f"  Text ends with: ...{full_text[-100:]}")
                                # Check pattern match
                                import re
                                pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
                                match = re.match(pattern, full_text, re.DOTALL)
                                print(f"  Pattern match: {match is not None}")
                                if not match:
                                    print(f"  Checking if has <think>: {'<think>' in full_text}")
                                    print(f"  Checking if has </think>: {'</think>' in full_text}")
                                    print(f"  Checking if has <answer>: {'<answer>' in full_text}")
                                    print(f"  Checking if has </answer>: {'</answer>' in full_text}")
                                print("="*50)
                        output_reward_func = reward_func(
                            prompts=in_prompts,
                            completions=completions,
                            max_length=self.arguments.max_sequence_length,
                            batch=batch,
                            completion_lengths=jax.device_get(completion_lengths_per_seq),
                        )
                        rew = jnp.array(output_reward_func, dtype="f4")
                        if jax.process_index() == 0 and i == 0:  # Debug first reward function only
                            print(f"  Reward output: {output_reward_func}")
                            print(f"  Reward array: {rew}")
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()
            
            with capture_time() as grouped_comp_time_fn:
                rewards = rewards_per_func.sum(axis=1)
                advantages = (
                    rewards
                    - jnp.mean(
                        rewards.reshape(-1, self.num_generations),
                        axis=-1,
                    ).repeat(self.num_generations, axis=0)
                ) / (
                    jnp.std(
                        rewards.reshape(-1, self.num_generations),
                        axis=-1,
                    ).repeat(self.num_generations, axis=0)
                    + 1e-4
                )
            grouped_comp_time = grouped_comp_time_fn()
        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.mean(completion_lengths_per_seq)  # Average for metrics
        
        # Simplified debug output to prevent TPU coordination issues
        if jax.process_index() == 0:
            try:
                step_val = jax.device_get(state.step)
                print(f"DEBUG: Step {step_val} - Generation time: {generation_time:.1f}s")
                print(f"  Rewards: shape={rewards.shape}")
                # Skip complex array operations that can cause coordination issues
            except Exception as e:
                print(f"DEBUG: Basic check failed - {e}")
            
        metrics_dict = {
            "rewards": jnp.mean(rewards, -1),
            "completion_length": completion_length,
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
        }
        for i, reward_func in enumerate(self.reward_funcs):
            _name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            metrics_dict[_name] = jnp.mean(rewards_per_func[:, i])
        if self.log_table is not None and jax.process_index() == 0:
            cur_step = int(jax.device_get(state.step))
            # Build local-only host arrays to avoid cross-host device_get
            def _to_local_host(x):
                if isinstance(x, jax.Array):
                    try:
                        is_addr = getattr(x, "is_fully_addressable", None)
                        if callable(is_addr) and is_addr():
                            return jax.device_get(x)
                    except Exception:
                        pass
                    shards = getattr(x, "addressable_shards", None)
                    if shards:
                        # Prefer returning the first local shard to avoid cross-device concat
                        return jax.device_get(shards[0].data)
                return x
            local_completion_ids = _to_local_host(completion_ids)
            local_comp_lens = _to_local_host(completion_lengths_per_seq)
            decoded_text = self.processing_class.batch_decode(local_completion_ids)
            individual_lengths = local_comp_lens
            print(f"DEBUG: WandB table logging EVERY STEP - step {cur_step}")
            print(f"DEBUG: Logging {len(decoded_text)} generations to WandB table")
            if len(decoded_text) > 0:
                print(f"DEBUG: Sample generation (first 100 chars): {decoded_text[0][:100]}...")
            print(f"DEBUG: Individual lengths: {individual_lengths}")
            for text, length in zip(decoded_text, individual_lengths, strict=False):
                self.log_table.add_data(text, generation_time, float(length), cur_step)
            print(f"DEBUG: Calling wandb.log with table containing {len(self.log_table.data)} rows")
            wandb.log({"generations": self.log_table}, step=cur_step)
            print("DEBUG: WandB log call completed")

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
        # Convert JAX arrays to float for metrics dict to match return type
        processed_metrics_dict = {}
        for key, value in metrics_dict.items():
            if hasattr(value, 'item'):  # JAX array or numpy array
                try:
                    processed_metrics_dict[key] = float(value.item())
                except Exception as e:
                    # If TPU halted, use a default value or skip the metric
                    logger.warning(f"Failed to convert metric '{key}' to float: {e}")
                    processed_metrics_dict[key] = 0.0  # Default fallback value
            else:
                processed_metrics_dict[key] = value
                
        # Ensure all arrays are moved to host memory (unsharded) before returning
        # This is necessary because the training step expects empty_sharding on inputs
        # and arrays from generation/computation may have device sharding that conflicts
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
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
