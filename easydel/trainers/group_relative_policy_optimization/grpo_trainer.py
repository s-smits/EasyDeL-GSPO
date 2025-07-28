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
        from .adaptive_mesh import get_adaptive_step_partition_spec
        adaptive_step_spec = get_adaptive_step_partition_spec(
            arguments.total_batch_size,
            force_tensor_parallel=arguments.force_tensor_parallel,
            mini_batch_size=arguments.mini_batch_size
        )
        
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
        adaptive_spec = get_adaptive_sharding_spec(
            self.arguments.total_batch_size,
            force_tensor_parallel=self.arguments.force_tensor_parallel,
            mini_batch_size=self.arguments.mini_batch_size
        )
        input_sharding = NamedSharding(
            mesh=mesh,
            spec=adaptive_spec
        )
        
        @ejit(
            in_shardings=(self.state_shardings, input_sharding, input_sharding),
            out_shardings=(empty_sharding, input_sharding, input_sharding),
        )
        def generate(state: EasyDeLState, input_ids, attention_mask):
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
                    num_return_sequences=self.num_generations,
                    do_sample=True,
                    use_cache=False,
                )
                
                sequences = module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                ).sequences
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
            in_shardings=(self.state_shardings, empty_sharding),
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
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

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
            static_argnames=("graphdef"),
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
            
            # Try multiple solutions to convert numpy arrays to JAX arrays
            conversion_success = False
            
            # Solution 1: Simple jnp.asarray
            if not conversion_success:
                try:
                    prompt_ids = jnp.asarray(prompt_ids)
                    prompt_mask = jnp.asarray(prompt_mask)
                    print(f"SUCCESS: Solution 1 (jnp.asarray) worked on worker {jax.process_index()}")
                    conversion_success = True
                except Exception as e:
                    print(f"FAILED: Solution 1 (jnp.asarray) failed on worker {jax.process_index()}: {e}")
            
            # Solution 2: jax.device_put with replicated sharding
            if not conversion_success:
                try:
                    mesh = self.model.mesh
                    replicated_sharding = NamedSharding(mesh=mesh, spec=PartitionSpec())
                    prompt_ids = jax.device_put(jnp.asarray(prompt_ids), replicated_sharding)
                    prompt_mask = jax.device_put(jnp.asarray(prompt_mask), replicated_sharding)
                    print(f"SUCCESS: Solution 2 (device_put + replicated) worked on worker {jax.process_index()}")
                    conversion_success = True
                except Exception as e:
                    print(f"FAILED: Solution 2 (device_put + replicated) failed on worker {jax.process_index()}: {e}")
            
            # Solution 3: jax.make_array_from_process_local_data
            if not conversion_success:
                try:
                    mesh = self.model.mesh
                    replicated_sharding = NamedSharding(mesh=mesh, spec=PartitionSpec())
                    prompt_ids = jax.make_array_from_process_local_data(replicated_sharding, prompt_ids)
                    prompt_mask = jax.make_array_from_process_local_data(replicated_sharding, prompt_mask)
                    print(f"SUCCESS: Solution 3 (make_array_from_process_local_data) worked on worker {jax.process_index()}")
                    conversion_success = True
                except Exception as e:
                    print(f"FAILED: Solution 3 (make_array_from_process_local_data) failed on worker {jax.process_index()}: {e}")
            
            # Solution 4: jax.make_array_from_callback
            if not conversion_success:
                try:
                    mesh = self.model.mesh
                    replicated_sharding = NamedSharding(mesh=mesh, spec=PartitionSpec())
                    
                    def _make_prompt_ids():
                        return jnp.asarray(prompt_ids)
                    def _make_prompt_mask():
                        return jnp.asarray(prompt_mask)
                    
                    prompt_ids = jax.make_array_from_callback(prompt_ids.shape, replicated_sharding, _make_prompt_ids)
                    prompt_mask = jax.make_array_from_callback(prompt_mask.shape, replicated_sharding, _make_prompt_mask)
                    print(f"SUCCESS: Solution 4 (make_array_from_callback) worked on worker {jax.process_index()}")
                    conversion_success = True
                except Exception as e:
                    print(f"FAILED: Solution 4 (make_array_from_callback) failed on worker {jax.process_index()}: {e}")
            
            if not conversion_success:
                print(f"ERROR: All conversion solutions failed on worker {jax.process_index()}")
                raise RuntimeError("Failed to convert numpy arrays to JAX arrays")

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
                # Properly shard tokens according to step_partition_spec before passing to compute_refmodel_logps
                sharded_prompt_completion_ids = with_sharding_constraint(
                    prompt_completion_ids, self.arguments.step_partition_spec
                )
                sharded_mask = with_sharding_constraint(
                    jnp.concatenate([ridmask, completion_mask], -1), self.arguments.step_partition_spec
                )
                ref_per_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    sharded_prompt_completion_ids,
                    sharded_mask,
                )
            token_logps_time = token_logps_time_fn()
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
        
        # DEBUG: Check if training is actually happening
        if jax.process_index() == 0:
            print(f"DEBUG TRAINING CHECK:")
            print(f"  Step: {jax.device_get(state.step)}")
            print(f"  Rewards shape: {rewards.shape}, mean: {jnp.mean(rewards):.4f}, std: {jnp.std(rewards):.4f}")
            print(f"  Completion lengths: min={jnp.min(completion_lengths_per_seq)}, max={jnp.max(completion_lengths_per_seq)}, mean={jnp.mean(completion_lengths_per_seq):.1f}")
            print(f"  Advantages: mean={jnp.mean(advantages):.4f}, std={jnp.std(advantages):.4f}")
            print(f"  Note: Advantage magnitudes (median, 95th percentile) are now visible in progress bar")
            print(f"  Generation time: {generation_time:.1f}s")
            # Check if we're getting diverse outputs
            unique_lengths = len(jnp.unique(completion_lengths_per_seq))
            print(f"  Unique completion lengths: {unique_lengths}/{len(completion_lengths_per_seq)} (diversity check)")
            
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
        if self.log_table is not None:
            cur_step = jax.device_get(state.step)
            # HARDCODED: Always log to WandB table every step
            if jax.process_index() == 0:
                print(f"DEBUG: WandB table logging EVERY STEP - step {cur_step}")
            decoded_text = self.processing_class.batch_decode(jax.device_get(completion_ids))
            individual_lengths = jax.device_get(completion_lengths_per_seq)
            if jax.process_index() == 0:
                print(f"DEBUG: Logging {len(decoded_text)} generations to WandB table")
                print(f"DEBUG: Sample generation (first 100 chars): {decoded_text[0][:100]}...")
                print(f"DEBUG: Individual lengths: {individual_lengths}")
            for text, length in zip(decoded_text, individual_lengths, strict=False):
                self.log_table.add_data(text, generation_time, float(length), cur_step)
            if jax.process_index() == 0:
                print(f"DEBUG: Calling wandb.log with table containing {len(self.log_table.data)} rows")
            wandb.log({"generations": self.log_table}, step=cur_step)
            if jax.process_index() == 0:
                print("DEBUG: WandB log call completed")

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
        # Convert JAX arrays to float for metrics dict to match return type
        processed_metrics_dict = {}
        for key, value in metrics_dict.items():
            if hasattr(value, 'item'):  # JAX array or numpy array
                processed_metrics_dict[key] = float(value.item())
            else:
                processed_metrics_dict[key] = value
        
        return (
            {
                "prompt_ids": self._all_gather(prompt_ids),
                "prompt_mask": self._all_gather(prompt_mask),
                "completion_ids": self._all_gather(completion_ids),
                "completion_mask": self._all_gather(completion_mask),
                "ref_per_token_logps": self._all_gather(ref_per_token_logps),
                "advantages": advantages,
            },
            processed_metrics_dict,
        )

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
