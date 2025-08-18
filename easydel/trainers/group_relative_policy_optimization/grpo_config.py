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
import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@auto_pytree
class GRPOConfig(TrainingArguments):
    """
    Configuration class for the GRPOTrainer.
    """

    trainer_prefix: str | None = field(
        default="grpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "The maximum length of the prompt."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "The maximum length of the completion."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for dataset processing."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The learning rate."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "The beta parameter for GRPO."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to periodically sync the reference model with the policy model."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={"help": "The alpha parameter for mixing the reference model with the policy model."},
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={"help": "The number of steps between syncing the reference model."},
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for training."},
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "whenever to skip extracting prompt from dataset."},
    )
    force_tensor_parallel: int | None = field(
        default=None,
        metadata={
            "help": "Force tensor parallelism dimension. Enables sub-batch processing for more "
            "efficient parallel training. E.g., tp=2 creates 2 models on 2 TPUs each with 4 TPUs total."
        },
    )
    force_data_parallel: int | None = field(
        default=None,
        metadata={
            "help": "Force data parallelism dimension. When used with --force_tensor_parallel, must satisfy dp * tp <= num_devices."
        },
    )
    mini_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Minimum batch size per model instance when using tensor parallelism. "
            "Must be >= 1. Used to control how batches are distributed across model instances."
        },
    )
    num_return_sequences: int = field(
        default=4,
        metadata={
            "help": "Number of completions generated per prompt per process. "
            "Total rollouts per process per step = total_batch_size × num_return_sequences. "
            "Global total ≈ that × data_parallel_processes (if >1)."
        },
    )

    # Backward-compatible aliases for clearer semantics
    completions_per_prompt: int | None = field(
        default=None,
        metadata={
            "help": "Alias for num_return_sequences. If provided, overrides num_return_sequences. "
            "Represents how many completions are generated per prompt per process."
        },
    )

    top_p: float = field(
        default=0.95,
        metadata={
            "help": "Top-p (nucleus) sampling threshold. Tokens are sampled from the smallest possible set whose "
            "cumulative probability exceeds this value."
        },
    )

    top_k: int = field(
        default=50,
        metadata={"help": "Top-k sampling threshold. Limits sampling to the top-k most probable tokens at each step."},
    )

    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Sampling temperature. Higher values (e.g., >1.0) produce more random outputs, while "
            "lower values (e.g., <1.0) make the output more deterministic."
        },
    )

    advantage_epsilon: float = field(
        default=1e-6,
        metadata={
            "help": "Minimum standard deviation threshold for computing advantages. Groups with std < epsilon "
            "will have their advantages set to 0 to avoid numerical instability. Typical values: 1e-6 to 1e-3. "
            "Critical for avoiding tiny advantages (e.g., 1e-8) that provide no learning signal. "
            "After per-group standardization, advantages should typically have median |A| ~ 0.5-1.5 "
            "and 95th percentile |A| ~ 2-5. If you see many low-variance groups, consider: "
            "(1) using more diverse prompts, (2) increasing num_return_sequences, "
            "(3) using continuous rewards instead of binary, or (4) increasing this epsilon."
        },
    )

    rollout_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": "Chunk size for processing rollouts to reduce memory usage during generation. "
            "If None or <= 0, defaults to min(2, num_return_sequences). Larger values use more memory "
            "but may be faster, while smaller values reduce memory usage. Does not change total rollouts."
        },
    )

    # Backward-compatible alias for rollout_chunk_size
    completions_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": "Alias for rollout_chunk_size. If provided, overrides rollout_chunk_size. "
            "Controls how many completions are generated per chunk per prompt."
        },
    )

    # Memory-optimized microbatching: process one completion per prompt per microbatch
    microbatch_one_completion: bool = field(
        default=False,
        metadata={
            "help": "If True, set gradient_accumulation_steps = num_return_sequences so each microbatch holds one completion per prompt. "
            "Also defaults completions_chunk_size (rollout_chunk_size) to 1 for lowest generation memory."
        },
    )

    # High-level rollout targets
    rollouts_per_prompt: int | None = field(
        default=None,
        metadata={
            "help": "High-level alias to set the number of completions per prompt (maps to num_return_sequences)."
        },
    )
    rollouts_per_step: int | None = field(
        default=None,
        metadata={
            "help": "Target total rollouts per global step across all data-parallel workers. "
            "Automatically derives num_return_sequences from DP and total_batch_size."
        },
    )

    def __post_init__(self):
        """Post initialization to set dependent parameters."""
        self.max_sequence_length = self.max_prompt_length + self.max_completion_length
        
        # Apply aliases with gentle override semantics
        if self.rollouts_per_prompt is not None:
            self.num_return_sequences = int(max(1, self.rollouts_per_prompt))
        if self.completions_per_prompt is not None:
            # Prefer the clearer alias if user set it
            self.num_return_sequences = int(self.completions_per_prompt)
        if self.completions_chunk_size is not None:
            self.rollout_chunk_size = int(self.completions_chunk_size)

        # If enabled, only tune rollout chunking for minimal generation memory
        if self.microbatch_one_completion:
            # Generate one completion per chunk to bound peak KV/logit memory
            self.rollout_chunk_size = 1

        # If user set a global target for rollouts per step, derive num_return_sequences
        if self.rollouts_per_step is not None:
            try:
                # Best-effort DP detection without forcing backend init
                from .training_configurations import _safe_process_count  # type: ignore
                dp = int(max(1, _safe_process_count()))
            except Exception:
                dp = 1
            per_process_target = max(1, int((self.rollouts_per_step + dp - 1) // dp))  # ceil div
            # Derive nrs so that total_batch_size * nrs >= per_process_target
            nrs = int(max(1, (per_process_target + self.total_batch_size - 1) // self.total_batch_size))
            self.num_return_sequences = nrs
            # If user wants microbatch-per-completion, set accum steps implicitly
            if self.microbatch_one_completion:
                self.gradient_accumulation_steps = nrs

        # Validate tensor parallelism configuration
        if self.force_tensor_parallel is not None:
            if self.force_tensor_parallel < 1:
                raise ValueError("force_tensor_parallel must be >= 1")
            
            if self.mini_batch_size is not None and self.mini_batch_size < 1:
                raise ValueError("mini_batch_size must be >= 1 when specified")
            
            # Default mini_batch_size to 1 if not specified with TP
            if self.mini_batch_size is None:
                self.mini_batch_size = 1

        # Validate data parallelism configuration
        if self.force_data_parallel is not None:
            if self.force_data_parallel < 1:
                raise ValueError("force_data_parallel must be >= 1")

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    __hash__ = hash_fn
