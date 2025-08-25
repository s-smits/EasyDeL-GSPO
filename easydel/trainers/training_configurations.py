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

import datetime
import functools
import io
import json
import os
import re
import tempfile
import traceback
import typing as tp
import warnings
from copy import deepcopy
from dataclasses import field

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
from eformer.optimizers import OptimizerFactory, SchedulerConfig
from eformer.pytree import auto_pytree
from jax.sharding import PartitionSpec

from easydel.infra.errors import EasyDeLTimerError
from easydel.infra.etils import (
    AVAILABLE_OPTIMIZERS,
    AVAILABLE_SCHEDULERS,
    AVAILABLE_SPARSE_MODULE_TYPES,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
)
from easydel.infra.loss_utils import LossConfig
from easydel.utils import EasyPath, EasyPathLike
from easydel.utils.compiling_utils import hash_fn
from easydel.utils.helpers import get_logger

from .metrics import MetricsHistogram, compute_weight_stats
from .utils import JaxDistributedConfig

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from flax.metrics.tensorboard import SummaryWriter  # type:ignore
    from jax import Array  # type:ignore
    from torch import Tensor  # type:ignore
else:
    Array, Tensor = [tp.Any] * 2


MetricsType = dict[str, float | list | tuple | np.ndarray | Array | Tensor]
logger = get_logger(__name__)


def get_safe_arr(xs):
    if isinstance(xs, np.generic | jax.Array):
        if xs.size == 1:  # Only try .item() on size-1 arrays
            return xs.item()
        return xs
    return xs


# Constants
AVAILABLE_BACKENDS: list[str] = ["cpu", "gpu", "tpu", None]


# --- Safe JAX helpers to avoid initializing TPU in child processes ---
def _safe_int_from_env(var_name: str, default: int | None) -> int | None:
    val = os.getenv(var_name)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


def _safe_process_index() -> int:
    """Best-effort process index without forcing JAX backend init.

    Priority:
      1) JAX_PROCESS_INDEX env var
      2) jax.process_index() (guarded)
      3) default 0
    """
    env_idx = _safe_int_from_env("JAX_PROCESS_INDEX", None)
    if env_idx is not None:
        return env_idx
    try:
        return jax.process_index()
    except Exception:
        return 0


def _safe_process_count() -> int:
    """Best-effort process count without forcing JAX backend init.

    Priority:
      1) JAX_PROCESS_COUNT env var
      2) jax.process_count() (guarded)
      3) default 1
    """
    env_cnt = _safe_int_from_env("JAX_PROCESS_COUNT", None)
    if env_cnt is not None:
        return env_cnt
    try:
        return jax.process_count()
    except Exception:
        return 1


@auto_pytree
class TrainingArguments:
    auto_shard_states: bool = field(
        default=True,
        metadata={"help": "Whether to automatically shard model states across devices."},
    )
    aux_loss_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable the auxiliary loss."},
    )
    backend: str | None = field(
        default=None,
        metadata={"help": "The JAX backend to use (e.g., 'cpu', 'gpu', 'tpu').  If None, JAX will choose."},
    )
    clip_grad: float | None = field(
        default=None,
        metadata={"help": "The value at which to clip the gradients."},
    )
    custom_scheduler: tp.Callable[[int], tp.Any] | None = field(
        default=None,
        metadata={"help": "A custom scheduler function that takes the current step as input."},
    )
    dataloader_num_workers: int | None = field(
        default=0,
        metadata={"help": "The number of workers to use for the dataloader."},
    )
    dataloader_pin_memory: bool | None = field(
        default=False,
        metadata={"help": "Whether to pin memory for the dataloader."},
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training."},
    )
    do_last_save: bool = field(
        default=True,
        metadata={"help": "Whether to save the model at the end of training."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    eval_batch_size: int | None = field(
        default=None,
        metadata={"help": "The batch size to use for evaluation."},
    )
    evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "Run evaluation every X steps."},
    )
    extra_optimizer_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional keyword arguments to pass to the optimizer."},
    )
    frozen_parameters: str | None = field(
        default=None,
        metadata={"help": "A regex pattern of parameters to freeze (not train)."},
    )
    grain_shard_index: int | None = field(
        default=None,
        metadata={"help": "sharding index to be used for grain dataloaders in both train and eval steps. If None, auto-detects from jax.process_index()."},
    )
    grain_shard_count: int | None = field(
        default=None,
        metadata={"help": "sharding count to be used for grain dataloaders in both train and eval steps. If None, auto-detects from jax.process_count()."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of steps to accumulate gradients over."},
    )
    ids_to_pop_from_dataset: list[str] | None = field(
        default_factory=list,
        metadata={"help": "A list of dataset columns to remove before training."},
    )
    is_fine_tuning: bool = field(
        default=True,
        metadata={"help": "Whether the training is a fine-tuning run."},
    )
    init_tx: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the training state."},
    )
    jax_distributed_config: dict | None = field(
        default=None,
        metadata={"help": "Configuration for JAX distributed training."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The learning rate."},
    )
    learning_rate_end: float | None = field(
        default=None,
        metadata={"help": "The final learning rate for linear decay schedulers."},
    )
    log_all_workers: bool = field(
        default=False,
        metadata={"help": "Whether to log metrics from all workers in a distributed setup."},
    )
    log_global: bool = field(
        default=False,
        metadata={
            "help": "If True, perform cross-process allgather of certain diagnostics (e.g., completion lengths) "
            "to log global views. This introduces collectives in the host preprocessing path; enable only when "
            "you need full global logs. Default False for maximal stability."
        },
    )
    log_grad_norms: bool = field(
        default=True,
        metadata={"help": "Whether to log gradient norms."},
    )
    report_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to report metrics to a logger."},
    )
    log_steps: int = field(
        default=10,
        metadata={"help": "Log metrics every X steps."},
    )
    log_logprobs_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to log logprobs metrics."},
    )
    verify_dataset_sharding: bool = field(
        default=True,
        metadata={
            "help": "If True, perform a one-time cross-process verification on step 0 to detect duplicate prompts across DP workers."
        },
    )
    dataset_sharding_tolerance: float = field(
        default=0.05,
        metadata={
            "help": "Maximum allowed ratio of duplicate prompts before raising an error (default: 0.05 = 5%). Set to 0.0 for strict checking."
        },
    )
    ensure_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": "If True, filter duplicate prompts in each batch to ensure training quality."
        },
    )
    loss_config: LossConfig | None = field(
        default=None,
        metadata={"help": "Configuration for the loss function."},
    )
    low_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether to try to minimize memory usage."},
    )
    max_evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of evaluation steps."},
    )
    max_sequence_length: int | None = field(
        default=4096,
        metadata={"help": "The maximum sequence length."},
    )
    max_training_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of training steps."},
    )
    per_epoch_training_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of training step per each epoch."},
    )
    per_epoch_evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of evaluation step per each epoch."},
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "The name of the model."},
    )
    model_parameters: dict | None = field(
        default=None,
        metadata={"help": "Model architecture config"},
    )
    metrics_to_show_in_rich_pbar: list[str] | None = field(
        default=None,
        metadata={"help": "Metrics to display in the rich progress bar."},
    )
    num_train_epochs: int = field(
        default=10,
        metadata={"help": "The number of training epochs."},
    )
    offload_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to offload the dataset to CPU or disk."},
    )
    offload_device_type: str = field(
        default="cpu",
        metadata={"help": "The device type to offload the dataset to (cpu or disk)."},
    )
    offload_device_index: int = field(
        default=0,
        metadata={"help": "The device index to offload the dataset to."},
    )
    optimizer: AVAILABLE_OPTIMIZERS = field(
        default=EasyDeLOptimizers.ADAMW,
        metadata={"help": "The optimizer to use."},
    )
    performance_mode: bool = field(
        default=False,
        metadata={"help": "Whether to enable performance mode (e.g., XLA compilation)."},
    )
    pruning_module: tp.Any = field(
        default=None,
        metadata={"help": "The pruning module to use."},
    )
    process_zero_is_admin: bool = field(
        default=True,
        metadata={"help": "Whether the process with rank 0 is the admin process."},
    )
    progress_bar_type: tp.Literal["tqdm", "rich", "json"] = field(
        default="tqdm",
        metadata={"help": "The type of progress bar to use."},
    )
    remove_ckpt_after_load: bool = field(
        default=False,
        metadata={"help": "Whether to remove the checkpoint after loading it."},
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    report_steps: int = field(
        default=5,
        metadata={"help": "Report metrics every X steps."},
    )
    save_directory: str = field(
        default="EasyDeL-Checkpoints",
        metadata={"help": "The directory to save checkpoints to."},
    )
    save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "Whether to save the optimizer state along with the model."},
    )
    save_steps: int | None = field(
        default=None,
        metadata={"help": "Save a checkpoint every X steps."},
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={"help": "The maximum number of checkpoints to keep."},
    )
    scheduler: AVAILABLE_SCHEDULERS = field(
        default=EasyDeLSchedulers.NONE,
        metadata={"help": "The scheduler to use."},
    )
    shuffle_seed_train: int = field(
        default=64871,
        metadata={"help": "seed used for trainer dataloader shuffle."},
    )
    sparsify_module: bool = field(
        default=False,
        metadata={"help": "Whether to sparsify the model."},
    )
    sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = field(
        default="bcoo",
        metadata={"help": "The type of sparse module to use."},
    )
    state_apply_fn_kwarguments_to_model: dict | None = field(
        default=None,
        metadata={"help": "Keyword arguments to pass to the state apply function."},
    )
    step_partition_spec: PartitionSpec = field(
        default=PartitionSpec(("dp", "fsdp"), "sp"),
        metadata={"help": "The partition specification for the training step."},
    )
    step_start_point: int | None = field(
        default=None,
        metadata={"help": "The step to start training from (for resuming)."},
    )
    shuffle_train_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    total_batch_size: int = field(
        default=32,
        metadata={"help": "The total batch size."},
    )
    training_time_limit: str | None = field(
        default=None,
        metadata={"help": "The maximum training time (e.g., '1d', '2h30m')."},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "Whether to train on the input data."},
    )
    trainer_prefix: str | None = field(
        default=None,
        metadata={"help": "default prefix name for trainer."},
    )
    truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={"help": "The truncation mode to use."},
    )
    tx_mu_dtype: jnp.dtype | None = field(
        default=None,
        metadata={"help": "The dtype to use for the `tx.mu` variable."},
    )
    track_memory: bool | float = field(
        default=False,
        metadata={"help": "Whether to track memory usage. If a float, it sets the memory tracking interval in seconds."},
    )
    use_data_collactor: bool = field(
        default=True,
        metadata={"help": "Whether to use a data collator."},
    )
    use_grain: bool = field(
        default=True,
        metadata={"help": "Whether to use grain instead of `tensorflow-datasets`."},
    )
    # Grain fine-tuning knobs to avoid multiprocessing shared-memory issues
    grain_worker_count: int = field(
        default=1,
        metadata={
            "help": "Grain DataLoader worker processes per host. Set to 1 to avoid shared-memory races on TPU."
        },
    )
    grain_worker_buffer_size: int = field(
        default=1,
        metadata={
            "help": "Per-worker in-flight buffer size. Keep small to reduce SHM footprint."
        },
    )
    grain_prefetch_buffer_size: int = field(
        default=64,
        metadata={
            "help": "Host prefetch buffer size. Lower values reduce shared memory usage."
        },
    )
    grain_read_threads: int = field(
        default=1,
        metadata={
            "help": "Threads used by Grain ReadOptions. Keep 1 to avoid contention."
        },
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to use Weights & Biases for logging."},
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print verbose output."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "The Weights & Biases entity."},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": "Optional Weights & Biases project name. If unset, defaults to EasyDeL-<trainer_prefix>-<model_name>."},
    )
    wandb_name: str | None = field(
        default=None,
        metadata={"help": "The Weights & Biases run name."},
    )
    # --- Hugging Face Hub integration ---
    hub_repo_id: str | None = field(
        default=None,
        metadata={"help": "Hugging Face Hub repository id to upload checkpoints to. If None, uses model_name."},
    )
    hub_private: bool = field(
        default=True,
        metadata={"help": "Create/upload to a private repository on Hugging Face Hub."},
    )
    push_checkpoints_to_hub: bool | None = field(
        default=None,
        metadata={"help": "If True, upload every checkpoint to Hugging Face Hub. If None, auto-enable when HF_TOKEN is set."},
    )
    hub_token_env: str = field(
        default="HF_TOKEN",
        metadata={"help": "Environment variable name to read the Hugging Face Hub token from."},
    )
    hub_path_in_repo_prefix: str = field(
        default="checkpoints",
        metadata={"help": "Path prefix inside the HF repo where checkpoints are uploaded."},
    )
    hub_keep_n_checkpoints: int | None = field(
        default=None,
        metadata={"help": "If set, keep only the most recent N checkpoints in the remote HF repo (best-effort)."},
    )
    continue_from_hf: bool = field(
        default=False,
        metadata={"help": "If True, resume training from the latest checkpoint found on Hugging Face Hub for hub_repo_id."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "The number of warmup steps for the learning rate scheduler."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay value."},
    )
    weight_distribution_pattern: str = field(
        default=r".*",
        metadata={"help": "The pattern to use to extract weight distribution."},
    )
    weight_distribution_log_steps: int = field(
        default=50,
        metadata={"help": "log weight distribution every X steps."},
    )

    _can_log_metrics: bool | None = None

    @property
    def can_log_metrics(self):
        if self._can_log_metrics is None:
            if not self.is_process_zero and not self.log_all_workers:
                return False
            return self.report_metrics
        return self._can_log_metrics

    @can_log_metrics.setter
    def can_log_metrics(self, val):
        self._can_log_metrics = val

    @property
    def offload_device(self):
        return jax.devices(self.offload_device_type)[self.offload_device_index]

    @property
    def training_time_seconds(self) -> int | None:
        if self.training_time_limit is None:
            return None
        return self._time_to_seconds(self.training_time_limit)

    @functools.cached_property
    def is_process_zero(self):
        return _safe_process_index() == 0

    def __post_init__(self):
        """
        Validates the configuration, sets up distributed training,
        initializes the optimizer, configures logging.
        This method is automatically called after the object is initialized.
        """
        self._validate_config()
        self._setup_distributed()
        self._setup_optimizer()
        self._setup_logging()
        self._ensure_variables()

    def _validate_config(self):
        """
        Performs validation checks on the provided configuration settings.
        Raises ValueError if any configuration is invalid.
        """
        assert self.gradient_accumulation_steps > 0, "`gradient_accumulation_steps` can't be lower than 1."

        if self.backend not in AVAILABLE_BACKENDS:
            raise ValueError(f"Backend {self.backend} is not recognized. Available backends: {AVAILABLE_BACKENDS}")

    def _setup_distributed(self):
        """
        Sets up JAX distributed training based on the chosen backend and sharding configuration.
        Determines the number of available devices and sets up the device mesh.
        """
        try:
            JaxDistributedConfig.initialize(self.jax_distributed_config)
        except Exception as e:
            logger.info(f"Skipping JAX distributed initialization in this process: {e}")
        
        # Auto-configure dataset sharding:
        # - If mesh dims exist, shard by effective DP (min(mesh dp, process_count)) only (replicate across FSDP)
        # - Otherwise, shard per process
        if self.grain_shard_index is None or self.grain_shard_count is None:
            proc_idx = _safe_process_index()
            proc_cnt = _safe_process_count()

            dp = None
            mesh_dims = getattr(self, "mesh_dims", None)
            if isinstance(mesh_dims, (tuple, list)) and len(mesh_dims) >= 1:
                try:
                    dp = int(mesh_dims[0])
                except Exception:
                    dp = None

            if dp and dp > 0:
                # Use effective DP based on active processes to avoid missing shards when dp>process_count
                effective_dp = max(1, min(dp, proc_cnt))
                self.grain_shard_count = effective_dp
                self.grain_shard_index = proc_idx % effective_dp
            else:
                self.grain_shard_index = proc_idx if self.grain_shard_index is None else self.grain_shard_index
                self.grain_shard_count = proc_cnt if self.grain_shard_count is None else self.grain_shard_count
            
        # Log multi-worker configuration
        if _safe_process_count() > 1:
            logger.info(
                f"Multi-worker training detected: {_safe_process_count()} workers, "
                f"this is worker {_safe_process_index()}, "
                f"grain shard {self.grain_shard_index}/{self.grain_shard_count}"
            )
        else:
            logger.info(
                f"Dataset sharding configured for {_safe_process_count()} workers: "
                f"shard_index={self.grain_shard_index}, shard_count={self.grain_shard_count}"
            )

    def _setup_optimizer(self):
        """
        Configures the optimizer and learning rate scheduler based on the provided arguments.
        Sets up the optimizer_kwargs dictionary.
        """
        extra_optimizer_kwargs = self.extra_optimizer_kwargs if self.extra_optimizer_kwargs is not None else {}
        self.optimizer_kwargs = {
            "learning_rate": self.learning_rate,
            "learning_rate_end": self.learning_rate_end,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_decay": self.weight_decay,
            "steps": self.max_training_steps,
            "clip_grad": self.clip_grad,
            "mu_dtype": self.tx_mu_dtype,
            **extra_optimizer_kwargs,
        }

    def _setup_logging(self):
        """
        Sets up logging for training using TensorBoard and Weights & Biases.
        Handles warnings if performance mode is enabled and disables WandB logging accordingly.
        """
        if self.use_wandb and self.performance_mode:
            logger.info("WandB logging disabled due to performance mode")
            self.use_wandb = False
        if self.report_metrics and self.performance_mode:
            logger.info("Metrics reporting disabled due to performance mode")
            self._can_log_metrics = False

    def _ensure_variables(self):
        """
        Checks and sets up variables for start.
        """
        if isinstance(self.step_partition_spec, str):
            self.step_partition_spec = eval(self.step_partition_spec)
        elif not isinstance(self.step_partition_spec, PartitionSpec):
            self.step_partition_spec = PartitionSpec(*tuple(self.step_partition_spec))

        self.step_start_point = self.step_start_point or 0
        self.eval_batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.total_batch_size
        if self.loss_config is None:
            self.loss_config = LossConfig()
        if isinstance(self.loss_config, dict):
            self.loss_config = LossConfig(**self.loss_config)

    @staticmethod
    def _time_to_seconds(time_str: str) -> int:
        """
        Converts a time string in the format "50min" or "23h" to seconds.

        Args:
            time_str (str): The time string to convert.

        Returns:
            int: The equivalent time in seconds.
        """
        match = re.match(r"(\d+)\s*(h|hour|hours|min|m|minutes|s|sec|seconds)", time_str.lower())
        if not match:
            raise ValueError("Invalid time format. Use `50min` for minutes, `23h` for hours, or `30s` for seconds.")
        value, unit = match.groups()
        unit_to_seconds = {
            "h": 3600,
            "hour": 3600,
            "hours": 3600,
            "min": 60,
            "m": 60,
            "minutes": 60,
            "s": 1,
            "sec": 1,
            "seconds": 1,
        }.get(unit.lower())
        return int(value) * unit_to_seconds

    def get_path(self) -> EasyPathLike:
        """
        Returns the path to the checkpoint directory.

        Returns:
            EasyPathLike: The path to the checkpoint directory.
        """
        if self.process_zero_is_admin:
            if self.is_process_zero:
                return tp.cast(EasyPathLike, EasyPath(self.save_directory) / (self.model_name or "model"))
            else:
                return tp.cast(EasyPathLike, EasyPath("/dev/null"))
        return tp.cast(EasyPathLike, EasyPath(self.save_directory) / (self.model_name or "model"))

    def ensure_checkpoint_path(self):
        """
        Creates the checkpoint directory if it doesn't exist.
        """
        if self.process_zero_is_admin:
            if self.is_process_zero:
                path = self.get_path()
                path.mkdir(parents=True, exist_ok=True)
        else:
            path = self.get_path()
            path.mkdir(parents=True, exist_ok=True)

    def get_optimizer_and_scheduler(self, steps: int | None = None):
        """
        Returns the configured optimizer and learning rate scheduler.

        Args:
            steps (tp.Optional[int]): The number of training steps.
                If not provided, uses the value from `self.optimizer_kwargs`.

        Returns:
            tuple: A tuple containing the optimizer and scheduler.
        """

        self.optimizer_kwargs["steps"] = steps or self.optimizer_kwargs["steps"]
        optimizer_kwargs = deepcopy(self.optimizer_kwargs)
        scheduler = optimizer_kwargs.pop("scheduler", None)
        if scheduler == "none":
            scheduler = None
        if scheduler == EasyDeLSchedulers.NONE:
            scheduler = None
        scheduler_config = SchedulerConfig(
            scheduler_type=scheduler,
            steps=optimizer_kwargs.pop("steps"),
            learning_rate=optimizer_kwargs.pop("learning_rate"),
            learning_rate_end=optimizer_kwargs.pop("learning_rate_end"),
            warmup_steps=optimizer_kwargs.pop("warmup_steps"),
            exponent=optimizer_kwargs.pop("exponent", 1),
        )
        optimizer_kwargs.pop("gradient_accumulation_steps", 0)
        optimizer, scheduler = OptimizerFactory.create(
            optimizer_type=optimizer_kwargs.pop("optimizer"),
            scheduler_config=scheduler_config,
            clip_grad=optimizer_kwargs.pop("clip_grad"),
            weight_decay=optimizer_kwargs.pop("weight_decay"),
            custom_scheduler=self.custom_scheduler,
            **optimizer_kwargs,
        )
        return optimizer, scheduler

    def get_streaming_checkpointer(self):
        """
        Returns the checkpoint manager, responsible for saving model checkpoints.

        Returns:
            CheckpointManager: The checkpoint manager.
        """

        from easydel.utils.checkpoint_managers import CheckpointManager

        return CheckpointManager(
            checkpoint_dir=self.get_path(),
            save_optimizer_state=self.save_optimizer_state,
            verbose=self.verbose,
        )

    @functools.cached_property
    def _tensorboard(self):
        from flax.metrics.tensorboard import SummaryWriter  # type:ignore

        path = self._get_save_directory(create=True)
        if path is None:
            return None
        if str(path).startswith("gs://"):
            return None
        return SummaryWriter(log_dir=str(path))

    def get_tensorboard(self) -> SummaryWriter | None:
        """
        Returns the TensorBoard SummaryWriter, used for logging metrics.

        Returns:
            flax.metrics.tensorboard.SummaryWriter: The TensorBoard SummaryWriter.
        """
        try:
            return self._tensorboard
        except ModuleNotFoundError:
            return None

    def get_wandb_init(self):
        """
        Initializes Weights & Biases for experiment tracking if enabled.

        Returns:
            tp.Optional[wandb.sdk.wandb_run.Run]: The WandB run object if initialized, else None.
        """
        if self.can_log_metrics:
            if not self.use_wandb or wandb is None:
                warnings.warn(
                    "you have used `use_wandb=True` but you haven't install wandb.",
                    stacklevel=1,
                )
                return None
            wandb_name = self.wandb_name
            prefix = self.trainer_prefix
            if prefix is None:
                prefix = ""
            else:
                prefix = "-" + prefix
            if wandb_name is None:
                _time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                wandb_name = f"{self.model_name.lower()}-{_time}"

            project_name = self.wandb_project or f"EasyDeL{prefix}-{self.model_name}"

            return wandb.init(
                project=project_name,
                config=self.to_dict(),
                save_code=True,
                name=wandb_name,
                tags=["EasyDeL", "Jax", "Train", "LLM", "VLM"],
                entity=self.wandb_entity,
            )
        return None

    def ensure_training_time_limit(self, time_passed):
        if self.training_time_limit is not None and time_passed > self._time_to_seconds(self.training_time_limit):
            raise EasyDeLTimerError("Time Out")

    def log_metrics(
        self,
        metrics: MetricsType,
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Logs training metrics to Weights & Biases and/or TensorBoard.

        Args:
          metrics (tp.Dict[str, tp.Union[float, tp.List, tp.Tuple, np.ndarray, 'jnp.ndarray', 'torch.Tensor']]):
            A dictionary where keys are metric names and values are metric values.
          step (int): The current training step or iteration.
        """

        if self.can_log_metrics:
            filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
            metrics = {self._restructure_metric_name(k): get_safe_arr(v) for k, v in filtered_metrics.items()}
            self._log_to_wandb(metrics, step, log_as)
            self._log_to_tensorboard(metrics, step, log_as)

    def _restructure_metric_name(self, metric_name: str) -> str:
        """
        Restructures the metric name for logging.

        Args:
          metric_name (str): The original metric name.

        Returns:
          str: The restructured metric name.
        """
        if metric_name.startswith("train/grad_norm/"):
            return metric_name.replace("train/grad_norm/", "grad_norm/")
        return metric_name

    def log_weight_distribution(self, state, step: int):
        """Log weight distribution histograms and statistics.

        Args:
            state: Model state containing parameters
            step: Current training step
        """
        if self.weight_distribution_log_steps > 0 and ((step % self.weight_distribution_log_steps) == 0):
            stats = compute_weight_stats(state.graphstate, self.weight_distribution_pattern)

            # Best-effort global aggregation; guard behind log_global and swallow issues to avoid TPU divergence
            if getattr(self, "log_global", False):
                try:
                    stats = jax.experimental.multihost_utils.process_allgather(stats)
                except Exception:
                    try:
                        if _safe_process_index() == 0:
                            print("WARN: process_allgather failed for weight stats; continuing with local stats")
                    except Exception:
                        ...

            metrics = {}
            for key, histogram in stats.items():
                try:
                    if isinstance(histogram, MetricsHistogram):
                        path = key.replace("/", ".")
                        metrics[f"weights-histogram/{path}"] = (
                            histogram.bin_counts.tolist(),
                            histogram.bin_edges.tolist(),
                        )

                        base_path = path.replace("/histogram", "")
                        metrics[f"weights-information/{base_path}/mean"] = float(histogram.mean)
                        metrics[f"weights-information/{base_path}/std"] = histogram.std.item()
                        metrics[f"weights-information/{base_path}/min"] = histogram.min.item()
                        metrics[f"weights-information/{base_path}/max"] = histogram.max.item()
                    else:
                        path = key.replace("/", ".")
                        metrics[f"weights-information/{path}"] = histogram
                except Exception as e:
                    traceback.print_exc()
                    raise e
            self.log_metrics(metrics, step)

    def _log_to_wandb(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to Weights & Biases (wandb).

        This method processes the given metrics and logs them to wandb if it's enabled and properly initialized.

        Args:
            metrics: A dictionary of metrics to log. Keys are metric names, values are the metric values.
            step: The current step or iteration number.
            log_as: How to log the metrics
                (None for regular logging, "summary" for wandb.summary, "config" for wandb.config)
        """
        if self.use_wandb and wandb is not None:
            if log_as == "summary":
                wandb.summary.update(metrics)
            elif log_as == "config":
                wandb.config.update(metrics)
            else:
                wandb_metrics = {}
                for key, value in metrics.items():
                    try:
                        if isinstance(value, tuple) and len(value) == 2:
                            bin_counts, bin_edges = value
                            if isinstance(bin_counts, list | jax.Array) and isinstance(bin_edges, list | jax.Array):
                                bin_counts = np.array(bin_counts).reshape(-1)
                                bin_edges = np.array(bin_edges).reshape(-1)
                                np_histogram = (bin_counts, bin_edges)

                                wandb_metrics[key] = wandb.Histogram(np_histogram=np_histogram)
                                continue

                        wandb_metrics[key] = (
                            self._create_wandb_histogram(value)
                            if isinstance(value, float | int | list | tuple | np.generic | jax.Array)
                            else value
                        )

                    except Exception as e:
                        warnings.warn(f"Failed to log metric {key} to wandb: {e}", stacklevel=3)

                wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}

                try:
                    wandb.log(wandb_metrics, step=step)
                except Exception as e:
                    warnings.warn(f"Failed to log metrics to wandb: {e}", stacklevel=3)

    def _log_to_tensorboard(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: A dictionary of metrics to log
            step: The current step or iteration number
            log_as: Currently not used for TensorBoard
        """
        summary_writer = self.get_tensorboard()
        if summary_writer is not None:
            for key, value in metrics.items():
                try:
                    if isinstance(value, float | int):
                        summary_writer.scalar(key, value, step)
                    elif isinstance(value, tuple) and len(value) == 2:
                        bin_counts, bin_edges = value
                        if isinstance(bin_counts, list | jax.Array) and isinstance(bin_edges, list | jax.Array):
                            bin_counts = np.array(bin_counts)
                            bin_edges = np.array(bin_edges)
                            values = []
                            for i, count in enumerate(bin_counts):
                                if i < len(bin_edges) - 1:
                                    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                                    values.extend([bin_center] * int(count))

                            if values:
                                summary_writer.histogram(key, np.array(values), step)
                    elif isinstance(value, list | np.ndarray | jnp.ndarray):
                        summary_writer.histogram(key, np.array(value), step)
                except Exception as e:
                    warnings.warn(f"Failed to log metric {key} to TensorBoard: {e}", stacklevel=1)
                finally:
                    summary_writer.flush()

    def _create_wandb_histogram(self, value):
        """
        Create a wandb.Histogram object from the given value.

        Args:
            value: The value to convert into a wandb.Histogram

        Returns:
            wandb.Histogram or None: A wandb.Histogram object if successful, None if an error occurs
        """
        try:
            if isinstance(value, jax.Array | np.generic):
                value = np.array(jax.device_get(value))
                if value.dtype in [np.bfloat16]:
                    value = value.astype(np.float32)
                value = value.astype(np.float16)
                return wandb.Histogram(value)
            return value
        except Exception as e:
            warnings.warn(f"Failed to create wandb histogram: {e}", stacklevel=1)
            return None

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        return json.loads(EasyPath(json_file).read_text())

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        config_dict["trainer_config_class"] = self.__class__.__name__
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_arguments(cls, json_file: str | os.PathLike):
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """

        config_dict = cls._dict_from_json_file(json_file)
        return cls.load_from_json(config_dict)

    @classmethod
    def load_from_json(cls, config_dict):
        if "trainer_config_class" in config_dict.keys():
            import easydel as ed

            cls = getattr(ed, config_dict.pop("trainer_config_class"))
            assert cls is not None, "We couldn't clearify the trainer config class from provided json."
        return cls(**config_dict)

    def save_arguments(self, json_file_path: str | os.PathLike | EasyPathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        EasyPath(json_file_path).write_text(self.to_json_string())

    def _get_save_directory(self, create: bool = True) -> EasyPathLike | None:
        if self.process_zero_is_admin and not self.is_process_zero:
            return None
        if create:
            self.ensure_checkpoint_path()
        return self.get_path()

    def _get_save_directory_milestone(self, step, create: bool = True) -> EasyPathLike:
        directory_name = f"run-{step}"
        savedir = self._get_save_directory(create=create)
        if savedir is None:
            return tp.cast(EasyPathLike, EasyPath("/dev/null"))
        save_directory = savedir / directory_name
        if create:
            save_directory.mkdir(exist_ok=True, parents=True)
        return save_directory

    __hash__ = hash_fn
