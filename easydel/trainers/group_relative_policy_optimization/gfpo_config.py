# gfpo_config.py

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from .grpo_config import GRPOConfig


@auto_pytree
class GFPOConfig(GRPOConfig):
    """
    Configuration class for the GFPO (Group Filtered Policy Optimization) Trainer.

    GFPO modifies GRPO by sampling G responses per prompt, filtering to k responses
    according to a metric (e.g., shortest length or highest reward/length), and
    computing advantages only over the retained subset. Non-retained responses
    receive zero advantage.
    """

    trainer_prefix: str | None = field(
        default="gfpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )

    # GFPO parameters
    gfpo_group_size: int = field(
        default=16,
        metadata={
            "help": "Total number of responses (G) to generate per prompt each step."
        },
    )

    gfpo_retain_count: int = field(
        default=8,
        metadata={
            "help": "Number of responses (k) to retain per prompt for training. Must be <= G."
        },
    )

    gfpo_metric: str = field(
        default="length",
        metadata={
            "help": "Filtering metric. 'length' retains shortest; 'token_efficiency' retains highest reward/length.",
        },
    )

    gfpo_adaptive: bool = field(
        default=False,
        metadata={
            "help": "Enable adaptive k based on per-prompt difficulty (average reward).",
        },
    )

    def __post_init__(self):
        """Post initialization to validate GFPO config and set generation count."""
        super().__post_init__()

        if self.gfpo_group_size < 1:
            raise ValueError("gfpo_group_size must be >= 1")
        if self.gfpo_retain_count < 1:
            raise ValueError("gfpo_retain_count must be >= 1")
        if self.gfpo_group_size < self.gfpo_retain_count:
            raise ValueError(
                f"gfpo_group_size ({self.gfpo_group_size}) must be >= gfpo_retain_count ({self.gfpo_retain_count})"
            )
        if self.gfpo_metric not in ["length", "token_efficiency"]:
            raise ValueError(
                f"gfpo_metric must be one of ['length', 'token_efficiency'], got {self.gfpo_metric}"
            )

        # Ensure we actually generate G completions per prompt
        self.num_return_sequences = int(self.gfpo_group_size)

    __hash__ = hash_fn


