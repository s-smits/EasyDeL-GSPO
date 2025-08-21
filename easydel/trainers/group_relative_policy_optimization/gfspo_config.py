# gfspo_config.py

from __future__ import annotations

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from .gspo_config import GSPOConfig


@auto_pytree
class GFSPOConfig(GSPOConfig):
    """
    Combined configuration for GFSPO (GFPO + GSPO).

    - Inherits GSPO's sequence-level importance sampling controls
    - Adds GFPO's group filtering controls

    Defaults are chosen to be reasonable for both strategies.
    """

    trainer_prefix: str | None = field(
        default="gfspotrainer",
        metadata={"help": "default prefix name for trainer."},
    )

    # GFPO parameters
    gfpo_group_size: int = field(
        default=16,
        metadata={
            "help": "Total number of responses (G) to generate per prompt each step.",
        },
    )

    gfpo_retain_count: int = field(
        default=8,
        metadata={
            "help": "Number of responses (k) to retain per prompt for training. Must be <= G.",
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
        """Validate settings and set dependent parameters."""
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

        # For GSPO, sequence-level is the recommended default
        if getattr(self, "importance_sampling_level", None) is None:
            self.importance_sampling_level = "sequence"

    __hash__ = hash_fn


def config(**kwargs) -> GFSPOConfig:
    """Convenience factory for building a GFSPOConfig."""
    return GFSPOConfig(**kwargs)


