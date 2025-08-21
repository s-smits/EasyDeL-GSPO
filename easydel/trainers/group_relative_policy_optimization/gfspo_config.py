# gfspo_config.py

from __future__ import annotations

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from .gspo_config import GSPOConfig
from .gfpo_config import enforce_gfpo_constraints


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
        enforce_gfpo_constraints(self)

        # For GSPO, sequence-level is the recommended default
        if getattr(self, "importance_sampling_level", None) is None:
            self.importance_sampling_level = "sequence"

    __hash__ = hash_fn


def config(**kwargs) -> GFSPOConfig:
    """Convenience factory for building a GFSPOConfig."""
    return GFSPOConfig(**kwargs)


