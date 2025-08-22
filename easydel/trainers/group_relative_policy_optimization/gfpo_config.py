# gfpo_config.py

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from .grpo_config import GRPOConfig


def enforce_gfpo_constraints(cfg: "GFPOConfig") -> None:
    """Shared validation logic for GFPO-style configs.

    Ensures sensible minima and relationships between G (group size) and k (retain count),
    and normalizes generation count to match group size.
    """
    try:
        print(f"DEBUG: Enforcing GFPO constraints - group_size={getattr(cfg, 'gfpo_group_size', None)}, retain_count={getattr(cfg, 'gfpo_retain_count', None)}")
        
        if getattr(cfg, "gfpo_group_size", None) is None:
            raise ValueError("gfpo_group_size must be set")
        if getattr(cfg, "gfpo_retain_count", None) is None:
            raise ValueError("gfpo_retain_count must be set")

        # Hard minima to avoid degenerate training
        if int(cfg.gfpo_group_size) < 2:
            raise ValueError(f"gfpo_group_size must be >= 2, got {cfg.gfpo_group_size}")
        if int(cfg.gfpo_retain_count) < 2:
            raise ValueError(
                f"gfpo_retain_count must be >= 2 (k=1 makes standardized advantages undefined), got {cfg.gfpo_retain_count}"
            )

        # Relationship constraints
        if int(cfg.gfpo_group_size) < int(cfg.gfpo_retain_count):
            raise ValueError(
                f"gfpo_group_size ({cfg.gfpo_group_size}) must be >= gfpo_retain_count ({cfg.gfpo_retain_count})"
            )

        # Supported metrics
        if getattr(cfg, "gfpo_metric", None) not in ["length", "token_efficiency"]:
            raise ValueError(
                f"gfpo_metric must be one of ['length', 'token_efficiency'], got {getattr(cfg, 'gfpo_metric', None)}"
            )

        # Ensure we generate G completions per prompt
        try:
            cfg.num_return_sequences = int(cfg.gfpo_group_size)
            print(f"DEBUG: Set num_return_sequences={cfg.num_return_sequences}")
        except Exception as e:
            print(f"DEBUG: Failed to set num_return_sequences: {e}")
            cfg.num_return_sequences = cfg.gfpo_group_size
        
        print("DEBUG: GFPO constraints enforced successfully")
    except Exception as e:
        print(f"DEBUG: Failed to enforce GFPO constraints: {e}")
        raise


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
        try:
            print(f"DEBUG: GFPOConfig post_init - gfpo_group_size={self.gfpo_group_size}, gfpo_retain_count={self.gfpo_retain_count}")
            super().__post_init__()
            enforce_gfpo_constraints(self)
            print("DEBUG: GFPOConfig post_init completed successfully")
        except Exception as e:
            print(f"DEBUG: GFPOConfig post_init failed: {e}")
            raise

    __hash__ = hash_fn


