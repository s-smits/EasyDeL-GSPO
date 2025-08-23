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

        # Relationship constraints (enforce strict filtering: k < G)
        if int(cfg.gfpo_group_size) <= int(cfg.gfpo_retain_count):
            raise ValueError(
                f"gfpo_group_size ({cfg.gfpo_group_size}) must be > gfpo_retain_count ({cfg.gfpo_retain_count}) to avoid trivial full retention"
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

    # Token-efficiency stability
    gfpo_efficiency_epsilon: float = field(
        default=1e-8,
        metadata={
            "help": "Small epsilon for token efficiency division to avoid divide-by-zero.",
        },
    )

    gfpo_adaptive: bool = field(
        default=True,
        metadata={
            "help": "Enable adaptive k based on per-prompt difficulty (average reward).",
        },
    )

    # Adaptive strategy selection
    gfpo_adaptive_method: str = field(
        default="rolling",
        metadata={
            "help": "Adaptive difficulty method: 'rolling' (history percentiles) or 'ema' (running percentiles).",
        },
    )

    # Adaptive difficulty controls (Algorithm 2)
    gfpo_adaptive_warmup_steps: int = field(
        default=10,
        metadata={
            "help": "Warmup steps retaining k=8 for all prompts before using percentiles.",
        },
    )

    gfpo_adaptive_k_map: dict[str, int] = field(
        default_factory=lambda: {"very_hard": 8, "hard": 8, "medium": 6, "easy": 4},
        metadata={
            "help": "Difficulty bucket to k mapping used after warmup (Algorithm 2).",
        },
    )

    gfpo_adaptive_history_max: int = field(
        default=20000,
        metadata={
            "help": "Maximum number of historical per-prompt difficulties to retain in the rolling buffer.",
        },
    )

    # EMA config (used when gfpo_adaptive_method == 'ema')
    gfpo_adaptive_ema_alpha: float = field(
        default=0.1,
        metadata={
            "help": "Smoothing factor for EMA of running percentiles (0 < alpha <= 1).",
        },
    )

    def __post_init__(self):
        """Post initialization to validate GFPO config and set generation count."""
        try:
            print(f"DEBUG: GFPOConfig post_init - gfpo_group_size={self.gfpo_group_size}, gfpo_retain_count={self.gfpo_retain_count}")
            super().__post_init__()
            enforce_gfpo_constraints(self)
            # Validate adaptive fields
            if not isinstance(self.gfpo_adaptive_warmup_steps, int) or self.gfpo_adaptive_warmup_steps < 0:
                raise ValueError("gfpo_adaptive_warmup_steps must be a non-negative integer")
            if self.gfpo_adaptive_method not in ["rolling", "ema"]:
                raise ValueError(
                    f"gfpo_adaptive_method must be 'rolling' or 'ema', got {self.gfpo_adaptive_method}"
                )
            if not isinstance(self.gfpo_adaptive_k_map, dict):
                raise ValueError("gfpo_adaptive_k_map must be a dict")
            for _k in ["very_hard", "hard", "medium", "easy"]:
                if _k not in self.gfpo_adaptive_k_map:
                    raise ValueError(f"gfpo_adaptive_k_map missing key: {_k}")
                if int(self.gfpo_adaptive_k_map[_k]) < 1:
                    raise ValueError(f"gfpo_adaptive_k_map[{_k}] must be >= 1")
            if int(self.gfpo_adaptive_history_max) < 100:
                raise ValueError("gfpo_adaptive_history_max must be >= 100")
            if not (0 < float(self.gfpo_adaptive_ema_alpha) <= 1.0):
                raise ValueError("gfpo_adaptive_ema_alpha must be in (0, 1]")
            print("DEBUG: GFPOConfig post_init completed successfully")
        except Exception as e:
            print(f"DEBUG: GFPOConfig post_init failed: {e}")
            raise

    __hash__ = hash_fn


