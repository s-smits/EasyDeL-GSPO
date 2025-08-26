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
        default=True,
        metadata={
            "help": "Enable adaptive k based on per-prompt difficulty (average reward).",
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

    # Weighted-shrinkage options (used by GFSPOWShrinkTrainer)
    use_wshrink: bool = field(
        default=False,
        metadata={
            "help": "Use weighted-shrinkage preprocessing (GFSPOWShrinkTrainer).",
        },
    )
    gfpo_shrinkage_alpha: float = field(
        default=0.5,
        metadata={
            "help": "Shrinkage mixing coefficient alpha in [0,1].",
        },
    )
    gfpo_sigma_floor_c: float = field(
        default=0.25,
        metadata={
            "help": "Sigma floor constant c; adds (c^2)/(n_eff-1) to variance.",
        },
    )

    # Soft mask and log-space clipping options
    gfpo_soft_mask: bool = field(
        default=False,
        metadata={
            "help": "Use temperature-softmax soft top-k mask scaled to k.",
        },
    )
    gfpo_soft_temperature: float = field(
        default=0.5,
        metadata={
            "help": "Temperature for soft top-k mask (smaller = sharper).",
        },
    )
    clip_in_log_space: bool = field(
        default=True,
        metadata={
            "help": "Clip sequence-level log-importance ratios in log space.",
        },
    )
    log_clip_epsilon: float = field(
        default=0.2,
        metadata={
            "help": "Epsilon for log-space clipping of sequence log-ratios.",
        },
    )
    importance_weight_normalization: str = field(
        default="mean",
        metadata={
            "help": "Normalization for sequence IS weights: 'none'|'mean'|'ess'",
        },
    )
    gfpo_metric_switch_clip: float = field(
        default=0.35,
        metadata={
            "help": "If clipped_fraction > this threshold, switch metric to 'length'; switch back when <0.20.",
        },
    )

    # Optional differential KL scales (selected vs off-selected). If left at 1.0, behavior matches beta.
    beta_sel_scale: float = field(
        default=1.0,
        metadata={
            "help": "Scale factor for KL penalty on selected sequences (multiplied by beta).",
        },
    )
    beta_off_scale: float = field(
        default=1.0,
        metadata={
            "help": "Scale factor for KL penalty on off-selected sequences (multiplied by beta).",
        },
    )

    def __post_init__(self):
        """Validate settings and set dependent parameters."""
        try:
            print(f"DEBUG: GFSPOConfig post_init - gfpo_group_size={self.gfpo_group_size}, importance_sampling_level={getattr(self, 'importance_sampling_level', None)}")
            super().__post_init__()
            enforce_gfpo_constraints(self)

            # For GSPO, sequence-level is the recommended default
            if getattr(self, "importance_sampling_level", None) is None:
                self.importance_sampling_level = "sequence"
                print("DEBUG: Set default importance_sampling_level=sequence")
            # Validate adaptive fields
            if not isinstance(self.gfpo_adaptive_warmup_steps, int) or self.gfpo_adaptive_warmup_steps < 0:
                raise ValueError("gfpo_adaptive_warmup_steps must be a non-negative integer")
            if not isinstance(self.gfpo_adaptive_k_map, dict):
                raise ValueError("gfpo_adaptive_k_map must be a dict")
            for _k in ["very_hard", "hard", "medium", "easy"]:
                if _k not in self.gfpo_adaptive_k_map:
                    raise ValueError(f"gfpo_adaptive_k_map missing key: {_k}")
                if int(self.gfpo_adaptive_k_map[_k]) < 1:
                    raise ValueError(f"gfpo_adaptive_k_map[{_k}] must be >= 1")
            if int(self.gfpo_adaptive_history_max) < 100:
                raise ValueError("gfpo_adaptive_history_max must be >= 100")
            
            print("DEBUG: GFSPOConfig post_init completed successfully")
        except Exception as e:
            print(f"DEBUG: GFSPOConfig post_init failed: {e}")
            raise

    __hash__ = hash_fn


def config(**kwargs) -> GFSPOConfig:
    """Convenience factory for building a GFSPOConfig."""
    return GFSPOConfig(**kwargs)
