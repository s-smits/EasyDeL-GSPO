"""GFSPO config: GSPO + GFPO controls, kept minimal and safe by default."""

from __future__ import annotations

from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from .gspo_config import GSPOConfig
from .gfpo_config import enforce_gfpo_constraints


@auto_pytree
class GFSPOConfig(GSPOConfig):
    trainer_prefix: str | None = field(
        default="gfspotrainer",
        metadata={"help": "default prefix name for trainer."},
    )

    # GFPO parameters
    gfpo_group_size: int = field(
        default=16,
        metadata={"help": "Total number of responses (G) per prompt each step."},
    )
    gfpo_retain_count: int = field(
        default=8,
        metadata={"help": "Number of responses (k) to retain per prompt; must be < G."},
    )
    gfpo_metric: str = field(
        default="length",
        metadata={"help": "'length' (shorter) or 'token_efficiency' (reward/length)."},
    )
    gfpo_adaptive: bool = field(
        default=True,
        metadata={"help": "Enable adaptive k via difficulty percentiles."},
    )
    gfpo_adaptive_warmup_steps: int = field(
        default=10,
        metadata={"help": "Warmup steps with fixed k before adapting."},
    )
    gfpo_adaptive_k_map: dict[str, int] = field(
        default_factory=lambda: {"very_hard": 8, "hard": 8, "medium": 6, "easy": 4},
        metadata={"help": "Bucket→k mapping used after warmup."},
    )
    gfpo_adaptive_history_max: int = field(
        default=20000,
        metadata={"help": "Max history length for rolling difficulty buffer."},
    )

    # Soft masking + shrinkage controls
    gfpo_soft_mask: bool = field(
        default=False,
        metadata={"help": "Use soft mask in [0,1] that sums to k per prompt."},
    )
    gfpo_soft_temperature: float = field(
        default=0.5,
        metadata={"help": "Temperature for soft mask (smaller=sharper)."},
    )
    gfpo_shrinkage_alpha: float = field(
        default=0.5,
        metadata={"help": "Shrinkage strength toward full-group stats (0..1)."},
    )
    gfpo_sigma_floor_c: float = field(
        default=0.25,
        metadata={"help": "Variance floor parameter; adds (c^2)/(n_eff-1)."},
    )

    # Metric switch threshold (optional)
    gfpo_metric_switch_clip: float = field(
        default=0.35,
        metadata={"help": "Switch to 'length' if clipped_fraction > threshold."},
    )

    # Dynamic epsilon (trust region) coupling via ESS
    use_ess_epsilon: bool = field(
        default=False,
        metadata={"help": "If True, scale epsilon by (ESS/G)^power using previous-step ESS."},
    )
    epsilon_ess_power: float = field(
        default=0.5,
        metadata={"help": "Power applied to normalized ESS when scaling epsilon (0.5 => sqrt)."},
    )
    epsilon_min_scale: float = field(
        default=0.1,
        metadata={"help": "Lower bound for epsilon_scale (safety clamp)."},
    )
    epsilon_max_scale: float = field(
        default=1.0,
        metadata={"help": "Upper bound for epsilon_scale (safety clamp)."},
    )

    # KL target controller (multiplicative on beta)
    kl_target: float = field(
        default=0.02,
        metadata={"help": "Target average per-sequence KL to maintain (approx)."},
    )
    beta_update_rate: float = field(
        default=0.1,
        metadata={"help": "Controller rate eta in beta_{t+1}=beta_t*exp(eta*(KL-KL_target))."},
    )
    beta_min_scale: float = field(
        default=0.1,
        metadata={"help": "Lower clamp on beta_scale multiplier (relative to static beta)."},
    )
    beta_max_scale: float = field(
        default=10.0,
        metadata={"help": "Upper clamp on beta_scale multiplier (relative to static beta)."},
    )

    def __post_init__(self):
        try:
            print(
                f"DEBUG: GFSPOConfig post_init - gfpo_group_size={self.gfpo_group_size}, "
                f"retain={self.gfpo_retain_count}"
            )
            super().__post_init__()
            enforce_gfpo_constraints(self)

            # For GSPO, sequence-level is the recommended default
            if getattr(self, "importance_sampling_level", None) is None:
                self.importance_sampling_level = "sequence"
            # Validate soft/shrinkage controls
            if float(self.gfpo_soft_temperature) <= 0:
                raise ValueError("gfpo_soft_temperature must be > 0")
            if not (0.0 <= float(self.gfpo_shrinkage_alpha) <= 1.0):
                raise ValueError("gfpo_shrinkage_alpha must be in [0,1]")
            if int(self.gfpo_adaptive_history_max) < 100:
                raise ValueError("gfpo_adaptive_history_max must be >= 100")
            if float(self.epsilon_min_scale) <= 0 or float(self.epsilon_max_scale) <= 0:
                raise ValueError("epsilon_min_scale and epsilon_max_scale must be > 0")
            if float(self.epsilon_min_scale) > float(self.epsilon_max_scale):
                raise ValueError("epsilon_min_scale must be <= epsilon_max_scale")
            if float(self.beta_min_scale) <= 0 or float(self.beta_max_scale) <= 0:
                raise ValueError("beta_min_scale and beta_max_scale must be > 0")
            if float(self.beta_min_scale) > float(self.beta_max_scale):
                raise ValueError("beta_min_scale must be <= beta_max_scale")
            print("DEBUG: GFSPOConfig post_init completed successfully")
        except Exception as e:
            print(f"DEBUG: GFSPOConfig post_init failed: {e}")
            raise

    __hash__ = hash_fn


def config(**kwargs) -> GFSPOConfig:
    return GFSPOConfig(**kwargs)
