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

from .grpo_config import GRPOConfig


@auto_pytree
class DAPOConfig(GRPOConfig):
    """
    Configuration class for the DAPO (Diversity-Aware Policy Optimization) Trainer.
    
    DAPO improves upon GRPO with four key components:
    1. Asymmetric clipping (Clip-Higher) to prevent entropy collapse
    2. Dynamic sampling to avoid gradient vanishing
    3. Token-level policy gradient loss to prevent length bias
    4. Overlong reward shaping for graduated soft punishment
    """

    trainer_prefix: str | None = field(
        default="dapotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    
    # Asymmetric clipping parameters (Clip-Higher)
    clip_ratio_low: float = field(
        default=0.2,
        metadata={"help": "Lower clipping ratio (ε_low = 0.2), same as GRPO."},
    )
    clip_ratio_high: float = field(
        default=0.28,
        metadata={"help": "Higher clipping ratio (ε_high = 0.28) for asymmetric clipping."},
    )
    
    # Dynamic sampling parameters
    enable_dynamic_sampling: bool = field(
        default=True,
        metadata={"help": "Enable dynamic resampling when all rewards are uniform."},
    )
    max_resample_attempts: int = field(
        default=3,
        metadata={"help": "Maximum number of resampling attempts for dynamic sampling."},
    )
    min_accuracy_variance: float = field(
        default=0.1,
        metadata={"help": "Minimum variance threshold for dynamic sampling."},
    )
    
    # Token-level loss parameters
    use_token_level_loss: bool = field(
        default=True,
        metadata={"help": "Use token-level policy gradient loss to prevent length bias."},
    )
    
    # Overlong reward shaping parameters
    enable_overlong_reward_shaping: bool = field(
        default=True,
        metadata={"help": "Enable graduated soft punishment for overlong sequences."},
    )
    overlong_buffer_length: int = field(
        default=4096,
        metadata={"help": "Buffer length for soft punishment (default 4096 tokens)."},
    )
    overlong_penalty_scale: float = field(
        default=0.1,
        metadata={"help": "Scale factor for overlong penalty."},
    )
    
    # Remove KL divergence as in DAPO (beta=0.0 by default)
    beta: float = field(
        default=0.0,
        metadata={"help": "The beta parameter for KL penalty. DAPO typically uses 0.0."},
    )

    def __post_init__(self):
        """Post initialization to set dependent parameters."""
        super().__post_init__()
        
        # Validate DAPO-specific parameters
        if self.clip_ratio_high <= self.clip_ratio_low:
            raise ValueError(f"clip_ratio_high ({self.clip_ratio_high}) must be greater than clip_ratio_low ({self.clip_ratio_low})")
        
        if self.overlong_buffer_length <= 0:
            raise ValueError(f"overlong_buffer_length must be positive, got {self.overlong_buffer_length}")

    __hash__ = hash_fn 