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
class GSPOConfig(GRPOConfig):
    """
    Configuration class for the GSPO (Group Sequence Policy Optimization) Trainer.
    
    GSPO modifies GRPO by computing importance sampling weights at the sequence level
    instead of per-token. This leads to more stable training when using sequence-level
    rewards, as shown in the GSPO paper (https://huggingface.co/papers/2507.18071).
    
    Key benefits:
    1. More stable training with sequence-level rewards
    2. Better alignment between optimization objective and reward signal
    3. Reduced variance in gradient estimates
    """

    trainer_prefix: str | None = field(
        default="gspotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    
    # GSPO-specific parameters
    importance_sampling_level: str = field(
        default="sequence",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the 'token' or 'sequence' level. "
            "'token' keeps the raw per-token log-probability ratios (one weight per token). 'sequence' averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. "
            "GSPO uses 'sequence' level by default."
        },
    )
    
    epsilon: float = field(
        default=0.2,
        metadata={"help": "The epsilon parameter for PPO-style clipping. Same as GRPO default."},
    )

    def __post_init__(self):
        """Post initialization to set dependent parameters."""
        super().__post_init__()
        
        # Validate GSPO-specific parameters
        if self.importance_sampling_level not in ["token", "sequence"]:
            raise ValueError(
                f"importance_sampling_level must be 'token' or 'sequence', got {self.importance_sampling_level}"
            )
        
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        
        # Note: advantage_epsilon is inherited from GRPOConfig and is critical for GSPO
        # as sequence-level rewards often have low variance within groups

    __hash__ = hash_fn 