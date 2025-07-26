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
class DRGRPOConfig(GRPOConfig):
    """
    Configuration class for the DR GRPO (GRPO Done Right) Trainer.
    
    DR GRPO corrects fundamental biases in GRPO's optimization:
    1. Eliminates length-dependent normalization to prevent length bias
    2. Removes standard deviation scaling to treat all questions equally
    3. Uses constant normalization for more stable training
    
    This implementation follows Zichen Liu's "GRPO Done Right" approach 
    that achieved 43.3% AIME 2024 accuracy with significantly less compute.
    """

    trainer_prefix: str | None = field(
        default="drgrpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    
    # DR GRPO normalization parameters
    use_constant_normalization: bool = field(
        default=True,
        metadata={"help": "Use constant normalization instead of length-dependent normalization."},
    )
    constant_normalization_factor: float = field(
        default=1.0,
        metadata={"help": "Constant normalization factor to replace sequence length normalization."},
    )
    
    # Disable standard deviation scaling
    disable_std_scaling: bool = field(
        default=True,
        metadata={"help": "Disable standard deviation scaling in advantage computation."},
    )
    advantage_epsilon: float = field(
        default=1e-4,
        metadata={"help": "Small epsilon for numerical stability in advantage computation."},
    )
    
    # DR GRPO specific optimization parameters
    use_advantage_whitening: bool = field(
        default=False,
        metadata={"help": "Apply advantage whitening for more stable training."},
    )
    whitening_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for advantage whitening numerical stability."},
    )

    def __post_init__(self):
        """Post initialization to set dependent parameters."""
        super().__post_init__()
        
        # Validate DR GRPO-specific parameters
        if self.constant_normalization_factor <= 0:
            raise ValueError(f"constant_normalization_factor must be positive, got {self.constant_normalization_factor}")
        
        if self.advantage_epsilon <= 0:
            raise ValueError(f"advantage_epsilon must be positive, got {self.advantage_epsilon}")

    __hash__ = hash_fn 