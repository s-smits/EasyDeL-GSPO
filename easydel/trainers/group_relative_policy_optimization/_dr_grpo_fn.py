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

import flax
import flax.nnx
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully
from ._fn import get_per_token_logps


def compute_dr_grpo_advantages(
    rewards: jnp.ndarray,
    num_generations: int,
    use_constant_normalization: bool = True,
    constant_normalization_factor: float = 1.0,
    disable_std_scaling: bool = True,
    advantage_epsilon: float = 1e-4,
    use_advantage_whitening: bool = False,
    whitening_epsilon: float = 1e-8,
) -> jnp.ndarray:
    """
    Compute advantages with DR GRPO corrections to eliminate optimization biases.
    
    Key corrections:
    1. Eliminates length-dependent normalization (length bias)
    2. Removes standard deviation scaling (std bias)
    3. Uses constant normalization for stability
    
    Args:
        rewards: Reward values [batch_size * num_generations]
        num_generations: Number of generations per prompt
        use_constant_normalization: Use constant factor instead of length normalization
        constant_normalization_factor: Constant normalization factor
        disable_std_scaling: Disable standard deviation scaling
        advantage_epsilon: Epsilon for numerical stability
        use_advantage_whitening: Apply advantage whitening
        whitening_epsilon: Epsilon for whitening numerical stability
    
    Returns:
        Computed advantages with corrected normalization
    """
    # Reshape rewards to group by prompt
    rewards_grouped = rewards.reshape(-1, num_generations)
    
    # Compute baselines (mean reward per group)
    baselines = jnp.mean(rewards_grouped, axis=-1)
    
    # Compute raw advantages (reward - baseline)
    raw_advantages = rewards - baselines.repeat(num_generations, axis=0)
    
    if disable_std_scaling:
        # DR GRPO: No standard deviation scaling - treats all questions equally
        if use_constant_normalization:
            # Use constant normalization instead of std scaling
            advantages = raw_advantages / constant_normalization_factor
        else:
            # No scaling - leave raw advantages unchanged
            advantages = raw_advantages
    else:
        # Original GRPO behavior (for comparison/ablation)
        grouped_std = jnp.std(rewards_grouped, axis=-1)
        advantages = raw_advantages / (grouped_std.repeat(num_generations, axis=0) + advantage_epsilon)
    
    if use_advantage_whitening:
        # Optional: Apply advantage whitening for more stable training
        # Normalize advantages to have zero mean and unit variance
        advantages_mean = jnp.mean(advantages)
        advantages_std = jnp.std(advantages)
        advantages = (advantages - advantages_mean) / (advantages_std + whitening_epsilon)
    
    return advantages


def dr_grpo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    use_constant_normalization: bool = True,
    constant_normalization_factor: float = 1.0,
    disable_std_scaling: bool = True,
    advantage_epsilon: float = 1e-4,
    use_advantage_whitening: bool = False,
    whitening_epsilon: float = 1e-8,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
) -> tuple[EasyDeLState, LossMetrics]:
    """
    DR GRPO training step with corrected normalization to eliminate optimization biases.
    
    Following Zichen Liu's "GRPO Done Right" approach:
    1. Eliminates length-dependent normalization
    2. Removes standard deviation scaling
    3. Uses constant normalization for stability
    """
    # Determine batch size, minibatch size, and enforce partition spec.
    batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        module = flax.nnx.merge(state.graphdef, tree, state.graphother)

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            advantages,
        ) = (
            minibatch["prompt_ids"],
            minibatch["prompt_mask"],
            minibatch["completion_ids"],
            minibatch["completion_mask"],
            minibatch["advantages"],
        )

        input_ids = jnp.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)

        per_token_logps = get_per_token_logps(module, input_ids, attention_mask, prompt_ids.shape[-1])
        ref_per_token_logps = minibatch["ref_per_token_logps"]
        
        # KL divergence computation
        per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Policy gradient loss with DR GRPO advantage computation
        # Note: advantages are already computed with DR GRPO corrections in preprocessing
        per_token_loss = jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps)) * jnp.expand_dims(
            advantages, 1
        )
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        
        if use_constant_normalization:
            # DR GRPO: Use constant normalization instead of length-dependent normalization
            loss = jnp.mean(jnp.sum(per_token_loss * completion_mask, axis=1) / constant_normalization_factor)
        else:
            # Original GRPO behavior (for comparison)
            comps = jnp.sum(completion_mask, axis=1)
            loss = jnp.mean(jnp.sum(per_token_loss * completion_mask, axis=1) / jnp.maximum(comps, 1.0))
        
        # Compute metrics
        mean_kl = jnp.mean(jnp.sum(per_token_kl * completion_mask, axis=1) / jnp.maximum(jnp.sum(completion_mask, axis=1), 1.0))

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "mean_kl": mean_kl,
                "ref_per_token_logps": jnp.mean(ref_per_token_logps),
                "advantages": jnp.mean(advantages),
                "use_constant_normalization": float(use_constant_normalization),
                "constant_normalization_factor": constant_normalization_factor,
                "disable_std_scaling": float(disable_std_scaling),
            },
        )

    # Compute gradients and metrics across minibatches.
    if is_training:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        state = update_state_respectfully(
            state=state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=state.step,
                gradients=gradients,
            ),
        )
        return state, metrics
    else:
        _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics 