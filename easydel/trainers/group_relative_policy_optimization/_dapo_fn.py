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


def apply_overlong_reward_shaping(
    rewards: jnp.ndarray,
    completion_lengths: jnp.ndarray,
    max_length: int,
    buffer_length: int = 4096,
    penalty_scale: float = 0.1,
) -> jnp.ndarray:
    """
    Apply graduated soft punishment for overlong sequences instead of hard truncation.
    
    Args:
        rewards: Original rewards [batch_size]
        completion_lengths: Length of each completion [batch_size]
        max_length: Maximum allowed length
        buffer_length: Buffer zone for soft punishment
        penalty_scale: Scale factor for penalty
    
    Returns:
        Shaped rewards with soft punishment for overlong sequences
    """
    # Calculate how much each sequence exceeds the max length
    excess_length = jnp.maximum(0, completion_lengths - max_length)
    
    # Apply graduated penalty within buffer zone
    penalty = jnp.minimum(excess_length / buffer_length, 1.0) * penalty_scale
    
    # Apply soft punishment
    shaped_rewards = rewards - penalty
    
    return shaped_rewards


def dapo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    use_token_level_loss: bool = True,
    enable_overlong_reward_shaping: bool = True,
    overlong_buffer_length: int = 4096,
    overlong_penalty_scale: float = 0.1,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
# Returns state & metrics in training mode, metrics only in eval mode.
) -> tp.Union[tuple[EasyDeLState, LossMetrics], LossMetrics]:
    """
    DAPO training step with four key improvements over GRPO:
    1. Asymmetric clipping (Clip-Higher) 
    2. Token-level policy gradient loss
    3. Overlong reward shaping
    4. Support for dynamic sampling (handled in trainer)
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
        
        # Compute probability ratios
        log_ratio = per_token_logps - ref_per_token_logps
        ratio = jnp.exp(log_ratio)
        
        # Asymmetric clipping (Clip-Higher) - key DAPO innovation
        # Allows greater exploration for low-probability tokens while maintaining stability
        clipped_ratio = jnp.clip(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
        
        # Apply overlong reward shaping if enabled
        if enable_overlong_reward_shaping:
            completion_lengths = jnp.sum(completion_mask, axis=1)
            max_length = completion_ids.shape[-1]
            shaped_advantages = apply_overlong_reward_shaping(
                advantages,
                completion_lengths,
                max_length,
                overlong_buffer_length,
                overlong_penalty_scale,
            )
        else:
            shaped_advantages = advantages

        # Compute policy gradient loss terms
        pg_loss1 = -shaped_advantages[:, None] * ratio
        pg_loss2 = -shaped_advantages[:, None] * clipped_ratio
        per_token_loss = jnp.maximum(pg_loss1, pg_loss2)
        
        # KL divergence computation (typically beta=0 in DAPO)
        per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Add KL penalty
        per_token_loss = per_token_loss + beta * per_token_kl
        
        if use_token_level_loss:
            # Token-level loss: Equal weighting across all tokens regardless of sequence length
            # This prevents length bias where longer sequences get diluted learning signals
            total_tokens = jnp.sum(completion_mask)
            loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(total_tokens, 1.0)
        else:
            # Original GRPO sample-level loss
            comps = jnp.sum(completion_mask, axis=1)
            loss = jnp.mean(jnp.sum(per_token_loss * completion_mask, axis=1) / jnp.maximum(comps, 1.0))
        
        # Compute metrics
        mean_kl = jnp.mean(jnp.sum(per_token_kl * completion_mask, axis=1) / jnp.maximum(jnp.sum(completion_mask, axis=1), 1.0))
        mean_ratio = jnp.mean(ratio)
        clipped_fraction = jnp.mean((jnp.abs(ratio - clipped_ratio) > 1e-6).astype(jnp.float32))
        
        # Distributional comparison of policy vs reference (sequence-averaged logprobs)
        policy_seq_logps = jnp.sum(per_token_logps * completion_mask, axis=1)
        ref_seq_logps = jnp.sum(ref_per_token_logps * completion_mask, axis=1)
        
        dist_stats = {
            "dist/policy_seq_logps_mean": jnp.mean(policy_seq_logps),
            "dist/policy_seq_logps_std": jnp.std(policy_seq_logps),
            "dist/ref_seq_logps_mean": jnp.mean(ref_seq_logps),
            "dist/ref_seq_logps_std": jnp.std(ref_seq_logps),
            "dist/logp_diff_mean": jnp.mean(policy_seq_logps - ref_seq_logps),
            "dist/logp_diff_std": jnp.std(policy_seq_logps - ref_seq_logps),
            "dist/per_token_logps_mean": jnp.mean(per_token_logps),
            "dist/per_token_kl_mean": jnp.mean(per_token_kl * completion_mask) / jnp.mean(completion_mask),
        }

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "mean_kl": mean_kl,
                "mean_ratio": mean_ratio,
                "clipped_fraction": clipped_fraction,
                "ref_per_token_logps": jnp.mean(ref_per_token_logps),
                "advantages_mean": jnp.mean(shaped_advantages),
                "advantage_median_abs": jnp.median(jnp.abs(shaped_advantages)),
                "advantage_95th_percentile_abs": jnp.percentile(jnp.abs(shaped_advantages), 95),
                "clip_ratio_low": clip_ratio_low,
                "clip_ratio_high": clip_ratio_high,
                **dist_stats,
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
    else:  # evaluation ⇒ state is unchanged, only metrics are needed
        _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics  # type: ignore[return-value] (Union return)


def check_batch_diversity(batch: tp.Mapping[str, jax.Array], num_generations: int, min_variance: float = 0.1) -> bool:
    """
    Check if batch has sufficient reward diversity for training.
    Part of DAPO's dynamic sampling strategy.
    
    Args:
        batch: Training batch containing rewards/advantages
        num_generations: Number of generations per prompt
        min_variance: Minimum variance threshold
    
    Returns:
        True if batch has sufficient diversity, False otherwise
    """
    if "advantages" not in batch:
        return True  # Skip check if no advantages available
    
    advantages = batch["advantages"]
    
    # Reshape to group by prompt
    advantages_grouped = advantages.reshape(-1, num_generations)
    
    # Check if any group has all same rewards (accuracy 0 or 1)
    group_variances = jnp.var(advantages_grouped, axis=1)
    
    # Return True if all groups have sufficient variance
    # Add small epsilon to avoid FP equality issues when variance is exactly zero
    return bool(jnp.all(group_variances >= (min_variance - 1e-6))) 