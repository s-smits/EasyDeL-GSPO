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


def gspo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    importance_sampling_level: str = "sequence",
    epsilon: float = 0.2,
    clip_in_log_space: bool = True,
    log_clip_epsilon: float = 0.2,
    iw_norm_mode: str = "mean",  # none|mean|ess
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
# Same mixed return convention as the other helpers
) -> tp.Union[tuple[EasyDeLState, LossMetrics], LossMetrics]:
    """
    GSPO (Group Sequence Policy Optimization) training step.
    
    Key difference from GRPO: Computes importance sampling weights at the sequence level
    instead of per-token, leading to more stable training for sequence-level rewards.
    
    Args:
        state: The current training state
        batch: Training batch containing prompts and completions
        num_generations: Number of generations per prompt
        beta: KL penalty coefficient
        importance_sampling_level: "token" or "sequence" level importance sampling
        epsilon: Clipping epsilon for PPO-style objective
        loss_config: Loss configuration
        learning_rate_fn: Learning rate schedule
        partition_spec: Partitioning specification for distributed training
        gradient_accumulation_steps: Number of gradient accumulation steps
        is_training: Whether in training mode
    
    Returns:
        Updated state and loss metrics
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

        # Repeat prompts to match completions if needed so leading dims align
        if prompt_ids.shape[0] != completion_ids.shape[0]:
            repeat_factor = completion_ids.shape[0] // prompt_ids.shape[0]
            prompt_ids_rep = prompt_ids.repeat(repeat_factor, 0)
            prompt_mask_rep = prompt_mask.repeat(repeat_factor, 0)
        else:
            prompt_ids_rep = prompt_ids
            prompt_mask_rep = prompt_mask

        input_ids = jnp.concatenate([prompt_ids_rep, completion_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_mask_rep, completion_mask], axis=1)

        per_token_logps = get_per_token_logps(module, input_ids, attention_mask, prompt_ids.shape[-1])
        ref_per_token_logps = minibatch["ref_per_token_logps"]
        
        # Compute log ratios at token level
        log_ratio = per_token_logps - ref_per_token_logps
        
        # Optional per-sequence lengths (precomputed) to avoid recomputing
        lengths = minibatch.get("completion_lengths")
        if lengths is None:
            lengths = jnp.maximum(jnp.sum(completion_mask, axis=1), 1.0)
        else:
            lengths = jnp.maximum(lengths, 1.0)

        # Determine effective epsilon (can be overridden per-minibatch without recompiles)
        _eps_scale = minibatch.get("epsilon_scale", None)
        if _eps_scale is not None:
            eps_eff = jnp.asarray(_eps_scale, dtype=jnp.float32)
        else:
            eps_eff = jnp.asarray(epsilon, dtype=jnp.float32)

        # GSPO: Compute importance sampling weights based on specified level
        if importance_sampling_level == "token":
            # Standard GRPO: per-token importance weights
            log_importance_weights = log_ratio
            ratio = jnp.exp(log_importance_weights)
            clipped_ratio = jnp.clip(ratio, 1 - eps_eff, 1 + eps_eff)
            clipped_frac_mask = (jnp.abs(ratio - clipped_ratio) > 1e-6) * completion_mask
            clipfrac = jnp.mean(clipped_frac_mask.astype(jnp.float32))
        elif importance_sampling_level == "sequence":
            # GSPO: sequence-level importance weights
            # Average log ratios across valid tokens to get single weight per sequence
            seq_log_ratios = (log_ratio * completion_mask).sum(axis=1) / lengths
            # Optional log-space clipping for stability (prevents extreme ratios before exp)
            if clip_in_log_space:
                seq_log_ratios = jnp.clip(seq_log_ratios, -log_clip_epsilon, log_clip_epsilon)

            # Stage 1: exponentiate to ratios
            B = advantages.shape[0] // int(num_generations)
            G = int(num_generations)
            seq_lr_grouped = jnp.reshape(seq_log_ratios, (B, G))
            w = jnp.exp(seq_lr_grouped)
            if iw_norm_mode == "mean":
                w = w / jnp.maximum(jnp.mean(w, axis=1, keepdims=True), 1e-8)
            elif iw_norm_mode == "ess":
                s1 = jnp.sum(w, axis=1, keepdims=True)
                s2 = jnp.sum(w * w, axis=1, keepdims=True)
                ess = (s1 * s1) / jnp.maximum(s2, 1e-8)
                scale = jnp.sqrt(ess / float(G))
                w = (w / jnp.maximum(jnp.mean(w, axis=1, keepdims=True), 1e-8)) * scale
            else:
                # no normalization
                pass
            ratio = jnp.reshape(w, (-1, 1))
            # Stage 2: ratio clipping regardless of log clipping
            clipped_ratio = jnp.clip(ratio, 1 - eps_eff, 1 + eps_eff)
            # Clipping fraction: max of log-clip hits and ratio-clip hits
            clip_hits_log = (jnp.abs(seq_log_ratios) >= log_clip_epsilon).astype(jnp.float32) if clip_in_log_space else jnp.zeros_like(seq_log_ratios, dtype=jnp.float32)
            clip_hits_ratio = (jnp.abs(ratio - clipped_ratio) > 1e-6).astype(jnp.float32)
            clipfrac = jnp.maximum(jnp.mean(clip_hits_log), jnp.mean(clip_hits_ratio))
        else:
            raise ValueError(f"Unknown importance_sampling_level: {importance_sampling_level}")
        
        # Compute policy gradient loss
        pg_loss1 = -advantages[:, None] * ratio
        pg_loss2 = -advantages[:, None] * clipped_ratio
        per_token_loss = jnp.maximum(pg_loss1, pg_loss2)
        
        # KL divergence computation (same as GRPO)
        per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Optional selection weights (for GFPO/GFSPO variants); shape (B,)
        sel_w = minibatch.get("selection_weights")
        if sel_w is not None:
            sel_w = sel_w.astype(jnp.float32)
            sel_w_bcast = sel_w[:, None]
        else:
            sel_w_bcast = 1.0

        # Differential KL penalty: beta for selected vs off-selected
        beta_sel_scale = minibatch.get("beta_sel_scale")
        beta_off_scale = minibatch.get("beta_off_scale")
        if beta_sel_scale is None:
            beta_sel_scale = 1.0
        if beta_off_scale is None:
            beta_off_scale = 1.0
        beta_vec = beta * (beta_sel_scale * sel_w_bcast + beta_off_scale * (1.0 - sel_w_bcast))
        per_token_loss = per_token_loss + beta_vec * per_token_kl

        # Weighted compute by selection_weights and completion_mask
        weighted_mask = completion_mask * sel_w_bcast
        eff_comps = jnp.sum(weighted_mask, axis=1)

        # Compute loss
        loss = jnp.mean(jnp.sum(per_token_loss * weighted_mask, axis=1) / jnp.maximum(eff_comps, 1.0))

        # Compute metrics
        mean_kl = jnp.mean(jnp.sum(per_token_kl * weighted_mask, axis=1) / jnp.maximum(eff_comps, 1.0))
        
        # Clipping metrics computation
        if importance_sampling_level == "sequence":
            clipped_fraction = clipfrac
            mean_ratio = jnp.mean(ratio[:, 0])
        else:
            clipped_fraction = jnp.mean(((jnp.abs(ratio - clipped_ratio) > 1e-6) * completion_mask).astype(jnp.float32))
            mean_ratio = jnp.mean(ratio * completion_mask) / jnp.mean(completion_mask)

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

        # Compute advantage statistics for progress bar
        advantage_median_abs = jnp.median(jnp.abs(advantages))
        advantage_95th_percentile_abs = jnp.percentile(jnp.abs(advantages), 95)

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "mean_ratio": mean_ratio,
                "clipped_fraction": clipped_fraction,
                "advantages_mean": jnp.mean(advantages),
                "advantage_median_abs": advantage_median_abs,
                "advantage_95th_percentile_abs": advantage_95th_percentile_abs,
                # Convert string to numeric value for JAX compatibility
                "importance_sampling_level_seq": jnp.float32(1.0 if importance_sampling_level == "sequence" else 0.0),
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
        return metrics  # type: ignore[return-value] 
