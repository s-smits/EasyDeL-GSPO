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
    # Apply per-leaf sharding constraints to avoid rank/spec mismatches across hosts
    try:
        bdim = partition_spec[0] if isinstance(partition_spec, PartitionSpec) else None
        spec_1d = PartitionSpec(bdim) if bdim is not None else PartitionSpec()
    except Exception:
        spec_1d = PartitionSpec()

    def _constrain_leaf(x):
        try:
            if hasattr(x, "ndim") and int(x.ndim) == 1:
                return with_sharding_constraint(x, spec_1d)
            else:
                return with_sharding_constraint(x, partition_spec)
        except Exception:
            return with_sharding_constraint(x, partition_spec)

    batch = jax.tree_util.tree_map(_constrain_leaf, batch)

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
            eps_eff = jnp.asarray(epsilon, dtype=jnp.float32) * jnp.asarray(_eps_scale, dtype=jnp.float32)
        else:
            eps_eff = jnp.asarray(epsilon, dtype=jnp.float32)
        # Ensure epsilon broadcasts across tokens (B, T) safely
        if hasattr(eps_eff, "ndim") and eps_eff.ndim == 1:
            eps_eff = eps_eff[:, None]

        # Optional runtime beta scaling (host-updated, avoids recompiles)
        _beta_scale = minibatch.get("beta_scale", None)
        if _beta_scale is not None:
            beta_eff = jnp.asarray(beta, dtype=jnp.float32) * jnp.asarray(_beta_scale, dtype=jnp.float32)
        else:
            beta_eff = jnp.asarray(beta, dtype=jnp.float32)
        # Ensure beta broadcasts across tokens (B, T) safely
        if hasattr(beta_eff, "ndim") and beta_eff.ndim == 1:
            beta_eff = beta_eff[:, None]

        # GSPO: Compute importance sampling weights based on specified level
        if importance_sampling_level == "token":
            # Standard GRPO: per-token importance weights
            log_importance_weights = log_ratio
            ratio = jnp.exp(log_importance_weights)
            clipped_ratio = jnp.clip(ratio, 1 - eps_eff, 1 + eps_eff)
            # Mask-aware clipping fraction: count over valid tokens only
            clipped_frac_mask = (jnp.abs(ratio - clipped_ratio) > 1e-6).astype(jnp.float32) * completion_mask
            clipfrac = jnp.sum(clipped_frac_mask) / jnp.maximum(jnp.sum(completion_mask), 1.0)
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

        # Add KL penalty (selection masking applied via weighted_mask below)
        per_token_loss = per_token_loss + beta_eff * per_token_kl

        # Weighted compute by selection_weights and completion_mask
        weighted_mask = completion_mask * sel_w_bcast
        eff_comps = jnp.sum(weighted_mask, axis=1)

        # Compute loss
        loss = jnp.mean(jnp.sum(per_token_loss * weighted_mask, axis=1) / jnp.maximum(eff_comps, 1.0))

        # Compute metrics: clipping fraction (masked for token mode) and mean ratio (masked if token mode)
        if importance_sampling_level == "token":
            clipped_fraction = clipfrac
            mean_ratio = jnp.sum(ratio * completion_mask) / jnp.maximum(jnp.sum(completion_mask), 1.0)
        else:
            clipped_fraction = jnp.mean((jnp.abs(ratio - clipped_ratio) > 1e-6).astype(jnp.float32))
            mean_ratio = jnp.mean(ratio)

        # Compute advantage statistics for progress bar
        advantage_median_abs = jnp.median(jnp.abs(advantages))
        advantage_95th_percentile_abs = jnp.percentile(jnp.abs(advantages), 95)

        # Sequence-level diagnostics: average KL per sequence and ESS (if sequence mode)
        kl_mean = jnp.array(0.0, dtype=jnp.float32)
        ess_mean = jnp.array(0.0, dtype=jnp.float32)
        if importance_sampling_level == "sequence":
            seq_kl = jnp.sum(per_token_kl * completion_mask, axis=1) / lengths
            kl_mean = jnp.mean(seq_kl)
            # Reuse grouped weights w if available, else recompute quickly
            if 'w' in locals():
                s1 = jnp.sum(w, axis=1)
                s2 = jnp.sum(w * w, axis=1)
                ess = (s1 * s1) / jnp.maximum(s2, 1e-8)
                ess_mean = jnp.mean(ess / float(G))  # normalize by G for interpretability

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "mean_ratio": mean_ratio,
                "clipped_fraction": clipped_fraction,
                "kl/mean": kl_mean,
                "ess/mean_norm": ess_mean,
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
