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

RewardFunc = tp.Union[EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa


def _compute_quantile_grid_stats(x: jnp.ndarray, y: jnp.ndarray, num_quantiles: int = 41) -> dict:
    """
    Compute quantile-based distances between two 1D samples using fixed quantile grid.

    Returns a dict containing:
      - qq_l2: Mean squared difference between quantile functions
      - w1: Approximate Wasserstein-1 distance via |Qx - Qy| integrated over p
    """
    # Ensure float32 for numerical stability
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)

    # Use interior quantiles to reduce tail noise; grid in (0,1)
    probs = jnp.linspace(0.02, 0.98, num_quantiles, dtype=jnp.float32)
    qx = jnp.quantile(x, probs, method="linear")
    qy = jnp.quantile(y, probs, method="linear")

    diff = qx - qy
    qq_l2 = jnp.mean(diff * diff)
    # Trapezoidal rule to integrate |Qx - Qy| over p in [0,1]
    absdiff = jnp.abs(diff)
    w1 = jnp.trapz(absdiff, probs)
    return {"qq_l2": qq_l2, "w1": w1}


def _ks_statistic_approx(x: jnp.ndarray, y: jnp.ndarray, grid_size: int = 129) -> jnp.ndarray:
    """
    Approximate two-sample Kolmogorov–Smirnov statistic on a fixed value grid.
    Uses an evenly spaced grid between min and max of pooled samples to avoid dynamic shapes.
    """
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    vmin = jnp.minimum(jnp.min(x), jnp.min(y))
    vmax = jnp.maximum(jnp.max(x), jnp.max(y))
    # Handle degenerate case to avoid NaNs
    vmax = jnp.where(jnp.equal(vmin, vmax), vmin + 1e-6, vmax)
    grid = jnp.linspace(vmin, vmax, grid_size, dtype=jnp.float32)

    # Empirical CDFs on the grid via broadcasting
    Fx = jnp.mean((x[:, None] <= grid[None, :]).astype(jnp.float32), axis=0)
    Fy = jnp.mean((y[:, None] <= grid[None, :]).astype(jnp.float32), axis=0)
    return jnp.max(jnp.abs(Fx - Fy))


def _epps_singleton_stat(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Epps–Singleton-style statistic using a small set of frequencies, avoiding complex dtypes.
    T = sum_k [(Ex cos(t_k X) - Ey cos(t_k Y))^2 + (Ex sin(t_k X) - Ey sin(t_k Y))^2]
    """
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    ts = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float32)

    def _stat_for_t(t):
        cx = jnp.mean(jnp.cos(t * x))
        sx = jnp.mean(jnp.sin(t * x))
        cy = jnp.mean(jnp.cos(t * y))
        sy = jnp.mean(jnp.sin(t * y))
        return (cx - cy) ** 2 + (sx - sy) ** 2

    vals = jax.vmap(_stat_for_t)(ts)
    return jnp.sum(vals)


def _sequence_average(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Compute per-sequence average over valid tokens using a binary mask.
    Returns a 1D array of shape [num_sequences].
    """
    # Sum over sequence axis then divide by valid lengths (avoid divide-by-zero)
    lengths = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    summed = jnp.sum(values * mask, axis=1)
    return summed / lengths


def compute_two_sample_stats_1d(x: jnp.ndarray, y: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """
    Compute a suite of lightweight, distribution-free two-sample statistics for 1D samples.

    Returned keys:
      - dist/ks: Kolmogorov–Smirnov sup |F1 - F2|
      - dist/w1: Wasserstein-1 via quantile approximation
      - dist/qq_l2: L2 between quantile functions
      - dist/es: Epps–Singleton statistic
    """
    results: dict[str, jnp.ndarray] = {}
    try:
        qstats = _compute_quantile_grid_stats(x, y, num_quantiles=41)
        results["dist/w1"] = qstats["w1"]
        results["dist/qq_l2"] = qstats["qq_l2"]
    except Exception:
        # Best-effort: skip quantile-based metrics
        print("Warning: Quantile-based metrics failed")
        pass
    try:
        ks = _ks_statistic_approx(x, y, grid_size=129)
        results["dist/ks"] = ks
    except Exception:
        print("Warning: KS statistic approximation failed")
        # Skip KS if grid/CDF fails
        pass
    try:
        es = _epps_singleton_stat(x, y)
        results["dist/es"] = es
    except Exception:
        # Skip ES if trig/means fail
        print("Warning: ES statistic failed")
        pass
    # Ensure at least one metric is present to avoid empty dicts in logging
    if not results:
        results = {"dist/qq_l2": jnp.array(0.0, dtype=jnp.float32)}
    return results


def get_per_token_logps(model, input_ids, attention_mask, prompt_length):
    """
    Get per-token log probabilities using the model outputs.

    Args:
        model: The language model
        input_ids: Input token ids [batch_size, seq_len]
        attention_mask: Input masks [batch_size, seq_len]
        prompt_length: Length of the prompt
    """

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs = compute_per_token_logps(logits, input_ids, prompt_length)
    return token_log_probs


def compute_per_token_logps(logits, input_ids, prompt_length):
    """
    Compute per-token log probabilities in a vectorized way.

    Args:
        logits: Pre-trimmed logits [batch_size, seq_len, vocab_size]
        input_ids: Input token ids [batch_size, seq_len]
        prompt_length: Length of the prompt
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_ids = input_ids[:, prompt_length:]
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(target_ids, axis=-1),
        axis=-1,
    )
    token_log_probs = jnp.squeeze(token_log_probs, axis=-1)
    return token_log_probs


def grpo_step(
    state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
) -> tp.Union[tuple[EasyDeLState, LossMetrics], LossMetrics]:
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
        per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        per_token_loss = jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps)) * jnp.expand_dims(
            advantages, 1
        )
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        comps = jnp.sum(completion_mask, axis=1)
        loss = jnp.mean(jnp.sum(per_token_loss * completion_mask, axis=1) / comps)
        mean_kl = jnp.mean(jnp.sum(per_token_kl * completion_mask, axis=1) / comps)

        # Compute advantage statistics for progress bar
        advantage_median_abs = jnp.median(jnp.abs(advantages))
        advantage_95th_percentile_abs = jnp.percentile(jnp.abs(advantages), 95)
        
        # Distributional comparison of policy vs reference (sequence-averaged logprobs)
        seq_mean_policy = _sequence_average(per_token_logps, completion_mask)
        seq_mean_ref = _sequence_average(ref_per_token_logps, completion_mask)
        dist_stats = compute_two_sample_stats_1d(seq_mean_policy, seq_mean_ref)

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics={
                "mean_kl": mean_kl,
                "ref_per_token_logps": jnp.mean(ref_per_token_logps),
                "advantages_mean": jnp.mean(advantages),
                "advantage_median_abs": advantage_median_abs,
                "advantage_95th_percentile_abs": advantage_95th_percentile_abs,
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
    else:
        _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics  # type: ignore[return-value]
