"""GFPO host-only filtering utilities shared by GFPO and GFSPO trainers.

This mixin centralizes the stable, TPU-safe top-k mask construction and small
GFPO metrics computed on host. It avoids device launches in preprocessing and
keeps identical behavior across trainers.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp


class GFPOFilterMixin:
    """Host-only GFPO filtering helpers for reuse in GFPO/GFSPO trainers."""

    # Expect the concrete trainer to provide `arguments` and optional `model_state`.

    def _gfpo_build_mask_host(
        self,
        rewards_grouped: jnp.ndarray,  # (B, G)
        lengths_grouped: jnp.ndarray,  # (B, G)
    ) -> jnp.ndarray:
        """Build top-k mask per prompt on host, returning jnp.float32 (B, G).

        - Supports 'length' (ascending) and 'token_efficiency' (descending) metrics.
        - Adaptive k via warmup + (rolling or EMA) percentiles computed on host.
        - Clamps k to [1, G-1] to avoid degenerate full retention.
        - Proc0-only debug uses host computations.
        """
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore

        try:
            rg = jax.device_get(rewards_grouped)
            lg = jax.device_get(lengths_grouped)
        except Exception:
            rg, lg = rewards_grouped, lengths_grouped  # type: ignore

        try:
            bsz, gsize = int(rg.shape[0]), int(rg.shape[1])
        except Exception:
            bsz, gsize = int(rewards_grouped.shape[0]), int(rewards_grouped.shape[1])

        # Metric on host
        if getattr(self.arguments, "gfpo_metric", "length") == "length":
            scores_h = lg
            ascending = True
        else:
            eps_eff = float(getattr(self.arguments, "gfpo_efficiency_epsilon", 1e-8))
            try:
                scores_h = rg / _np.maximum(lg, eps_eff)  # type: ignore
            except Exception:
                scores_h = rg / (lg + eps_eff)
            ascending = False

        # k-per-prompt on host
        if not getattr(self.arguments, "gfpo_adaptive", False):
            k_per = [int(self.arguments.gfpo_retain_count)] * bsz
        else:
            # Warmup
            try:
                model_state = getattr(self, "model_state", None)
                if model_state is not None and hasattr(model_state, "step"):
                    state_step = int(jax.device_get(model_state.step))
                else:
                    state_step = 0
            except Exception:
                state_step = 0
            warmup = int(getattr(self.arguments, "gfpo_adaptive_warmup_steps", 10))
            if state_step < warmup:
                k0 = int(self.arguments.gfpo_adaptive_k_map.get("very_hard", 8))
                k_per = [k0] * bsz
            else:
                km = self.arguments.gfpo_adaptive_k_map
                k_vh = int(km.get("very_hard", 8))
                k_h = int(km.get("hard", 8))
                k_m = int(km.get("medium", 6))
                k_e = int(km.get("easy", 4))
                try:
                    avg_rewards = _np.mean(rg, axis=1) if _np is not None else rg.mean(axis=1)
                except Exception:
                    avg_rewards = jax.device_get(jnp.mean(rewards_grouped, axis=1))
                method = getattr(self.arguments, "gfpo_adaptive_method", "rolling")
                if method == "ema":
                    try:
                        cur = _np.percentile(avg_rewards, [25.0, 50.0, 75.0])
                    except Exception:
                        cur = jax.device_get(
                            jnp.percentile(jnp.asarray(avg_rewards), jnp.array([25.0, 50.0, 75.0]))
                        )
                    alpha = float(getattr(self.arguments, "gfpo_adaptive_ema_alpha", 0.1))
                    if not hasattr(self, "_running_percentiles"):
                        self._running_percentiles = cur
                    else:
                        self._running_percentiles = (1.0 - alpha) * self._running_percentiles + alpha * cur
                    q25, q50, q75 = [float(x) for x in self._running_percentiles]
                else:
                    if not hasattr(self, "_difficulty_buffer"):
                        self._difficulty_buffer = []
                    try:
                        self._difficulty_buffer.extend([float(x) for x in list(avg_rewards)])
                    except Exception:
                        pass
                    max_hist = int(getattr(self.arguments, "gfpo_adaptive_history_max", 20000))
                    if len(self._difficulty_buffer) > max_hist:
                        self._difficulty_buffer = self._difficulty_buffer[-max_hist:]
                    if len(self._difficulty_buffer) < 40:
                        q25 = q50 = q75 = None
                    else:
                        try:
                            q25, q50, q75 = _np.percentile(
                                _np.asarray(self._difficulty_buffer, dtype=_np.float32), [25.0, 50.0, 75.0]
                            )
                        except Exception:
                            arr = jnp.asarray(self._difficulty_buffer, dtype=jnp.float32)
                            q25, q50, q75 = [
                                float(x)
                                for x in jax.device_get(
                                    jnp.percentile(arr, jnp.array([25.0, 50.0, 75.0]))
                                )
                            ]
                k_per = []
                for r in (avg_rewards.tolist() if hasattr(avg_rewards, "tolist") else list(avg_rewards)):
                    if q25 is None:
                        k_per.append(k_vh)
                    elif r < q25:
                        k_per.append(k_vh)
                    elif r < q50:
                        k_per.append(k_h)
                    elif r < q75:
                        k_per.append(k_m)
                    else:
                        k_per.append(k_e)

        # Clamp and build mask via host argsort
        upper = max(1, int(gsize) - 1)
        k_per = [int(max(1, min(upper, int(x)))) for x in k_per]

        try:
            idx_sorted = _np.argsort(scores_h, axis=1) if ascending else _np.argsort(-scores_h, axis=1)
        except Exception:
            idx_sorted = jax.device_get(
                jnp.argsort(jnp.asarray(scores_h), axis=1)
                if ascending
                else jnp.argsort(-jnp.asarray(scores_h), axis=1)
            )
        mask_h = _np.zeros((bsz, gsize), dtype=_np.float32)
        for i in range(bsz):
            ki = int(k_per[i])
            mask_h[i, idx_sorted[i, :ki]] = 1.0

        # Proc0-only debug
        try:
            if jax.process_index() == 0:
                m_sum = float(mask_h.sum())
                exp_sum = float(sum(k_per))
                print(
                    f"DEBUG: GFPO filter params: G={int(gsize)}, k_fixed={int(getattr(self.arguments,'gfpo_retain_count',-1))}, "
                    f"adaptive={bool(getattr(self.arguments,'gfpo_adaptive', False))}"
                )
                print(f"DEBUG: GFPO mask sum={m_sum}, expected_sum≈{exp_sum} (bsz={int(bsz)})")
        except Exception:
            pass

        return jnp.asarray(mask_h, dtype=jnp.float32)

    def _gfpo_compute_metrics_host(
        self, mask: jnp.ndarray, lengths_grouped: jnp.ndarray
    ) -> dict[str, float]:
        """Compute small GFPO metrics on host for stability."""
        try:
            import numpy as _np
            mh = jax.device_get(mask)
            lg_h = jax.device_get(lengths_grouped)
            retention_rate = float(_np.mean(mh))
            total_selected = float(max(1.0, float(_np.sum(mh))))
            avg_retained_length = float((_np.sum(lg_h * mh)) / total_selected)
            return {
                "gfpo/retention_rate": retention_rate,
                "gfpo/avg_retained_length": avg_retained_length,
            }
        except Exception:
            return {}

