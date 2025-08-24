# gfpo_trainer.py

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from ..trainer_protocol import TrainerConfigureFunctionOutput
from ._gfpo_fn import gfpo_step
from .gfpo_config import GFPOConfig
from .grpo_trainer import GRPOTrainer, RewardFunc

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

logger = get_logger(__name__)


class GFPOTrainer(GRPOTrainer):
    """
    GFPO (Group Filtered Policy Optimization) Trainer.

    Extends GRPO by sampling G responses per prompt, filtering to k responses using a
    metric (length or reward/length), and computing standardized advantages only over
    the retained subset per prompt. Rejected responses receive zero advantage.
    """

    arguments: GFPOConfig  # type hinting

    def _dbg(self, msg: str) -> None:
        try:
            if jax.process_index() == 0 and getattr(self.arguments, "verbose", True):
                print(msg)
        except Exception:
            ...

    def _dbg_shape(self, name: str, arr) -> None:
        try:
            self._dbg(f"{name}.shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        except Exception:
            ...

    def _dbg_head(self, name: str, arr, n: int = 5) -> None:
        try:
            if jax.process_index() != 0 or not getattr(self.arguments, "verbose", True):
                return
            head = None
            try:
                head = jax.device_get(arr)[:n]
            except Exception:
                try:
                    head = arr[:n]
                except Exception:
                    head = None
            self._dbg(f"{name}: head={getattr(head, 'tolist', lambda: head)()} shape={getattr(arr, 'shape', None)}")
        except Exception:
            ...

    def __init__(
        self,
        arguments: GFPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        try:
            if jax.process_index() == 0:
                print(f"GFPOTrainer.__init__ called with arguments type: {type(arguments)}")
        except Exception:
            ...
        assert arguments is not None, (
            "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
        )
        assert isinstance(arguments, GFPOConfig), f"arguments type must be `GFPOConfig` but got {type(arguments)}"

        self._dbg("Calling super().__init__ in GFPOTrainer")
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )

        # Ensure local alias to config type
        self.arguments = arguments
        self._dbg(f"GFPOTrainer: self.arguments set to {type(self.arguments)}")

        try:
            self._dbg(f"DEBUG: Initializing GFPO trainer - G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}")
            logger.info(
                f"Initialized GFPO trainer: G={arguments.gfpo_group_size}, k={arguments.gfpo_retain_count}, metric={arguments.gfpo_metric}, adaptive={arguments.gfpo_adaptive}"
            )
        except Exception as e:
            print(f"DEBUG: Failed to log GFPO trainer initialization: {e}")
            logger.warning(f"Failed to log GFPO trainer initialization: {e}")

    def _filter_mask_per_prompt(
        self,
        rewards_grouped: jnp.ndarray,  # (B, G)
        lengths_grouped: jnp.ndarray,  # (B, G)
    ) -> jnp.ndarray:
        # Local debug helpers to work even when called from GFSPOTrainer
        try:
            _proc0 = jax.process_index() == 0
        except Exception:
            _proc0 = True
        try:
            _verbose = getattr(self.arguments, "verbose", True)
        except Exception:
            _verbose = True

        def _dbg(msg: str) -> None:
            try:
                if _proc0 and _verbose:
                    print(msg)
            except Exception:
                ...

        def _dbg_shape(name: str, arr) -> None:
            try:
                _dbg(f"{name}.shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
            except Exception:
                ...

        def _dbg_head(name: str, arr, n: int = 5) -> None:
            try:
                if not (_proc0 and _verbose):
                    return
                head = None
                try:
                    head = jax.device_get(arr)[:n]
                except Exception:
                    try:
                        head = arr[:n]
                    except Exception:
                        head = None
                _dbg(f"{name}: head={getattr(head, 'tolist', lambda: head)()} shape={getattr(arr, 'shape', None)}")
            except Exception:
                ...

        _dbg(f"GFPOTrainer._filter_mask_per_prompt called")
        _dbg_shape("rewards_grouped", rewards_grouped)
        _dbg_shape("lengths_grouped", lengths_grouped)
        bsz, gsize = rewards_grouped.shape
        _dbg(f"bsz: {bsz}, gsize: {gsize}")
        # Debug and sanity checks for configuration propagation
        try:
            configured_G = int(getattr(self.arguments, "gfpo_group_size", gsize))
            _dbg(f"configured_G: {configured_G}")
            if configured_G != gsize and jax.process_index() == 0:
                print(
                    f"DEBUG: GFPO group size mismatch: configured_G={configured_G} but grouped G={gsize}; "
                    f"will use grouped size for masking."
                )
        except Exception as e:
            _dbg(f"Exception in group size check: {e}")
        # Compute scores per configured metric
        _dbg(f"gfpo_metric: {getattr(self.arguments, 'gfpo_metric', None)}")
        if self.arguments.gfpo_metric == "length":
            # Lower is better
            _dbg("Metric is length, lower is better")
            scores = lengths_grouped
            ascending = True
        else:
            # token_efficiency = reward / length; Higher is better
            _dbg("Metric is not length, using reward/length (token_efficiency)")
            scores = rewards_grouped / jnp.maximum(
                lengths_grouped, jnp.float32(getattr(self.arguments, "gfpo_efficiency_epsilon", 1e-8))
            )
            ascending = False

        # Determine k per prompt (fixed or adaptive per Algorithm 2)
        _dbg(f"gfpo_adaptive: {getattr(self.arguments, 'gfpo_adaptive', None)}")
        if not self.arguments.gfpo_adaptive:
            _dbg("gfpo_adaptive is False, using fixed k_per_prompt")
            # TPU-safe: enforce int32 dtype for k array
            k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_retain_count), dtype=jnp.int32)
            _dbg_head("k_per_prompt(fixed)", k_per_prompt)
        else:
            _dbg("gfpo_adaptive is True, using adaptive k_per_prompt")
            # Method 1 (rolling history on CPU) per Algorithm 2
            avg_rewards = jnp.mean(rewards_grouped, axis=1)
            _dbg_shape("avg_rewards", avg_rewards)
            try:
                # Host step value for warmup logic
                try:
                    # Try to get step from model_state if available, otherwise use 0
                    model_state = getattr(self, 'model_state', None)
                    _dbg(f"model_state available: {model_state is not None}")
                    if model_state is not None and hasattr(model_state, 'step'):
                        state_step = int(jax.device_get(model_state.step))
                        _dbg(f"state_step from model_state: {state_step}")
                    else:
                        state_step = 0
                        _dbg("model_state missing or no step; using step=0")
                except Exception as e:
                    _dbg(f"Exception in getting state_step: {e}")
                    state_step = 0
                warmup_steps = int(getattr(self.arguments, 'gfpo_adaptive_warmup_steps', 10))
                _dbg(f"warmup_steps: {warmup_steps}")
                if state_step < warmup_steps:
                    _dbg("Warmup phase: using very_hard bucket for k_per_prompt")
                    k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_adaptive_k_map.get('very_hard', 8)), dtype=jnp.int32)
                    _dbg_head("k_per_prompt(warmup)", k_per_prompt)
                else:
                    method = getattr(self.arguments, 'gfpo_adaptive_method', 'rolling')
                    _dbg(f"gfpo_adaptive_method: {method}")
                    if method == 'ema':
                        _dbg("Using EMA for adaptive k_per_prompt")
                        current_percentiles = jnp.percentile(
                            avg_rewards, jnp.array([25.0, 50.0, 75.0], dtype=jnp.float32)
                        )
                        _dbg(f"current_percentiles: {jax.device_get(current_percentiles).tolist()}")
                        alpha = jnp.float32(getattr(self.arguments, 'gfpo_adaptive_ema_alpha', 0.1))
                        _dbg(f"EMA alpha: {alpha}")
                        if not hasattr(self, '_running_percentiles'):
                            _dbg("Initializing _running_percentiles")
                            self._running_percentiles = current_percentiles
                        else:
                            _dbg("Updating _running_percentiles")
                            self._running_percentiles = (1.0 - alpha) * self._running_percentiles + alpha * current_percentiles
                        q25, q50, q75 = self._running_percentiles
                        _dbg(f"Running percentiles: {jax.device_get(jnp.array([q25, q50, q75])).tolist()}")
                        km = self.arguments.gfpo_adaptive_k_map
                        _dbg(f"k_map: {km}")
                        k_vh = int(km.get('very_hard', 8))
                        k_h = int(km.get('hard', 8))
                        k_m = int(km.get('medium', 6))
                        k_e = int(km.get('easy', 4))
                        _dbg(f"k_vh={k_vh}, k_h={k_h}, k_m={k_m}, k_e={k_e}")
                        k_per_prompt = jnp.where(
                            avg_rewards < q25, k_vh,
                            jnp.where(
                                avg_rewards < q50, k_h,
                                jnp.where(avg_rewards < q75, k_m, k_e),
                            ),
                        )
                        _dbg_head("k_per_prompt(ema)", k_per_prompt)
                        k_per_prompt = k_per_prompt.astype(jnp.int32)
                    else:
                        _dbg("Using rolling buffer for adaptive k_per_prompt")
                        # Initialize CPU-side rolling buffer
                        if not hasattr(self, '_difficulty_buffer'):
                            _dbg("Initializing _difficulty_buffer")
                            self._difficulty_buffer = []  # python list on CPU
                        # Append current batch avg rewards to buffer (CPU)
                        try:
                            avg_rewards_cpu = [float(x) for x in jax.device_get(avg_rewards[:8])]
                        except Exception:
                            avg_rewards_cpu = []
                        _dbg(f"Appending avg_rewards head to _difficulty_buffer: {avg_rewards_cpu} (total so far={len(self._difficulty_buffer)})")
                        self._difficulty_buffer.extend(avg_rewards_cpu)
                        # Trim to max history
                        max_hist = int(getattr(self.arguments, 'gfpo_adaptive_history_max', 20000))
                        _dbg(f"max_hist: {max_hist}, current buffer length: {len(self._difficulty_buffer)}")
                        if len(self._difficulty_buffer) > max_hist:
                            _dbg("Trimming _difficulty_buffer to max_hist")
                            self._difficulty_buffer = self._difficulty_buffer[-max_hist:]

                        # Require minimal history to compute stable percentiles
                        if len(self._difficulty_buffer) < 40:
                            _dbg("Not enough history, using very_hard bucket for k_per_prompt")
                            k_per_prompt = jnp.full(
                                (bsz,), int(self.arguments.gfpo_adaptive_k_map.get('very_hard', 8)), dtype=jnp.int32
                            )
                            _dbg_head("k_per_prompt(history)" , k_per_prompt)
                        else:
                            _dbg("Enough history, computing percentiles for k_per_prompt")
                            hist = jnp.asarray(self._difficulty_buffer, dtype=jnp.float32)
                            q25, q50, q75 = jnp.percentile(hist, jnp.array([25.0, 50.0, 75.0], dtype=jnp.float32))
                            _dbg(f"History percentiles: {jax.device_get(jnp.array([q25, q50, q75])).tolist()}")
                            km = self.arguments.gfpo_adaptive_k_map
                            _dbg(f"k_map: {km}")
                            k_vh = int(km.get('very_hard', 8))
                            k_h = int(km.get('hard', 8))
                            k_m = int(km.get('medium', 6))
                            k_e = int(km.get('easy', 4))
                            _dbg(f"k_vh={k_vh}, k_h={k_h}, k_m={k_m}, k_e={k_e}")
                            k_per_prompt = jnp.where(
                                avg_rewards < q25, k_vh,
                                jnp.where(
                                    avg_rewards < q50, k_h,
                                    jnp.where(avg_rewards < q75, k_m, k_e),
                                ),
                            )
                            _dbg_head("k_per_prompt(rolling)", k_per_prompt)
                            k_per_prompt = k_per_prompt.astype(jnp.int32)
            except Exception as e:
                _dbg(f"Exception in adaptive k_per_prompt: {e}")
                # Safe fallback: fixed k
                k_per_prompt = jnp.full((bsz,), int(self.arguments.gfpo_retain_count), dtype=jnp.int32)
                _dbg_head("k_per_prompt(fallback)", k_per_prompt)

        # Clamp k to valid range [1, G-1] to avoid degenerate retention==1.0 when possible
        try:
            upper = max(1, int(gsize) - 1)
            _dbg(f"Clamping k_per_prompt to [1, {upper}]")
            k_per_prompt = jnp.clip(k_per_prompt, 1, upper).astype(jnp.int32)
            _dbg_head("k_per_prompt(clamped)", k_per_prompt)
            if jax.process_index() == 0:
                # If any k equals G, log a hint
                try:
                    num_full = int(jnp.sum(k_per_prompt >= upper))
                    _dbg(f"num_full (k_per_prompt >= upper): {num_full}")
                except Exception as e:
                    _dbg(f"Exception in counting num_full: {e}")
                    num_full = -1
                if num_full and num_full > 0:
                    print(
                        f"DEBUG: GFPO k reached upper bound for {num_full}/{int(bsz)} prompts (G={int(gsize)}, upper={upper}). "
                        "Consider lowering gfpo_retain_count or adjusting adaptive k-map."
                    )
        except Exception as e:
            _dbg(f"Exception in clamping k_per_prompt: {e}")

        # Build mask per prompt by argsort
        _dbg("Building mask per prompt by argsort (showing first 2 prompts only)")
        mask = jnp.zeros((bsz, gsize), dtype=jnp.float32)
        for i in range(bsz):
            # Guard against pathological k values
            try:
                k_i = int(k_per_prompt[i])
                if i < 2:
                    _dbg(f"Prompt {i}: k_i={k_i}")
            except Exception as e:
                if i < 2:
                    _dbg(f"Exception in getting k_i for prompt {i}: {e}")
                k_i = int(self.arguments.gfpo_retain_count)
            k_i = max(1, min(int(gsize), k_i))
            if i < 2:
                _dbg(f"Prompt {i}: final k_i: {k_i}")
            if ascending:
                chosen = jnp.argsort(scores[i])[:k_i]
                if i < 2:
                    try:
                        _dbg(f"Prompt {i}: ascending, chosen head={jax.device_get(chosen)[:8].tolist()}")
                    except Exception:
                        ...
            else:
                chosen = jnp.argsort(-scores[i])[:k_i]
                if i < 2:
                    try:
                        _dbg(f"Prompt {i}: descending, chosen head={jax.device_get(chosen)[:8].tolist()}")
                    except Exception:
                        ...
            mask = mask.at[i, chosen].set(1.0)
            # Do not print mask rows to avoid huge logs

        # Debug: show mask sum versus expected retained count
        try:
            if jax.process_index() == 0:
                try:
                    expected_sum = float(jnp.sum(k_per_prompt))
                    _dbg(f"expected_sum: {expected_sum}")
                except Exception as e:
                    _dbg(f"Exception in computing expected_sum: {e}")
                    expected_sum = -1.0
                try:
                    mask_sum = float(jnp.sum(mask))
                    _dbg(f"mask_sum: {mask_sum}")
                except Exception as e:
                    _dbg(f"Exception in computing mask_sum: {e}")
                    mask_sum = -1.0
                print(
                    f"DEBUG: GFPO filter params: G={int(gsize)}, k_fixed={int(getattr(self.arguments,'gfpo_retain_count',-1))}, "
                    f"adaptive={bool(getattr(self.arguments,'gfpo_adaptive', False))}"
                )
                print(
                    f"DEBUG: GFPO mask sum={mask_sum}, expected_sum≈{expected_sum} (bsz={int(bsz)})"
                )
        except Exception as e:
            _dbg(f"Exception in mask sum debug: {e}")
        _dbg("Returning mask from _filter_mask_per_prompt")
        return mask

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        self._dbg(f"GFPOTrainer._preprocess_batch_input | is_train={is_train}")
        grpo_batch, metrics_dict = super()._preprocess_batch_input(state, batch, is_train)
        self._dbg(f"grpo_batch keys: {list(grpo_batch.keys())}")

        # Shapes
        try:
            num_prompts = int(batch["input_ids"].shape[0])
            self._dbg(f"num_prompts: {num_prompts}")
        except Exception as e:
            self._dbg(f"Exception in getting num_prompts: {e}")
            num_prompts = 0
        G = int(self.arguments.gfpo_group_size)
        self._dbg(f"G (gfpo_group_size): {G}")

        # Prefer true rewards if provided by parent; fallback to reconstructing from advantages
        eps = jnp.float32(self.arguments.advantage_epsilon)
        self._dbg(f"advantage_epsilon: {float(eps)}")
        if "rewards" in grpo_batch:
            self._dbg("Found 'rewards' in grpo_batch")
            rewards_arr = grpo_batch["rewards"]
            self._dbg_shape("rewards_arr", rewards_arr)
            try:
                rewards_grouped = rewards_arr.reshape(num_prompts, G)
                self._dbg_shape("rewards_grouped", rewards_grouped)
            except Exception as e:
                self._dbg(f"Exception in reshaping rewards_arr: {e}")
                # Fallback to reconstruction
                advantages = grpo_batch["advantages"].reshape(num_prompts, G)
                self._dbg_shape("advantages", advantages)
                mean_per_prompt = jnp.mean(advantages, axis=1, keepdims=True)
                self._dbg_shape("mean_per_prompt", mean_per_prompt)
                std_per_prompt = jnp.std(advantages, axis=1, keepdims=True)
                self._dbg_shape("std_per_prompt", std_per_prompt)
                std_per_prompt = jnp.maximum(std_per_prompt, eps)
                self._dbg_shape("std_per_prompt (clamped)", std_per_prompt)
                rewards_grouped = advantages * std_per_prompt + mean_per_prompt
                self._dbg_shape("rewards_grouped (reconstructed)", rewards_grouped)
        else:
            self._dbg("No 'rewards' in grpo_batch, reconstructing from 'advantages'")
            advantages = grpo_batch["advantages"].reshape(num_prompts, G)
            self._dbg_shape("advantages", advantages)
            mean_per_prompt = jnp.mean(advantages, axis=1, keepdims=True)
            self._dbg_shape("mean_per_prompt", mean_per_prompt)
            std_per_prompt = jnp.std(advantages, axis=1, keepdims=True)
            self._dbg_shape("std_per_prompt", std_per_prompt)
            std_per_prompt = jnp.maximum(std_per_prompt, eps)
            self._dbg_shape("std_per_prompt (clamped)", std_per_prompt)
            rewards_grouped = advantages * std_per_prompt + mean_per_prompt
            self._dbg_shape("rewards_grouped (reconstructed)", rewards_grouped)

        # Compute completion lengths
        lengths = jnp.sum(grpo_batch["completion_mask"], axis=-1)
        self._dbg_shape("lengths", lengths)
        lengths_grouped = lengths.reshape(num_prompts, G)
        self._dbg_shape("lengths_grouped", lengths_grouped)

        # Build selection mask per prompt and compute GFPO advantages per prompt
        self._dbg("Calling _filter_mask_per_prompt")
        mask = self._filter_mask_per_prompt(rewards_grouped, lengths_grouped)  # (B, G)
        self._dbg_shape("mask", mask)

        # Selected subset statistics per prompt (mu_S, sigma_S) computed from selected only
        selected_count = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        self._dbg_shape("selected_count", selected_count)
        selected_sum = jnp.sum(rewards_grouped * mask, axis=1, keepdims=True)
        self._dbg_shape("selected_sum", selected_sum)
        mu_S = selected_sum / selected_count
        self._dbg_shape("mu_S", mu_S)

        # Variance with Bessel's correction on selected entries only
        diff_sq = jnp.where(mask > 0.0, (rewards_grouped - mu_S) ** 2, 0.0)
        self._dbg_shape("diff_sq", diff_sq)
        denom = jnp.maximum(selected_count - 1.0, 1.0)
        self._dbg_shape("denom", denom)
        sigma_S = jnp.sqrt(jnp.sum(diff_sq, axis=1, keepdims=True) / denom)
        self._dbg_shape("sigma_S (raw)", sigma_S)
        sigma_S = jnp.maximum(sigma_S, eps)
        self._dbg_shape("sigma_S (clamped)", sigma_S)

        # A^(m) = ((R - mu_S) / sigma_S) * m
        advantages_gfpo = ((rewards_grouped - mu_S) / sigma_S) * mask
        grpo_batch["advantages"] = advantages_gfpo.reshape(-1)
        self._dbg_shape("advantages (final)", grpo_batch["advantages"])

        # Minimal GFPO metrics
        try:
            retention_rate = float(jnp.mean(mask))
            self._dbg(f"retention_rate: {retention_rate:.3f}")
            avg_retained_length = float(
                jnp.sum(lengths * mask.reshape(-1)) / float(jnp.maximum(jnp.sum(mask), 1.0))
            )
            self._dbg(f"avg_retained_length: {avg_retained_length:.1f}")
            print(f"DEBUG: GFPO metrics - retention_rate={retention_rate:.3f}, avg_retained_length={avg_retained_length:.1f}")
            metrics_dict["gfpo/retention_rate"] = retention_rate
            metrics_dict["gfpo/avg_retained_length"] = avg_retained_length
        except Exception as e:
            self._dbg(f"DEBUG: Failed to compute GFPO metrics: {e}")
            pass

        self._dbg("Returning from _preprocess_batch_input")
        return grpo_batch, metrics_dict

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        self._dbg("GFPOTrainer.configure_functions called")
        """
        Configure training and evaluation step functions using gfpo_step. The loss is
        identical to GRPO; GFPO behavior arises from preprocessing that filters and
        standardizes advantages per retained subset.
        """
        parent = super().configure_functions()
        self._dbg("Called super().configure_functions, got parent")

        from easydel.utils.compiling_utils import ejit
        from jax.sharding import NamedSharding, PartitionSpec
        import inspect as _inspect

        mesh = self.model.mesh
        self._dbg(f"mesh ready for GFPO: {mesh}")
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)
        self._dbg(f"empty_sharding: {empty_sharding}")

        # Reuse static args built by the parent (num_generations, beta, loss_config, ...)
        _sig = _inspect.signature(gfpo_step)
        self._dbg(f"gfpo_step signature: {_sig}")
        _max_pos_index = len(_sig.parameters) - 1
        self._dbg(f"_max_pos_index: {_max_pos_index}")
        _end = min(2 + len(self._train_shared_fn_static_args), _max_pos_index + 1)
        self._dbg(f"_end: {_end}")
        static_argnames = tuple(range(2, _end))
        self._dbg(f"static_argnames: {static_argnames}")

        self._dbg("Compiling sharded_training_step_function with ejit")
        sharded_training_step_function = ejit(
            gfpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )
        self._dbg("Compiled sharded_training_step_function")

        self._dbg("Compiling sharded_evaluation_step_function with ejit")
        sharded_evaluation_step_function = ejit(
            gfpo_step,
            in_shardings=(self.state_shardings, None),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )
        self._dbg("Compiled sharded_evaluation_step_function")

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames
        self._dbg("Set static_argnums_ for both sharded functions")

        self._dbg("Returning TrainerConfigureFunctionOutput from configure_functions")
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=parent.mesh,
            checkpoint_manager=parent.checkpoint_manager,
        )