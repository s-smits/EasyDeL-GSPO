#!/bin/bash
# GSPO Training Script with TPU Optimizations
#
# Usage: ./run_gspo_training.sh [DATASET] [CURRICULUM_MATH]
# 
# Arguments:
#   DATASET        - Dataset to use: 'math-ds' or 'gsm8k-ds' (default: math-ds)
#   CURRICULUM_MATH - Enable curriculum learning: 'true' or 'false' (default: false)
#
# Examples:
#   ./run_gspo_training.sh math-ds true    # Enable curriculum learning on math dataset
#   ./run_gspo_training.sh math-ds false   # Disable curriculum learning
#   ./run_gspo_training.sh gsm8k-ds        # Use GSM8K dataset (curriculum learning has no effect)

# Set environment variables for TPU
export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off  # For better debugging if needed
export PYTHONUNBUFFERED=1
export GRAIN_WORKER_COUNT=0
export AUTO_INIT_JAX=0
export WANDB_MODE=disabled
# Let JAX auto-detect process count/index from the TPU runtime
# Only normalize if they're explicitly set to 'None' string
if [ "${JAX_PROCESS_COUNT:-}" = "None" ]; then
  unset JAX_PROCESS_COUNT
fi
if [ "${JAX_PROCESS_INDEX:-}" = "None" ]; then
  unset JAX_PROCESS_INDEX
fi

# Auto-configure JAX multi-worker (TPU Pod) if coordinator not set and hostname follows *-w-<rank>
if [ -z "${JAX_COORDINATOR_ADDRESS:-}" ]; then
  HN="$(hostname)"
  if [[ "$HN" =~ (.*-w-)([0-9]+)$ ]]; then
    BASE="${BASH_REMATCH[1]}"
    RANK="${BASH_REMATCH[2]}"
    COORD_NAME="${BASE}0"
    # Prefer IPv4 to avoid multi-line/IPv6 link-local issues
    COORD_IP="$(getent ahostsv4 "$COORD_NAME" | awk 'NR==1 {print $1}')"
    if [ -z "$COORD_IP" ]; then
      # Fallback: resolve first address and strip scope if any
      COORD_IP="$(getent hosts "$COORD_NAME" | awk 'NR==1 {print $1}' | sed 's/%.*//')"
    fi
    if [ -n "$COORD_IP" ]; then
      export JAX_COORDINATOR_ADDRESS="${COORD_IP}:8476"
      export JAX_PROCESS_INDEX="${RANK}"
      export JAX_PROCESS_COUNT="${ED_NUM_PROCS:-4}"
      echo "Auto JAX distributed: coord=$JAX_COORDINATOR_ADDRESS index=$JAX_PROCESS_INDEX count=$JAX_PROCESS_COUNT"
    else
      echo "WARN: Could not resolve coordinator address for $COORD_NAME"
    fi
  fi
fi
# Pull latest changes and install
echo "Setting up environment..."
git pull origin working
git fetch origin working && git reset --hard origin/working
uv pip install -e . --quiet
uv pip install "math-verify[antlr4_13_2]" --quiet || true

# Navigate to project directory
cd /home/air/EasyDeL-GSPO

echo "Starting GSPO training with optimized configuration..."

# Parse command line arguments
DATASET="${1:-math-ds}"
CURRICULUM_MATH="${2:-false}"
echo "Using dataset: ${DATASET}"
echo "Curriculum math: ${CURRICULUM_MATH}"

python3.11 easydel/scripts/finetune/gsm8k_math_gspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset ${DATASET} \
  --curriculum_math ${CURRICULUM_MATH} \
  --total_batch_size 2 \
  --num_return_sequences 8 \
  --rollout_chunk_size 1 \
  --num_train_epochs 2 \
  --max_prompt_length 512 \
  --max_completion_length 4096 \
  --learning_rate 2e-6 \
  --dataset_use_pct 10 \
  --force_tensor_parallel 4 \
  --use_grain true \
  --grain_worker_count 0 \
  --grain_worker_buffer_size 0 \
  --grain_read_threads 1 \
  --grain_prefetch_buffer_size 128 \
  --use_wandb false \
  --log_logprobs_metrics false \
  --log_global true \
  --log_steps 1 \
  --save_steps 100 \
  --do_eval false \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 8 \
  --beta 0.04 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --advantage_epsilon 1e-6