#!/bin/bash
# GSPO Training Script with TPU Optimizations (8-worker metrics support)
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

# Pull latest changes and install
echo "Setting up environment..."
git pull origin main 2>/dev/null || true
uv pip install -e . --quiet
uv pip install "math-verify[antlr4_13_2]" --quiet || true

# Navigate to project directory
cd /home/air/EasyDeL-GSPO

echo "Starting GSPO training with optimized configuration..."

git pull origin math-only-improved || true
uv pip install -e . --quiet
uv pip install "math-verify[antlr4_13_2]" --quiet || true

export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off  # For better debugging if needed

cd /home/air/EasyDeL-GSPO

echo "Starting GSPO training with optimized configuration..."

# Parse command line arguments
DATASET="${1:-math-ds}"
CURRICULUM_MATH="${2:-false}"
echo "Using dataset: ${DATASET}"
echo "Curriculum math: ${CURRICULUM_MATH}"

# Metrics and logging: ensure correct aggregation for 8 workers
# Set log_global to true for correct global logging aggregation
LOG_GLOBAL_VAL=${LOG_GLOBAL:-true}
echo "LOG_GLOBAL: ${LOG_GLOBAL_VAL}"

# Set force_data_parallel to 8 for 8-worker setup
FORCE_DATA_PARALLEL=8

python3.11 easydel/scripts/finetune/gsm8k_math_gspo.py \
  --repo_id "Qwen/Qwen3-0.6B" \
  --dataset ${DATASET} \
  --curriculum_math ${CURRICULUM_MATH} \
  --total_batch_size 2 \
  --num_return_sequences 2 \
  --rollout_chunk_size 4 \
  --num_train_epochs 2 \
  --max_prompt_length 512 \
  --max_completion_length 5120 \
  --learning_rate 2e-6 \
  --dataset_use_pct 10 \
  --force_tensor_parallel 4 \
  --force_data_parallel 8 \
  --log_logprobs_metrics false \
  --log_global ${LOG_GLOBAL_VAL} \
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

echo "Training completed!"
