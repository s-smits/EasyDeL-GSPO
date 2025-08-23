#!/bin/bash
# GFSPO Training Script with TPU Optimizations
#
# Usage: ./run_gfspo_training.sh [DATASET] [CURRICULUM_MATH]
# 
# Arguments:
#   DATASET        - Dataset to use: 'math-ds' or 'gsm8k-ds' (default: math-ds)
#   CURRICULUM_MATH - Enable curriculum learning: 'true' or 'false' (default: false)
#
# Examples:
#   ./run_gfspo_training.sh math-ds true    # Enable curriculum learning on math dataset
#   ./run_gfspo_training.sh math-ds false   # Disable curriculum learning
#   ./run_gfspo_training.sh gsm8k-ds        # Use GSM8K dataset (curriculum learning has no effect)

# Set environment variables for TPU
export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off

echo "Setting up environment..."
git pull origin math-only-improved 2>/dev/null || true
uv pip install -e . --quiet
uv pip install "math-verify[antlr4_13_2]" --quiet || true

cd /home/air/EasyDeL-GSPO

# Activate virtual environment if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "Starting GFSPO training with optimized configuration..."

# Parse command line arguments
DATASET="${1:-math-ds}"
CURRICULUM_MATH="${2:-false}"

# If DATASET is math-ds, force curriculum math to true
if [ "$DATASET" = "math-ds" ]; then
  CURRICULUM_MATH="true"
fi

echo "Using dataset: ${DATASET}"
echo "Curriculum math: ${CURRICULUM_MATH}"

# Note: dataset_use_rate now uses fractions (1.0 = 100%, 0.1 = 10%)
python3.11 easydel/scripts/finetune/gsm8k_math_gfspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset ${DATASET} \
  --curriculum_math ${CURRICULUM_MATH} \
  --total_batch_size 2 \
  --gfpo_group_size 4 \
  --gfpo_retain_count 2 \
  --rollout_chunk_size 1 \
  --num_train_epochs 4 \
  --max_prompt_length 512 \
  --max_completion_length 5120 \
  --learning_rate 2e-6 \
  --dataset_use_rate 1.0 \
  --force_tensor_parallel 4 \
  --force_data_parallel 2 \
  --log_logprobs_metrics false \
  --log_global true \
  --log_steps 1 \
  --save_steps 50 \
  --do_eval false \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 16 \
  --beta 0.04 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --advantage_epsilon 1e-6

echo "Training completed!"
