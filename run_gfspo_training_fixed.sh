#!/bin/bash
# GFSPO Training Script with Fixes for Repetitive Generation
#
# FIXES APPLIED:
# 1. Reduced max_completion_length from 6144 to 1024 (better for small models)
# 2. Increased temperature from 0.7 to 1.0 (break repetitive patterns)  
# 3. Reduced top_p from 0.95 to 0.9 (more focused sampling)
# 4. Reduced top_k from 50 to 30 (smaller vocabulary)
# 5. Reduced gfpo_group_size from 8 to 4 (less stress on small model)
# 6. Reduced learning_rate from 2e-6 to 1e-6 (more stable training)
# 7. Added repetition_penalty to prevent loops
#
# Usage: ./run_gfspo_training_fixed.sh [DATASET] [CURRICULUM_MATH]

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

echo "Starting GFSPO training with FIXED configuration for repetitive generation..."

# Parse command line arguments
DATASET="${1:-math-ds}"
CURRICULUM_MATH="${2:-false}"

# If DATASET is math-ds, force curriculum math to true
if [ "$DATASET" = "math-ds" ]; then
  CURRICULUM_MATH="true"
fi

echo "Using dataset: ${DATASET}"
echo "Curriculum math: ${CURRICULUM_MATH}"

echo "FIXES APPLIED:"
echo "- Reduced max_completion_length: 6144 → 1024"
echo "- Increased temperature: 0.7 → 1.0" 
echo "- Reduced top_p: 0.95 → 0.9"
echo "- Reduced top_k: 50 → 30"
echo "- Reduced gfpo_group_size: 8 → 4"
echo "- Reduced learning_rate: 2e-6 → 1e-6"

python3.11 easydel/scripts/finetune/gsm8k_math_gfspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset ${DATASET} \
  --curriculum_math ${CURRICULUM_MATH} \
  --total_batch_size 2 \
  --gfpo_group_size 4 \
  --gfpo_retain_count 2 \
  --rollout_chunk_size 2 \
  --num_train_epochs 2 \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --learning_rate 1e-6 \
  --dataset_use_rate 10 \
  --force_tensor_parallel 4 \
  --force_data_parallel 8 \
  --log_logprobs_metrics false \
  --log_global true \
  --log_steps 1 \
  --save_steps 50 \
  --do_eval false \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 16 \
  --beta 0.04 \
  --temperature 1.0 \
  --top_p 0.9 \
  --top_k 30 \
  --advantage_epsilon 1e-6

echo "Fixed training completed!"

