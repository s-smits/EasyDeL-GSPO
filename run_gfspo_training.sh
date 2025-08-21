#!/bin/bash
# GFSPO Training Script with TPU Optimizations

# Set environment variables for TPU
export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off

echo "Setting up environment..."
git pull origin main 2>/dev/null || true
uv pip install -e . --quiet
uv pip install "math-verify[antlr4_13_2]" --quiet || true

cd /home/air/EasyDeL-GSPO

# Activate virtual environment if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "Starting GFSPO training with optimized configuration..."

DATASET=${DATASET:-gsm8k}

python easydel/scripts/finetune/gsm8k_math_gfspo.py \
  --repo_id "Qwen/Qwen3-0.6B" \
  --dataset ${DATASET} \
  --total_batch_size 2 \
  --gfpo_group_size 16 \
  --gfpo_retain_count 8 \
  --rollout_chunk_size 4 \
  --num_train_epochs 2 \
  --max_prompt_length 512 \
  --max_completion_length 5632 \
  --learning_rate 2e-6 \
  --dataset_use_rate 10 \
  --force_tensor_parallel 4 \
  --force_data_parallel 8 \
  --log_logprobs_metrics false \
  --log_global true \
  --log_steps 1 \
  --save_steps 100 \
  --do_eval false \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --beta 0.04 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --advantage_epsilon 1e-6

echo "Training completed!"


