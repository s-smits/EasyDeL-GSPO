#!/bin/bash
# GSPO Training Script with TPU Optimizations

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

# Run simplified GSPO training focused on GSM8K or MATH. Choose dataset via DATASET env var.
# Example: DATASET=math ./run_gspo_training.sh
DATASET=${DATASET:-gsm8k}

python easydel/scripts/finetune/gsm8k_math_gspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset $DATASET \
  --total_batch_size 4 \
  --num_return_sequences 8 \
  --rollout_chunk_size 8 \
  --num_train_epochs 1 \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --learning_rate 2e-6 \
  --dataset_use_rate 10 \
  --force_tensor_parallel 4 \
  --log_logprobs_metrics false \
  --log_global true \
  --log_steps 1 \
  --save_steps 4 \
  --evaluation_steps 100 \
  --do_eval false \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --beta 0.04 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --advantage_epsilon 1e-6

#   --force_data_parallel 1 \

echo "Training completed!"
