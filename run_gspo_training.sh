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

echo "Setting up environment..."
# Set environment variables for TPU
export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off
# Show DEBUG logs from EasyDeL when debug flags are enabled (safe to always enable during debugging)
export EASYDEL_LOG_LEVEL=DEBUG

# Navigate to project directory and pull latest changes (non-fatal)
cd /home/air/EasyDeL-GSPO || exit 1
git pull origin gfspo-wshrink 2>/dev/null || true

# Install project and math-verify dependency
uv pip install -e . --quiet
uv pip install "math-verify[antlr4_13_2]" --quiet || true

echo "Starting GSPO training with optimized configuration..."

# Parse command line arguments
DATASET="${1:-math-ds}"
CURRICULUM_MATH="${2:-false}"

# If DATASET is math-ds, enable curriculum learning (mirrors GFSPO script behavior)
if [ "$DATASET" = "math-ds" ]; then
  CURRICULUM_MATH="true"
fi

echo "Using dataset: ${DATASET}"
echo "Curriculum math: ${CURRICULUM_MATH}"

# Metrics and logging: ensure correct aggregation for 8 workers
# Set log_global to true for correct global logging aggregation
LOG_GLOBAL_VAL=${LOG_GLOBAL:-true}
echo "LOG_GLOBAL: ${LOG_GLOBAL_VAL}"

#!/usr/bin/env bash

python3.11 easydel/scripts/finetune/gsm8k_math_gspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset ${DATASET} \
  --curriculum_math ${CURRICULUM_MATH} \
  --total_batch_size 1 \
  --num_return_sequences 8 \
  --rollout_chunk_size 1 \
  --num_train_epochs 2 \
  --max_prompt_length 512 \
  --max_completion_length 4096 \
  --learning_rate 2e-6 \
  --dataset_use_pct 10 \
  --force_tensor_parallel 4 \
  --force_data_parallel 2 \
  --report_steps 1 \
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
  --advantage_epsilon 1e-6 \
  --logprob_analysis_enable false \
  --sync_multihost_phases true \
  --sync_ref_model true \
  --ref_model_sync_steps 4 \
  --sync_ref_model_on_step_start true \
  --ref_sync_copy_graphother true \
  --logprob_alignment_check_on_sync false \
  --cap_rollout_chunk_to_tp true \
  --verbose true \
  --debug_enable true \
  --debug_jit_print_every_n 1 \
  --debug_rank0_only true \
  --debug_dump_batch_shapes true

echo "Training completed!"
