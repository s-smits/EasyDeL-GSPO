#!/bin/bash
# Safe GSPO Multi-Host Testing Configuration
#
# This configuration prioritizes stability over performance for testing
# multi-host synchronization fixes

echo "Testing safe GSPO configuration with multi-host synchronization fixes..."

# Navigate to project directory
cd /home/air/EasyDeL-GSPO || exit 1

# Core safety settings for multi-host
export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off
# Force deterministic compilation
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_deterministic_ops=true"

echo "Starting safe test with minimal configuration..."

python3.11 easydel/scripts/finetune/gsm8k_math_gspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset "math-ds" \
  --curriculum_math false \
  --total_batch_size 1 \
  --num_return_sequences 1 \
  --rollout_chunk_size 1 \
  --num_train_epochs 1 \
  --max_steps 2 \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --learning_rate 2e-6 \
  --dataset_use_pct 1 \
  --force_tensor_parallel 4 \
  --force_data_parallel 2 \
  --report_steps 1 \
  --log_steps 1 \
  --save_steps 100 \
  --do_eval false \
  --gradient_accumulation_steps 1 \
  --beta 0.04 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --advantage_epsilon 1e-6 \
  --sync_ref_model true \
  --ref_model_sync_steps 1 \
  --sync_ref_model_on_step_start true \
  --ref_model_sync_strategy "hard" \
  --ref_sync_copy_graphother true \
  --logprob_alignment_check_on_sync false \
  --sync_multihost_phases true \
  --cap_rollout_chunk_to_tp true \
  --logprob_analysis_enable false \
  --verbose true \
  --use_wandb false

echo "Safe test completed!"
