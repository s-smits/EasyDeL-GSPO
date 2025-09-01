#!/bin/bash
# GSPO Training Script (multi-host ready)
#
# Usage:
#   # Single host (1 process):
#   ./run_gspo_training.sh [DATASET] [CURRICULUM]
#
#   # Multi-host (DP>1): run on each host with unique PROC_ID and same COORD_ADDR
#   COORD_ADDR="host0:12345" NPROCS=4 PROC_ID=0 ./run_gspo_training.sh gsm8k-ds false
#   COORD_ADDR="host0:12345" NPROCS=4 PROC_ID=1 ./run_gspo_training.sh gsm8k-ds false
#   COORD_ADDR="host0:12345" NPROCS=4 PROC_ID=2 ./run_gspo_training.sh gsm8k-ds false
#   COORD_ADDR="host0:12345" NPROCS=4 PROC_ID=3 ./run_gspo_training.sh gsm8k-ds false
#
# Environment variables (optional):
#   COORD_ADDR  - JAX coordinator address (host:port), required when NPROCS>1.
#   NPROCS      - Number of JAX processes (data-parallel workers). Default: 1
#   PROC_ID     - Rank of this process in [0..NPROCS-1]. Default: 0
#   FORCE_DP    - Desired data parallel degree. Default: NPROCS
#   FORCE_TP    - Desired tensor parallel degree. Default: 4
#   BATCH       - Per-process prompt batch size. Default: 4
#   NRS         - num_return_sequences per prompt. Default: 4
#   MAX_PROMPT  - Max prompt tokens. Default: 512
#   MAX_COMP    - Max completion tokens. Default: 1024
#   USE_WANDB   - Enable wandb logging (true/false). Default: false
#   LOG_GLOBAL  - Enable cross-process global logging (true/false). Default: false

set -euo pipefail

export JAX_PLATFORMS=tpu
export JAX_TRACEBACK_FILTERING=off
# Disable best-effort global host aggregations inside reward functions for stability
export EASYDEL_DISABLE_GLOBAL_AGG=1

DATASET="${1:-gsm8k-ds}"
CURR="${2:-false}"

COORD_ADDR="${COORD_ADDR:-localhost:12355}"
NPROCS="${NPROCS:-1}"
PROC_ID="${PROC_ID:-0}"
FORCE_DP="${FORCE_DP:-$NPROCS}"
FORCE_TP="${FORCE_TP:-4}"
BATCH="${BATCH:-4}"
NRS="${NRS:-4}"
MAX_PROMPT="${MAX_PROMPT:-512}"
MAX_COMP="${MAX_COMP:-1024}"
USE_WANDB="${USE_WANDB:-false}"
LOG_GLOBAL="${LOG_GLOBAL:-false}"

# Enable JAX distributed only when multi-process
INIT_DIST=false
if [ "${NPROCS}" -gt 1 ]; then
  INIT_DIST=true
  if [ -z "${COORD_ADDR}" ]; then
    echo "ERROR: COORD_ADDR must be set when NPROCS>1" >&2
    exit 1
  fi
fi

echo "Starting GSPO training with optimized configuration..."
echo "Dataset: ${DATASET} | Curriculum: ${CURR}"
echo "DP=${FORCE_DP} TP=${FORCE_TP} NPROCS=${NPROCS} PROC_ID=${PROC_ID}"
echo "Batch=${BATCH} NRS=${NRS} MaxPrompt=${MAX_PROMPT} MaxComp=${MAX_COMP}"

# Optional: ensure editable install + math-verify
if command -v uv >/dev/null 2>&1; then
  uv pip install -e . --quiet || true
  uv pip install "math-verify[antlr4_13_2]" --quiet || true
fi

python3.11 easydel/scripts/finetune/gsm8k_math_gspo.py \
  --repo_id "Qwen/Qwen3-1.7B" \
  --dataset "${DATASET}" \
  --curriculum_math "${CURR}" \
  --total_batch_size "${BATCH}" \
  --num_return_sequences "${NRS}" \
  --rollout_chunk_size 1 \
  --num_train_epochs 2 \
  --max_prompt_length "${MAX_PROMPT}" \
  --max_completion_length "${MAX_COMP}" \
  --learning_rate 2e-6 \
  --dataset_use_pct 1.0 \
  --force_tensor_parallel "${FORCE_TP}" \
  --force_data_parallel "${FORCE_DP}" \
  --log_logprobs_metrics false \
  --log_global "${LOG_GLOBAL}" \
  --use_wandb "${USE_WANDB}" \
  --log_steps 1 \
  --save_steps 100 \
  --do_eval false \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --beta 0.04 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --advantage_epsilon 1e-6 \
  --jax_distributed_config.initialize_jax_distributed "${INIT_DIST}" \
  --jax_distributed_config.coordinator_address "${COORD_ADDR}" \
  --jax_distributed_config.num_processes "${NPROCS}" \
  --jax_distributed_config.process_id "${PROC_ID}"

echo "Training completed!"
