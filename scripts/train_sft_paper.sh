#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Omni-7B}"
SFT_TRAIN_JSONL="${SFT_TRAIN_JSONL:?Set SFT_TRAIN_JSONL to the MS-Swift training JSONL path}"
SFT_VAL_JSONL="${SFT_VAL_JSONL:-}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:?Set SFT_OUTPUT_DIR to the output directory}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

LR="${LR:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
SAVE_STEPS="${SAVE_STEPS:-250}"
EVAL_STEPS="${EVAL_STEPS:-250}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

cmd=(
  "${SWIFT_BIN}" sft
  --model "${BASE_MODEL}"
  --train_type lora
  --dataset "${SFT_TRAIN_JSONL}"
  --output_dir "${SFT_OUTPUT_DIR}"
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH}"
  --learning_rate "${LR}"
  --num_train_epochs "${NUM_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --target_modules "${TARGET_MODULES}"
  --warmup_ratio "${WARMUP_RATIO}"
  --weight_decay 0.1
  --adam_beta1 0.9
  --adam_beta2 0.95
  --lr_scheduler_type cosine
  --save_steps "${SAVE_STEPS}"
  --logging_steps 10
)

if [[ -n "${SFT_VAL_JSONL}" ]]; then
  cmd+=(
    --val_dataset "${SFT_VAL_JSONL}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --eval_steps "${EVAL_STEPS}"
    --eval_strategy steps
  )
fi

echo "[train_sft_paper] project_root=${PROJECT_ROOT}"
printf '[train_sft_paper] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
