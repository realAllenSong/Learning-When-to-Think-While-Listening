#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DAPO_INPUT_JSONL="${DAPO_INPUT_JSONL:?Set DAPO_INPUT_JSONL to the controller training JSONL path}"
SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:?Set SFT_ADAPTER_DIR to the SFT LoRA adapter directory}"
DAPO_RUN_DIR="${DAPO_RUN_DIR:?Set DAPO_RUN_DIR to the output run directory}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Omni-7B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLICY_SERVICE_PYTHON="${POLICY_SERVICE_PYTHON:-${PYTHON_BIN}}"
POLICY_ENDPOINT="${POLICY_ENDPOINT:-}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-}"
MANAGE_POLICY_SERVICE="${MANAGE_POLICY_SERVICE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
REWARD_STACK="${REWARD_STACK:-6reward}"
DAPO_MAX_STEPS="${DAPO_MAX_STEPS:-1000}"
DAPO_WARMUP_STEPS="${DAPO_WARMUP_STEPS:-50}"
POLICY_SERVICE_TP="${POLICY_SERVICE_TP:-2}"
POLICY_ROLLOUT_WORKERS="${POLICY_ROLLOUT_WORKERS:-4}"

manage_policy_flag="--manage-policy-service"
if [[ "${MANAGE_POLICY_SERVICE}" == "0" ]]; then
  manage_policy_flag="--no-manage-policy-service"
fi

reward_args=()
case "${REWARD_STACK}" in
  4reward|4|base)
    reward_args=()
    ;;
  5reward|5|think)
    reward_args=(
      --use-judge
      --reward-use-think-judge
      --no-reward-use-consistency-judge
      --reward-rt-prompt-version thought_quality_judge_prompt
    )
    ;;
  6reward|6|full)
    reward_args=(
      --use-judge
      --reward-use-think-judge
      --reward-use-consistency-judge
      --reward-rt-prompt-version thought_quality_judge_prompt
      --reward-rc-prompt-version chain_consistency_judge_prompt
    )
    ;;
  *)
    echo "Unsupported REWARD_STACK=${REWARD_STACK}. Use 4reward, 5reward, or 6reward." >&2
    exit 2
    ;;
esac

cmd=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/run_grpo.py"
  --input "${DAPO_INPUT_JSONL}"
  --run-dir "${DAPO_RUN_DIR}"
  --backend omni-vllm
  --phase 2
  --group-size 8
  --batch-size 1
  --max-steps "${DAPO_MAX_STEPS}"
  --warmup-steps "${DAPO_WARMUP_STEPS}"
  --epsilon 0.20
  --epsilon-high 0.28
  --dynamic-sample
  --max-resample-times 3
  --trainable
  --actor-model "${BASE_MODEL}"
  --actor-processor "${BASE_MODEL}"
  --actor-reference-model "${BASE_MODEL}"
  --actor-init-adapter "${SFT_ADAPTER_DIR}"
  --actor-reference-adapter "${SFT_ADAPTER_DIR}"
  --actor-use-lora
  --actor-lora-rank 8
  --actor-lora-alpha 32
  --actor-lora-dropout 0.0
  --actor-lora-target-modules all-linear
  --actor-dtype bfloat16
  --actor-learning-rate 4e-7
  --actor-optimizer adamw
  --actor-grad-clip-norm 1.0
  --actor-loss-mode dapo-token
  --actor-kl-beta 0.01
  --actor-credit-assignment hybrid-local
  --actor-hybrid-alpha 0.5
  --actor-turn-batch-size 2
  --policy-max-think-tokens 48
  --policy-max-answer-tokens 48
  --policy-think-temperature 1.0
  --policy-answer-temperature 0.2
  --policy-think-top-p 0.95
  --policy-answer-top-p 0.9
  --policy-temperature-jitter 0.2
  --policy-rollout-workers "${POLICY_ROLLOUT_WORKERS}"
  --policy-audio-window-mode full_prefix
  "${manage_policy_flag}"
  --policy-service-python "${POLICY_SERVICE_PYTHON}"
  --policy-service-initial-model "${BASE_MODEL}"
  --policy-service-lora-path "${SFT_ADAPTER_DIR}"
  --policy-service-enable-lora
  --policy-service-max-model-len 8192
  --policy-service-tensor-parallel-size "${POLICY_SERVICE_TP}"
  --reward-format-scale 1.0
  --reward-sync-scale 1.0
  --reward-update-scale 3.0
  --reward-think-scale 1.0
  --reward-consistency-bonus 0.45
  --sync-free-final-think-tokens 6
  --reward-sync-final-think-token-alpha 0.30
  --reward-sync-final-think-token-penalty-cap 3.0
  --reward-final-short-correct-bonus-scale 0.4
  --reward-final-short-correct-min-tokens 3
  --reward-final-short-correct-max-tokens 6
  --reward-use-update
  --judge-model Qwen/Qwen3.6-35B-A3B
  "${reward_args[@]}"
  --use-reference-answer-fallback
  --checkpoint-every 100
  --full-checkpoint-every 100
  --wandb-mode "${WANDB_MODE}"
)

if [[ -n "${POLICY_ENDPOINT}" ]]; then
  cmd+=(--policy-endpoint "${POLICY_ENDPOINT}")
fi
if [[ -n "${JUDGE_ENDPOINT}" ]]; then
  cmd+=(--judge-endpoint "${JUDGE_ENDPOINT}")
fi

echo "[train_dapo_paper] project_root=${PROJECT_ROOT}"
printf '[train_dapo_paper] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
