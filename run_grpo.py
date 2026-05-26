#!/usr/bin/env python3
"""DAPO-style trainer for the wait-think-answer controller."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any, Dict

from rewards import RewardConfig
from prompts.reward_prompts import (
    CHAIN_CONSISTENCY_JUDGE_PROMPT,
    THOUGHT_QUALITY_JUDGE_PROMPT,
)
from prompts.policy_prompts import (
    DEFAULT_FINAL_POLICY_PROMPT_VERSION,
    DEFAULT_POLICY_PROMPT_VERSION,
    SUPPORTED_POLICY_PROMPT_VERSIONS,
)
from training import (
    HeuristicCaptionPolicyBackend,
    LocalVLLMServiceConfig,
    LocalVLLMServiceController,
    OmniVLLMPolicyBackend,
    OmniActorUpdateConfig,
    TeacherPolicyBackend,
    TrainableOmniActorUpdater,
    TrainerConfig,
    StreamingGRPOTrainer,
    apply_question_visibility_override,
    load_pcot_dataset,
)

DEFAULT_BASE_MODEL_PATH = os.environ.get(
    "WTA_BASE_MODEL",
    "Qwen/Qwen2.5-Omni-7B",
)
DEFAULT_AUDIO_ONLY_SFT_ADAPTER_PATH = os.environ.get(
    "WTA_SFT_ADAPTER",
    "",
)
DEFAULT_ROLLOUT_LORA_NAME = "wait-think-answer-sft"
DEFAULT_POLICY_SERVICE_SERVED_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_ACTOR_DEVICE_MAP = '{"":1}'
DEFAULT_ACTOR_REFERENCE_DEVICE_MAP = '{"":2}'
DEFAULT_ACTOR_MODEL_MAX_MEMORY = "1:150GiB"
DEFAULT_ACTOR_REFERENCE_MAX_MEMORY = "2:150GiB"


def _enable_signal_tracebacks() -> None:
    """Allow live stack dumps from a running training job via SIGUSR1/2."""
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        return

    armed: list[str] = []
    for name in ("SIGUSR1", "SIGUSR2"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True, chain=True)
        except (OSError, RuntimeError, ValueError):
            continue
        armed.append(name)

    if armed:
        print(
            f"[run_grpo] faulthandler ready on pid={os.getpid()} via {', '.join(armed)}",
            file=sys.stderr,
            flush=True,
        )


def build_actor_updater(args):
    if not args.trainable:
        return None
    if args.backend != "omni-vllm":
        raise ValueError("--trainable currently requires backend=omni-vllm")

    return TrainableOmniActorUpdater(
        config=OmniActorUpdateConfig(
            model_name=args.actor_model or args.policy_model,
            processor_name=args.actor_processor or args.actor_model or args.policy_model,
            reference_model_name=args.actor_reference_model or args.policy_model,
            init_adapter_path=args.actor_init_adapter,
            reference_adapter_path=args.actor_reference_adapter,
            use_lora=args.actor_use_lora,
            lora_rank=args.actor_lora_rank,
            lora_alpha=args.actor_lora_alpha,
            lora_dropout=args.actor_lora_dropout,
            lora_target_modules=args.actor_lora_target_modules,
            learning_rate=args.actor_learning_rate,
            weight_decay=args.actor_weight_decay,
            device_map=args.actor_device_map,
            reference_device_map=args.actor_reference_device_map,
            model_max_memory=args.actor_model_max_memory,
            reference_max_memory=args.actor_reference_max_memory,
            dtype=args.actor_dtype,
            grad_clip_norm=args.actor_grad_clip_norm,
            sampling_rate=args.actor_sampling_rate,
            max_turn_examples=args.actor_max_turn_examples,
            turn_batch_size=args.actor_turn_batch_size,
            kl_beta=args.actor_kl_beta,
            gradient_checkpointing=args.actor_gradient_checkpointing,
            cache_audio_arrays=not args.no_actor_audio_cache,
            update_epochs=args.actor_update_epochs,
            epsilon_low=args.epsilon,
            epsilon_high=args.epsilon_high,
            loss_mode=args.actor_loss_mode,
            overlong_shaping=not args.no_actor_overlong_shaping,
            overlong_threshold_tokens=args.actor_overlong_threshold_tokens,
            overlong_penalty_slope=args.actor_overlong_penalty_slope,
            overlong_penalty_cap=args.actor_overlong_penalty_cap,
            warmup_steps=args.warmup_steps,
            credit_assignment=args.actor_credit_assignment,
            hybrid_alpha=args.actor_hybrid_alpha,
            resume_checkpoint=args.resume_full_checkpoint,
            checkpoint_mode=args.checkpoint_mode,
            optimizer_name=args.actor_optimizer,
            load_optimizer_state=not args.resume_only_model,
            deepspeed_enabled=args.actor_deepspeed,
            deepspeed_zero_stage=args.actor_deepspeed_zero_stage,
            deepspeed_micro_batch_size=args.actor_deepspeed_micro_batch_size,
            deepspeed_gradient_accumulation_steps=args.actor_deepspeed_gradient_accumulation_steps,
            deepspeed_config_json=args.actor_deepspeed_config_json,
            deepspeed_offload_optimizer_device=args.actor_deepspeed_offload_optimizer_device,
            deepspeed_offload_param_device=args.actor_deepspeed_offload_param_device,
            answer_fallback_enabled=bool(args.use_reference_answer_fallback),
            answer_fallback_max_new_tokens=args.reference_answer_fallback_max_new_tokens,
        )
    )


def build_policy_backend(args):
    if args.backend == "teacher":
        return TeacherPolicyBackend()
    if args.backend == "heuristic":
        return HeuristicCaptionPolicyBackend(
            no_think_prob=args.no_think_prob,
            correct_answer_prob=args.correct_answer_prob,
            max_think_words=args.max_think_words,
        )
    if args.backend == "omni-vllm":
        service_controller = None
        if args.manage_policy_service:
            service_controller = LocalVLLMServiceController(
                config=LocalVLLMServiceConfig(
                    python_bin=args.policy_service_python,
                    host=args.policy_service_host,
                    port=args.policy_service_port,
                    served_model_name=args.policy_service_served_model_name or args.policy_model,
                    cuda_visible_devices=args.policy_service_cuda_visible_devices,
                    dtype=args.policy_service_dtype,
                    gpu_memory_utilization=args.policy_service_gpu_memory_utilization,
                    max_model_len=args.policy_service_max_model_len,
                    tensor_parallel_size=args.policy_service_tensor_parallel_size,
                    trust_remote_code=not args.no_policy_service_trust_remote_code,
                    ready_timeout_sec=args.policy_service_ready_timeout_sec,
                    enable_lora=bool(args.policy_service_enable_lora),
                    max_loras=args.policy_service_max_loras,
                    lora_name=args.policy_service_lora_name or args.policy_model,
                    lora_base_model_name=args.policy_service_lora_base_model_name or args.policy_service_served_model_name or args.policy_model,
                ),
                initial_model=args.policy_service_initial_model or args.policy_model,
                initial_lora_path=args.policy_service_lora_path,
            )
        return OmniVLLMPolicyBackend(
            endpoint=args.policy_endpoint,
            model_name=args.policy_model,
            api_key=args.policy_api_key,
            timeout_sec=args.policy_timeout_sec,
            max_think_tokens=args.policy_max_think_tokens,
            max_answer_tokens=args.policy_max_answer_tokens,
            think_temperature=args.policy_think_temperature,
            answer_temperature=args.policy_answer_temperature,
            think_top_p=args.policy_think_top_p,
            answer_top_p=args.policy_answer_top_p,
            temperature_jitter=args.policy_temperature_jitter,
            rollout_workers=args.policy_rollout_workers,
            prompt_version=args.policy_prompt_version,
            final_think_prompt_version=args.policy_final_think_prompt_version,
            final_answer_prompt_version=args.policy_final_answer_prompt_version,
            audio_window_mode=args.policy_audio_window_mode,
            overlap_chunks=args.policy_overlap_chunks,
            min_audio_window_sec=args.policy_min_audio_window_sec,
            force_wait_before_sec=args.policy_force_wait_before_sec,
            answer_audio_output=bool(args.policy_answer_audio_output),
            answer_audio_speaker=args.policy_answer_audio_speaker,
            answer_audio_onset_prior_seconds=args.policy_answer_audio_onset_prior_seconds,
            question_visible_from_text=False,
            updater=build_actor_updater(args),
            service_controller=service_controller,
        )
    raise ValueError("Unsupported backend: {}".format(args.backend))


def _resolve_checkpoint_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file() and path.name == "metadata.json":
        return path.parent
    if path.is_file():
        return path.with_suffix("")
    return path


def _load_resume_metadata(path_str: str) -> tuple[str, Dict[str, Any]]:
    checkpoint_dir = _resolve_checkpoint_dir(path_str)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("resume checkpoint is missing metadata.json: {}".format(checkpoint_dir))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(checkpoint_dir), payload


def _upgrade_legacy_reward_prompt_args(args):
    """Fill in public-release reward defaults for paper-aligned DAPO runs."""
    flag = str(os.environ.get("WTA_REWARD_DEFAULTS", "1")).strip().lower()
    prompt_flag = str(os.environ.get("WTA_REWARD_PROMPTS", "1")).strip().lower()
    disabled_values = {"0", "false", "no", "off"}
    if flag in disabled_values:
        return args
    if prompt_flag not in disabled_values:
        args.reward_rt_prompt_version = THOUGHT_QUALITY_JUDGE_PROMPT
        args.reward_rc_prompt_version = CHAIN_CONSISTENCY_JUDGE_PROMPT
    if float(getattr(args, "reward_answer_shape_penalty_scale", 0.0) or 0.0) <= 0.0:
        args.reward_answer_shape_penalty_scale = 1.0
    if float(getattr(args, "reward_final_short_correct_bonus_scale", 0.0) or 0.0) <= 0.0:
        args.reward_final_short_correct_bonus_scale = 0.4
    if not hasattr(args, "reward_final_short_correct_min_tokens") or int(
        getattr(args, "reward_final_short_correct_min_tokens", 999) or 999
    ) > 3:
        args.reward_final_short_correct_min_tokens = 3
    if not hasattr(args, "reward_final_short_correct_max_tokens") or int(
        getattr(args, "reward_final_short_correct_max_tokens", 999) or 999
    ) > 6:
        args.reward_final_short_correct_max_tokens = 6
    if float(getattr(args, "reward_final_short_pairwise_bonus_scale", 0.0) or 0.0) <= 0.0:
        args.reward_final_short_pairwise_bonus_scale = 0.6
    if int(getattr(args, "reward_think_quality_gate_free_final_think_tokens", 999) or 999) > 6:
        args.reward_think_quality_gate_free_final_think_tokens = 6
    if float(getattr(args, "reward_think_quality_gate_final_length_weight", 0.0) or 0.0) <= 1.2:
        args.reward_think_quality_gate_final_length_weight = 1.8
    if int(getattr(args, "sync_free_final_think_tokens", 999) or 999) > 6:
        args.sync_free_final_think_tokens = 6
    if float(getattr(args, "reward_sync_final_think_token_alpha", 0.0) or 0.0) <= 0.18:
        args.reward_sync_final_think_token_alpha = 0.30
    if float(getattr(args, "reward_sync_final_think_token_penalty_cap", 0.0) or 0.0) <= 1.5:
        args.reward_sync_final_think_token_penalty_cap = 3.0
    return args


def main():
    _enable_signal_tracebacks()
    parser = argparse.ArgumentParser(description="Wait-think-answer DAPO trainer")
    parser.add_argument("--input", required=True, help="Controller training JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset rows")
    parser.add_argument("--group-size", type=int, default=8, help="Number of rollout samples generated per prompt")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompt groups accumulated into one optimizer step")
    parser.add_argument("--prompt-batch-workers", type=int, default=1, help="Concurrent workers used to collect prompt groups inside one optimizer step")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", "--epsilon_high", dest="epsilon_high", type=float, default=0.28)
    parser.add_argument(
        "--trajectory-advantage-source",
        default="total",
        choices=["total", "outcome"],
        help="Rollout-level reward source used for group-normalized trajectory advantages",
    )
    parser.add_argument("--dynamic-sample", "--dynamic_sample", dest="dynamic_sample", action="store_true")
    parser.add_argument("--no-dynamic-sample", dest="dynamic_sample", action="store_false")
    parser.add_argument("--max-resample-times", "--max_resample_times", dest="max_resample_times", type=int, default=3)
    parser.add_argument("--dynamic-sample-min-std", "--dynamic_sample_min_std", dest="dynamic_sample_min_std", type=float, default=1e-6)
    parser.add_argument("--dynamic-sample-min-format-pass-rollouts", type=int, default=2)
    parser.add_argument("--dynamic-sample-min-final-think-raw-valid-rollouts", type=int, default=1)
    parser.add_argument("--dynamic-sample-min-pre-eof-think-rollouts", type=int, default=1)
    parser.add_argument("--health-observed-kl-warn", type=float, default=2.0)
    parser.add_argument("--health-observed-kl-critical", type=float, default=5.0)
    parser.add_argument("--health-entropy-warn", type=float, default=0.3)
    parser.add_argument("--health-entropy-critical", type=float, default=0.1)
    parser.add_argument("--health-clip-fraction-warn", type=float, default=0.3)
    parser.add_argument("--health-clip-fraction-critical", type=float, default=0.5)
    parser.add_argument("--health-wait-rate-low-warn", type=float, default=0.1)
    parser.add_argument("--health-wait-rate-high-warn", type=float, default=0.9)
    parser.add_argument("--health-wait-rate-low-critical", type=float, default=0.05)
    parser.add_argument("--health-wait-rate-high-critical", type=float, default=0.95)
    parser.add_argument("--health-wait-rate-critical-start-step", type=int, default=300)
    parser.add_argument("--health-teacher-precision-warn", type=float, default=0.15)
    parser.add_argument("--health-teacher-precision-critical", type=float, default=0.08)
    parser.add_argument("--health-teacher-recall-warn", type=float, default=0.15)
    parser.add_argument("--health-teacher-recall-critical", type=float, default=0.08)
    parser.add_argument("--health-predicted-to-target-ratio-warn", type=float, default=3.0)
    parser.add_argument("--health-predicted-to-target-ratio-critical", type=float, default=5.0)
    parser.add_argument("--health-critical-patience", type=int, default=3)
    parser.add_argument("--health-critical-warmup-steps", type=int, default=50)
    parser.add_argument("--health-warn-only", dest="health_warn_only", action="store_true")
    parser.add_argument("--no-health-warn-only", dest="health_warn_only", action="store_false")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save checkpoint every N steps; 0 disables checkpoint writes")
    parser.add_argument("--checkpoint-keep", type=int, default=3, help="How many regular rollout-reload checkpoints to keep on disk")
    parser.add_argument("--checkpoint-keep-best", type=int, default=5, help="How many best regular checkpoints to keep in addition to the newest checkpoints")
    parser.add_argument("--checkpoint-score-key", default="checkpoint_selection_score", help="Summary metric key used to rank best regular checkpoints")
    parser.add_argument("--checkpoint-score-minimize", action="store_true", help="Rank regular checkpoints by minimizing --checkpoint-score-key instead of maximizing it")
    parser.add_argument("--candidate-checkpoint-keep-best", type=int, default=10, help="How many top-scoring per-step candidate policy checkpoints to keep")
    parser.add_argument("--candidate-checkpoint-alt-score-key", default="mean_total", help="Secondary summary metric key used to rank extra candidate checkpoints")
    parser.add_argument("--candidate-checkpoint-alt-keep-best", type=int, default=10, help="How many extra candidate checkpoints to keep using the secondary score key")
    parser.add_argument("--candidate-checkpoint-alt-score-minimize", action="store_true", help="Rank secondary candidate checkpoints by minimizing --candidate-checkpoint-alt-score-key instead of maximizing it")
    parser.add_argument("--candidate-checkpoint-bucket-size", type=int, default=100, help="Optional step bucket size for preserving additional candidate checkpoints per time window")
    parser.add_argument("--candidate-checkpoint-keep-per-bucket", type=int, default=5, help="How many top-scoring candidate checkpoints to keep within each candidate bucket")
    parser.add_argument("--candidate-checkpoint-alt-bucket-size", type=int, default=100, help="Optional step bucket size for preserving additional secondary-score candidate checkpoints per time window")
    parser.add_argument("--candidate-checkpoint-alt-keep-per-bucket", type=int, default=5, help="How many secondary-score candidate checkpoints to keep within each secondary bucket")
    parser.add_argument("--candidate-checkpoint-user-goal-score-key", default="", help="Optional user-goal metric key used to preserve extra candidate checkpoints")
    parser.add_argument("--candidate-checkpoint-user-goal-keep-best", type=int, default=0, help="How many user-goal-ranked candidate checkpoints to keep")
    parser.add_argument("--candidate-checkpoint-user-goal-score-minimize", action="store_true", help="Rank user-goal candidate checkpoints by minimizing instead of maximizing")
    parser.add_argument("--candidate-checkpoint-user-goal-bucket-size", type=int, default=0, help="Optional step bucket size for preserving user-goal candidate checkpoints")
    parser.add_argument("--candidate-checkpoint-user-goal-keep-per-bucket", type=int, default=0, help="How many user-goal candidate checkpoints to keep within each bucket")
    parser.add_argument("--candidate-checkpoint-score-key", default="candidate_checkpoint_score", help="Summary metric key used to rank candidate checkpoints")
    parser.add_argument("--candidate-checkpoint-score-minimize", action="store_true", help="Rank candidate checkpoints by minimizing --candidate-checkpoint-score-key instead of maximizing it")
    parser.add_argument("--full-checkpoint-every", type=int, default=100, help="Save a full recoverable checkpoint every N steps; 0 disables sparse full checkpoints")
    parser.add_argument("--full-checkpoint-keep", type=int, default=2, help="How many full checkpoints to keep on disk")
    parser.add_argument("--full-checkpoint-keep-best", type=int, default=3, help="How many best full checkpoints to keep in addition to the newest checkpoints")
    parser.add_argument("--full-checkpoint-score-key", default="checkpoint_selection_score", help="Summary metric key used to rank best full checkpoints")
    parser.add_argument("--full-checkpoint-score-minimize", action="store_true", help="Rank full checkpoints by minimizing --full-checkpoint-score-key instead of maximizing it")
    parser.add_argument("--reload-policy-on-checkpoint", dest="reload_policy_on_checkpoint", action="store_true", help="Reload the rollout policy service after each regular checkpoint")
    parser.add_argument("--no-reload-policy-on-checkpoint", dest="reload_policy_on_checkpoint", action="store_false", help="Keep the rollout policy service running across regular checkpoints")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-order-mode", default="sequential", choices=["sequential", "shuffle", "balanced_interleave"])
    parser.add_argument("--sample-order-bucket-keys", default="topic,difficulty", help="Comma-separated difficulty_metadata keys used for balanced interleave bucketing")
    parser.add_argument("--sample-order-seed", type=int, default=-1, help="Seed for sample ordering; negative values fall back to --seed")
    parser.add_argument("--resume-full-checkpoint", default="", help="Resume actor model/optimizer state from a saved full checkpoint directory")
    parser.add_argument("--resume-only-model", action="store_true", help="When resuming, restore model/adapters but do not restore optimizer state")
    parser.add_argument("--backend", default="omni-vllm", choices=["heuristic", "teacher", "omni-vllm"])
    parser.add_argument("--run-dir", default="", help="Output directory under runs/")
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    parser.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false")
    parser.add_argument("--wandb-project", default="wait-think-answer")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-notes", default="")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--trainable", dest="trainable", action="store_true", help="Enable the first non-dry-run actor optimizer path")
    parser.add_argument("--no-trainable", dest="trainable", action="store_false", help="Disable trainable actor updates")
    parser.add_argument("--use-judge", action="store_true", help="Enable LLM judge for R_t/R_c")
    parser.add_argument("--reward-use-think-judge", dest="reward_use_think_judge", action="store_true", help="Enable judge-based R_t scoring")
    parser.add_argument("--no-reward-use-think-judge", dest="reward_use_think_judge", action="store_false", help="Disable judge-based R_t scoring")
    parser.add_argument("--reward-use-consistency-judge", dest="reward_use_consistency_judge", action="store_true", help="Enable judge-based R_c scoring")
    parser.add_argument("--no-reward-use-consistency-judge", dest="reward_use_consistency_judge", action="store_false", help="Disable judge-based R_c scoring")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--judge-endpoint", default="", help="Optional HTTP judge service endpoint")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-batch-size", type=int, default=1)
    parser.add_argument(
        "--judge-http-concurrency",
        type=int,
        default=8,
        help="Concurrent HTTP requests to the judge endpoint; local Transformers fallback still uses judge-batch-size",
    )
    parser.add_argument("--use-reference-answer-fallback", dest="use_reference_answer_fallback", action="store_true", help="Use the updater/reference Omni+LoRA model as an R_a fallback judge for deterministic-fail final answers")
    parser.add_argument("--no-reference-answer-fallback", dest="use_reference_answer_fallback", action="store_false", help="Disable reference-model R_a fallback judging")
    parser.add_argument("--reference-answer-fallback-max-new-tokens", type=int, default=96)
    parser.add_argument("--raw-captions", action="store_true", help="Disable reward-side caption cleanup")
    parser.add_argument("--no-save-rollouts", action="store_true", help="Skip per-step rollout JSON payloads")
    parser.add_argument("--skip-path-check", dest="skip_path_check", action="store_true", help="Skip chunk path existence checks")
    parser.add_argument("--no-skip-path-check", dest="skip_path_check", action="store_false", help="Require chunk path existence checks")
    parser.add_argument("--require-teacher", action="store_true", help="Require think_annotations in input JSONL")
    parser.add_argument("--correct-answer-prob", type=float, default=0.55)
    parser.add_argument("--no-think-prob", type=float, default=0.35)
    parser.add_argument("--max-think-words", type=int, default=8)
    parser.add_argument("--reward-sync-alpha", type=float, default=0.10)
    parser.add_argument("--sync-free-memory-tokens", type=int, default=12)
    parser.add_argument(
        "--reward-accuracy-mode",
        default="difficulty_aware_v1",
        choices=["legacy_quadrant", "difficulty_aware_v1"],
        help="Accuracy/depth reward mode",
    )
    parser.add_argument("--reward-state-floor-tokens", type=int, default=3)
    parser.add_argument("--reward-depth-normalizer-tokens", type=int, default=6)
    parser.add_argument("--reward-difficulty-default", type=float, default=0.5)
    parser.add_argument("--reward-difficulty-margin", type=float, default=0.1)
    parser.add_argument("--reward-lambda-easy", type=float, default=0.5)
    parser.add_argument("--reward-lambda-hard", type=float, default=1.0)
    parser.add_argument("--reward-correct-reward", type=float, default=2.0)
    parser.add_argument("--reward-wrong-penalty", type=float, default=-2.0)
    parser.add_argument("--reward-think-fallback", type=float, default=0.5)
    parser.add_argument("--reward-format-scale", type=float, default=1.0)
    parser.add_argument("--reward-think-scale", type=float, default=1.0)
    parser.add_argument("--reward-consistency-bonus", type=float, default=0.25)
    parser.add_argument("--reward-think-quality-gate-scale", type=float, default=0.0)
    parser.add_argument("--reward-think-quality-gate-target-rate", type=float, default=0.13)
    parser.add_argument("--reward-think-quality-gate-rt-good", type=float, default=0.15)
    parser.add_argument("--reward-think-quality-gate-rc-good", type=float, default=0.25)
    parser.add_argument("--reward-think-quality-gate-quality-floor", type=float, default=0.65)
    parser.add_argument("--reward-think-quality-gate-free-final-think-tokens", type=int, default=6)
    parser.add_argument("--reward-think-quality-gate-final-rt-good", type=float, default=0.50)
    parser.add_argument("--reward-think-quality-gate-overthink-weight", type=float, default=1.0)
    parser.add_argument("--reward-think-quality-gate-final-length-weight", type=float, default=1.8)
    parser.add_argument("--reward-answer-shape-penalty-scale", type=float, default=1.0)
    parser.add_argument("--reward-final-short-correct-bonus-scale", type=float, default=0.4)
    parser.add_argument("--reward-final-short-correct-min-tokens", type=int, default=3)
    parser.add_argument("--reward-final-short-correct-max-tokens", type=int, default=6)
    parser.add_argument("--reward-final-short-pairwise-bonus-scale", type=float, default=0.6)
    parser.add_argument("--reward-sync-scale", type=float, default=1.0)
    parser.add_argument("--reward-prediction-scale", type=float, default=0.2)
    parser.add_argument(
        "--reward-rt-prompt-version",
        default=THOUGHT_QUALITY_JUDGE_PROMPT,
        choices=[THOUGHT_QUALITY_JUDGE_PROMPT],
    )
    parser.add_argument(
        "--reward-rc-prompt-version",
        default=CHAIN_CONSISTENCY_JUDGE_PROMPT,
        choices=[CHAIN_CONSISTENCY_JUDGE_PROMPT],
    )
    parser.add_argument("--reward-use-update", dest="reward_use_update", action="store_true", help="Enable update-timing reward R_u")
    parser.add_argument("--no-reward-use-update", dest="reward_use_update", action="store_false", help="Disable update-timing reward R_u")
    parser.add_argument("--reward-update-scale", type=float, default=3.0)
    parser.add_argument("--reward-update-fallback", type=float, default=0.0)
    parser.add_argument("--reward-update-true-positive-reward", type=float, default=1.25)
    parser.add_argument("--reward-update-true-negative-reward", type=float, default=0.9)
    parser.add_argument("--reward-update-false-positive-penalty", type=float, default=1.5)
    parser.add_argument("--reward-update-false-negative-penalty", type=float, default=1.25)
    parser.add_argument("--reward-update-tolerance-ticks", type=int, default=2)
    parser.add_argument("--reward-update-target-threshold", type=float, default=0.5)
    parser.add_argument("--reward-update-fn-rate-penalty", type=float, default=1.0)
    parser.add_argument("--reward-update-fp-rate-penalty", type=float, default=1.0)
    parser.add_argument("--reward-update-over-prediction-penalty", type=float, default=1.0)
    parser.add_argument("--reward-update-under-prediction-penalty", type=float, default=10.0)
    parser.add_argument("--reward-update-precision-beta", type=float, default=0.9)
    parser.add_argument("--reward-update-progress-power", type=float, default=1.3)
    parser.add_argument("--reward-update-wait-target-start", type=float, default=0.70)
    parser.add_argument("--reward-update-wait-tolerance-start", type=float, default=0.18)
    parser.add_argument("--reward-update-wait-tolerance-end", type=float, default=0.05)
    parser.add_argument("--reward-update-wait-over-target-penalty", type=float, default=2.0)
    parser.add_argument("--reward-update-wait-under-target-penalty", type=float, default=0.75)
    parser.add_argument("--reward-update-true-negative-reward-start", type=float, default=0.35)
    parser.add_argument("--reward-update-under-prediction-penalty-start", type=float, default=2.0)
    parser.add_argument("--reward-update-precision-beta-start", type=float, default=1.1)
    parser.add_argument("--reward-update-teacher-anchor-end", type=float, default=0.35)
    parser.add_argument("--reward-update-policy-correct-reward", type=float, default=1.0)
    parser.add_argument("--reward-update-policy-wrong-penalty", type=float, default=1.5)
    parser.add_argument("--reward-update-policy-think-density-penalty", type=float, default=8.0)
    parser.add_argument("--reward-update-policy-lag-penalty", type=float, default=0.2)
    parser.add_argument("--reward-update-policy-lag-normalizer", type=float, default=2.0)
    parser.add_argument("--reward-update-policy-sparse-target-easy", type=float, default=0.06)
    parser.add_argument("--reward-update-policy-sparse-target-hard", type=float, default=0.15)
    parser.add_argument("--reward-update-policy-sparse-tolerance", type=float, default=0.03)
    parser.add_argument("--reward-update-policy-zero-think-wrong-penalty", type=float, default=0.5)
    parser.add_argument("--reward-update-policy-zero-think-correct-penalty", type=float, default=0.8)
    parser.add_argument("--reward-update-policy-medium-threshold", type=float, default=0.45)
    parser.add_argument("--reward-update-policy-hard-threshold", type=float, default=0.75)
    parser.add_argument("--reward-update-policy-zero-think-medium-multiplier", type=float, default=2.0)
    parser.add_argument("--reward-update-policy-zero-think-hard-multiplier", type=float, default=3.0)
    parser.add_argument("--reward-update-policy-recall-zero-medium-multiplier", type=float, default=1.25)
    parser.add_argument("--reward-update-policy-recall-zero-hard-multiplier", type=float, default=2.0)
    parser.add_argument("--reward-update-policy-target-hit-bonus-medium", type=float, default=0.15)
    parser.add_argument("--reward-update-policy-target-hit-bonus-hard", type=float, default=0.30)
    parser.add_argument("--reward-sync-eof-wait-penalty", type=float, default=0.60)
    parser.add_argument("--reward-sync-answer-alpha", type=float, default=0.02)
    parser.add_argument("--sync-free-answer-tokens", type=int, default=16)
    parser.add_argument("--reward-sync-final-think-token-alpha", type=float, default=0.30)
    parser.add_argument("--sync-free-final-think-tokens", type=int, default=6)
    parser.add_argument("--reward-sync-final-think-token-penalty-cap", type=float, default=3.0)
    parser.add_argument("--reward-sync-latency-token-alpha", type=float, default=0.0)
    parser.add_argument("--sync-free-latency-tokens", type=int, default=0)
    parser.add_argument("--reward-sync-post-eof-wall-clock-alpha", type=float, default=0.0)
    parser.add_argument("--reward-sync-free-post-eof-wall-clock-seconds", type=float, default=0.20)
    parser.add_argument("--reward-sync-text-first-token-alpha", type=float, default=0.0)
    parser.add_argument("--reward-sync-free-text-first-token-seconds", type=float, default=0.20)
    parser.add_argument("--reward-sync-effective-text-first-token-alpha", type=float, default=0.0)
    parser.add_argument("--reward-sync-free-effective-text-first-token-seconds", type=float, default=0.20)
    parser.add_argument("--reward-sync-effective-response-onset-alpha", type=float, default=0.0)
    parser.add_argument("--reward-sync-free-effective-response-onset-seconds", type=float, default=0.20)

    parser.add_argument("--policy-endpoint", default="", help="OpenAI-compatible audio chat endpoint for omni-vllm")
    parser.add_argument("--policy-model", default=DEFAULT_ROLLOUT_LORA_NAME)
    parser.add_argument(
        "--policy-prompt-version",
        default=DEFAULT_POLICY_PROMPT_VERSION,
        choices=SUPPORTED_POLICY_PROMPT_VERSIONS,
    )
    parser.add_argument(
        "--policy-final-think-prompt-version",
        default=DEFAULT_FINAL_POLICY_PROMPT_VERSION,
        choices=SUPPORTED_POLICY_PROMPT_VERSIONS,
    )
    parser.add_argument(
        "--policy-final-answer-prompt-version",
        default=DEFAULT_FINAL_POLICY_PROMPT_VERSION,
        choices=SUPPORTED_POLICY_PROMPT_VERSIONS,
    )
    parser.add_argument("--policy-api-key", default="EMPTY")
    parser.add_argument("--policy-timeout-sec", type=float, default=180.0)
    parser.add_argument("--policy-max-think-tokens", type=int, default=48)
    parser.add_argument("--policy-max-answer-tokens", type=int, default=48)
    parser.add_argument("--policy-think-temperature", type=float, default=0.7)
    parser.add_argument("--policy-answer-temperature", type=float, default=0.2)
    parser.add_argument("--policy-think-top-p", type=float, default=0.95)
    parser.add_argument("--policy-answer-top-p", type=float, default=0.9)
    parser.add_argument("--policy-temperature-jitter", type=float, default=0.0)
    parser.add_argument("--policy-rollout-workers", type=int, default=4, help="Concurrent rollout workers per group for omni-vllm")
    parser.add_argument("--policy-audio-window-mode", default="full_prefix", choices=["full_prefix", "since_last_think"])
    parser.add_argument("--policy-overlap-chunks", type=int, default=1)
    parser.add_argument("--policy-min-audio-window-sec", type=float, default=2.0)
    parser.add_argument(
        "--policy-force-wait-before-sec",
        type=float,
        default=0.0,
        help="Force pre-EOF controller ticks before this audio-prefix end time to emit <wait/> during rollout",
    )
    parser.add_argument("--policy-answer-audio-output", type=int, choices=[0, 1], default=0)
    parser.add_argument("--policy-answer-audio-speaker", default="Chelsie")
    parser.add_argument("--policy-answer-audio-onset-prior-seconds", type=float, default=0.30)
    parser.add_argument("--manage-policy-service", dest="manage_policy_service", action="store_true", help="Launch and manage the local vLLM rollout service from the trainer process")
    parser.add_argument("--no-manage-policy-service", dest="manage_policy_service", action="store_false", help="Use an already-running policy service instead of launching one")
    parser.add_argument("--policy-service-python", default="python3", help="Python binary used to launch the managed local vLLM service")
    parser.add_argument("--policy-service-initial-model", default=DEFAULT_BASE_MODEL_PATH, help="Optional initial model path for the managed rollout service")
    parser.add_argument("--policy-service-served-model-name", default=DEFAULT_POLICY_SERVICE_SERVED_MODEL_NAME, help="Served base-model alias exposed by the managed rollout service")
    parser.add_argument("--policy-service-host", default="127.0.0.1")
    parser.add_argument("--policy-service-port", type=int, default=8100)
    parser.add_argument("--policy-service-cuda-visible-devices", default="0,3", help="CUDA_VISIBLE_DEVICES for the managed rollout service")
    parser.add_argument("--policy-service-dtype", default="bfloat16")
    parser.add_argument("--policy-service-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--policy-service-max-model-len", type=int, default=8192)
    parser.add_argument("--policy-service-tensor-parallel-size", type=int, default=2)
    parser.add_argument("--policy-service-ready-timeout-sec", type=float, default=900.0)
    parser.add_argument("--policy-service-enable-lora", dest="policy_service_enable_lora", action="store_true", help="Serve rollout policy as base model + static LoRA instead of a merged full model")
    parser.add_argument("--no-policy-service-enable-lora", dest="policy_service_enable_lora", action="store_false", help="Serve a merged rollout model instead of a static LoRA")
    parser.add_argument("--policy-service-max-loras", type=int, default=1, help="Maximum number of static LoRAs exposed by the managed rollout service")
    parser.add_argument("--policy-service-lora-name", default=DEFAULT_ROLLOUT_LORA_NAME, help="LoRA alias used by rollout requests")
    parser.add_argument("--policy-service-lora-path", default=DEFAULT_AUDIO_ONLY_SFT_ADAPTER_PATH, help="Path to the rollout LoRA adapter directory or checkpoint dir")
    parser.add_argument("--policy-service-lora-base-model-name", default=DEFAULT_POLICY_SERVICE_SERVED_MODEL_NAME, help="Optional base-model name recorded on the static rollout LoRA")
    parser.add_argument("--no-policy-service-trust-remote-code", action="store_true")

    parser.add_argument("--actor-model", default=DEFAULT_BASE_MODEL_PATH, help="Optional trainable actor model; defaults to --policy-model")
    parser.add_argument("--actor-processor", default=DEFAULT_BASE_MODEL_PATH, help="Optional processor name; defaults to actor/policy model")
    parser.add_argument("--actor-reference-model", default=DEFAULT_BASE_MODEL_PATH, help="Optional fixed reference model path/name for KL")
    parser.add_argument("--actor-init-adapter", default=DEFAULT_AUDIO_ONLY_SFT_ADAPTER_PATH, help="Optional SFT LoRA adapter directory used to initialize the trainable actor")
    parser.add_argument("--actor-reference-adapter", default=DEFAULT_AUDIO_ONLY_SFT_ADAPTER_PATH, help="Optional LoRA adapter directory attached to the KL reference model")
    parser.add_argument("--actor-device-map", default=DEFAULT_ACTOR_DEVICE_MAP, help="Device map for the trainable actor")
    parser.add_argument("--actor-reference-device-map", default=DEFAULT_ACTOR_REFERENCE_DEVICE_MAP, help="Optional device map for the fixed KL reference model")
    parser.add_argument("--actor-model-max-memory", default=DEFAULT_ACTOR_MODEL_MAX_MEMORY, help="Optional max_memory map for actor loading, e.g. '2:170GiB,3:170GiB'")
    parser.add_argument("--actor-reference-max-memory", default=DEFAULT_ACTOR_REFERENCE_MAX_MEMORY, help="Optional max_memory map for reference model loading")
    parser.add_argument("--actor-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--actor-use-lora", dest="actor_use_lora", action="store_true", help="Train LoRA adapters instead of full thinker parameters")
    parser.add_argument("--no-actor-use-lora", dest="actor_use_lora", action="store_false", help="Train full thinker parameters instead of LoRA adapters")
    parser.add_argument("--actor-lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--actor-lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--actor-lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--actor-lora-target-modules", default="all-linear", help="LoRA target modules, e.g. 'all-linear' or a comma-separated list")
    parser.add_argument("--actor-learning-rate", type=float, default=4e-7)
    parser.add_argument("--actor-weight-decay", type=float, default=0.0)
    parser.add_argument("--actor-optimizer", default="adamw", choices=["adamw"])
    parser.add_argument("--actor-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--actor-sampling-rate", type=int, default=16000)
    parser.add_argument("--actor-max-turn-examples", type=int, default=0, help="Optional cap on turn examples per group")
    parser.add_argument("--actor-turn-batch-size", type=int, default=2, help="Training-side micro-batch size for turn examples within the sequence-level updater")
    parser.add_argument("--actor-update-epochs", type=int, default=1, help="Number of PPO/GRPO-style optimizer passes over sampled turn records")
    parser.add_argument(
        "--actor-loss-mode",
        choices=["grpo-sequence", "dapo-token"],
        default="dapo-token",
        help="Actor update objective: sequence-level GRPO-style loss or token-level DAPO-style loss",
    )
    parser.add_argument(
        "--no-actor-overlong-shaping",
        action="store_true",
        help="Disable explicit overlong shaping for the token-level DAPO objective.",
    )
    parser.add_argument("--actor-overlong-threshold-tokens", type=int, default=32)
    parser.add_argument("--actor-overlong-penalty-slope", type=float, default=0.03)
    parser.add_argument("--actor-overlong-penalty-cap", type=float, default=1.0)
    parser.add_argument("--actor-kl-beta", type=float, default=0.01, help="Optional reverse-KL penalty weight")
    parser.add_argument("--actor-credit-assignment", default="hybrid-local", choices=["sequence", "hybrid-local"], help="Turn credit assignment mode for the local trainable actor")
    parser.add_argument("--actor-hybrid-alpha", type=float, default=0.5, help="Outcome/process mixing weight for hybrid-local credit assignment")
    parser.add_argument("--actor-gradient-checkpointing", action="store_true", help="Enable gradient checkpointing for the trainable actor")
    parser.add_argument("--no-actor-audio-cache", action="store_true", help="Disable per-update audio waveform caching inside the actor updater")
    parser.add_argument("--actor-deepspeed", action="store_true", help="Wrap the trainable actor in a DeepSpeed engine for optimizer/ZeRO handling")
    parser.add_argument("--actor-deepspeed-zero-stage", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--actor-deepspeed-micro-batch-size", type=int, default=0, help="Optional DeepSpeed train_micro_batch_size_per_gpu override; 0 reuses --actor-turn-batch-size")
    parser.add_argument("--actor-deepspeed-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--actor-deepspeed-config-json", default="", help="Optional path to a JSON DeepSpeed config; overrides the auto-generated config")
    parser.add_argument("--actor-deepspeed-offload-optimizer-device", default="", choices=["", "cpu", "nvme"])
    parser.add_argument("--actor-deepspeed-offload-param-device", default="", choices=["", "cpu", "nvme"])
    parser.add_argument(
        "--checkpoint-mode",
        default="full",
        choices=["full", "model-only", "metadata-only"],
        help="Checkpoint payload mode for the local trainable actor",
    )
    parser.set_defaults(
        actor_use_lora=True,
        dynamic_sample=True,
        health_warn_only=True,
        manage_policy_service=True,
        policy_service_enable_lora=True,
        reload_policy_on_checkpoint=True,
        reward_use_update=True,
        reward_use_think_judge=False,
        reward_use_consistency_judge=False,
        skip_path_check=True,
        trainable=True,
        use_reference_answer_fallback=True,
        wandb_enabled=True,
    )
    args = _upgrade_legacy_reward_prompt_args(parser.parse_args())

    if args.reward_use_think_judge or args.reward_use_consistency_judge:
        args.use_judge = True

    enable_think_judge = bool(args.use_judge and (args.reward_use_think_judge or (not args.reward_use_think_judge and not args.reward_use_consistency_judge)))
    enable_consistency_judge = bool(args.use_judge and (args.reward_use_consistency_judge or (not args.reward_use_think_judge and not args.reward_use_consistency_judge)))

    resume_step = 0
    if args.resume_full_checkpoint:
        args.resume_full_checkpoint, resume_metadata = _load_resume_metadata(args.resume_full_checkpoint)
        resume_step = int(resume_metadata.get("step", 0) or 0)
        if resume_step >= args.max_steps:
            raise ValueError(
                "--resume-full-checkpoint step {} is not smaller than --max-steps {}".format(
                    resume_step,
                    args.max_steps,
                )
            )
        if args.manage_policy_service and not args.policy_service_initial_model:
            args.policy_service_initial_model = args.resume_full_checkpoint

    if args.backend == "omni-vllm" and not args.policy_endpoint:
        if args.manage_policy_service:
            args.policy_endpoint = "http://{}:{}/v1".format(args.policy_service_host, args.policy_service_port)
        else:
            raise ValueError("--policy-endpoint is required for backend=omni-vllm")

    samples = load_pcot_dataset(
        path=args.input,
        limit=args.limit,
        clean_captions=not args.raw_captions,
        require_teacher=args.require_teacher or args.backend == "teacher",
        check_paths=not args.skip_path_check,
    )
    samples = apply_question_visibility_override(
        samples,
        False,
    )
    print("[run_grpo] controller_visibility=audio_only", file=sys.stderr, flush=True)

    reward_config = RewardConfig(
        format_scale=args.reward_format_scale,
        think_scale=args.reward_think_scale,
        consistency_bonus=args.reward_consistency_bonus,
        think_quality_gate_scale=args.reward_think_quality_gate_scale,
        think_quality_gate_target_rate=args.reward_think_quality_gate_target_rate,
        think_quality_gate_rt_good=args.reward_think_quality_gate_rt_good,
        think_quality_gate_rc_good=args.reward_think_quality_gate_rc_good,
        think_quality_gate_quality_floor=args.reward_think_quality_gate_quality_floor,
        think_quality_gate_free_final_think_tokens=args.reward_think_quality_gate_free_final_think_tokens,
        think_quality_gate_final_rt_good=args.reward_think_quality_gate_final_rt_good,
        think_quality_gate_overthink_weight=args.reward_think_quality_gate_overthink_weight,
        think_quality_gate_final_length_weight=args.reward_think_quality_gate_final_length_weight,
        answer_shape_penalty_scale=args.reward_answer_shape_penalty_scale,
        final_short_correct_bonus_scale=args.reward_final_short_correct_bonus_scale,
        final_short_correct_min_tokens=args.reward_final_short_correct_min_tokens,
        final_short_correct_max_tokens=args.reward_final_short_correct_max_tokens,
        final_short_pairwise_bonus_scale=args.reward_final_short_pairwise_bonus_scale,
        sync_scale=args.reward_sync_scale,
        prediction_scale=args.reward_prediction_scale,
        use_think=enable_think_judge and args.phase >= 2,
        use_consistency=enable_consistency_judge and args.phase >= 2,
        use_sync=args.phase >= 1,
        use_prediction=args.phase >= 3,
        use_update=args.reward_use_update,
        use_reference_answer_fallback=bool(args.use_reference_answer_fallback),
        accuracy_mode=args.reward_accuracy_mode,
        state_floor_tokens=args.reward_state_floor_tokens,
        depth_normalizer_tokens=args.reward_depth_normalizer_tokens,
        difficulty_default=args.reward_difficulty_default,
        difficulty_margin=args.reward_difficulty_margin,
        lambda_easy=args.reward_lambda_easy,
        lambda_hard=args.reward_lambda_hard,
        correct_reward=args.reward_correct_reward,
        wrong_penalty=args.reward_wrong_penalty,
        think_fallback=args.reward_think_fallback,
        rt_prompt_version=args.reward_rt_prompt_version,
        rc_prompt_version=args.reward_rc_prompt_version,
        sync_alpha=args.reward_sync_alpha,
        sync_free_memory_tokens=args.sync_free_memory_tokens,
        update_scale=args.reward_update_scale,
        update_fallback=args.reward_update_fallback,
        update_true_positive_reward=args.reward_update_true_positive_reward,
        update_true_negative_reward=args.reward_update_true_negative_reward,
        update_false_positive_penalty=args.reward_update_false_positive_penalty,
        update_false_negative_penalty=args.reward_update_false_negative_penalty,
        update_tolerance_ticks=args.reward_update_tolerance_ticks,
        update_target_threshold=args.reward_update_target_threshold,
        update_false_negative_rate_penalty=args.reward_update_fn_rate_penalty,
        update_false_positive_rate_penalty=args.reward_update_fp_rate_penalty,
        update_over_prediction_penalty=args.reward_update_over_prediction_penalty,
        update_under_prediction_penalty=args.reward_update_under_prediction_penalty,
        update_precision_beta=args.reward_update_precision_beta,
        update_progress_power=args.reward_update_progress_power,
        update_wait_target_start=args.reward_update_wait_target_start,
        update_wait_tolerance_start=args.reward_update_wait_tolerance_start,
        update_wait_tolerance_end=args.reward_update_wait_tolerance_end,
        update_wait_over_target_penalty=args.reward_update_wait_over_target_penalty,
        update_wait_under_target_penalty=args.reward_update_wait_under_target_penalty,
        update_true_negative_reward_start=args.reward_update_true_negative_reward_start,
        update_under_prediction_penalty_start=args.reward_update_under_prediction_penalty_start,
        update_precision_beta_start=args.reward_update_precision_beta_start,
        update_teacher_anchor_end=args.reward_update_teacher_anchor_end,
        update_policy_correct_reward=args.reward_update_policy_correct_reward,
        update_policy_wrong_penalty=args.reward_update_policy_wrong_penalty,
        update_policy_think_density_penalty=args.reward_update_policy_think_density_penalty,
        update_policy_lag_penalty=args.reward_update_policy_lag_penalty,
        update_policy_lag_normalizer=args.reward_update_policy_lag_normalizer,
        update_policy_sparse_target_easy=args.reward_update_policy_sparse_target_easy,
        update_policy_sparse_target_hard=args.reward_update_policy_sparse_target_hard,
        update_policy_sparse_tolerance=args.reward_update_policy_sparse_tolerance,
        update_policy_zero_think_wrong_penalty=args.reward_update_policy_zero_think_wrong_penalty,
        update_policy_zero_think_correct_penalty=args.reward_update_policy_zero_think_correct_penalty,
        update_policy_medium_threshold=args.reward_update_policy_medium_threshold,
        update_policy_hard_threshold=args.reward_update_policy_hard_threshold,
        update_policy_zero_think_medium_multiplier=args.reward_update_policy_zero_think_medium_multiplier,
        update_policy_zero_think_hard_multiplier=args.reward_update_policy_zero_think_hard_multiplier,
        update_policy_recall_zero_medium_multiplier=args.reward_update_policy_recall_zero_medium_multiplier,
        update_policy_recall_zero_hard_multiplier=args.reward_update_policy_recall_zero_hard_multiplier,
        update_policy_target_hit_bonus_medium=args.reward_update_policy_target_hit_bonus_medium,
        update_policy_target_hit_bonus_hard=args.reward_update_policy_target_hit_bonus_hard,
        sync_eof_wait_penalty=args.reward_sync_eof_wait_penalty,
        sync_answer_alpha=args.reward_sync_answer_alpha,
        sync_free_answer_tokens=args.sync_free_answer_tokens,
        sync_final_think_token_alpha=args.reward_sync_final_think_token_alpha,
        sync_free_final_think_tokens=args.sync_free_final_think_tokens,
        sync_final_think_token_penalty_cap=args.reward_sync_final_think_token_penalty_cap,
        sync_latency_token_alpha=args.reward_sync_latency_token_alpha,
        sync_free_latency_tokens=args.sync_free_latency_tokens,
        sync_post_eof_wall_clock_alpha=args.reward_sync_post_eof_wall_clock_alpha,
        sync_free_post_eof_wall_clock_seconds=args.reward_sync_free_post_eof_wall_clock_seconds,
        sync_text_first_token_alpha=args.reward_sync_text_first_token_alpha,
        sync_free_text_first_token_seconds=args.reward_sync_free_text_first_token_seconds,
        sync_effective_text_first_token_alpha=args.reward_sync_effective_text_first_token_alpha,
        sync_free_effective_text_first_token_seconds=args.reward_sync_free_effective_text_first_token_seconds,
        sync_effective_response_onset_alpha=args.reward_sync_effective_response_onset_alpha,
        sync_free_effective_response_onset_seconds=args.reward_sync_free_effective_response_onset_seconds,
    )
    trainer_config = TrainerConfig(
        input_path=args.input,
        group_size=args.group_size,
        batch_size=args.batch_size,
        prompt_batch_workers=args.prompt_batch_workers,
        max_steps=args.max_steps,
        resume_step=resume_step,
        advantage_source=args.trajectory_advantage_source,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        checkpoint_keep_best=args.checkpoint_keep_best,
        checkpoint_score_key=args.checkpoint_score_key,
        checkpoint_score_maximize=not bool(args.checkpoint_score_minimize),
        candidate_checkpoint_keep_best=args.candidate_checkpoint_keep_best,
        candidate_checkpoint_alt_score_key=args.candidate_checkpoint_alt_score_key,
        candidate_checkpoint_alt_keep_best=args.candidate_checkpoint_alt_keep_best,
        candidate_checkpoint_alt_score_maximize=not bool(args.candidate_checkpoint_alt_score_minimize),
        candidate_checkpoint_bucket_size=args.candidate_checkpoint_bucket_size,
        candidate_checkpoint_keep_per_bucket=args.candidate_checkpoint_keep_per_bucket,
        candidate_checkpoint_alt_bucket_size=args.candidate_checkpoint_alt_bucket_size,
        candidate_checkpoint_alt_keep_per_bucket=args.candidate_checkpoint_alt_keep_per_bucket,
        candidate_checkpoint_user_goal_score_key=args.candidate_checkpoint_user_goal_score_key,
        candidate_checkpoint_user_goal_keep_best=args.candidate_checkpoint_user_goal_keep_best,
        candidate_checkpoint_user_goal_score_maximize=not bool(args.candidate_checkpoint_user_goal_score_minimize),
        candidate_checkpoint_user_goal_bucket_size=args.candidate_checkpoint_user_goal_bucket_size,
        candidate_checkpoint_user_goal_keep_per_bucket=args.candidate_checkpoint_user_goal_keep_per_bucket,
        candidate_checkpoint_score_key=args.candidate_checkpoint_score_key,
        candidate_checkpoint_score_maximize=not bool(args.candidate_checkpoint_score_minimize),
        full_checkpoint_every=args.full_checkpoint_every,
        full_checkpoint_keep=args.full_checkpoint_keep,
        full_checkpoint_keep_best=args.full_checkpoint_keep_best,
        full_checkpoint_score_key=args.full_checkpoint_score_key,
        full_checkpoint_score_maximize=not bool(args.full_checkpoint_score_minimize),
        reload_policy_on_checkpoint=args.reload_policy_on_checkpoint,
        phase=args.phase,
        seed=args.seed,
        use_judge=args.use_judge,
        use_reference_answer_fallback=bool(args.use_reference_answer_fallback),
        run_dir=args.run_dir,
        save_rollouts=not args.no_save_rollouts,
        dry_run=not args.trainable,
        sample_order_mode=args.sample_order_mode,
        sample_order_bucket_keys=args.sample_order_bucket_keys,
        sample_order_seed=(args.seed if int(args.sample_order_seed) < 0 else int(args.sample_order_seed)),
        dynamic_sample=bool(args.dynamic_sample),
        max_resample_times=args.max_resample_times,
        dynamic_sample_min_std=args.dynamic_sample_min_std,
        dynamic_sample_min_format_pass_rollouts=args.dynamic_sample_min_format_pass_rollouts,
        dynamic_sample_min_final_think_raw_valid_rollouts=args.dynamic_sample_min_final_think_raw_valid_rollouts,
        dynamic_sample_min_pre_eof_think_rollouts=args.dynamic_sample_min_pre_eof_think_rollouts,
        health_observed_kl_warn=args.health_observed_kl_warn,
        health_observed_kl_critical=args.health_observed_kl_critical,
        health_entropy_warn=args.health_entropy_warn,
        health_entropy_critical=args.health_entropy_critical,
        health_clip_fraction_warn=args.health_clip_fraction_warn,
        health_clip_fraction_critical=args.health_clip_fraction_critical,
        health_wait_rate_low_warn=args.health_wait_rate_low_warn,
        health_wait_rate_high_warn=args.health_wait_rate_high_warn,
        health_wait_rate_low_critical=args.health_wait_rate_low_critical,
        health_wait_rate_high_critical=args.health_wait_rate_high_critical,
        health_wait_rate_critical_start_step=args.health_wait_rate_critical_start_step,
        health_teacher_precision_warn=args.health_teacher_precision_warn,
        health_teacher_precision_critical=args.health_teacher_precision_critical,
        health_teacher_recall_warn=args.health_teacher_recall_warn,
        health_teacher_recall_critical=args.health_teacher_recall_critical,
        health_predicted_to_target_ratio_warn=args.health_predicted_to_target_ratio_warn,
        health_predicted_to_target_ratio_critical=args.health_predicted_to_target_ratio_critical,
        health_critical_patience=args.health_critical_patience,
        health_critical_warmup_steps=args.health_critical_warmup_steps,
        health_warn_only=bool(args.health_warn_only),
        wandb_enabled=bool(args.wandb_enabled) and str(args.wandb_mode or "online").lower() != "disabled",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        policy_think_temperature=args.policy_think_temperature,
        policy_temperature_jitter=args.policy_temperature_jitter,
        policy_prompt_version=args.policy_prompt_version,
        policy_final_think_prompt_version=args.policy_final_think_prompt_version,
        policy_final_answer_prompt_version=args.policy_final_answer_prompt_version,
        policy_audio_window_mode=args.policy_audio_window_mode,
        policy_force_wait_before_sec=args.policy_force_wait_before_sec,
        policy_question_visible_from_text=False,
        policy_service_cuda_visible_devices=args.policy_service_cuda_visible_devices,
        policy_service_tensor_parallel_size=args.policy_service_tensor_parallel_size,
        policy_service_port=args.policy_service_port,
    )

    judge = None
    if args.use_judge:
        from rewards.judge import LLMJudge

        judge = LLMJudge.get_instance(
            model_name=args.judge_model,
            endpoint=args.judge_endpoint,
            api_key=args.judge_api_key,
            batch_size=args.judge_batch_size,
            http_concurrency=args.judge_http_concurrency,
        )
        judge.start()

    policy_backend = build_policy_backend(args)
    answer_fallback_judge = None
    if args.use_reference_answer_fallback:
        updater = getattr(policy_backend, "updater", None)
        if updater is not None and hasattr(updater, "judge_answer_equivalence_batch"):
            answer_fallback_judge = updater
        else:
            print(
                "[run_grpo] reference answer fallback requested but no updater-backed reference judge is available; disabling",
                file=sys.stderr,
                flush=True,
            )

    trainer = StreamingGRPOTrainer(
        samples=samples,
        policy_backend=policy_backend,
        reward_config=reward_config,
        trainer_config=trainer_config,
        judge=judge,
        answer_fallback_judge=answer_fallback_judge,
        run_dir=args.run_dir or None,
    )

    try:
        final_state = trainer.run()
    finally:
        if judge is not None:
            judge.stop()

    print(json.dumps({"run_dir": str(trainer.run_dir), "final_state": final_state}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
