"""
Streaming GRPO trainer.

This trainer supports both:
- dry-run rollout/reward/logging validation
- the first non-dry-run actor optimizer smoke path

It is still not the final production online RL loop: rollout-service reload,
KV-cache reuse, and full KL/reference-model training remain separate work.
"""

import concurrent.futures
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rewards import RewardConfig, compute_rewards_batch, format_reward_summary

from .checkpointing import (
    checkpoint_would_enter_bucket_top_k,
    checkpoint_would_enter_top_k,
    prune_step_checkpoints,
    prune_step_checkpoints_recent_and_best,
    ranked_step_checkpoints,
)
from .sample_order import reorder_items


_PLACEHOLDER_PATTERNS = (
    "collecting evidence",
    "audio continues",
    "still listening",
    "audio playing",
    "final reasoning state",
)

_META_THINK_PATTERNS = (
    "tone",
    "pause",
    "hesitat",
    "speaker hesitat",
    "speaker sounds",
    "speaker seems",
    "speaker tone",
    "speaker pauses",
    "sounds unsure",
    "sounds certain",
    "question asks",
    "question wants",
    "the question",
    "options are",
    "need answer",
    "task asks",
    "audio asks",
)

_THINK_ACTION_PATTERN = re.compile(r"\s*<think>(.*?)</think>\s*", re.IGNORECASE | re.DOTALL)

_WANDB_CORE_METRIC_KEYS = frozenset(
    {
        "answer_correct_rate",
        "format_pass_rate",
        "mean_total",
        "mean_total_reward",
        "mean_R_f",
        "mean_R_a",
        "mean_R_t",
        "mean_R_t_final",
        "mean_R_t_final_judged",
        "mean_R_u",
        "mean_R_s",
        "mean_R_c",
        "mean_wait_rate",
        "mean_think_rate",
        "mean_final_think_token_count",
        "mean_response_latency_tokens",
        "mean_effective_text_first_token_seconds",
        "mean_post_eof_wall_clock_seconds",
        "mean_teacher_anchor_weight",
        "mean_teacher_score",
        "mean_policy_score",
        "mean_teacher_boundary_precision",
        "mean_teacher_boundary_recall",
        "update_observed_kl",
        "update_clip_fraction",
        "update_entropy",
        "candidate_checkpoint_score",
        "checkpoint_selection_score",
        "user_goal_checkpoint_score",
        "think_rate_target_gap",
        "think_rate_target_penalty",
        "think_rate_low_penalty",
        "think_rate_high_penalty",
        "think_rate_quality_factor",
        "final_think_token_penalty",
        "final_think_token_hard_penalty",
        "final_think_token_soft_cost",
        "final_think_token_ideal_bonus",
        "mean_final_short_pairwise_adjustment",
        "final_think_raw_valid_rate",
        "final_think_fallback_rate",
        "answer_leak_rate",
        "meta_think_rate",
        "final_think_placeholder_rate",
        "final_think_meta_rate",
        "protocol_violation_penalty",
        "rt_quality_gate_penalty",
        "rt_final_quality_gate_penalty",
        "rc_quality_gate_penalty",
        "reasoning_state_quality_penalty",
        "health_stop_triggered",
    }
)

_REWARD_METRIC_KEYS = ("R_f", "R_a", "R_t", "R_t_final", "R_t_final_judged", "R_u", "R_s", "R_c")
_CONTROLLER_METRIC_KEYS = (
    "answer_rule_correct",
    "answer_fallback_invoked",
    "answer_fallback_rescued",
    "answer_fallback_short_circuit_no_final_answer",
    "wait_rate",
    "think_rate",
    "updates_per_episode",
    "teacher_boundary_recall",
    "teacher_boundary_precision",
    "teacher_boundary_f1",
    "teacher_wait_rate",
    "scheduled_wait_target",
    "scheduled_wait_tolerance",
    "wait_over_target_gap",
    "wait_under_target_gap",
    "wait_corridor_penalty",
    "teacher_anchor_weight",
    "teacher_score",
    "policy_score",
    "policy_think_density",
    "policy_lag_cost",
    "policy_sparse_target_density",
    "policy_sparse_tolerance",
    "policy_overthink_gap",
    "policy_underthink_gap",
    "post_eof_wall_clock_seconds",
    "post_eof_wall_clock_penalty",
    "text_first_token_wall_clock_seconds",
    "text_first_token_penalty",
    "effective_text_first_token_seconds",
    "effective_text_first_token_penalty",
    "text_streaming_supported",
    "effective_response_onset_seconds",
    "effective_response_onset_penalty",
    "response_onset_seconds",
    "answer_generation_wall_clock_seconds",
    "controller_total_wall_clock_seconds",
    "final_think_generation_wall_clock_seconds",
    "final_think_token_count",
    "final_think_token_penalty",
    "final_short_pairwise_adjustment",
    "response_latency_token_proxy",
    "sync_think_verbosity_penalty",
    "sync_answer_length_penalty",
    "sync_symbolic_eof_wait_penalty",
    "sync_token_latency_penalty",
    "false_negative_update_rate",
    "false_positive_update_rate",
    "predicted_to_target_ratio",
    "over_prediction_excess",
    "episodes_with_zero_think_before_eof",
    "eof_to_answer_lag",
    "answer_start_delay_steps",
)


def normalize_advantages(totals: List[float]) -> List[float]:
    if not totals:
        return []

    mean = sum(totals) / float(len(totals))
    variance = sum((value - mean) ** 2 for value in totals) / float(len(totals))
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.0 for _ in totals]
    return [(value - mean) / std for value in totals]


def select_advantage_values(
    rewards: List[Dict[str, float]],
    *,
    source: str = "total",
) -> List[float]:
    normalized_source = str(source or "total").strip().lower()
    if normalized_source == "outcome":
        return [float(reward.get("R_outcome", reward.get("total", 0.0))) for reward in rewards]
    return [float(reward.get("total", 0.0)) for reward in rewards]


def _final_shortness_score(final_tokens: float) -> float:
    """Piecewise preference for compact final answer cues."""
    tokens = float(final_tokens)
    if tokens <= 0.0:
        return -1.0
    if tokens <= 6.0:
        return 1.0
    if tokens <= 8.0:
        return 0.25
    if tokens <= 10.0:
        return -0.50
    return -1.0


def _is_final_short_pairwise_eligible(reward: Dict[str, float]) -> bool:
    if float(reward.get("R_a", 0.0)) <= 0.0:
        return False
    if float(reward.get("answer_shape_penalty", 0.0)) < 0.0:
        return False
    if float(reward.get("final_think_token_count", 0.0)) <= 0.0:
        return False
    if float(reward.get("R_t_final_judged", 0.0)) > 0.0 and float(reward.get("R_t_final", 0.0)) < 0.50:
        return False
    return True


def _apply_final_short_pairwise_adjustments(rewards: List[Dict[str, float]], *, scale: float) -> None:
    """Make shorter correct final cues win within the same prompt group."""
    for reward in rewards:
        reward.setdefault("final_short_pairwise_adjustment", 0.0)

    scale = float(scale or 0.0)
    if scale <= 0.0:
        return

    eligible = [
        (idx, _final_shortness_score(float(reward.get("final_think_token_count", 0.0))))
        for idx, reward in enumerate(rewards)
        if _is_final_short_pairwise_eligible(reward)
    ]
    if len(eligible) < 2:
        return

    mean_score = sum(score for _idx, score in eligible) / float(len(eligible))
    for idx, score in eligible:
        adjustment = scale * (score - mean_score)
        rewards[idx]["final_short_pairwise_adjustment"] = float(adjustment)
        rewards[idx]["R_outcome"] = (
            float(rewards[idx].get("R_outcome", rewards[idx].get("total", 0.0))) + adjustment
        )
        rewards[idx]["total"] = float(rewards[idx].get("total", 0.0)) + adjustment


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return {"mean": mean, "std": math.sqrt(max(0.0, variance))}


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _word_count(text: str) -> int:
    return len([token for token in str(text or "").strip().split() if token])


def _placeholder_ratio(thinks: List[str]) -> float:
    if not thinks:
        return 0.0
    hits = 0
    for think in thinks:
        normalized = _normalize_text(think)
        if not normalized or any(pattern in normalized for pattern in _PLACEHOLDER_PATTERNS):
            hits += 1
    return hits / float(len(thinks))


def _is_valid_raw_think_action(text: str) -> bool:
    match = _THINK_ACTION_PATTERN.fullmatch(str(text or "").strip())
    if match is None:
        return False
    body = str(match.group(1) or "").strip()
    if not body:
        return False
    lowered = str(text or "").lower()
    if "<answer>" in lowered or "</answer>" in lowered or "<wait" in lowered:
        return False
    return True


def _think_body_from_action(text: str) -> str:
    match = _THINK_ACTION_PATTERN.search(str(text or ""))
    if match is None:
        return ""
    return str(match.group(1) or "").strip()


def _step_model_action_text(step: Dict[str, Any]) -> str:
    model_raw = str(step.get("model_raw_output", "") or "").strip()
    if model_raw:
        return model_raw
    return str(step.get("raw_output", "") or "").strip()


def _step_controller_action_text(step: Dict[str, Any]) -> str:
    normalized = str(step.get("normalized_output", "") or "").strip()
    if normalized:
        return normalized
    return _step_model_action_text(step)


def _is_final_think_step(step: Dict[str, Any]) -> bool:
    timing = dict(step.get("timing") or {})
    return str(step.get("turn_type", "")).strip().lower() == "think" and bool(timing.get("is_final_think"))


def _rollout_final_think_raw_valid(rollout: Any) -> bool:
    steps = list(getattr(rollout, "steps", []) or [])
    final_steps = []
    for step in steps:
        timing = dict(getattr(step, "timing", {}) or {})
        turn_type = str(getattr(step, "turn_type", "") or "").strip().lower()
        if turn_type == "think" and bool(timing.get("is_final_think")):
            final_steps.append(step)
    if not final_steps:
        return False
    final_step = final_steps[-1]
    timing = dict(getattr(final_step, "timing", {}) or {})
    raw_valid = timing.get("final_think_raw_valid")
    if raw_valid is not None:
        return bool(raw_valid)
    return _is_valid_raw_think_action(str(getattr(final_step, "raw_output", "") or ""))


def _rollout_has_valid_pre_eof_think(rollout: Any) -> bool:
    steps = list(getattr(rollout, "steps", []) or [])
    for step in steps:
        timing = dict(getattr(step, "timing", {}) or {})
        if bool(timing.get("is_final_think")):
            continue
        turn_type = str(getattr(step, "turn_type", "") or "").strip().lower()
        if turn_type != "think":
            continue
        raw_output = str(getattr(step, "raw_output", "") or "").strip()
        if raw_output and _is_valid_raw_think_action(raw_output):
            return True
        if not raw_output and _is_valid_raw_think_action(str(getattr(step, "normalized_output", "") or "")):
            return True
    return False


def _dynamic_protocol_gate_detail(
    *,
    rollouts: List[Any],
    rewards: List[Dict[str, Any]],
    min_format_pass_rollouts: int,
    min_final_think_raw_valid_rollouts: int,
    min_pre_eof_think_rollouts: int,
) -> Dict[str, Any]:
    group_size = max(len(rollouts), len(rewards))
    format_required = min(group_size, max(0, int(min_format_pass_rollouts or 0)))
    final_required = min(group_size, max(0, int(min_final_think_raw_valid_rollouts or 0)))
    pre_eof_required = min(group_size, max(0, int(min_pre_eof_think_rollouts or 0)))
    format_pass_count = sum(1 for reward in rewards if float(reward.get("R_f", 1.0)) > 0.0)
    final_valid_count = sum(1 for rollout in rollouts if _rollout_final_think_raw_valid(rollout))
    pre_eof_think_count = sum(1 for rollout in rollouts if _rollout_has_valid_pre_eof_think(rollout))
    format_failed = bool(format_required > 0 and format_pass_count < format_required)
    final_failed = bool(final_required > 0 and final_valid_count < final_required)
    pre_eof_failed = bool(pre_eof_required > 0 and pre_eof_think_count < pre_eof_required)
    reasons = []
    if format_failed:
        reasons.append("format_pass_count<{}".format(format_required))
    if final_failed:
        reasons.append("final_think_raw_valid_count<{}".format(final_required))
    if pre_eof_failed:
        reasons.append("pre_eof_think_rollout_count<{}".format(pre_eof_required))
    return {
        "failed": bool(format_failed or final_failed or pre_eof_failed),
        "format_pass_count": int(format_pass_count),
        "format_pass_required": int(format_required),
        "final_think_raw_valid_count": int(final_valid_count),
        "final_think_raw_valid_required": int(final_required),
        "pre_eof_think_rollout_count": int(pre_eof_think_count),
        "pre_eof_think_rollout_required": int(pre_eof_required),
        "reason": ",".join(reasons),
    }


def _is_meta_or_placeholder_think(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if any(pattern in normalized for pattern in _PLACEHOLDER_PATTERNS):
        return True
    return any(pattern in normalized for pattern in _META_THINK_PATTERNS)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _think_rate_quality_factor(*, rt_value: float, rc_value: float) -> float:
    """How much checkpoint ranking should trust a high think-rate.

    High-rate thinking is only acceptable when the judge signals that the thinks
    are useful and the chain is supported. Low-quality overthinking still gets
    penalized hard.
    """

    return _clip01(0.60 * (float(rt_value) / 0.25) + 0.40 * (float(rc_value) / 0.60))


def _think_rate_preference_penalty(
    mean_think_rate: float,
    *,
    rt_value: float,
    rc_value: float,
) -> Dict[str, float]:
    """Ranking penalty for the user target: sparse facts, adaptive reasoning.

    Preferred global band is 5-15%. Under 5% is treated as an all-wait risk.
    15-20% is a mild overthink zone. Above 20% is strong unless R_t/R_c
    indicate that the extra thinking is genuinely high quality.
    """

    rate = max(0.0, float(mean_think_rate))
    quality = _think_rate_quality_factor(rt_value=rt_value, rc_value=rc_value)

    if rate < 0.05:
        low_penalty = 0.35 + 0.65 * ((0.05 - rate) / 0.05)
    else:
        low_penalty = 0.0

    if rate <= 0.15:
        high_raw = 0.0
    elif rate <= 0.20:
        high_raw = 0.20 * ((rate - 0.15) / 0.05)
    else:
        high_raw = 0.20 + 0.80 * min(1.0, (rate - 0.20) / 0.15)
    high_discount = 1.0 - 0.55 * quality
    high_penalty = high_raw * high_discount

    if 0.05 <= rate <= 0.15:
        target_gap = 0.0
    elif rate < 0.05:
        target_gap = 0.05 - rate
    else:
        target_gap = rate - 0.15

    return {
        "target": 0.13,
        "target_low": 0.05,
        "target_high": 0.15,
        "strong_low": 0.05,
        "mild_high": 0.20,
        "quality_factor": float(quality),
        "low_penalty": float(low_penalty),
        "high_penalty": float(high_penalty),
        "target_gap": float(target_gap),
        "total_penalty": float(low_penalty + high_penalty),
    }


def _build_step_monitoring_summary(group_payloads: List[Dict]) -> Dict[str, float]:
    reward_values = {key: [] for key in _REWARD_METRIC_KEYS}
    controller_values = {key: [] for key in _CONTROLLER_METRIC_KEYS}
    rollout_totals: List[float] = []
    rollout_outcomes: List[float] = []
    rollout_ra: List[float] = []
    rollout_rf: List[float] = []
    think_lengths: List[float] = []
    placeholder_ratios: List[float] = []
    meta_think_ratios: List[float] = []
    final_think_placeholder_flags: List[float] = []
    final_think_meta_flags: List[float] = []
    final_think_raw_valid_flags: List[float] = []
    final_think_fallback_flags: List[float] = []
    answer_leak_flags: List[float] = []
    repeated_hits = 0
    delta_hits = 0
    delta_total = 0
    n_rollouts = 0

    for group_payload in group_payloads:
        for rollout_payload in group_payload.get("rollouts", []):
            n_rollouts += 1
            reward = rollout_payload.get("reward", {})
            rollout = rollout_payload.get("rollout", {})

            for key in _REWARD_METRIC_KEYS:
                reward_values[key].append(float(reward.get(key, 0.0)))
            for key in _CONTROLLER_METRIC_KEYS:
                controller_values[key].append(float(reward.get(key, 0.0)))

            rollout_totals.append(float(reward.get("total", 0.0)))
            rollout_outcomes.append(float(reward.get("R_outcome", reward.get("total", 0.0))))
            rollout_ra.append(float(reward.get("R_a", 0.0)))
            rollout_rf.append(float(reward.get("R_f", 0.0)))

            thinks = [str(item) for item in rollout.get("thinks", [])]
            if thinks:
                think_lengths.extend(float(_word_count(think)) for think in thinks)
                placeholder_ratios.append(_placeholder_ratio(thinks))
                meta_think_ratios.append(
                    sum(1.0 for think in thinks if _is_meta_or_placeholder_think(think)) / float(len(thinks))
                )
                normalized = [_normalize_text(think) for think in thinks]
                for prev, curr in zip(normalized[:-1], normalized[1:]):
                    delta_total += 1
                    if prev == curr:
                        repeated_hits += 1
                    else:
                        delta_hits += 1

            rollout_answer_leak = False
            final_think_steps = []
            step_thinks: List[str] = []
            for step in rollout.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                turn_type = str(step.get("turn_type", "")).strip().lower()
                timing = dict(step.get("timing") or {})
                model_action = _step_model_action_text(step)
                controller_action = _step_controller_action_text(step)
                lowered_model_action = model_action.lower()
                if turn_type != "answer" and (
                    "<answer>" in lowered_model_action
                    or "</answer>" in lowered_model_action
                    or str(timing.get("final_think_raw_turn_type", "")).strip().lower() == "answer"
                ):
                    rollout_answer_leak = True
                if turn_type == "think":
                    body = str(step.get("think", "") or "").strip()
                    if not body:
                        body = _think_body_from_action(controller_action)
                    if body:
                        step_thinks.append(body)
                if _is_final_think_step(step):
                    final_think_steps.append(step)

            if step_thinks and not thinks:
                think_lengths.extend(float(_word_count(think)) for think in step_thinks)
                placeholder_ratios.append(_placeholder_ratio(step_thinks))
                meta_think_ratios.append(
                    sum(1.0 for think in step_thinks if _is_meta_or_placeholder_think(think))
                    / float(len(step_thinks))
                )

            if final_think_steps:
                final_step = final_think_steps[-1]
                timing = dict(final_step.get("timing") or {})
                final_body = str(final_step.get("think", "") or "").strip()
                if not final_body:
                    final_body = _think_body_from_action(_step_controller_action_text(final_step))
                final_think_placeholder_flags.append(_placeholder_ratio([final_body]))
                final_think_meta_flags.append(1.0 if _is_meta_or_placeholder_think(final_body) else 0.0)
                raw_valid = timing.get("final_think_raw_valid")
                if raw_valid is None:
                    raw_valid = _is_valid_raw_think_action(_step_model_action_text(final_step))
                final_think_raw_valid_flags.append(1.0 if bool(raw_valid) else 0.0)
                final_think_fallback_flags.append(1.0 if bool(timing.get("final_think_fallback_used")) else 0.0)
            if rollout.get("steps"):
                answer_leak_flags.append(1.0 if rollout_answer_leak else 0.0)

    summary: Dict[str, float] = {
        "n_rollouts": int(n_rollouts),
        "mean_think_length": (sum(think_lengths) / float(len(think_lengths))) if think_lengths else 0.0,
        "placeholder_rate": (sum(placeholder_ratios) / float(len(placeholder_ratios))) if placeholder_ratios else 0.0,
        "meta_think_rate": (sum(meta_think_ratios) / float(len(meta_think_ratios))) if meta_think_ratios else 0.0,
        "final_think_placeholder_rate": (
            sum(final_think_placeholder_flags) / float(len(final_think_placeholder_flags))
            if final_think_placeholder_flags
            else 0.0
        ),
        "final_think_meta_rate": (
            sum(final_think_meta_flags) / float(len(final_think_meta_flags))
            if final_think_meta_flags
            else 0.0
        ),
        "final_think_raw_valid_rate": (
            sum(final_think_raw_valid_flags) / float(len(final_think_raw_valid_flags))
            if final_think_raw_valid_flags
            else 1.0
        ),
        "final_think_fallback_rate": (
            sum(final_think_fallback_flags) / float(len(final_think_fallback_flags))
            if final_think_fallback_flags
            else 0.0
        ),
        "answer_leak_rate": (
            sum(answer_leak_flags) / float(len(answer_leak_flags)) if answer_leak_flags else 0.0
        ),
        "repeated_think_rate": (repeated_hits / float(delta_total)) if delta_total else 0.0,
        "think_delta_rate": (delta_hits / float(delta_total)) if delta_total else 0.0,
    }

    for key, values in reward_values.items():
        stats = _mean_std(values)
        summary["mean_{}".format(key)] = stats["mean"]
        summary["std_{}".format(key)] = stats["std"]

    total_stats = _mean_std(rollout_totals)
    summary["mean_total"] = total_stats["mean"]
    summary["std_total"] = total_stats["std"]

    outcome_stats = _mean_std(rollout_outcomes)
    summary["mean_outcome"] = outcome_stats["mean"]
    summary["std_outcome"] = outcome_stats["std"]

    for key, values in controller_values.items():
        summary["mean_{}".format(key)] = _mean_std(values)["mean"]
    summary["mean_answer_start_delay_steps"] = summary.get("mean_eof_to_answer_lag", 0.0)
    if "mean_final_think_token_count" in summary:
        summary["mean_response_latency_tokens"] = summary.get("mean_final_think_token_count", 0.0)
    else:
        summary["mean_response_latency_tokens"] = summary.get("mean_response_latency_token_proxy", 0.0)
    summary["mean_post_eof_answer_wall_clock_seconds"] = summary.get("mean_post_eof_wall_clock_seconds", 0.0)
    summary["mean_text_first_token_wall_clock_seconds"] = summary.get("mean_text_first_token_wall_clock_seconds", 0.0)
    summary["mean_effective_text_first_token_seconds"] = summary.get("mean_effective_text_first_token_seconds", 0.0)
    summary["mean_effective_response_onset_seconds"] = summary.get("mean_effective_response_onset_seconds", 0.0)
    summary["mean_final_think_generation_wall_clock_seconds"] = summary.get(
        "mean_final_think_generation_wall_clock_seconds", 0.0
    )

    answer_correct_rate = (
        sum(1.0 for value in rollout_ra if float(value) > 0.0) / float(len(rollout_ra))
        if rollout_ra
        else 0.0
    )
    format_pass_rate = (
        sum(1.0 for value in rollout_rf if float(value) > 0.0) / float(len(rollout_rf))
        if rollout_rf
        else 0.0
    )
    positive_total_rate = (
        sum(1.0 for value in rollout_totals if float(value) > 0.0) / float(len(rollout_totals))
        if rollout_totals
        else 0.0
    )
    ratio_value = max(1e-6, float(summary.get("mean_predicted_to_target_ratio", 0.0) or 0.0))
    ratio_penalty = abs(math.log(ratio_value))
    zero_think_rate = float(summary.get("mean_episodes_with_zero_think_before_eof", 0.0) or 0.0)
    delay_penalty = float(summary.get("mean_answer_start_delay_steps", 0.0) or 0.0)
    mean_wait_rate = float(summary.get("mean_wait_rate", 0.0) or 0.0)
    mean_think_rate = float(summary.get("mean_think_rate", 0.0) or 0.0)
    mean_think_length = float(summary.get("mean_think_length", 0.0) or 0.0)
    mean_final_think_token_count = float(summary.get("mean_final_think_token_count", 0.0) or 0.0)
    rt_value = float(summary.get("mean_R_t", 0.0) or 0.0)
    rt_final_value = float(summary.get("mean_R_t_final", 0.0) or 0.0)
    rt_final_judged_value = float(summary.get("mean_R_t_final_judged", 0.0) or 0.0)
    rc_value = float(summary.get("mean_R_c", 0.0) or 0.0)
    teacher_precision_value = float(summary.get("mean_teacher_boundary_precision", 0.0) or 0.0)
    teacher_recall_value = float(summary.get("mean_teacher_boundary_recall", 0.0) or 0.0)
    teacher_f1_value = float(summary.get("mean_teacher_boundary_f1", 0.0) or 0.0)
    post_eof_wall_clock_seconds = float(summary.get("mean_post_eof_wall_clock_seconds", 0.0) or 0.0)
    effective_text_first_token_seconds = float(
        summary.get("mean_effective_text_first_token_seconds", 0.0) or 0.0
    )
    text_streaming_supported = float(summary.get("mean_text_streaming_supported", 0.0) or 0.0)
    if text_streaming_supported > 0.0 or effective_text_first_token_seconds > 0.0:
        effective_latency_seconds = effective_text_first_token_seconds
    else:
        effective_latency_seconds = float(summary.get("mean_effective_response_onset_seconds", 0.0) or 0.0)
    entropy_value = float(summary.get("update_entropy", 0.0) or 0.0)
    entropy_collapse_penalty = max(0.0, 0.35 - entropy_value)
    clip_fraction_value = float(summary.get("update_clip_fraction", 0.0) or 0.0)
    clip_excess_penalty = max(0.0, clip_fraction_value - 0.25)
    observed_kl_value = float(summary.get("update_observed_kl", 0.0) or 0.0)
    kl_excess_penalty = max(0.0, observed_kl_value - 0.02)
    mean_completion_tokens = float(summary.get("update_mean_completion_tokens", 0.0) or 0.0)
    completion_length_penalty = max(0.0, (mean_completion_tokens - 48.0) / 48.0)
    underthink_penalty = max(0.0, 0.5 - mean_think_length)
    wait_extreme_penalty = max(0.0, mean_wait_rate - 0.95) + max(0.0, 0.05 - mean_wait_rate)
    think_rate_preference = _think_rate_preference_penalty(
        mean_think_rate,
        rt_value=rt_value,
        rc_value=rc_value,
    )
    think_rate_target = think_rate_preference["target"]
    think_rate_target_gap = think_rate_preference["target_gap"]
    think_rate_target_penalty = think_rate_preference["total_penalty"]
    # Paper target: final-think should be a short answer cue. We now rank
    # 3-6 tokens as ideal; 7-8 is only tolerable, >8 is visibly worse, and
    # >10 is a hard latency smell unless accuracy/quality are exceptional.
    final_think_token_penalty = max(0.0, (mean_final_think_token_count - 6.0) / 2.0)
    final_think_token_hard_penalty = max(0.0, (mean_final_think_token_count - 10.0) / 2.0)
    final_think_token_soft_cost = min(max(0.0, mean_final_think_token_count), 6.0) / 6.0
    final_think_token_ideal_bonus = 1.0 if 3.0 <= mean_final_think_token_count <= 6.0 else 0.0
    final_think_raw_valid_rate = float(summary.get("final_think_raw_valid_rate", 1.0) or 0.0)
    final_think_fallback_rate = float(summary.get("final_think_fallback_rate", 0.0) or 0.0)
    answer_leak_rate = float(summary.get("answer_leak_rate", 0.0) or 0.0)
    placeholder_rate = float(summary.get("placeholder_rate", 0.0) or 0.0)
    meta_think_rate = float(summary.get("meta_think_rate", 0.0) or 0.0)
    final_think_placeholder_rate = float(summary.get("final_think_placeholder_rate", 0.0) or 0.0)
    final_think_meta_rate = float(summary.get("final_think_meta_rate", 0.0) or 0.0)
    final_think_raw_invalid_penalty = max(0.0, 1.0 - final_think_raw_valid_rate)
    final_think_fallback_penalty = max(0.0, final_think_fallback_rate)
    answer_leak_penalty = max(0.0, answer_leak_rate)
    judge_quality_available = bool(
        rt_final_judged_value > 0.0 or rt_value > 0.0 or rt_final_value > 0.0 or rc_value > 0.0
    )
    rt_quality_gate_penalty = (
        max(0.0, (0.12 - rt_value) / 0.12)
        if judge_quality_available and mean_think_rate >= 0.05
        else 0.0
    )
    rt_final_quality_gate_penalty = (
        max(0.0, (0.25 - rt_final_value) / 0.25)
        if rt_final_judged_value > 0.0
        else 0.0
    )
    rc_quality_gate_penalty = (
        max(0.0, (0.15 - rc_value) / 0.15)
        if judge_quality_available and mean_think_rate >= 0.05
        else 0.0
    )
    protocol_violation_penalty = (
        1.75 * final_think_raw_invalid_penalty
        + 1.25 * final_think_fallback_penalty
        + 1.50 * answer_leak_penalty
    )
    reasoning_state_quality_penalty = (
        rt_quality_gate_penalty
        + rt_final_quality_gate_penalty
        + rc_quality_gate_penalty
        + 0.75 * placeholder_rate
        + 0.75 * meta_think_rate
        + 2.25 * final_think_placeholder_rate
        + 2.25 * final_think_meta_rate
    )
    checkpoint_selection_score = (
        2.0 * answer_correct_rate
        + 0.75 * format_pass_rate
        + 0.50 * positive_total_rate
        + 0.75 * rt_value
        + 0.50 * rt_final_value
        + 0.50 * rc_value
        + 0.10 * final_think_token_ideal_bonus
        - 0.15 * delay_penalty
        - 0.15 * effective_latency_seconds
        - 0.25 * ratio_penalty
        - 0.25 * zero_think_rate
        - 0.30 * think_rate_target_penalty
        - 0.30 * final_think_token_penalty
        - 0.20 * final_think_token_hard_penalty
        - 0.05 * final_think_token_soft_cost
        - 1.25 * protocol_violation_penalty
        - 0.75 * reasoning_state_quality_penalty
        - 0.25 * entropy_collapse_penalty
        - 0.40 * clip_excess_penalty
        - 2.00 * kl_excess_penalty
        - 0.15 * completion_length_penalty
    )
    candidate_checkpoint_score = (
        2.5 * answer_correct_rate
        + 1.0 * format_pass_rate
        + 0.50 * positive_total_rate
        + 1.25 * rt_value
        + 1.00 * rt_final_value
        + 0.75 * rc_value
        + 0.20 * final_think_token_ideal_bonus
        + 0.75 * teacher_precision_value
        + 0.75 * teacher_recall_value
        + 0.75 * teacher_f1_value
        - 0.20 * delay_penalty
        - 0.20 * effective_latency_seconds
        - 0.15 * post_eof_wall_clock_seconds
        - 0.35 * ratio_penalty
        - 0.30 * zero_think_rate
        - 0.75 * underthink_penalty
        - 0.50 * wait_extreme_penalty
        - 0.50 * think_rate_target_penalty
        - 0.45 * final_think_token_penalty
        - 0.35 * final_think_token_hard_penalty
        - 0.10 * final_think_token_soft_cost
        - 1.75 * protocol_violation_penalty
        - 1.00 * reasoning_state_quality_penalty
        - 0.25 * entropy_collapse_penalty
        - 0.40 * clip_excess_penalty
        - 2.00 * kl_excess_penalty
        - 0.15 * completion_length_penalty
    )
    user_goal_checkpoint_score = (
        3.0 * answer_correct_rate
        + 1.5 * format_pass_rate
        + 1.50 * rt_value
        + 1.50 * rt_final_value
        + 1.00 * rc_value
        + 0.35 * final_think_token_ideal_bonus
        - 0.20 * delay_penalty
        - 0.10 * effective_latency_seconds
        - 4.50 * think_rate_target_penalty
        - 1.50 * final_think_token_penalty
        - 1.00 * final_think_token_hard_penalty
        - 0.30 * final_think_token_soft_cost
        - 3.00 * protocol_violation_penalty
        - 2.00 * reasoning_state_quality_penalty
    )
    summary["answer_correct_rate"] = answer_correct_rate
    summary["format_pass_rate"] = format_pass_rate
    summary["positive_total_rate"] = positive_total_rate
    summary["teacher_alignment_score"] = (
        teacher_precision_value + teacher_recall_value + teacher_f1_value
    ) / 3.0
    summary["underthink_penalty"] = underthink_penalty
    summary["wait_extreme_penalty"] = wait_extreme_penalty
    summary["entropy_collapse_penalty"] = entropy_collapse_penalty
    summary["clip_excess_penalty"] = clip_excess_penalty
    summary["kl_excess_penalty"] = kl_excess_penalty
    summary["completion_length_penalty"] = completion_length_penalty
    summary["think_rate_target_gap"] = think_rate_target_gap
    summary["think_rate_target_penalty"] = think_rate_target_penalty
    summary["think_rate_target"] = think_rate_target
    summary["think_rate_target_low"] = think_rate_preference["target_low"]
    summary["think_rate_target_high"] = think_rate_preference["target_high"]
    summary["think_rate_strong_low"] = think_rate_preference["strong_low"]
    summary["think_rate_mild_high"] = think_rate_preference["mild_high"]
    summary["think_rate_quality_factor"] = think_rate_preference["quality_factor"]
    summary["think_rate_low_penalty"] = think_rate_preference["low_penalty"]
    summary["think_rate_high_penalty"] = think_rate_preference["high_penalty"]
    summary["final_think_token_penalty"] = final_think_token_penalty
    summary["final_think_token_hard_penalty"] = final_think_token_hard_penalty
    summary["final_think_token_soft_cost"] = final_think_token_soft_cost
    summary["final_think_token_ideal_bonus"] = final_think_token_ideal_bonus
    summary["final_think_raw_invalid_penalty"] = final_think_raw_invalid_penalty
    summary["final_think_fallback_penalty"] = final_think_fallback_penalty
    summary["answer_leak_penalty"] = answer_leak_penalty
    summary["final_think_placeholder_penalty"] = final_think_placeholder_rate
    summary["final_think_meta_penalty"] = final_think_meta_rate
    summary["protocol_violation_penalty"] = protocol_violation_penalty
    summary["rt_quality_gate_penalty"] = rt_quality_gate_penalty
    summary["rt_final_quality_gate_penalty"] = rt_final_quality_gate_penalty
    summary["rc_quality_gate_penalty"] = rc_quality_gate_penalty
    summary["reasoning_state_quality_penalty"] = reasoning_state_quality_penalty
    summary["checkpoint_selection_score"] = checkpoint_selection_score
    summary["candidate_checkpoint_score"] = candidate_checkpoint_score
    summary["user_goal_checkpoint_score"] = user_goal_checkpoint_score

    return summary


@dataclass
class TrainerConfig:
    input_path: str
    group_size: int = 8
    batch_size: int = 1
    prompt_batch_workers: int = 0
    max_steps: int = 10
    resume_step: int = 0
    phase: int = 1
    seed: int = 7
    use_judge: bool = False
    use_reference_answer_fallback: bool = False
    checkpoint_every: int = 1
    checkpoint_keep: int = 1
    checkpoint_keep_best: int = 0
    checkpoint_score_key: str = "mean_total"
    checkpoint_score_maximize: bool = True
    candidate_checkpoint_keep_best: int = 0
    candidate_checkpoint_alt_score_key: str = ""
    candidate_checkpoint_alt_keep_best: int = 0
    candidate_checkpoint_alt_score_maximize: bool = True
    candidate_checkpoint_bucket_size: int = 0
    candidate_checkpoint_keep_per_bucket: int = 0
    candidate_checkpoint_alt_bucket_size: int = 0
    candidate_checkpoint_alt_keep_per_bucket: int = 0
    candidate_checkpoint_user_goal_score_key: str = ""
    candidate_checkpoint_user_goal_keep_best: int = 0
    candidate_checkpoint_user_goal_score_maximize: bool = True
    candidate_checkpoint_user_goal_bucket_size: int = 0
    candidate_checkpoint_user_goal_keep_per_bucket: int = 0
    candidate_checkpoint_score_key: str = "candidate_checkpoint_score"
    candidate_checkpoint_score_maximize: bool = True
    full_checkpoint_every: int = 0
    full_checkpoint_keep: int = 1
    full_checkpoint_keep_best: int = 0
    full_checkpoint_score_key: str = "mean_total"
    full_checkpoint_score_maximize: bool = True
    reload_policy_on_checkpoint: bool = False
    save_rollouts: bool = True
    dry_run: bool = True
    advantage_source: str = "total"
    run_name: str = ""
    run_dir: str = ""
    sample_order_mode: str = "sequential"
    sample_order_bucket_keys: str = "topic,difficulty"
    sample_order_seed: int = 7
    dynamic_sample: bool = False
    max_resample_times: int = 0
    dynamic_sample_min_std: float = 1e-6
    dynamic_sample_min_format_pass_rollouts: int = 2
    dynamic_sample_min_final_think_raw_valid_rollouts: int = 1
    dynamic_sample_min_pre_eof_think_rollouts: int = 1
    health_observed_kl_warn: float = 2.0
    health_observed_kl_critical: float = 5.0
    health_entropy_warn: float = 0.3
    health_entropy_critical: float = 0.1
    health_clip_fraction_warn: float = 0.3
    health_clip_fraction_critical: float = 0.5
    health_wait_rate_low_warn: float = 0.1
    health_wait_rate_high_warn: float = 0.9
    health_wait_rate_low_critical: float = 0.05
    health_wait_rate_high_critical: float = 0.95
    health_wait_rate_critical_start_step: int = 300
    health_teacher_precision_warn: float = 0.15
    health_teacher_precision_critical: float = 0.08
    health_teacher_recall_warn: float = 0.15
    health_teacher_recall_critical: float = 0.08
    health_predicted_to_target_ratio_warn: float = 3.0
    health_predicted_to_target_ratio_critical: float = 5.0
    health_critical_patience: int = 3
    health_critical_warmup_steps: int = 50
    health_warn_only: bool = False
    wandb_enabled: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_name: str = ""
    wandb_tags: str = ""
    wandb_notes: str = ""
    wandb_mode: str = "online"
    policy_think_temperature: float = 0.0
    policy_temperature_jitter: float = 0.0
    policy_prompt_version: str = ""
    policy_final_think_prompt_version: str = ""
    policy_final_answer_prompt_version: str = ""
    policy_audio_window_mode: str = ""
    policy_force_wait_before_sec: float = 0.0
    policy_question_visible_from_text: bool = False
    policy_service_cuda_visible_devices: str = ""
    policy_service_tensor_parallel_size: int = 0
    policy_service_port: int = 0


class StreamingGRPOTrainer:
    def __init__(
        self,
        samples,
        policy_backend,
        reward_config: RewardConfig,
        trainer_config: TrainerConfig,
        judge=None,
        answer_fallback_judge=None,
        run_dir: Optional[str] = None,
    ):
        bucket_keys = [
            key.strip()
            for key in str(trainer_config.sample_order_bucket_keys or "").split(",")
            if key.strip()
        ] or ["topic", "difficulty"]
        self.samples = reorder_items(
            list(samples),
            mode=str(trainer_config.sample_order_mode or "sequential"),
            bucket_keys=tuple(bucket_keys),
            seed=int(trainer_config.sample_order_seed),
        )
        self.policy_backend = policy_backend
        self.reward_config = reward_config
        self.trainer_config = trainer_config
        self.judge = judge
        self.answer_fallback_judge = answer_fallback_judge
        self.sample_cursor = max(0, int(trainer_config.resume_step)) * max(1, int(trainer_config.batch_size))

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        default_name = trainer_config.run_name or "grpo_{}".format(timestamp)
        self.run_dir = Path(run_dir or trainer_config.run_dir or ("runs/" + default_name)).resolve()
        self.steps_dir = self.run_dir / "steps"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.policy_candidate_dir = self.ckpt_dir / "policy_candidates"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.run_dir / "summary.jsonl"
        self.run_state_path = self.run_dir / "run_state.json"

        config_payload = {
            "trainer_config": asdict(trainer_config),
            "reward_config": asdict(reward_config),
            "policy_backend": policy_backend.name,
        }
        updater = getattr(policy_backend, "updater", None)
        updater_config = getattr(updater, "config", None)
        if updater_config is not None:
            try:
                config_payload["actor_config"] = asdict(updater_config)
            except Exception:
                config_payload["actor_config"] = {"repr": repr(updater_config)}
        self._write_json(
            self.run_dir / "config.json",
            config_payload,
        )
        self._config_payload = config_payload
        self._write_run_state(status="initialized", current_step=max(0, int(trainer_config.resume_step)))
        self._health_critical_streak = 0
        self._effective_update_steps = 0
        self._wandb_run = None
        self._metric_history: Dict[str, List[float]] = {
            "observed_kl": [],
            "entropy": [],
            "clip_fraction": [],
            "wait_rate": [],
            "teacher_precision": [],
            "teacher_recall": [],
            "predicted_to_target_ratio": [],
            "final_think_raw_valid_rate": [],
            "final_think_fallback_rate": [],
            "answer_leak_rate": [],
            "placeholder_rate": [],
            "meta_think_rate": [],
            "R_t": [],
            "R_t_final": [],
            "R_c": [],
        }
        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        if not bool(self.trainer_config.wandb_enabled):
            return
        try:
            import wandb
        except Exception as exc:
            self._write_run_state(
                status="initialized",
                current_step=max(0, int(self.trainer_config.resume_step)),
                extra={"wandb_init_error": repr(exc)},
            )
            return

        tags = [
            item.strip()
            for item in str(self.trainer_config.wandb_tags or "").split(",")
            if item.strip()
        ]
        run_name = str(self.trainer_config.wandb_name or self.trainer_config.run_name or self.run_dir.name).strip()
        init_kwargs: Dict[str, Any] = {
            "project": str(self.trainer_config.wandb_project or "wait-think-answer"),
            "name": run_name,
            "dir": str(self.run_dir),
            "config": dict(self._config_payload),
            "mode": str(self.trainer_config.wandb_mode or "online"),
            "tags": tags,
            "notes": str(self.trainer_config.wandb_notes or ""),
            "reinit": False,
        }
        entity = str(self.trainer_config.wandb_entity or "").strip()
        if entity:
            init_kwargs["entity"] = entity

        self._wandb_run = wandb.init(**init_kwargs)
        try:
            self._write_run_state(
                status="initialized",
                current_step=max(0, int(self.trainer_config.resume_step)),
                extra={"wandb_run_id": getattr(self._wandb_run, "id", "")},
            )
        except Exception:
            pass

    def _wandb_scalar_payload(self, step_summary: Dict[str, Any]) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for key, value in step_summary.items():
            if key not in _WANDB_CORE_METRIC_KEYS:
                continue
            if isinstance(value, bool):
                payload[key] = float(value)
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[key] = float(value)
        return payload

    def _metric_trend(self, name: str, value: Optional[float]) -> str:
        if value is None:
            return "na"
        history = self._metric_history.setdefault(name, [])
        history.append(float(value))
        if len(history) > 100:
            del history[:-100]
        if len(history) < 20:
            return "na"
        earlier = history[-20:-10]
        recent = history[-10:]
        earlier_mean = sum(earlier) / float(len(earlier))
        recent_mean = sum(recent) / float(len(recent))
        baseline = max(abs(earlier_mean), 1e-8)
        if recent_mean > earlier_mean + 0.1 * baseline:
            return "up"
        if recent_mean < earlier_mean - 0.1 * baseline:
            return "down"
        return "flat"

    def _evaluate_health(self, step_summary: Dict, *, step: int) -> Dict[str, object]:
        warnings: List[str] = []
        criticals: List[str] = []
        trends: Dict[str, str] = {}
        in_critical_warmup = int(step) <= max(0, int(self.trainer_config.health_critical_warmup_steps))
        wait_rate_critical_enabled = int(step) >= max(
            0, int(self.trainer_config.health_wait_rate_critical_start_step)
        )

        def _check_upper(name: str, value: Optional[float], warn_at: float, critical_at: float) -> None:
            if value is None:
                return
            trends[name] = self._metric_trend(name, value)
            if not in_critical_warmup and float(value) >= float(critical_at):
                criticals.append("{}={:.4f} >= {:.4f}".format(name, float(value), float(critical_at)))
            elif float(value) >= float(warn_at):
                warnings.append("{}={:.4f} >= {:.4f}".format(name, float(value), float(warn_at)))

        def _check_lower(name: str, value: Optional[float], warn_at: float, critical_at: float) -> None:
            if value is None:
                return
            trends[name] = self._metric_trend(name, value)
            if not in_critical_warmup and float(value) <= float(critical_at):
                criticals.append("{}={:.4f} <= {:.4f}".format(name, float(value), float(critical_at)))
            elif float(value) <= float(warn_at):
                warnings.append("{}={:.4f} <= {:.4f}".format(name, float(value), float(warn_at)))

        observed_kl = step_summary.get("update_observed_kl")
        entropy = step_summary.get("update_entropy")
        clip_fraction = step_summary.get("update_clip_fraction")
        wait_rate = step_summary.get("mean_wait_rate")
        teacher_precision = step_summary.get("mean_teacher_boundary_precision")
        teacher_recall = step_summary.get("mean_teacher_boundary_recall")
        predicted_to_target_ratio = step_summary.get("mean_predicted_to_target_ratio")
        final_think_raw_valid_rate = step_summary.get("final_think_raw_valid_rate")
        final_think_fallback_rate = step_summary.get("final_think_fallback_rate")
        answer_leak_rate = step_summary.get("answer_leak_rate")
        placeholder_rate = step_summary.get("placeholder_rate")
        meta_think_rate = step_summary.get("meta_think_rate")
        rt_value = step_summary.get("mean_R_t")
        rt_final_value = step_summary.get("mean_R_t_final")
        rt_final_judged_value = step_summary.get("mean_R_t_final_judged")
        rc_value = step_summary.get("mean_R_c")
        mean_think_rate = step_summary.get("mean_think_rate")

        _check_upper(
            "observed_kl",
            observed_kl if isinstance(observed_kl, (int, float)) else None,
            self.trainer_config.health_observed_kl_warn,
            self.trainer_config.health_observed_kl_critical,
        )
        _check_lower(
            "entropy",
            entropy if isinstance(entropy, (int, float)) else None,
            self.trainer_config.health_entropy_warn,
            self.trainer_config.health_entropy_critical,
        )
        _check_upper(
            "clip_fraction",
            clip_fraction if isinstance(clip_fraction, (int, float)) else None,
            self.trainer_config.health_clip_fraction_warn,
            self.trainer_config.health_clip_fraction_critical,
        )

        if isinstance(wait_rate, (int, float)):
            trends["wait_rate"] = self._metric_trend("wait_rate", wait_rate)
            if (
                not in_critical_warmup
                and wait_rate_critical_enabled
                and float(wait_rate) >= float(self.trainer_config.health_wait_rate_high_critical)
            ):
                criticals.append(
                    "wait_rate={:.4f} >= {:.4f}".format(
                        float(wait_rate), float(self.trainer_config.health_wait_rate_high_critical)
                    )
                )
            elif float(wait_rate) >= float(self.trainer_config.health_wait_rate_high_warn):
                warnings.append(
                    "wait_rate={:.4f} >= {:.4f}".format(
                        float(wait_rate), float(self.trainer_config.health_wait_rate_high_warn)
                    )
                )

            if (
                not in_critical_warmup
                and wait_rate_critical_enabled
                and float(wait_rate) <= float(self.trainer_config.health_wait_rate_low_critical)
            ):
                criticals.append(
                    "wait_rate={:.4f} <= {:.4f}".format(
                        float(wait_rate), float(self.trainer_config.health_wait_rate_low_critical)
                    )
                )
            elif float(wait_rate) <= float(self.trainer_config.health_wait_rate_low_warn):
                warnings.append(
                    "wait_rate={:.4f} <= {:.4f}".format(
                        float(wait_rate), float(self.trainer_config.health_wait_rate_low_warn)
                    )
                )

        if isinstance(teacher_precision, (int, float)):
            trends["teacher_precision"] = self._metric_trend("teacher_precision", teacher_precision)
            if (
                not in_critical_warmup
                and wait_rate_critical_enabled
                and float(teacher_precision) <= float(self.trainer_config.health_teacher_precision_critical)
            ):
                criticals.append(
                    "teacher_precision={:.4f} <= {:.4f}".format(
                        float(teacher_precision), float(self.trainer_config.health_teacher_precision_critical)
                    )
                )
            elif float(teacher_precision) <= float(self.trainer_config.health_teacher_precision_warn):
                warnings.append(
                    "teacher_precision={:.4f} <= {:.4f}".format(
                        float(teacher_precision), float(self.trainer_config.health_teacher_precision_warn)
                    )
                )

        if isinstance(teacher_recall, (int, float)):
            trends["teacher_recall"] = self._metric_trend("teacher_recall", teacher_recall)
            if (
                not in_critical_warmup
                and wait_rate_critical_enabled
                and float(teacher_recall) <= float(self.trainer_config.health_teacher_recall_critical)
            ):
                criticals.append(
                    "teacher_recall={:.4f} <= {:.4f}".format(
                        float(teacher_recall), float(self.trainer_config.health_teacher_recall_critical)
                    )
                )
            elif float(teacher_recall) <= float(self.trainer_config.health_teacher_recall_warn):
                warnings.append(
                    "teacher_recall={:.4f} <= {:.4f}".format(
                        float(teacher_recall), float(self.trainer_config.health_teacher_recall_warn)
                    )
                )

        if isinstance(predicted_to_target_ratio, (int, float)):
            trends["predicted_to_target_ratio"] = self._metric_trend(
                "predicted_to_target_ratio",
                predicted_to_target_ratio,
            )
            if (
                not in_critical_warmup
                and wait_rate_critical_enabled
                and float(predicted_to_target_ratio)
                >= float(self.trainer_config.health_predicted_to_target_ratio_critical)
            ):
                criticals.append(
                    "predicted_to_target_ratio={:.4f} >= {:.4f}".format(
                        float(predicted_to_target_ratio),
                        float(self.trainer_config.health_predicted_to_target_ratio_critical),
                    )
                )
            elif float(predicted_to_target_ratio) >= float(self.trainer_config.health_predicted_to_target_ratio_warn):
                warnings.append(
                    "predicted_to_target_ratio={:.4f} >= {:.4f}".format(
                        float(predicted_to_target_ratio),
                        float(self.trainer_config.health_predicted_to_target_ratio_warn),
                    )
                )

        if isinstance(final_think_raw_valid_rate, (int, float)):
            trends["final_think_raw_valid_rate"] = self._metric_trend(
                "final_think_raw_valid_rate",
                final_think_raw_valid_rate,
            )
            if not in_critical_warmup and float(final_think_raw_valid_rate) < 0.90:
                criticals.append(
                    "final_think_raw_valid_rate={:.4f} < 0.9000".format(float(final_think_raw_valid_rate))
                )
            elif float(final_think_raw_valid_rate) < 0.98:
                warnings.append(
                    "final_think_raw_valid_rate={:.4f} < 0.9800".format(float(final_think_raw_valid_rate))
                )

        if isinstance(final_think_fallback_rate, (int, float)):
            trends["final_think_fallback_rate"] = self._metric_trend(
                "final_think_fallback_rate",
                final_think_fallback_rate,
            )
            if not in_critical_warmup and float(final_think_fallback_rate) > 0.10:
                criticals.append(
                    "final_think_fallback_rate={:.4f} > 0.1000".format(float(final_think_fallback_rate))
                )
            elif float(final_think_fallback_rate) > 0.02:
                warnings.append(
                    "final_think_fallback_rate={:.4f} > 0.0200".format(float(final_think_fallback_rate))
                )

        if isinstance(answer_leak_rate, (int, float)):
            trends["answer_leak_rate"] = self._metric_trend("answer_leak_rate", answer_leak_rate)
            if not in_critical_warmup and float(answer_leak_rate) > 0.0:
                criticals.append("answer_leak_rate={:.4f} > 0.0000".format(float(answer_leak_rate)))
            elif float(answer_leak_rate) > 0.0:
                warnings.append("answer_leak_rate={:.4f} > 0.0000".format(float(answer_leak_rate)))

        if isinstance(placeholder_rate, (int, float)):
            trends["placeholder_rate"] = self._metric_trend("placeholder_rate", placeholder_rate)
            if not in_critical_warmup and float(placeholder_rate) > 0.30:
                criticals.append("placeholder_rate={:.4f} > 0.3000".format(float(placeholder_rate)))
            elif float(placeholder_rate) > 0.10:
                warnings.append("placeholder_rate={:.4f} > 0.1000".format(float(placeholder_rate)))

        if isinstance(meta_think_rate, (int, float)):
            trends["meta_think_rate"] = self._metric_trend("meta_think_rate", meta_think_rate)
            if not in_critical_warmup and float(meta_think_rate) > 0.35:
                criticals.append("meta_think_rate={:.4f} > 0.3500".format(float(meta_think_rate)))
            elif float(meta_think_rate) > 0.15:
                warnings.append("meta_think_rate={:.4f} > 0.1500".format(float(meta_think_rate)))

        judge_quality_available = False
        if isinstance(rt_final_judged_value, (int, float)) and float(rt_final_judged_value) > 0.0:
            judge_quality_available = True
        if isinstance(rt_value, (int, float)) and float(rt_value) > 0.0:
            judge_quality_available = True
        if isinstance(rc_value, (int, float)) and float(rc_value) > 0.0:
            judge_quality_available = True
        think_rate_active = isinstance(mean_think_rate, (int, float)) and float(mean_think_rate) >= 0.05
        if judge_quality_available and think_rate_active:
            if isinstance(rt_value, (int, float)):
                trends["R_t"] = self._metric_trend("R_t", rt_value)
                if not in_critical_warmup and float(rt_value) < 0.08:
                    criticals.append("R_t={:.4f} < 0.0800".format(float(rt_value)))
                elif float(rt_value) < 0.12:
                    warnings.append("R_t={:.4f} < 0.1200".format(float(rt_value)))
            if isinstance(rt_final_value, (int, float)) and isinstance(rt_final_judged_value, (int, float)):
                trends["R_t_final"] = self._metric_trend("R_t_final", rt_final_value)
                if float(rt_final_judged_value) > 0.0:
                    if not in_critical_warmup and float(rt_final_value) < 0.15:
                        criticals.append("R_t_final={:.4f} < 0.1500".format(float(rt_final_value)))
                    elif float(rt_final_value) < 0.25:
                        warnings.append("R_t_final={:.4f} < 0.2500".format(float(rt_final_value)))
            if isinstance(rc_value, (int, float)):
                trends["R_c"] = self._metric_trend("R_c", rc_value)
                if not in_critical_warmup and float(rc_value) < 0.05:
                    criticals.append("R_c={:.4f} < 0.0500".format(float(rc_value)))
                elif float(rc_value) < 0.15:
                    warnings.append("R_c={:.4f} < 0.1500".format(float(rc_value)))

        status = "healthy"
        if criticals:
            status = "critical"
        elif warnings:
            status = "warning"

        return {
            "health_status": status,
            "health_warnings": warnings,
            "health_criticals": criticals,
            "health_critical": bool(criticals),
            "health_in_critical_warmup": bool(in_critical_warmup),
            "health_wait_rate_critical_enabled": bool(wait_rate_critical_enabled),
            "health_trends": trends,
        }

    def _write_json(self, path: Path, payload: Dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: Dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_checkpoint_pointer(self, name: str, payload: Dict) -> None:
        self._write_json(self.run_dir / name, payload)

    def _write_candidate_manifest(
        self,
        *,
        root_dir: Path,
        score_key: str,
        maximize: bool,
        output_name: str,
    ) -> None:
        ranked = ranked_step_checkpoints(
            str(root_dir),
            score_key=score_key,
            maximize=maximize,
        )
        self._write_json(
            self.run_dir / output_name,
            {
                "kind": "policy_candidate_manifest",
                "score_key": score_key,
                "maximize": bool(maximize),
                "candidates": ranked,
            },
        )

    def _maybe_save_policy_candidate(self, *, step: int, step_summary: Dict) -> Optional[Dict]:
        keep_best = max(0, int(self.trainer_config.candidate_checkpoint_keep_best or 0))
        alt_score_key = str(self.trainer_config.candidate_checkpoint_alt_score_key or "").strip()
        alt_keep_best = max(0, int(self.trainer_config.candidate_checkpoint_alt_keep_best or 0))
        bucket_size = max(0, int(self.trainer_config.candidate_checkpoint_bucket_size or 0))
        keep_per_bucket = max(0, int(self.trainer_config.candidate_checkpoint_keep_per_bucket or 0))
        alt_bucket_size = max(0, int(self.trainer_config.candidate_checkpoint_alt_bucket_size or 0))
        alt_keep_per_bucket = max(0, int(self.trainer_config.candidate_checkpoint_alt_keep_per_bucket or 0))
        user_goal_score_key = str(self.trainer_config.candidate_checkpoint_user_goal_score_key or "").strip()
        user_goal_keep_best = max(0, int(self.trainer_config.candidate_checkpoint_user_goal_keep_best or 0))
        user_goal_bucket_size = max(0, int(self.trainer_config.candidate_checkpoint_user_goal_bucket_size or 0))
        user_goal_keep_per_bucket = max(
            0,
            int(self.trainer_config.candidate_checkpoint_user_goal_keep_per_bucket or 0),
        )
        if (
            keep_best <= 0
            and alt_keep_best <= 0
            and user_goal_keep_best <= 0
            and (bucket_size <= 0 or keep_per_bucket <= 0)
            and (alt_bucket_size <= 0 or alt_keep_per_bucket <= 0)
            and (user_goal_bucket_size <= 0 or user_goal_keep_per_bucket <= 0)
        ):
            return None

        score_key = str(self.trainer_config.candidate_checkpoint_score_key or "candidate_checkpoint_score")
        score_value = step_summary.get(score_key)
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            return None
        alt_score_value = None
        enters_alt_topk = False
        enters_alt_bucket_topk = False
        if alt_score_key:
            raw_alt = step_summary.get(alt_score_key)
            if not isinstance(raw_alt, bool) and isinstance(raw_alt, (int, float)):
                alt_score_value = float(raw_alt)
        user_goal_score_value = None
        enters_user_goal_topk = False
        enters_user_goal_bucket_topk = False
        if user_goal_score_key:
            raw_user_goal = step_summary.get(user_goal_score_key)
            if not isinstance(raw_user_goal, bool) and isinstance(raw_user_goal, (int, float)):
                user_goal_score_value = float(raw_user_goal)

        self.policy_candidate_dir.mkdir(parents=True, exist_ok=True)
        enters_global_topk = checkpoint_would_enter_top_k(
            str(self.policy_candidate_dir),
            step=step,
            score=float(score_value),
            keep_best=keep_best,
            score_key=score_key,
            maximize=bool(self.trainer_config.candidate_checkpoint_score_maximize),
        )
        if alt_score_key and alt_keep_best > 0 and alt_score_value is not None:
            enters_alt_topk = checkpoint_would_enter_top_k(
                str(self.policy_candidate_dir),
                step=step,
                score=float(alt_score_value),
                keep_best=alt_keep_best,
                score_key=alt_score_key,
                maximize=bool(self.trainer_config.candidate_checkpoint_alt_score_maximize),
            )
        enters_bucket_topk = checkpoint_would_enter_bucket_top_k(
            str(self.policy_candidate_dir),
            step=step,
            score=float(score_value),
            bucket_size=bucket_size,
            keep_best_per_bucket=keep_per_bucket,
            score_key=score_key,
            maximize=bool(self.trainer_config.candidate_checkpoint_score_maximize),
        )
        if alt_score_key and alt_score_value is not None and alt_bucket_size > 0 and alt_keep_per_bucket > 0:
            enters_alt_bucket_topk = checkpoint_would_enter_bucket_top_k(
                str(self.policy_candidate_dir),
                step=step,
                score=float(alt_score_value),
                bucket_size=alt_bucket_size,
                keep_best_per_bucket=alt_keep_per_bucket,
                score_key=alt_score_key,
                maximize=bool(self.trainer_config.candidate_checkpoint_alt_score_maximize),
            )
        if user_goal_score_key and user_goal_score_value is not None and user_goal_keep_best > 0:
            enters_user_goal_topk = checkpoint_would_enter_top_k(
                str(self.policy_candidate_dir),
                step=step,
                score=float(user_goal_score_value),
                keep_best=user_goal_keep_best,
                score_key=user_goal_score_key,
                maximize=bool(self.trainer_config.candidate_checkpoint_user_goal_score_maximize),
            )
        if (
            user_goal_score_key
            and user_goal_score_value is not None
            and user_goal_bucket_size > 0
            and user_goal_keep_per_bucket > 0
        ):
            enters_user_goal_bucket_topk = checkpoint_would_enter_bucket_top_k(
                str(self.policy_candidate_dir),
                step=step,
                score=float(user_goal_score_value),
                bucket_size=user_goal_bucket_size,
                keep_best_per_bucket=user_goal_keep_per_bucket,
                score_key=user_goal_score_key,
                maximize=bool(self.trainer_config.candidate_checkpoint_user_goal_score_maximize),
            )
        if (
            not enters_global_topk
            and not enters_alt_topk
            and not enters_bucket_topk
            and not enters_alt_bucket_topk
            and not enters_user_goal_topk
            and not enters_user_goal_bucket_topk
        ):
            return None

        candidate_path = self.policy_candidate_dir / "step_{:06d}.json".format(step)
        artifact = self.policy_backend.save_checkpoint(
            str(candidate_path),
            step=step,
            checkpoint_mode="model-only",
        )
        metrics_path = self._write_checkpoint_metrics(Path(artifact.checkpoint_dir), step_summary)
        pruned = prune_step_checkpoints_recent_and_best(
            str(self.policy_candidate_dir),
            keep_recent=0,
            keep_best=keep_best,
            alt_score_key=alt_score_key,
            alt_keep_best=alt_keep_best,
            alt_maximize=bool(self.trainer_config.candidate_checkpoint_alt_score_maximize),
            bucket_size=bucket_size,
            keep_best_per_bucket=keep_per_bucket,
            alt_bucket_size=alt_bucket_size,
            alt_keep_best_per_bucket=alt_keep_per_bucket,
            user_goal_score_key=user_goal_score_key,
            user_goal_keep_best=user_goal_keep_best,
            user_goal_maximize=bool(self.trainer_config.candidate_checkpoint_user_goal_score_maximize),
            user_goal_bucket_size=user_goal_bucket_size,
            user_goal_keep_best_per_bucket=user_goal_keep_per_bucket,
            score_key=score_key,
            maximize=bool(self.trainer_config.candidate_checkpoint_score_maximize),
        )
        self._write_checkpoint_pointer(
            "latest_policy_candidate.json",
            {
                "kind": "policy_candidate",
                "step": step,
                "mode": artifact.mode,
                "checkpoint_dir": artifact.checkpoint_dir,
                "reloadable_model_path": artifact.reloadable_model_path,
                "metadata_path": artifact.metadata_path,
                "trainer_metrics_path": metrics_path,
                "score_key": score_key,
                "score_value": float(score_value),
                "alt_score_key": alt_score_key,
                "alt_score_value": alt_score_value,
                "user_goal_score_key": user_goal_score_key,
                "user_goal_score_value": user_goal_score_value,
                "enters_global_topk": bool(enters_global_topk),
                "enters_alt_topk": bool(enters_alt_topk),
                "enters_bucket_topk": bool(enters_bucket_topk),
                "enters_alt_bucket_topk": bool(enters_alt_bucket_topk),
                "enters_user_goal_topk": bool(enters_user_goal_topk),
                "enters_user_goal_bucket_topk": bool(enters_user_goal_bucket_topk),
            },
        )
        self._write_candidate_manifest(
            root_dir=self.policy_candidate_dir,
            score_key=score_key,
            maximize=bool(self.trainer_config.candidate_checkpoint_score_maximize),
            output_name="best_policy_candidates.json",
        )
        if alt_score_key and alt_keep_best > 0:
            self._write_candidate_manifest(
                root_dir=self.policy_candidate_dir,
                score_key=alt_score_key,
                maximize=bool(self.trainer_config.candidate_checkpoint_alt_score_maximize),
                output_name="best_policy_candidates_{}.json".format(alt_score_key),
            )
        if user_goal_score_key and user_goal_keep_best > 0:
            self._write_candidate_manifest(
                root_dir=self.policy_candidate_dir,
                score_key=user_goal_score_key,
                maximize=bool(self.trainer_config.candidate_checkpoint_user_goal_score_maximize),
                output_name="best_policy_candidates_{}.json".format(user_goal_score_key),
            )
        return {
            "kind": "policy_candidate",
            "step": step,
            "mode": artifact.mode,
            "checkpoint_dir": artifact.checkpoint_dir,
            "trainer_metrics_path": metrics_path,
            "score_key": score_key,
            "score_value": float(score_value),
            "alt_score_key": alt_score_key,
            "alt_score_value": alt_score_value,
            "user_goal_score_key": user_goal_score_key,
            "user_goal_score_value": user_goal_score_value,
            "enters_global_topk": bool(enters_global_topk),
            "enters_alt_topk": bool(enters_alt_topk),
            "enters_bucket_topk": bool(enters_bucket_topk),
            "enters_alt_bucket_topk": bool(enters_alt_bucket_topk),
            "enters_user_goal_topk": bool(enters_user_goal_topk),
            "enters_user_goal_bucket_topk": bool(enters_user_goal_bucket_topk),
            "pruned": pruned,
        }

    def _write_checkpoint_metrics(self, checkpoint_dir: Path, step_summary: Dict) -> str:
        metrics_payload = {
            key: value
            for key, value in step_summary.items()
            if isinstance(value, (int, float, str, bool, list, dict)) or value is None
        }
        metrics_payload["checkpoint_step"] = int(step_summary.get("step", 0) or 0)
        metrics_payload["mean_answer_start_delay_steps"] = float(
            metrics_payload.get(
                "mean_answer_start_delay_steps",
                metrics_payload.get("mean_eof_to_answer_lag", 0.0),
            )
        )
        metrics_path = checkpoint_dir / "trainer_metrics.json"
        self._write_json(metrics_path, metrics_payload)
        return str(metrics_path)

    def _write_run_state(self, status: str, current_step: int, extra: Optional[Dict] = None) -> None:
        payload = {
            "status": status,
            "current_step": int(current_step),
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
        }
        if extra:
            payload.update(extra)
        self._write_json(self.run_state_path, payload)

    def _next_batch_samples(self):
        batch = []
        for _ in range(self.trainer_config.batch_size):
            batch.append(self.samples[self.sample_cursor % len(self.samples)])
            self.sample_cursor += 1
        return batch

    def _collect_group_payload(self, sample, step_seed: int) -> Dict:
        best_payload = None
        best_outcome_std = -1.0
        attempt = 0
        max_attempts = max(0, int(self.trainer_config.max_resample_times or 0))

        while True:
            rollout_seed = step_seed + attempt * 1000003
            rollouts = self.policy_backend.rollout_group(
                sample=sample,
                group_size=self.trainer_config.group_size,
                seed=rollout_seed,
                phase=self.trainer_config.phase,
            )
            episodes = [rollout.to_episode(sample) for rollout in rollouts]
            rewards = compute_rewards_batch(
                episodes,
                self.reward_config,
                judge=self.judge,
                answer_fallback_judge=self.answer_fallback_judge,
            )
            _apply_final_short_pairwise_adjustments(
                rewards,
                scale=float(getattr(self.reward_config, "final_short_pairwise_bonus_scale", 0.0) or 0.0),
            )
            totals = [reward["total"] for reward in rewards]
            outcomes = [reward.get("R_outcome", reward["total"]) for reward in rewards]
            advantage_values = select_advantage_values(
                rewards,
                source=self.trainer_config.advantage_source,
            )
            advantages = normalize_advantages(advantage_values)

            if self.judge is not None:
                self.judge.flush()

            outcome_stats = _mean_std([float(value) for value in advantage_values])
            reward_low_signal = (
                outcome_stats["std"] < float(self.trainer_config.dynamic_sample_min_std)
                or not any(abs(float(value)) > 1e-8 for value in advantages)
            )
            protocol_gate = _dynamic_protocol_gate_detail(
                rollouts=rollouts,
                rewards=rewards,
                min_format_pass_rollouts=self.trainer_config.dynamic_sample_min_format_pass_rollouts,
                min_final_think_raw_valid_rollouts=(
                    self.trainer_config.dynamic_sample_min_final_think_raw_valid_rollouts
                ),
                min_pre_eof_think_rollouts=self.trainer_config.dynamic_sample_min_pre_eof_think_rollouts,
            )
            low_signal = bool(reward_low_signal or protocol_gate["failed"])

            best_idx = max(range(len(totals)), key=lambda idx: totals[idx]) if totals else 0
            payload = {
                "sample": {
                    "audio_id": sample.audio_id,
                    "question": sample.question,
                    "gt_answer": sample.gt_answer,
                },
                "group_size": len(rollouts),
                "totals": totals,
                "outcomes": outcomes,
                "advantage_source": str(self.trainer_config.advantage_source),
                "advantage_values": advantage_values,
                "advantages": advantages,
                "best_index": best_idx,
                "best_summary": format_reward_summary(rewards[best_idx]) if rewards else "",
                "outcome_std": float(outcome_stats["std"]),
                "resample_attempts": int(attempt),
                "dynamic_sample_low_signal": bool(low_signal),
                "dynamic_sample_reward_low_signal": bool(reward_low_signal),
                "protocol_gate_failed": bool(protocol_gate["failed"]),
                "protocol_gate_reason": str(protocol_gate["reason"]),
                "protocol_gate_format_pass_count": int(protocol_gate["format_pass_count"]),
                "protocol_gate_format_pass_required": int(protocol_gate["format_pass_required"]),
                "protocol_gate_final_think_raw_valid_count": int(
                    protocol_gate["final_think_raw_valid_count"]
                ),
                "protocol_gate_final_think_raw_valid_required": int(
                    protocol_gate["final_think_raw_valid_required"]
                ),
                "protocol_gate_pre_eof_think_rollout_count": int(
                    protocol_gate["pre_eof_think_rollout_count"]
                ),
                "protocol_gate_pre_eof_think_rollout_required": int(
                    protocol_gate["pre_eof_think_rollout_required"]
                ),
                "rollouts": [
                    {
                        "rollout": rollout.to_dict(),
                        "reward": reward,
                        "advantage": advantage,
                    }
                    for rollout, reward, advantage in zip(rollouts, rewards, advantages)
                ],
                "update_batch": {
                    "sample": sample,
                    "rollouts": rollouts,
                    "episodes": episodes,
                    "rewards": rewards,
                    "advantages": advantages,
                },
            }

            if payload["outcome_std"] > best_outcome_std:
                best_payload = payload
                best_outcome_std = payload["outcome_std"]

            if not self.trainer_config.dynamic_sample or not low_signal:
                payload["skip_update"] = False
                return payload
            if attempt >= max_attempts:
                assert best_payload is not None
                best_payload["skip_update"] = True
                best_payload["dynamic_sample_skipped"] = True
                best_payload["protocol_gate_skipped"] = bool(best_payload.get("protocol_gate_failed"))
                return best_payload
            attempt += 1

    def _collect_group_payloads(self, batch_samples: List[Any], *, step: int) -> List[Dict]:
        collection_plan = [
            (
                batch_index,
                sample,
                self.trainer_config.seed + step * 1000 + batch_index * 100,
            )
            for batch_index, sample in enumerate(batch_samples)
        ]
        prompt_batch_workers = max(0, int(self.trainer_config.prompt_batch_workers or 0))
        if (
            prompt_batch_workers <= 1
            or len(collection_plan) <= 1
            or self.judge is not None
            or self.answer_fallback_judge is not None
        ):
            return [
                self._collect_group_payload(sample=sample, step_seed=step_seed)
                for _, sample, step_seed in collection_plan
            ]

        max_workers = min(prompt_batch_workers, len(collection_plan))
        results: List[Optional[Dict]] = [None for _ in collection_plan]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._collect_group_payload, sample, step_seed): batch_index
                for batch_index, sample, step_seed in collection_plan
            }
            for future in concurrent.futures.as_completed(future_to_index):
                batch_index = future_to_index[future]
                results[batch_index] = future.result()
        return [payload for payload in results if payload is not None]

    def run(self) -> Dict:
        final_state = {}
        current_step = max(0, int(self.trainer_config.resume_step))
        completion_status = "completed"
        completion_extra: Dict[str, object] = {}
        self.policy_backend.start()
        try:
            start_step = max(0, int(self.trainer_config.resume_step)) + 1
            for step in range(start_step, self.trainer_config.max_steps + 1):
                step_started_at = time.perf_counter()
                current_step = step
                self.reward_config.progress_fraction = (step - 1) / float(max(1, self.trainer_config.max_steps))
                self.reward_config.current_step = int(step)
                self.reward_config.max_steps = int(self.trainer_config.max_steps)
                self._write_run_state(
                    status="running",
                    current_step=max(0, int(step - 1)),
                    extra={"in_progress_step": int(step), "step_phase": "rollout"},
                )
                batch_samples = self._next_batch_samples()
                group_payloads = []
                step_totals = []
                update_batches = []

                group_payloads = self._collect_group_payloads(batch_samples, step=step)
                self._write_run_state(
                    status="running",
                    current_step=max(0, int(step - 1)),
                    extra={"in_progress_step": int(step), "step_phase": "update"},
                )
                for group_payload in group_payloads:
                    update_batch = group_payload.pop("update_batch")
                    if not bool(group_payload.get("skip_update")):
                        update_batches.append(update_batch)
                    step_totals.extend(group_payload["totals"])

                step_has_update = bool(update_batches)
                update_info = self.policy_backend.update_step(update_batches, step_index=step)
                if step_has_update:
                    self._effective_update_steps += 1
                for group_payload in group_payloads:
                    group_payload["update_info"] = update_info

                monitoring_summary = _build_step_monitoring_summary(group_payloads)
                step_summary = {
                    "step": step,
                    "max_total": max(step_totals),
                    "min_total": min(step_totals),
                    "n_groups": len(group_payloads),
                    "dry_run": self.trainer_config.dry_run,
                    "update_info": update_info,
                    "effective_update_steps": int(self._effective_update_steps),
                    "step_had_effective_update": bool(step_has_update),
                    "n_effective_update_groups": int(len(update_batches)),
                    "n_dynamic_sample_skipped": sum(1 for payload in group_payloads if payload.get("skip_update")),
                    "n_protocol_gate_skipped": sum(
                        1 for payload in group_payloads if payload.get("protocol_gate_skipped")
                    ),
                    "n_protocol_gate_failed": sum(
                        1 for payload in group_payloads if payload.get("protocol_gate_failed")
                    ),
                    "mean_outcome_std": (
                        sum(float(payload.get("outcome_std", 0.0)) for payload in group_payloads)
                        / float(len(group_payloads))
                    ) if group_payloads else 0.0,
                    "mean_protocol_gate_format_pass_count": (
                        sum(int(payload.get("protocol_gate_format_pass_count", 0)) for payload in group_payloads)
                        / float(len(group_payloads))
                    ) if group_payloads else 0.0,
                    "mean_protocol_gate_final_think_raw_valid_count": (
                        sum(
                            int(payload.get("protocol_gate_final_think_raw_valid_count", 0))
                            for payload in group_payloads
                        )
                        / float(len(group_payloads))
                    ) if group_payloads else 0.0,
                    "mean_protocol_gate_pre_eof_think_rollout_count": (
                        sum(
                            int(payload.get("protocol_gate_pre_eof_think_rollout_count", 0))
                            for payload in group_payloads
                        )
                        / float(len(group_payloads))
                    ) if group_payloads else 0.0,
                    "mean_resample_attempts": (
                        sum(int(payload.get("resample_attempts", 0)) for payload in group_payloads)
                        / float(len(group_payloads))
                    ) if group_payloads else 0.0,
                    "step_wall_time_sec": float(time.perf_counter() - step_started_at),
                }
                step_summary.update(monitoring_summary)
                for key, value in update_info.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)) or value is None:
                        step_summary["update_{}".format(key)] = value
                # Keep a small set of stable aliases in the step summary so
                # notebooks/docs do not have to remember the internal
                # `mean_...` vs `update_...` naming split.
                step_summary["mean_total_reward"] = step_summary.get("mean_total")
                step_summary["observed_kl"] = step_summary.get("update_observed_kl")
                step_summary["clip_fraction"] = step_summary.get("update_clip_fraction")
                step_summary["entropy"] = step_summary.get("update_entropy")
                step_summary["teacher_boundary_precision"] = step_summary.get(
                    "mean_teacher_boundary_precision"
                )
                step_summary["teacher_boundary_recall"] = step_summary.get(
                    "mean_teacher_boundary_recall"
                )
                step_summary["predicted_to_target_ratio"] = step_summary.get(
                    "mean_predicted_to_target_ratio"
                )
                health_payload = self._evaluate_health(step_summary, step=step)
                step_summary.update(health_payload)
                for metric_name, trend in health_payload.get("health_trends", {}).items():
                    step_summary["health_trend_{}".format(metric_name)] = trend
                if bool(health_payload.get("health_critical")):
                    self._health_critical_streak += 1
                else:
                    self._health_critical_streak = 0
                step_summary["health_critical_streak"] = int(self._health_critical_streak)
                if (
                    bool(health_payload.get("health_critical"))
                    and not self.trainer_config.health_warn_only
                    and self._health_critical_streak >= max(1, int(self.trainer_config.health_critical_patience))
                ):
                    step_summary["health_stop_triggered"] = True
                    completion_status = "stopped_unhealthy"
                    completion_extra = {
                        "stop_reason": "health_threshold",
                        "health_criticals": list(health_payload.get("health_criticals", [])),
                    }

                checkpoint_events = []
                if self.trainer_config.checkpoint_every > 0 and step % self.trainer_config.checkpoint_every == 0:
                    snapshot_root = self.ckpt_dir / "policy_snapshots"
                    snapshot_root.mkdir(parents=True, exist_ok=True)
                    snapshot_path = snapshot_root / "step_{:06d}.json".format(step)
                    artifact = self.policy_backend.save_checkpoint(str(snapshot_path), step=step)
                    metrics_path = self._write_checkpoint_metrics(Path(artifact.checkpoint_dir), step_summary)
                    snapshot_event = {
                        "kind": "policy_snapshot",
                        "step": step,
                        "mode": artifact.mode,
                        "checkpoint_dir": artifact.checkpoint_dir,
                        "trainer_metrics_path": metrics_path,
                    }
                    should_reload = (
                        self.trainer_config.reload_policy_on_checkpoint
                        and step < self.trainer_config.max_steps
                    )
                    if should_reload:
                        snapshot_event["reload"] = self.policy_backend.reload_from_checkpoint(artifact)
                    elif self.trainer_config.reload_policy_on_checkpoint:
                        snapshot_event["reload_skipped"] = "final-step"
                    if self.trainer_config.checkpoint_keep > 0 or self.trainer_config.checkpoint_keep_best > 0:
                        snapshot_event["pruned"] = prune_step_checkpoints_recent_and_best(
                            str(snapshot_root),
                            keep_recent=max(0, int(self.trainer_config.checkpoint_keep)),
                            keep_best=max(0, int(self.trainer_config.checkpoint_keep_best)),
                            score_key=str(self.trainer_config.checkpoint_score_key or "mean_total"),
                            maximize=bool(self.trainer_config.checkpoint_score_maximize),
                        )
                    self._write_checkpoint_pointer(
                        "latest_policy_snapshot.json",
                        {
                            "kind": "policy_snapshot",
                            "step": step,
                            "mode": artifact.mode,
                            "checkpoint_dir": artifact.checkpoint_dir,
                            "reloadable_model_path": artifact.reloadable_model_path,
                            "metadata_path": artifact.metadata_path,
                            "trainer_metrics_path": metrics_path,
                        },
                    )
                    checkpoint_events.append(snapshot_event)

                if self.trainer_config.full_checkpoint_every > 0 and step % self.trainer_config.full_checkpoint_every == 0:
                    full_root = self.ckpt_dir / "full"
                    full_root.mkdir(parents=True, exist_ok=True)
                    full_path = full_root / "step_{:06d}.json".format(step)
                    artifact = self.policy_backend.save_checkpoint(str(full_path), step=step, checkpoint_mode="full")
                    metrics_path = self._write_checkpoint_metrics(Path(artifact.checkpoint_dir), step_summary)
                    pruned = []
                    if self.trainer_config.full_checkpoint_keep > 0 or self.trainer_config.full_checkpoint_keep_best > 0:
                        pruned = prune_step_checkpoints_recent_and_best(
                            str(full_root),
                            keep_recent=max(0, int(self.trainer_config.full_checkpoint_keep)),
                            keep_best=max(0, int(self.trainer_config.full_checkpoint_keep_best)),
                            score_key=str(self.trainer_config.full_checkpoint_score_key or "mean_total"),
                            maximize=bool(self.trainer_config.full_checkpoint_score_maximize),
                        )
                    self._write_checkpoint_pointer(
                        "latest_full_checkpoint.json",
                        {
                            "kind": "full_checkpoint",
                            "step": step,
                            "mode": artifact.mode,
                            "checkpoint_dir": artifact.checkpoint_dir,
                            "optimizer_path": artifact.optimizer_path,
                            "metadata_path": artifact.metadata_path,
                            "trainer_metrics_path": metrics_path,
                        },
                    )
                    checkpoint_events.append(
                        {
                            "kind": "full_checkpoint",
                            "step": step,
                            "mode": artifact.mode,
                            "checkpoint_dir": artifact.checkpoint_dir,
                            "trainer_metrics_path": metrics_path,
                            "pruned": pruned,
                        }
                    )

                candidate_event = self._maybe_save_policy_candidate(step=step, step_summary=step_summary)
                if candidate_event is not None:
                    checkpoint_events.append(candidate_event)

                if checkpoint_events:
                    step_summary["checkpoint_events"] = checkpoint_events

                if self.trainer_config.save_rollouts:
                    self._write_json(
                        self.steps_dir / "step_{:06d}.json".format(step),
                        {"summary": step_summary, "groups": group_payloads},
                    )

                self._append_jsonl(self.summary_path, step_summary)
                if self._wandb_run is not None:
                    try:
                        self._wandb_run.log(self._wandb_scalar_payload(step_summary), step=step)
                    except Exception:
                        pass
                final_state = step_summary
                self._write_run_state(status="running", current_step=step, extra={"latest_step_summary": step_summary})
                if bool(step_summary.get("health_stop_triggered")):
                    break
        except Exception as exc:
            self._write_run_state(
                status="failed",
                current_step=current_step,
                extra={
                    "error": repr(exc),
                    "last_completed_step": int(final_state.get("step", 0) or 0),
                },
            )
            raise
        finally:
            self.policy_backend.stop()
            if self._wandb_run is not None:
                try:
                    self._wandb_run.finish()
                except Exception:
                    pass

        final_extra = {"final_state": final_state}
        final_extra.update(completion_extra)
        self._write_run_state(
            status=completion_status,
            current_step=int(final_state.get("step", 0) or 0),
            extra=final_extra,
        )
        return final_state
