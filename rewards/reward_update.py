"""
R_u: update-timing reward and controller-side episode metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from .episode import PCoTEpisode
from .reward_accuracy import count_effective_tokens, get_required_depth


DEFAULT_UPDATE_FALLBACK = 0.0
DEFAULT_UPDATE_TRUE_POSITIVE_REWARD = 1.25
DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD = 0.9
DEFAULT_UPDATE_FALSE_POSITIVE_PENALTY = 1.5
DEFAULT_UPDATE_FALSE_NEGATIVE_PENALTY = 1.25
DEFAULT_UPDATE_MIN_THINK_TOKENS = 1
DEFAULT_UPDATE_TOLERANCE_TICKS = 1
DEFAULT_UPDATE_TARGET_THRESHOLD = 0.5
DEFAULT_UPDATE_FALSE_NEGATIVE_RATE_PENALTY = 1.0
DEFAULT_UPDATE_FALSE_POSITIVE_RATE_PENALTY = 1.0
DEFAULT_UPDATE_OVER_PREDICTION_PENALTY = 1.0
DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY = 1.1
DEFAULT_UPDATE_PRECISION_BETA = 0.9
DEFAULT_UPDATE_PROGRESS_POWER = 1.3
DEFAULT_UPDATE_WAIT_TARGET_START = 0.70
DEFAULT_UPDATE_WAIT_TOLERANCE_START = 0.18
DEFAULT_UPDATE_WAIT_TOLERANCE_END = 0.05
DEFAULT_UPDATE_WAIT_OVER_TARGET_PENALTY = 2.5
DEFAULT_UPDATE_WAIT_UNDER_TARGET_PENALTY = 0.75
DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD_START = 0.35
DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY_START = 1.5
DEFAULT_UPDATE_PRECISION_BETA_START = 1.1
DEFAULT_UPDATE_TEACHER_ANCHOR_END = 0.35
DEFAULT_UPDATE_POLICY_CORRECT_REWARD = 1.0
DEFAULT_UPDATE_POLICY_WRONG_PENALTY = 1.0
DEFAULT_UPDATE_POLICY_THINK_DENSITY_PENALTY = 8.0
DEFAULT_UPDATE_POLICY_LAG_PENALTY = 0.5
DEFAULT_UPDATE_POLICY_LAG_NORMALIZER = 2.0
DEFAULT_UPDATE_POLICY_SPARSE_TARGET_EASY = 0.06
DEFAULT_UPDATE_POLICY_SPARSE_TARGET_HARD = 0.15
DEFAULT_UPDATE_POLICY_SPARSE_TOLERANCE = 0.03
DEFAULT_UPDATE_POLICY_ZERO_THINK_WRONG_PENALTY = 0.5
DEFAULT_UPDATE_POLICY_ZERO_THINK_CORRECT_PENALTY = 0.2
DEFAULT_UPDATE_POLICY_MEDIUM_THRESHOLD = 0.45
DEFAULT_UPDATE_POLICY_HARD_THRESHOLD = 0.75
DEFAULT_UPDATE_POLICY_ZERO_THINK_MEDIUM_MULTIPLIER = 2.0
DEFAULT_UPDATE_POLICY_ZERO_THINK_HARD_MULTIPLIER = 3.0
DEFAULT_UPDATE_POLICY_RECALL_ZERO_MEDIUM_MULTIPLIER = 1.25
DEFAULT_UPDATE_POLICY_RECALL_ZERO_HARD_MULTIPLIER = 2.0
DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_MEDIUM = 0.15
DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_HARD = 0.30
DEFAULT_UPDATE_PHASE1_END_STEP = 120
DEFAULT_UPDATE_PHASE2_END_STEP = 300
DEFAULT_UPDATE_PHASE3_END_STEP = 500
DEFAULT_UPDATE_PHASE2_PROGRESS = 0.40
DEFAULT_UPDATE_PHASE3_PROGRESS = 0.75


@dataclass
class UpdateTimingDetail:
    score: float
    per_tick_scores: List[float]
    n_ticks: int
    think_count: int
    wait_count: int
    update_target_mass: float
    no_update_target_mass: float
    true_positive_mass: float
    true_negative_mass: float
    false_positive_mass: float
    false_negative_mass: float
    wait_rate: float
    think_rate: float
    updates_per_episode: float
    teacher_boundary_recall: float
    teacher_boundary_precision: float
    teacher_boundary_f1: float
    false_negative_update_rate: float
    false_positive_update_rate: float
    episodes_with_zero_think_before_eof: float
    eof_to_answer_lag: float
    has_targets: bool
    target_update_count: int
    predicted_update_count: int
    matched_update_count: int
    predicted_to_target_ratio: float
    over_prediction_excess: float
    under_prediction_gap: float
    update_tolerance_ticks: int
    teacher_wait_rate: float
    scheduled_wait_target: float
    scheduled_wait_tolerance: float
    wait_over_target_gap: float
    wait_under_target_gap: float
    wait_corridor_penalty: float
    scheduled_true_negative_reward: float
    scheduled_under_prediction_penalty: float
    scheduled_precision_beta: float
    teacher_anchor_weight: float
    teacher_score: float
    policy_score: float
    policy_think_density: float
    policy_lag_cost: float
    policy_sparse_target_density: float
    policy_sparse_tolerance: float
    policy_overthink_gap: float
    policy_underthink_gap: float
    schedule_progress: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "per_tick_scores": [float(value) for value in self.per_tick_scores],
            "n_ticks": int(self.n_ticks),
            "think_count": int(self.think_count),
            "wait_count": int(self.wait_count),
            "update_target_mass": float(self.update_target_mass),
            "no_update_target_mass": float(self.no_update_target_mass),
            "true_positive_mass": float(self.true_positive_mass),
            "true_negative_mass": float(self.true_negative_mass),
            "false_positive_mass": float(self.false_positive_mass),
            "false_negative_mass": float(self.false_negative_mass),
            "wait_rate": float(self.wait_rate),
            "think_rate": float(self.think_rate),
            "updates_per_episode": float(self.updates_per_episode),
            "teacher_boundary_recall": float(self.teacher_boundary_recall),
            "teacher_boundary_precision": float(self.teacher_boundary_precision),
            "teacher_boundary_f1": float(self.teacher_boundary_f1),
            "false_negative_update_rate": float(self.false_negative_update_rate),
            "false_positive_update_rate": float(self.false_positive_update_rate),
            "episodes_with_zero_think_before_eof": float(self.episodes_with_zero_think_before_eof),
            "eof_to_answer_lag": float(self.eof_to_answer_lag),
            "has_targets": bool(self.has_targets),
            "target_update_count": int(self.target_update_count),
            "predicted_update_count": int(self.predicted_update_count),
            "matched_update_count": int(self.matched_update_count),
            "predicted_to_target_ratio": float(self.predicted_to_target_ratio),
            "over_prediction_excess": float(self.over_prediction_excess),
            "under_prediction_gap": float(self.under_prediction_gap),
            "update_tolerance_ticks": int(self.update_tolerance_ticks),
            "teacher_wait_rate": float(self.teacher_wait_rate),
            "scheduled_wait_target": float(self.scheduled_wait_target),
            "scheduled_wait_tolerance": float(self.scheduled_wait_tolerance),
            "wait_over_target_gap": float(self.wait_over_target_gap),
            "wait_under_target_gap": float(self.wait_under_target_gap),
            "wait_corridor_penalty": float(self.wait_corridor_penalty),
            "scheduled_true_negative_reward": float(self.scheduled_true_negative_reward),
            "scheduled_under_prediction_penalty": float(self.scheduled_under_prediction_penalty),
            "scheduled_precision_beta": float(self.scheduled_precision_beta),
            "teacher_anchor_weight": float(self.teacher_anchor_weight),
            "teacher_score": float(self.teacher_score),
            "policy_score": float(self.policy_score),
            "policy_think_density": float(self.policy_think_density),
            "policy_lag_cost": float(self.policy_lag_cost),
            "policy_sparse_target_density": float(self.policy_sparse_target_density),
            "policy_sparse_tolerance": float(self.policy_sparse_tolerance),
            "policy_overthink_gap": float(self.policy_overthink_gap),
            "policy_underthink_gap": float(self.policy_underthink_gap),
            "schedule_progress": float(self.schedule_progress),
        }


def _clamp01(value: Any) -> float:
    try:
        return float(min(1.0, max(0.0, float(value))))
    except Exception:
        return 0.0


def _lerp(start: float, end: float, t: float) -> float:
    return float(start) + (float(end) - float(start)) * float(t)


def _schedule_progress(progress_fraction: float, progress_power: float) -> float:
    progress = _clamp01(progress_fraction)
    power = max(1e-6, float(progress_power))
    return float(progress**power)


def _piecewise_schedule_progress(
    *,
    current_step: int,
    max_steps: int,
    progress_fraction: float,
) -> float:
    total_steps = max(1, int(max_steps))
    step = max(0, int(current_step))
    if total_steps <= DEFAULT_UPDATE_PHASE1_END_STEP:
        return _clamp01(progress_fraction)

    phase1_end = min(total_steps, DEFAULT_UPDATE_PHASE1_END_STEP)
    phase2_end = min(total_steps, max(phase1_end + 1, DEFAULT_UPDATE_PHASE2_END_STEP))
    phase3_end = min(total_steps, max(phase2_end + 1, DEFAULT_UPDATE_PHASE3_END_STEP))

    if step <= phase1_end:
        return 0.0

    if step <= phase2_end:
        local_t = (step - phase1_end) / float(max(1, phase2_end - phase1_end))
        return _lerp(0.0, DEFAULT_UPDATE_PHASE2_PROGRESS, local_t)

    if step <= phase3_end:
        local_t = (step - phase2_end) / float(max(1, phase3_end - phase2_end))
        return _lerp(DEFAULT_UPDATE_PHASE2_PROGRESS, DEFAULT_UPDATE_PHASE3_PROGRESS, local_t)

    local_t = (step - phase3_end) / float(max(1, total_steps - phase3_end))
    return _lerp(DEFAULT_UPDATE_PHASE3_PROGRESS, 1.0, _clamp01(local_t))


def _event_actions_by_tick(episode: PCoTEpisode, n_ticks: int) -> List[str]:
    actions = ["wait" for _ in range(n_ticks)]
    if episode.rollout_events:
        for event in episode.rollout_events:
            kind = str(event.get("kind", "")).strip().lower()
            try:
                chunk_index = int(event.get("chunk_index", -1))
            except Exception:
                chunk_index = -1
            if not (0 <= chunk_index < n_ticks):
                continue
            if kind == "assistant_think":
                timing = event.get("timing")
                is_final_think = bool(event.get("is_final_think"))
                if isinstance(timing, dict):
                    is_final_think = is_final_think or bool(timing.get("is_final_think"))
                if is_final_think:
                    continue
                actions[chunk_index] = "think"
            elif kind == "assistant_wait":
                actions[chunk_index] = "wait"
        return actions

    thinks = list(getattr(episode, "thinks", []) or [])
    for idx in range(min(n_ticks, len(thinks))):
        if count_effective_tokens(thinks[idx]) >= DEFAULT_UPDATE_MIN_THINK_TOKENS:
            actions[idx] = "think"
    return actions


def _get_tick_count(episode: PCoTEpisode, metadata: Dict[str, Any]) -> int:
    for key in ("ingest_grid_ticks", "n_ingest_ticks", "n_ticks"):
        if key in metadata:
            try:
                value = int(metadata[key])
                if value > 0:
                    return value
            except Exception:
                pass
    return max(0, int(episode.n_chunks))


def _build_target_probabilities(n_ticks: int, metadata: Dict[str, Any]) -> List[float]:
    probabilities = [0.0 for _ in range(n_ticks)]
    boundary_after_tick = metadata.get("boundary_after_tick_t")
    if isinstance(boundary_after_tick, Sequence) and not isinstance(boundary_after_tick, (str, bytes)):
        for idx in range(min(n_ticks, len(boundary_after_tick))):
            probabilities[idx] = _clamp01(boundary_after_tick[idx])
        return probabilities

    update_tick_indices = metadata.get("update_tick_indices")
    if isinstance(update_tick_indices, Sequence) and not isinstance(update_tick_indices, (str, bytes)):
        confidence_values = metadata.get("update_tick_confidences") or metadata.get("update_tick_scores") or []
        if not isinstance(confidence_values, Sequence) or isinstance(confidence_values, (str, bytes)):
            confidence_values = []
        for pos, tick in enumerate(update_tick_indices):
            try:
                tick_idx = int(tick)
            except Exception:
                continue
            if not (0 <= tick_idx < n_ticks):
                continue
            confidence = _clamp01(confidence_values[pos]) if pos < len(confidence_values) else 1.0
            probabilities[tick_idx] = max(probabilities[tick_idx], confidence)
    return probabilities


def _build_target_ticks(
    n_ticks: int,
    metadata: Dict[str, Any],
    *,
    threshold: float,
) -> List[int]:
    target_ticks: Set[int] = set()
    update_tick_indices = metadata.get("update_tick_indices")
    if isinstance(update_tick_indices, Sequence) and not isinstance(update_tick_indices, (str, bytes)):
        for tick in update_tick_indices:
            try:
                tick_idx = int(tick)
            except Exception:
                continue
            if 0 <= tick_idx < n_ticks:
                target_ticks.add(tick_idx)
        if target_ticks:
            return sorted(target_ticks)

    boundary_after_tick = metadata.get("boundary_after_tick_t")
    if isinstance(boundary_after_tick, Sequence) and not isinstance(boundary_after_tick, (str, bytes)):
        for idx in range(min(n_ticks, len(boundary_after_tick))):
            if _clamp01(boundary_after_tick[idx]) >= threshold:
                target_ticks.add(idx)
    return sorted(target_ticks)


def _predicted_update_ticks(episode: PCoTEpisode, n_ticks: int) -> List[int]:
    actions = _event_actions_by_tick(episode, n_ticks)
    return [idx for idx, action in enumerate(actions) if action == "think"]


def _match_with_tolerance(
    predicted_ticks: Sequence[int],
    target_ticks: Sequence[int],
    *,
    tolerance_ticks: int,
) -> tuple[Dict[int, int], Set[int], Set[int]]:
    matched_targets: Dict[int, int] = {}
    matched_predicted: Set[int] = set()
    used_predicted: Set[int] = set()

    for target_tick in sorted(int(tick) for tick in target_ticks):
        best_predicted = None
        best_distance = None
        for predicted_tick in sorted(int(tick) for tick in predicted_ticks):
            if predicted_tick in used_predicted:
                continue
            distance = abs(predicted_tick - target_tick)
            if distance > tolerance_ticks:
                continue
            if best_distance is None or distance < best_distance:
                best_predicted = predicted_tick
                best_distance = distance
        if best_predicted is not None:
            matched_targets[target_tick] = best_predicted
            matched_predicted.add(best_predicted)
            used_predicted.add(best_predicted)

    return matched_targets, set(matched_targets.keys()), matched_predicted


def infer_eof_to_answer_lag(episode: PCoTEpisode, n_ticks: int) -> float:
    if not episode.rollout_events:
        if not episode.answer.strip():
            return -1.0
        return 0.0 if not episode.answer_appears_before_last_chunk() else -1.0

    has_explicit_eof = any(
        str(event.get("kind", "")).strip().lower() in {"user_eof", "stream_eof", "audio_end"}
        for event in episode.rollout_events
    )
    eof_seen = False
    actions_after_eof = 0
    heard_chunks = 0
    for event in episode.rollout_events:
        kind = str(event.get("kind", "")).strip().lower()
        timing = event.get("timing")
        is_final_think = bool(event.get("is_final_think"))
        if isinstance(timing, dict):
            is_final_think = is_final_think or bool(timing.get("is_final_think"))
        if kind in {"user_audio", "user_chunk"}:
            heard_chunks += 1
            if heard_chunks >= n_ticks and not has_explicit_eof:
                eof_seen = True
            continue
        if kind in {"user_eof", "stream_eof", "audio_end"}:
            eof_seen = True
            continue
        if kind == "assistant_answer":
            if not eof_seen:
                return -1.0
            return float(actions_after_eof)
        if eof_seen and kind == "assistant_wait":
            actions_after_eof += 1
            continue
        if eof_seen and kind == "assistant_think" and not is_final_think:
            actions_after_eof += 1
    return -1.0 if episode.answer.strip() else float(actions_after_eof)


def _difficulty_bucket(
    difficulty: float,
    *,
    medium_threshold: float,
    hard_threshold: float,
) -> str:
    difficulty = float(difficulty)
    if difficulty >= float(hard_threshold):
        return "hard"
    if difficulty >= float(medium_threshold):
        return "medium"
    return "easy"


def _difficulty_bucket_value(
    bucket: str,
    *,
    medium_value: float,
    hard_value: float,
    easy_value: float = 0.0,
) -> float:
    if bucket == "hard":
        return float(hard_value)
    if bucket == "medium":
        return float(medium_value)
    return float(easy_value)


def _compute_policy_score(
    episode: PCoTEpisode,
    *,
    has_targets: bool,
    teacher_recall: float,
    think_count: int,
    matched_count: int,
    n_ticks: int,
    eof_to_answer_lag: float,
    correct_reward: float,
    wrong_penalty: float,
    think_density_penalty: float,
    lag_penalty: float,
    lag_normalizer: float,
    sparse_target_easy: float,
    sparse_target_hard: float,
    sparse_tolerance: float,
    zero_think_wrong_penalty: float,
    zero_think_correct_penalty: float,
    medium_threshold: float,
    hard_threshold: float,
    zero_think_medium_multiplier: float,
    zero_think_hard_multiplier: float,
    recall_zero_medium_multiplier: float,
    recall_zero_hard_multiplier: float,
    target_hit_bonus_medium: float,
    target_hit_bonus_hard: float,
) -> tuple[float, float, float, float, float, float, float]:
    think_density = float(think_count) / float(max(1, n_ticks))
    lag_normalizer = max(1e-6, float(lag_normalizer))
    if eof_to_answer_lag < 0:
        lag_cost = 1.0
    else:
        lag_cost = min(1.0, max(0.0, float(eof_to_answer_lag)) / lag_normalizer)
    difficulty = float(get_required_depth(episode))
    sparse_target_density = (
        float(sparse_target_easy)
        + (float(sparse_target_hard) - float(sparse_target_easy)) * difficulty
    )
    sparse_tolerance = float(sparse_tolerance)
    overthink_gap = max(0.0, think_density - (sparse_target_density + sparse_tolerance))
    underthink_gap = max(0.0, (sparse_target_density - sparse_tolerance) - think_density)
    difficulty_bucket = _difficulty_bucket(
        difficulty,
        medium_threshold=medium_threshold,
        hard_threshold=hard_threshold,
    )
    teacher_positive_medium_hard = bool(has_targets) and difficulty_bucket in {"medium", "hard"}
    zero_think_multiplier = _difficulty_bucket_value(
        difficulty_bucket,
        medium_value=zero_think_medium_multiplier,
        hard_value=zero_think_hard_multiplier,
        easy_value=0.0,
    )
    recall_zero_multiplier = _difficulty_bucket_value(
        difficulty_bucket,
        medium_value=recall_zero_medium_multiplier,
        hard_value=recall_zero_hard_multiplier,
        easy_value=0.0,
    )
    target_hit_bonus = _difficulty_bucket_value(
        difficulty_bucket,
        medium_value=target_hit_bonus_medium,
        hard_value=target_hit_bonus_hard,
        easy_value=0.0,
    )

    if episode.is_correct():
        score = (
            float(correct_reward)
            - float(think_density_penalty) * (1.10 * float(overthink_gap) + 0.90 * float(underthink_gap))
            - float(lag_penalty) * float(lag_cost)
        )
        if teacher_positive_medium_hard and think_count == 0:
            score -= float(zero_think_multiplier) * float(zero_think_correct_penalty)
        elif teacher_positive_medium_hard and float(teacher_recall) <= 1e-8:
            score -= float(recall_zero_multiplier) * float(zero_think_correct_penalty)
        elif teacher_positive_medium_hard and matched_count > 0:
            score += float(target_hit_bonus)
    else:
        score = (
            -float(wrong_penalty)
            - float(think_density_penalty) * (0.50 * float(overthink_gap) + 2.00 * float(underthink_gap))
            - float(lag_penalty) * float(lag_cost)
        )
        if teacher_positive_medium_hard and think_count == 0:
            score -= float(zero_think_multiplier) * float(zero_think_wrong_penalty) * float(wrong_penalty)
        elif teacher_positive_medium_hard and float(teacher_recall) <= 1e-8:
            score -= float(recall_zero_multiplier) * float(zero_think_wrong_penalty) * float(wrong_penalty)
        elif teacher_positive_medium_hard and matched_count > 0:
            score += 0.5 * float(target_hit_bonus)
    return (
        float(score),
        float(think_density),
        float(lag_cost),
        float(sparse_target_density),
        float(sparse_tolerance),
        float(overthink_gap),
        float(underthink_gap),
    )


def compute_update_timing_detail(
    episode: PCoTEpisode,
    *,
    fallback: float = DEFAULT_UPDATE_FALLBACK,
    true_positive_reward: float = DEFAULT_UPDATE_TRUE_POSITIVE_REWARD,
    true_negative_reward: float = DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD,
    false_positive_penalty: float = DEFAULT_UPDATE_FALSE_POSITIVE_PENALTY,
    false_negative_penalty: float = DEFAULT_UPDATE_FALSE_NEGATIVE_PENALTY,
    tolerance_ticks: int = DEFAULT_UPDATE_TOLERANCE_TICKS,
    target_threshold: float = DEFAULT_UPDATE_TARGET_THRESHOLD,
    false_negative_rate_penalty: float = DEFAULT_UPDATE_FALSE_NEGATIVE_RATE_PENALTY,
    false_positive_rate_penalty: float = DEFAULT_UPDATE_FALSE_POSITIVE_RATE_PENALTY,
    over_prediction_penalty: float = DEFAULT_UPDATE_OVER_PREDICTION_PENALTY,
    under_prediction_penalty: float = DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY,
    precision_beta: float = DEFAULT_UPDATE_PRECISION_BETA,
    progress_fraction: float = 0.0,
    progress_power: float = DEFAULT_UPDATE_PROGRESS_POWER,
    current_step: int = 0,
    max_steps: int = 1000,
    wait_target_start: float = DEFAULT_UPDATE_WAIT_TARGET_START,
    wait_tolerance_start: float = DEFAULT_UPDATE_WAIT_TOLERANCE_START,
    wait_tolerance_end: float = DEFAULT_UPDATE_WAIT_TOLERANCE_END,
    wait_over_target_penalty: float = DEFAULT_UPDATE_WAIT_OVER_TARGET_PENALTY,
    wait_under_target_penalty: float = DEFAULT_UPDATE_WAIT_UNDER_TARGET_PENALTY,
    true_negative_reward_start: float = DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD_START,
    under_prediction_penalty_start: float = DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY_START,
    precision_beta_start: float = DEFAULT_UPDATE_PRECISION_BETA_START,
    teacher_anchor_end: float = DEFAULT_UPDATE_TEACHER_ANCHOR_END,
    policy_correct_reward: float = DEFAULT_UPDATE_POLICY_CORRECT_REWARD,
    policy_wrong_penalty: float = DEFAULT_UPDATE_POLICY_WRONG_PENALTY,
    policy_think_density_penalty: float = DEFAULT_UPDATE_POLICY_THINK_DENSITY_PENALTY,
    policy_lag_penalty: float = DEFAULT_UPDATE_POLICY_LAG_PENALTY,
    policy_lag_normalizer: float = DEFAULT_UPDATE_POLICY_LAG_NORMALIZER,
    policy_sparse_target_easy: float = DEFAULT_UPDATE_POLICY_SPARSE_TARGET_EASY,
    policy_sparse_target_hard: float = DEFAULT_UPDATE_POLICY_SPARSE_TARGET_HARD,
    policy_sparse_tolerance: float = DEFAULT_UPDATE_POLICY_SPARSE_TOLERANCE,
    policy_zero_think_wrong_penalty: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_WRONG_PENALTY,
    policy_zero_think_correct_penalty: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_CORRECT_PENALTY,
    policy_medium_threshold: float = DEFAULT_UPDATE_POLICY_MEDIUM_THRESHOLD,
    policy_hard_threshold: float = DEFAULT_UPDATE_POLICY_HARD_THRESHOLD,
    policy_zero_think_medium_multiplier: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_MEDIUM_MULTIPLIER,
    policy_zero_think_hard_multiplier: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_HARD_MULTIPLIER,
    policy_recall_zero_medium_multiplier: float = DEFAULT_UPDATE_POLICY_RECALL_ZERO_MEDIUM_MULTIPLIER,
    policy_recall_zero_hard_multiplier: float = DEFAULT_UPDATE_POLICY_RECALL_ZERO_HARD_MULTIPLIER,
    policy_target_hit_bonus_medium: float = DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_MEDIUM,
    policy_target_hit_bonus_hard: float = DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_HARD,
) -> UpdateTimingDetail:
    metadata = dict(getattr(episode, "controller_metadata", {}) or {})
    if not metadata:
        metadata = dict(getattr(episode, "difficulty_metadata", {}) or {})

    n_ticks = _get_tick_count(episode, metadata)
    if n_ticks <= 0:
        return UpdateTimingDetail(
            score=float(fallback),
            per_tick_scores=[],
            n_ticks=0,
            think_count=0,
            wait_count=0,
            update_target_mass=0.0,
            no_update_target_mass=0.0,
            true_positive_mass=0.0,
            true_negative_mass=0.0,
            false_positive_mass=0.0,
            false_negative_mass=0.0,
            wait_rate=0.0,
            think_rate=0.0,
            updates_per_episode=0.0,
            teacher_boundary_recall=0.0,
            teacher_boundary_precision=0.0,
            teacher_boundary_f1=0.0,
            false_negative_update_rate=0.0,
            false_positive_update_rate=0.0,
            episodes_with_zero_think_before_eof=0.0,
            eof_to_answer_lag=infer_eof_to_answer_lag(episode, 0),
            has_targets=False,
            target_update_count=0,
            predicted_update_count=0,
            matched_update_count=0,
            predicted_to_target_ratio=0.0,
            over_prediction_excess=0.0,
            under_prediction_gap=0.0,
            update_tolerance_ticks=int(max(0, tolerance_ticks)),
            teacher_wait_rate=0.0,
            scheduled_wait_target=float(wait_target_start),
            scheduled_wait_tolerance=float(wait_tolerance_start),
            wait_over_target_gap=0.0,
            wait_under_target_gap=0.0,
            wait_corridor_penalty=0.0,
            scheduled_true_negative_reward=float(true_negative_reward_start),
            scheduled_under_prediction_penalty=float(under_prediction_penalty_start),
            scheduled_precision_beta=float(precision_beta_start),
            teacher_anchor_weight=1.0,
            teacher_score=float(fallback),
            policy_score=0.0,
            policy_think_density=0.0,
            policy_lag_cost=0.0,
            policy_sparse_target_density=0.0,
            policy_sparse_tolerance=0.0,
            policy_overthink_gap=0.0,
            policy_underthink_gap=0.0,
            schedule_progress=0.0,
        )

    target_probabilities = _build_target_probabilities(n_ticks, metadata)
    target_ticks = _build_target_ticks(
        n_ticks,
        metadata,
        threshold=target_threshold,
    )
    has_targets = bool(target_ticks) or any(prob > 0.0 for prob in target_probabilities)
    actions = _event_actions_by_tick(episode, n_ticks)
    predicted_ticks = _predicted_update_ticks(episode, n_ticks)
    matched_targets, matched_target_set, matched_predicted_set = _match_with_tolerance(
        predicted_ticks,
        target_ticks,
        tolerance_ticks=max(0, int(tolerance_ticks)),
    )
    unmatched_target_set = set(target_ticks) - matched_target_set
    unmatched_predicted_set = set(predicted_ticks) - matched_predicted_set

    think_count = sum(1 for action in actions if action == "think")
    wait_count = max(0, n_ticks - think_count)
    wait_rate = wait_count / float(n_ticks)

    tp_mass = float(len(matched_target_set))
    fn_mass = float(len(unmatched_target_set))
    fp_mass = float(len(unmatched_predicted_set))
    update_target_mass = float(len(target_ticks))
    no_update_target_mass = float(max(0, n_ticks - len(target_ticks)))
    tn_mass = float(max(0, n_ticks - int(tp_mass + fn_mass + fp_mass)))

    smooth_progress = _schedule_progress(progress_fraction, progress_power)
    schedule_progress = _piecewise_schedule_progress(
        current_step=current_step,
        max_steps=max_steps,
        progress_fraction=smooth_progress,
    )
    teacher_wait_rate = no_update_target_mass / float(n_ticks)
    scheduled_wait_target = _lerp(wait_target_start, teacher_wait_rate, schedule_progress)
    scheduled_wait_tolerance = _lerp(wait_tolerance_start, wait_tolerance_end, schedule_progress)
    scheduled_true_negative_reward = _lerp(
        true_negative_reward_start,
        true_negative_reward,
        schedule_progress,
    )
    scheduled_under_prediction_penalty = _lerp(
        under_prediction_penalty_start,
        under_prediction_penalty,
        schedule_progress,
    )
    scheduled_precision_beta = _lerp(
        precision_beta_start,
        precision_beta,
        schedule_progress,
    )
    teacher_anchor_weight = _lerp(1.0, teacher_anchor_end, schedule_progress)

    teacher_per_tick_scores: List[float] = []
    for tick_idx, action in enumerate(actions):
        tick_score = 0.0
        if tick_idx in matched_predicted_set:
            tick_score += float(true_positive_reward)
        elif tick_idx in unmatched_predicted_set:
            tick_score -= float(false_positive_penalty)
        elif tick_idx in unmatched_target_set:
            tick_score -= float(false_negative_penalty)
        elif action == "wait":
            tick_score += float(scheduled_true_negative_reward)
        teacher_per_tick_scores.append(float(tick_score))

    predicted_count = len(predicted_ticks)
    target_count = len(target_ticks)
    matched_count = len(matched_target_set)

    if target_count == 0 and predicted_count == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        fn_rate = 0.0
        fp_rate = 0.0
        predicted_to_target_ratio = 0.0
        over_prediction_excess = 0.0
        under_prediction_gap = 0.0
        score = 1.0
    elif target_count == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        fn_rate = 0.0
        fp_rate = 1.0
        predicted_to_target_ratio = float(predicted_count)
        over_prediction_excess = min(1.0, float(predicted_count) / float(max(1, n_ticks)))
        under_prediction_gap = 0.0
        score = -float(false_positive_rate_penalty) - float(over_prediction_penalty) * float(over_prediction_excess)
    else:
        precision = matched_count / float(predicted_count) if predicted_count > 0 else 0.0
        recall = matched_count / float(target_count)
        if precision + recall > 1e-8:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        precision_beta_sq = float(max(1e-6, scheduled_precision_beta)) ** 2
        if precision > 0.0 and recall > 0.0:
            precision_biased_f = (1.0 + precision_beta_sq) * precision * recall / (
                precision_beta_sq * precision + recall
            )
        else:
            precision_biased_f = 0.0
        fn_rate = (target_count - matched_count) / float(target_count)
        fp_rate = (predicted_count - matched_count) / float(max(1, predicted_count))
        predicted_to_target_ratio = predicted_count / float(max(1, target_count))
        over_prediction_excess = max(0.0, min(4.0, predicted_to_target_ratio) - 1.0) / 3.0
        under_prediction_gap = max(0.0, 1.0 - min(1.0, predicted_to_target_ratio))
        score = (
            float(precision_biased_f)
            - float(false_negative_rate_penalty) * float(fn_rate)
            - float(false_positive_rate_penalty) * float(fp_rate)
            - float(over_prediction_penalty) * float(over_prediction_excess)
            - float(scheduled_under_prediction_penalty) * float(under_prediction_gap)
        )

    if not has_targets and n_ticks > 0 and predicted_count == 0:
        teacher_per_tick_scores = [float(scheduled_true_negative_reward) for _ in range(n_ticks)]
        if abs(score) < 1e-8:
            score = float(fallback)

    wait_over_target_gap = 0.0
    wait_under_target_gap = 0.0
    wait_corridor_penalty = 0.0
    if has_targets and n_ticks > 0:
        upper = min(1.0, scheduled_wait_target + scheduled_wait_tolerance)
        lower = max(0.0, scheduled_wait_target - scheduled_wait_tolerance)
        wait_over_target_gap = max(0.0, wait_rate - upper)
        wait_under_target_gap = max(0.0, lower - wait_rate)
        wait_corridor_penalty = (
            float(wait_over_target_penalty) * float(wait_over_target_gap)
            + float(wait_under_target_penalty) * float(wait_under_target_gap)
        )
        score -= float(wait_corridor_penalty)

    teacher_score = float(score)
    eof_to_answer_lag = infer_eof_to_answer_lag(episode, n_ticks)
    (
        policy_score,
        policy_think_density,
        policy_lag_cost,
        policy_sparse_target_density,
        policy_sparse_tolerance,
        policy_overthink_gap,
        policy_underthink_gap,
    ) = _compute_policy_score(
        episode,
        has_targets=has_targets,
        teacher_recall=recall,
        think_count=think_count,
        matched_count=matched_count,
        n_ticks=n_ticks,
        eof_to_answer_lag=eof_to_answer_lag,
        correct_reward=policy_correct_reward,
        wrong_penalty=policy_wrong_penalty,
        think_density_penalty=policy_think_density_penalty,
        lag_penalty=policy_lag_penalty,
        lag_normalizer=policy_lag_normalizer,
        sparse_target_easy=policy_sparse_target_easy,
        sparse_target_hard=policy_sparse_target_hard,
        sparse_tolerance=policy_sparse_tolerance,
        zero_think_wrong_penalty=policy_zero_think_wrong_penalty,
        zero_think_correct_penalty=policy_zero_think_correct_penalty,
        medium_threshold=policy_medium_threshold,
        hard_threshold=policy_hard_threshold,
        zero_think_medium_multiplier=policy_zero_think_medium_multiplier,
        zero_think_hard_multiplier=policy_zero_think_hard_multiplier,
        recall_zero_medium_multiplier=policy_recall_zero_medium_multiplier,
        recall_zero_hard_multiplier=policy_recall_zero_hard_multiplier,
        target_hit_bonus_medium=policy_target_hit_bonus_medium,
        target_hit_bonus_hard=policy_target_hit_bonus_hard,
    )
    policy_weight = max(0.0, 1.0 - float(teacher_anchor_weight))
    blended_score = (
        float(teacher_anchor_weight) * float(teacher_score)
        + float(policy_weight) * float(policy_score)
    )
    per_tick_scores = [
        float(teacher_anchor_weight) * float(teacher_tick)
        + float(policy_weight) * float(policy_score)
        for teacher_tick in teacher_per_tick_scores
    ]

    return UpdateTimingDetail(
        score=blended_score,
        per_tick_scores=per_tick_scores,
        n_ticks=n_ticks,
        think_count=think_count,
        wait_count=wait_count,
        update_target_mass=update_target_mass,
        no_update_target_mass=no_update_target_mass,
        true_positive_mass=tp_mass,
        true_negative_mass=tn_mass,
        false_positive_mass=fp_mass,
        false_negative_mass=fn_mass,
        wait_rate=wait_rate,
        think_rate=think_count / float(n_ticks),
        updates_per_episode=float(think_count),
        teacher_boundary_recall=recall,
        teacher_boundary_precision=precision,
        teacher_boundary_f1=f1,
        false_negative_update_rate=fn_rate,
        false_positive_update_rate=fp_rate,
        episodes_with_zero_think_before_eof=1.0 if think_count == 0 else 0.0,
        eof_to_answer_lag=eof_to_answer_lag,
        has_targets=has_targets,
        target_update_count=int(target_count),
        predicted_update_count=int(predicted_count),
        matched_update_count=int(matched_count),
        predicted_to_target_ratio=float(predicted_to_target_ratio),
        over_prediction_excess=float(over_prediction_excess),
        under_prediction_gap=float(under_prediction_gap),
        update_tolerance_ticks=int(max(0, tolerance_ticks)),
        teacher_wait_rate=float(teacher_wait_rate),
        scheduled_wait_target=float(scheduled_wait_target),
        scheduled_wait_tolerance=float(scheduled_wait_tolerance),
        wait_over_target_gap=float(wait_over_target_gap),
        wait_under_target_gap=float(wait_under_target_gap),
        wait_corridor_penalty=float(wait_corridor_penalty),
        scheduled_true_negative_reward=float(scheduled_true_negative_reward),
        scheduled_under_prediction_penalty=float(scheduled_under_prediction_penalty),
        scheduled_precision_beta=float(scheduled_precision_beta),
        teacher_anchor_weight=float(teacher_anchor_weight),
        teacher_score=float(teacher_score),
        policy_score=float(policy_score),
        policy_think_density=float(policy_think_density),
        policy_lag_cost=float(policy_lag_cost),
        policy_sparse_target_density=float(policy_sparse_target_density),
        policy_sparse_tolerance=float(policy_sparse_tolerance),
        policy_overthink_gap=float(policy_overthink_gap),
        policy_underthink_gap=float(policy_underthink_gap),
        schedule_progress=float(schedule_progress),
    )
