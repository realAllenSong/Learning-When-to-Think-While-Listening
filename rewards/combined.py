"""Combined reward assembly for wait-think-answer DAPO training.

The base reward stack is:

- `R_f`: format / controller validity
- `R_a`: final-answer correctness
- `R_u`: update-timing alignment
- `R_s`: latency / post-EOF wait pressure

Episode-level assembly implemented in this module:

  R_outcome
    = 1[R_a enabled] * R_a
    + 1[R_f enabled] * format_scale * R_f
    + 1[R_s enabled] * sync_scale * R_s
    + 1[R_p enabled] * prediction_scale * R_p
    + 1[R_a and R_c enabled and answer correct] * (R_a * consistency_bonus * R_c)

  total
    = R_outcome
    + 1[R_t enabled] * think_scale * R_t
    + 1[R_u enabled] * update_scale * R_u

Trainer-side credit assignment is then:

- The trainer group-normalizes `total` across the `G=8` rollouts of
  the same prompt to get the trajectory advantage
- if `hybrid-local` credit is enabled, `think`/`wait` turns blend:
    alpha * outcome_advantage + (1 - alpha) * local_process_advantage
  where the local process signal currently comes from `R_u_per_tick`
  (and `R_t_per_chunk` if think-judge rewards are enabled)
- answer turns use the trajectory/outcome advantage directly

This keeps early `wait` / `think` decisions tied to the quality and latency of
the final trajectory without pretending we already have a full trainer-level
per-token reward decomposition.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .episode import PCoTEpisode
from .reward_accuracy import (
    BatchBalanceContext,
    DEFAULT_ACCURACY_MODE,
    DEFAULT_DIFFICULTY_MARGIN,
    DEFAULT_DIFFICULTY_SCORE,
    DEFAULT_EXTRA_DEPTH_NORMALIZER,
    DEFAULT_MIN_EFFECTIVE_TOKENS,
    DEFAULT_STATE_FLOOR_TOKENS,
    compute_effort,
    reward_accuracy,
)
from .reward_consistency import reward_consistency_sync
from .reward_format import reward_format
from .reference_answer_fallback import prepare_reference_answer_fallback_batch
from .reward_sync import (
    DEFAULT_ANSWER_ALPHA,
    DEFAULT_EFFECTIVE_TEXT_FIRST_TOKEN_ALPHA,
    DEFAULT_EFFECTIVE_RESPONSE_ONSET_ALPHA,
    DEFAULT_FINAL_THINK_TOKEN_ALPHA,
    DEFAULT_FINAL_THINK_TOKEN_PENALTY_CAP,
    DEFAULT_FREE_ANSWER_TOKENS,
    DEFAULT_FREE_EFFECTIVE_TEXT_FIRST_TOKEN_SECONDS,
    DEFAULT_FREE_EFFECTIVE_RESPONSE_ONSET_SECONDS,
    DEFAULT_FREE_FINAL_THINK_TOKENS,
    DEFAULT_FREE_STATE_TOKENS,
    DEFAULT_FREE_LATENCY_TOKENS,
    DEFAULT_FREE_POST_EOF_WALL_CLOCK_SECONDS,
    DEFAULT_FREE_TEXT_FIRST_TOKEN_SECONDS,
    DEFAULT_LATENCY_TOKEN_ALPHA,
    DEFAULT_POST_EOF_WALL_CLOCK_ALPHA,
    DEFAULT_TEXT_FIRST_TOKEN_ALPHA,
    compute_sync_detail,
    infer_final_think_token_count,
    infer_response_latency_token_proxy,
)
from .reward_think import reward_think_sync_details, reward_think_sync_batch_details
from .reward_update import (
    DEFAULT_UPDATE_FALLBACK,
    DEFAULT_UPDATE_FALSE_NEGATIVE_PENALTY,
    DEFAULT_UPDATE_FALSE_NEGATIVE_RATE_PENALTY,
    DEFAULT_UPDATE_OVER_PREDICTION_PENALTY,
    DEFAULT_UPDATE_POLICY_HARD_THRESHOLD,
    DEFAULT_UPDATE_POLICY_CORRECT_REWARD,
    DEFAULT_UPDATE_POLICY_LAG_NORMALIZER,
    DEFAULT_UPDATE_POLICY_LAG_PENALTY,
    DEFAULT_UPDATE_POLICY_MEDIUM_THRESHOLD,
    DEFAULT_UPDATE_POLICY_RECALL_ZERO_HARD_MULTIPLIER,
    DEFAULT_UPDATE_POLICY_RECALL_ZERO_MEDIUM_MULTIPLIER,
    DEFAULT_UPDATE_POLICY_SPARSE_TARGET_EASY,
    DEFAULT_UPDATE_POLICY_SPARSE_TARGET_HARD,
    DEFAULT_UPDATE_POLICY_SPARSE_TOLERANCE,
    DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_HARD,
    DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_MEDIUM,
    DEFAULT_UPDATE_POLICY_THINK_DENSITY_PENALTY,
    DEFAULT_UPDATE_POLICY_WRONG_PENALTY,
    DEFAULT_UPDATE_POLICY_ZERO_THINK_CORRECT_PENALTY,
    DEFAULT_UPDATE_POLICY_ZERO_THINK_HARD_MULTIPLIER,
    DEFAULT_UPDATE_POLICY_ZERO_THINK_MEDIUM_MULTIPLIER,
    DEFAULT_UPDATE_POLICY_ZERO_THINK_WRONG_PENALTY,
    DEFAULT_UPDATE_PROGRESS_POWER,
    DEFAULT_UPDATE_PRECISION_BETA_START,
    DEFAULT_UPDATE_TEACHER_ANCHOR_END,
    DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY,
    DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY_START,
    DEFAULT_UPDATE_FALSE_POSITIVE_PENALTY,
    DEFAULT_UPDATE_FALSE_POSITIVE_RATE_PENALTY,
    DEFAULT_UPDATE_TARGET_THRESHOLD,
    DEFAULT_UPDATE_TOLERANCE_TICKS,
    DEFAULT_UPDATE_PRECISION_BETA,
    DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD_START,
    DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD,
    DEFAULT_UPDATE_TRUE_POSITIVE_REWARD,
    DEFAULT_UPDATE_WAIT_OVER_TARGET_PENALTY,
    DEFAULT_UPDATE_WAIT_TARGET_START,
    DEFAULT_UPDATE_WAIT_TOLERANCE_END,
    DEFAULT_UPDATE_WAIT_TOLERANCE_START,
    DEFAULT_UPDATE_WAIT_UNDER_TARGET_PENALTY,
    compute_update_timing_detail,
)
from prompts.reward_prompts import CHAIN_CONSISTENCY_JUDGE_PROMPT, THOUGHT_QUALITY_JUDGE_PROMPT

if TYPE_CHECKING:
    from .judge import LLMJudge


_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_FINAL_PLACEHOLDER_PATTERNS = (
    "final reasoning state",
    "collecting evidence",
    "audio continues",
    "still listening",
    "audio playing",
)
_FINAL_META_PATTERNS = (
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
    "the phrasing",
    "phrasing suggests",
    "specific fact",
    "single answer",
    "answer type",
    "options are",
    "need answer",
    "task asks",
    "audio asks",
)

_YES_NO_WORDS = frozenset(("yes", "no", "true", "false"))
_QUESTION_FORM_STARTERS = frozenset(
    (
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "should",
        "would",
        "could",
        "can",
        "will",
        "is",
        "are",
        "was",
        "were",
    )
)


@dataclass
class RewardConfig:
    format_scale: float = 0.5
    think_scale: float = 1.0
    consistency_bonus: float = 0.25

    sync_scale: float = 0.3
    prediction_scale: float = 0.2

    use_format: bool = True
    use_accuracy: bool = True
    use_reference_answer_fallback: bool = False
    use_think: bool = False
    use_consistency: bool = False
    use_sync: bool = False
    use_prediction: bool = False
    use_update: bool = False

    think_threshold: float = 0.3
    min_effective_tokens: int = DEFAULT_MIN_EFFECTIVE_TOKENS
    accuracy_mode: str = "difficulty_aware_v1"
    state_floor_tokens: int = DEFAULT_STATE_FLOOR_TOKENS
    depth_normalizer_tokens: int = DEFAULT_EXTRA_DEPTH_NORMALIZER
    difficulty_default: float = DEFAULT_DIFFICULTY_SCORE
    difficulty_margin: float = DEFAULT_DIFFICULTY_MARGIN
    lambda_easy: float = 0.5
    lambda_hard: float = 1.0
    correct_reward: float = 2.0
    wrong_penalty: float = -2.0
    rc_prompt_version: str = CHAIN_CONSISTENCY_JUDGE_PROMPT
    rt_prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT

    gpu_speed_tps: float = 50.0
    sync_alpha: float = 0.10
    sync_free_memory_tokens: int = DEFAULT_FREE_STATE_TOKENS
    sync_eof_wait_penalty: float = 0.5
    sync_answer_alpha: float = DEFAULT_ANSWER_ALPHA
    sync_free_answer_tokens: int = DEFAULT_FREE_ANSWER_TOKENS
    sync_final_think_token_alpha: float = DEFAULT_FINAL_THINK_TOKEN_ALPHA
    sync_free_final_think_tokens: int = DEFAULT_FREE_FINAL_THINK_TOKENS
    sync_final_think_token_penalty_cap: float = DEFAULT_FINAL_THINK_TOKEN_PENALTY_CAP
    sync_latency_token_alpha: float = DEFAULT_LATENCY_TOKEN_ALPHA
    sync_free_latency_tokens: int = DEFAULT_FREE_LATENCY_TOKENS
    sync_post_eof_wall_clock_alpha: float = DEFAULT_POST_EOF_WALL_CLOCK_ALPHA
    sync_free_post_eof_wall_clock_seconds: float = DEFAULT_FREE_POST_EOF_WALL_CLOCK_SECONDS
    sync_text_first_token_alpha: float = DEFAULT_TEXT_FIRST_TOKEN_ALPHA
    sync_free_text_first_token_seconds: float = DEFAULT_FREE_TEXT_FIRST_TOKEN_SECONDS
    sync_effective_text_first_token_alpha: float = DEFAULT_EFFECTIVE_TEXT_FIRST_TOKEN_ALPHA
    sync_free_effective_text_first_token_seconds: float = DEFAULT_FREE_EFFECTIVE_TEXT_FIRST_TOKEN_SECONDS
    sync_effective_response_onset_alpha: float = DEFAULT_EFFECTIVE_RESPONSE_ONSET_ALPHA
    sync_free_effective_response_onset_seconds: float = DEFAULT_FREE_EFFECTIVE_RESPONSE_ONSET_SECONDS

    think_fallback: float = 0.0
    update_fallback: float = DEFAULT_UPDATE_FALLBACK
    update_scale: float = 1.0
    update_true_positive_reward: float = DEFAULT_UPDATE_TRUE_POSITIVE_REWARD
    update_true_negative_reward: float = DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD
    update_false_positive_penalty: float = DEFAULT_UPDATE_FALSE_POSITIVE_PENALTY
    update_false_negative_penalty: float = DEFAULT_UPDATE_FALSE_NEGATIVE_PENALTY
    update_tolerance_ticks: int = DEFAULT_UPDATE_TOLERANCE_TICKS
    update_target_threshold: float = DEFAULT_UPDATE_TARGET_THRESHOLD
    update_false_negative_rate_penalty: float = DEFAULT_UPDATE_FALSE_NEGATIVE_RATE_PENALTY
    update_false_positive_rate_penalty: float = DEFAULT_UPDATE_FALSE_POSITIVE_RATE_PENALTY
    update_over_prediction_penalty: float = DEFAULT_UPDATE_OVER_PREDICTION_PENALTY
    update_under_prediction_penalty: float = DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY
    update_precision_beta: float = DEFAULT_UPDATE_PRECISION_BETA
    update_progress_power: float = DEFAULT_UPDATE_PROGRESS_POWER
    update_wait_target_start: float = DEFAULT_UPDATE_WAIT_TARGET_START
    update_wait_tolerance_start: float = DEFAULT_UPDATE_WAIT_TOLERANCE_START
    update_wait_tolerance_end: float = DEFAULT_UPDATE_WAIT_TOLERANCE_END
    update_wait_over_target_penalty: float = DEFAULT_UPDATE_WAIT_OVER_TARGET_PENALTY
    update_wait_under_target_penalty: float = DEFAULT_UPDATE_WAIT_UNDER_TARGET_PENALTY
    update_true_negative_reward_start: float = DEFAULT_UPDATE_TRUE_NEGATIVE_REWARD_START
    update_under_prediction_penalty_start: float = DEFAULT_UPDATE_UNDER_PREDICTION_PENALTY_START
    update_precision_beta_start: float = DEFAULT_UPDATE_PRECISION_BETA_START
    update_teacher_anchor_end: float = DEFAULT_UPDATE_TEACHER_ANCHOR_END
    update_policy_correct_reward: float = DEFAULT_UPDATE_POLICY_CORRECT_REWARD
    update_policy_wrong_penalty: float = DEFAULT_UPDATE_POLICY_WRONG_PENALTY
    update_policy_think_density_penalty: float = DEFAULT_UPDATE_POLICY_THINK_DENSITY_PENALTY
    update_policy_lag_penalty: float = DEFAULT_UPDATE_POLICY_LAG_PENALTY
    update_policy_lag_normalizer: float = DEFAULT_UPDATE_POLICY_LAG_NORMALIZER
    update_policy_sparse_target_easy: float = DEFAULT_UPDATE_POLICY_SPARSE_TARGET_EASY
    update_policy_sparse_target_hard: float = DEFAULT_UPDATE_POLICY_SPARSE_TARGET_HARD
    update_policy_sparse_tolerance: float = DEFAULT_UPDATE_POLICY_SPARSE_TOLERANCE
    update_policy_zero_think_wrong_penalty: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_WRONG_PENALTY
    update_policy_zero_think_correct_penalty: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_CORRECT_PENALTY
    update_policy_medium_threshold: float = DEFAULT_UPDATE_POLICY_MEDIUM_THRESHOLD
    update_policy_hard_threshold: float = DEFAULT_UPDATE_POLICY_HARD_THRESHOLD
    update_policy_zero_think_medium_multiplier: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_MEDIUM_MULTIPLIER
    update_policy_zero_think_hard_multiplier: float = DEFAULT_UPDATE_POLICY_ZERO_THINK_HARD_MULTIPLIER
    update_policy_recall_zero_medium_multiplier: float = DEFAULT_UPDATE_POLICY_RECALL_ZERO_MEDIUM_MULTIPLIER
    update_policy_recall_zero_hard_multiplier: float = DEFAULT_UPDATE_POLICY_RECALL_ZERO_HARD_MULTIPLIER
    update_policy_target_hit_bonus_medium: float = DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_MEDIUM
    update_policy_target_hit_bonus_hard: float = DEFAULT_UPDATE_POLICY_TARGET_HIT_BONUS_HARD
    consistency_default: float = 0.0
    think_quality_gate_scale: float = 0.0
    think_quality_gate_target_rate: float = 0.13
    think_quality_gate_rt_good: float = 0.15
    think_quality_gate_rc_good: float = 0.25
    think_quality_gate_quality_floor: float = 0.65
    think_quality_gate_free_final_think_tokens: int = 6
    think_quality_gate_final_rt_good: float = 0.50
    think_quality_gate_overthink_weight: float = 1.0
    think_quality_gate_final_length_weight: float = 1.8
    answer_shape_penalty_scale: float = 0.0
    final_short_correct_bonus_scale: float = 0.0
    final_short_correct_min_tokens: int = 3
    final_short_correct_max_tokens: int = 6
    final_short_pairwise_bonus_scale: float = 0.0
    progress_fraction: float = 0.0
    current_step: int = 0
    max_steps: int = 1000


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _is_yes_no_expected(question: str, reference_answer: str) -> bool:
    ref_tokens = _word_tokens(reference_answer)
    if ref_tokens and ref_tokens[0] in _YES_NO_WORDS:
        return True
    q_tokens = _word_tokens(question)
    yes_no_starters = {
        "do",
        "does",
        "did",
        "should",
        "would",
        "could",
        "can",
        "will",
        "is",
        "are",
        "was",
        "were",
    }
    return bool(q_tokens and q_tokens[0] in yes_no_starters)


def _contains_yes_no_answer(text: str) -> bool:
    return any(token in _YES_NO_WORDS for token in _word_tokens(text))


def _looks_question_form(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized.endswith("?"):
        return True
    tokens = _word_tokens(normalized)
    return bool(len(tokens) >= 3 and tokens[0] in _QUESTION_FORM_STARTERS)


def _looks_option_label_only(text: str) -> bool:
    normalized = _normalize_text(text)
    return bool(
        re.fullmatch(
            r"(?:the\s+)?(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|option\s*[a-e1-5]|choice\s*[a-e1-5]|[a-e])(?:\s+option)?",
            normalized,
        )
    )


def _final_think_text(episode: PCoTEpisode) -> str:
    if episode.rollout_events:
        final_text = ""
        for event in episode.rollout_events:
            if str(event.get("kind", "")).strip() != "assistant_think":
                continue
            timing = dict(event.get("timing") or {})
            is_final = bool(event.get("is_final_think") or timing.get("is_final_think"))
            if not is_final:
                continue
            direct = str(event.get("think") or "").strip()
            if direct:
                final_text = direct
                continue
            raw = str(event.get("normalized_output") or event.get("raw_output") or "").strip()
            match = _THINK_TAG_PATTERN.search(raw)
            final_text = str(match.group(1) or "").strip() if match else raw
        if final_text:
            return final_text
    thinks = [str(item or "").strip() for item in list(getattr(episode, "thinks", []) or []) if str(item or "").strip()]
    return thinks[-1] if thinks else ""


def compute_answer_shape_detail(episode: PCoTEpisode) -> Dict[str, float]:
    """Rule-based guardrail for malformed final answers."""
    question = str(getattr(episode, "question", "") or "")
    reference = str(getattr(episode, "gt_answer", "") or "")
    answer = str(getattr(episode, "answer", "") or "")
    final_think = _final_think_text(episode)

    answer_type_mismatch = 0.0
    final_think_answer_type_mismatch = 0.0
    question_form_answer = 1.0 if _looks_question_form(answer) else 0.0
    option_label_only = 1.0 if _looks_option_label_only(answer) and not _looks_option_label_only(reference) else 0.0

    if _is_yes_no_expected(question, reference):
        if not _contains_yes_no_answer(answer):
            answer_type_mismatch = 1.0
        if final_think and not _contains_yes_no_answer(final_think):
            final_think_answer_type_mismatch = 1.0

    penalty = 0.0
    penalty -= 1.0 * answer_type_mismatch
    penalty -= 0.5 * final_think_answer_type_mismatch
    penalty -= 0.75 * question_form_answer
    penalty -= 0.50 * option_label_only
    return {
        "penalty": float(penalty),
        "answer_type_mismatch": float(answer_type_mismatch),
        "final_think_answer_type_mismatch": float(final_think_answer_type_mismatch),
        "question_form_answer": float(question_form_answer),
        "option_label_only": float(option_label_only),
    }


def _is_final_meta_or_placeholder(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if any(pattern in normalized for pattern in _FINAL_PLACEHOLDER_PATTERNS):
        return True
    return any(pattern in normalized for pattern in _FINAL_META_PATTERNS)


def _compute_think_quality_gate_detail(
    episode: PCoTEpisode,
    config: RewardConfig,
    *,
    rt: float,
    rc: float,
    ru_detail: Dict[str, float],
    sync_detail: Dict[str, float],
) -> Dict[str, float]:
    """Internal R_t gate for "necessary, short, high-quality" thinking.

    This is intentionally one-sided: it only activates when the policy thinks more
    than the target band. Low think-rate trajectories are handled by R_u / R_a and
    are not punished here, which prevents this rule from rewarding all-wait collapse.
    """

    scale = max(0.0, float(config.think_quality_gate_scale))
    if scale <= 0.0:
        return {
            "adjustment": 0.0,
            "quality": 0.0,
            "quality_gap": 0.0,
            "low_quality_think_penalty": 0.0,
            "overthink_excess": 0.0,
            "overthink_penalty": 0.0,
            "final_quality": 0.0,
            "final_quality_gap": 0.0,
            "final_quality_penalty": 0.0,
            "final_meta_penalty": 0.0,
            "final_excess": 0.0,
            "final_penalty": 0.0,
        }

    think_rate = float(ru_detail.get("think_rate", 0.0))
    target_rate = max(0.0, float(config.think_quality_gate_target_rate))
    overthink_excess = max(0.0, think_rate - target_rate)

    rt_good = max(1e-6, float(config.think_quality_gate_rt_good))
    rc_good = max(1e-6, float(config.think_quality_gate_rc_good))
    quality = _clip01(0.70 * (float(rt) / rt_good) + 0.30 * (float(rc) / rc_good))
    quality_floor = _clip01(config.think_quality_gate_quality_floor)
    quality_gap = max(0.0, quality_floor - quality) / max(quality_floor, 1e-6)
    active_think_pressure = min(1.0, think_rate / max(target_rate, 1e-6)) if think_rate > 0.0 else 0.0
    low_quality_think_penalty = -scale * 0.35 * quality_gap * active_think_pressure

    # Normalize a +0.10 overshoot as "full" overthink pressure. This keeps the
    # term readable and prevents tiny deviations from dominating R_a / R_u.
    overthink_penalty = (
        -scale
        * max(0.0, float(config.think_quality_gate_overthink_weight))
        * quality_gap
        * min(1.0, overthink_excess / 0.10)
    )

    final_tokens = float(
        sync_detail.get("final_think_token_count", infer_final_think_token_count(episode))
    )
    final_free = max(0.0, float(config.think_quality_gate_free_final_think_tokens))
    final_excess = max(0.0, final_tokens - final_free) / max(final_free, 1.0)
    rt_final = float(sync_detail.get("rt_final_score", 0.0))
    final_judged = bool(sync_detail.get("rt_final_judged", 0.0))
    if final_judged:
        final_quality_raw = rt_final / max(1e-6, float(config.think_quality_gate_final_rt_good))
    else:
        # If the final think was not judged, fall back to the episode-level think
        # score but keep it conservative. Missing final-think should not look good.
        final_quality_raw = 0.5 * float(rt) / rt_good
    if episode.is_correct():
        final_quality_raw += 0.25 * (float(rc) / rc_good)
    final_quality = _clip01(final_quality_raw)
    final_text = _final_think_text(episode)
    final_meta_or_placeholder = _is_final_meta_or_placeholder(final_text)
    if final_meta_or_placeholder:
        # A short generic marker or task-commentary note is not a successful
        # low-latency answer cue. Force quality to zero even if a noisy judge
        # score is high, so the actor receives a real negative signal.
        final_quality = 0.0
    final_quality_gap = 1.0 - final_quality
    final_quality_penalty = -scale * 0.35 * final_quality_gap if final_judged else 0.0
    final_meta_penalty = -scale * 0.75 if final_meta_or_placeholder and final_judged else 0.0
    # Even a good final-think should stay compact. Keep the quality gate
    # mostly conditional, but preserve a small length pressure so equally good
    # final states prefer fewer tokens instead of drifting back to base-length
    # explanations.
    final_length_pressure = 0.60 + 0.40 * final_quality_gap
    final_penalty = (
        -scale
        * max(0.0, float(config.think_quality_gate_final_length_weight))
        * final_length_pressure
        * min(1.0, final_excess)
    )

    adjustment = (
        low_quality_think_penalty
        + overthink_penalty
        + final_quality_penalty
        + final_meta_penalty
        + final_penalty
    )
    return {
        "adjustment": float(adjustment),
        "quality": float(quality),
        "quality_gap": float(quality_gap),
        "low_quality_think_penalty": float(low_quality_think_penalty),
        "overthink_excess": float(overthink_excess),
        "overthink_penalty": float(overthink_penalty),
        "final_quality": float(final_quality),
        "final_quality_gap": float(final_quality_gap),
        "final_quality_penalty": float(final_quality_penalty),
        "final_meta_or_placeholder": float(bool(final_meta_or_placeholder)),
        "final_meta_penalty": float(final_meta_penalty),
        "final_length_pressure": float(final_length_pressure),
        "final_excess": float(final_excess),
        "final_penalty": float(final_penalty),
    }


def _empty_reward_dict(episode: PCoTEpisode, config: RewardConfig) -> Dict[str, float]:
    return {
        "R_f": 0.0,
        "R_a": 0.0,
        "R_t": 0.0,
        "R_t_raw": 0.0,
        "R_t_effective": 0.0,
        "R_t_quality_gate_adjustment": 0.0,
        "R_u": 0.0,
        "R_t_per_chunk": [0.0 for _ in range(episode.n_chunks)],
        "R_u_per_tick": [0.0 for _ in range(episode.n_chunks)],
        "R_t_judged_mask": [False for _ in range(episode.n_chunks)],
        "R_t_final": 0.0,
        "R_t_final_judged": 0.0,
        "R_c": 0.0,
        "R_s": 0.0,
        "R_p": 0.0,
        "think_quality_gate_quality": 0.0,
        "think_quality_gate_quality_gap": 0.0,
        "think_quality_gate_low_quality_think_penalty": 0.0,
        "think_quality_gate_overthink_excess": 0.0,
        "think_quality_gate_overthink_penalty": 0.0,
        "think_quality_gate_final_quality": 0.0,
        "think_quality_gate_final_quality_gap": 0.0,
        "think_quality_gate_final_quality_penalty": 0.0,
        "think_quality_gate_final_meta_penalty": 0.0,
        "think_quality_gate_final_excess": 0.0,
        "think_quality_gate_final_penalty": 0.0,
        "answer_rule_correct": 0.0,
        "answer_fallback_invoked": 0.0,
        "answer_fallback_rescued": 0.0,
        "answer_fallback_short_circuit_no_final_answer": 0.0,
        "answer_shape_penalty": 0.0,
        "answer_type_mismatch": 0.0,
        "final_think_answer_type_mismatch": 0.0,
        "question_form_answer": 0.0,
        "option_label_only": 0.0,
        "final_short_correct_bonus": 0.0,
        "final_short_pairwise_adjustment": 0.0,
        "final_think_token_count": 0.0,
        "final_think_token_penalty": 0.0,
        "final_think_generation_wall_clock_seconds": 0.0,
        "response_latency_token_proxy": 0.0,
        "post_eof_wall_clock_seconds": 0.0,
        "post_eof_wall_clock_penalty": 0.0,
        "text_first_token_wall_clock_seconds": 0.0,
        "text_first_token_penalty": 0.0,
        "effective_text_first_token_seconds": 0.0,
        "effective_text_first_token_penalty": 0.0,
        "text_streaming_supported": 0.0,
        "effective_response_onset_seconds": 0.0,
        "effective_response_onset_penalty": 0.0,
        "response_onset_seconds": 0.0,
        "answer_generation_wall_clock_seconds": 0.0,
        "controller_total_wall_clock_seconds": 0.0,
        "sync_think_verbosity_penalty": 0.0,
        "sync_answer_length_penalty": 0.0,
        "sync_symbolic_eof_wait_penalty": 0.0,
        "sync_token_latency_penalty": 0.0,
        "wait_rate": 0.0,
        "think_rate": 0.0,
        "updates_per_episode": 0.0,
        "teacher_boundary_recall": 0.0,
        "teacher_boundary_precision": 0.0,
        "teacher_boundary_f1": 0.0,
        "false_negative_update_rate": 0.0,
        "false_positive_update_rate": 0.0,
        "episodes_with_zero_think_before_eof": 0.0,
        "eof_to_answer_lag": 0.0,
        "target_update_count": 0,
        "predicted_update_count": 0,
        "matched_update_count": 0,
        "predicted_to_target_ratio": 0.0,
        "over_prediction_excess": 0.0,
        "under_prediction_gap": 0.0,
        "teacher_wait_rate": 0.0,
        "scheduled_wait_target": 0.0,
        "scheduled_wait_tolerance": 0.0,
        "wait_over_target_gap": 0.0,
        "wait_under_target_gap": 0.0,
        "wait_corridor_penalty": 0.0,
        "scheduled_true_negative_reward": 0.0,
        "scheduled_under_prediction_penalty": 0.0,
        "scheduled_precision_beta": 0.0,
        "teacher_anchor_weight": 0.0,
        "teacher_score": 0.0,
        "policy_score": 0.0,
        "policy_think_density": 0.0,
        "policy_lag_cost": 0.0,
        "policy_sparse_target_density": 0.0,
        "policy_sparse_tolerance": 0.0,
        "policy_overthink_gap": 0.0,
        "policy_underthink_gap": 0.0,
        "schedule_progress": 0.0,
        "R_outcome": 0.0,
        "total": 0.0,
    }


def _finalize_reward_dict(
    episode: PCoTEpisode,
    config: RewardConfig,
    *,
    rf: float,
    ra: float,
    rt: float,
    rt_per_chunk: List[float],
    rt_judged_mask: List[bool],
    ru_detail: Dict[str, float],
    rc: float,
    rs: float,
    rp: float,
    sync_detail: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    outcome = 0.0

    if config.use_accuracy:
        outcome += ra
        if config.use_consistency and episode.is_correct():
            outcome += ra * config.consistency_bonus * rc

    if config.use_format:
        outcome += config.format_scale * rf

    if config.use_sync:
        outcome += config.sync_scale * rs

    if config.use_prediction:
        outcome += config.prediction_scale * rp

    sync_detail = dict(sync_detail or {})
    answer_shape_detail = compute_answer_shape_detail(episode)
    answer_shape_penalty = 0.0
    if config.use_accuracy and config.answer_shape_penalty_scale > 0.0:
        answer_shape_penalty = (
            float(config.answer_shape_penalty_scale) * float(answer_shape_detail.get("penalty", 0.0))
        )
        outcome += answer_shape_penalty

    quality_gate_detail = _compute_think_quality_gate_detail(
        episode,
        config,
        rt=rt if config.use_think else 0.0,
        rc=rc if config.use_consistency else config.consistency_default,
        ru_detail=ru_detail,
        sync_detail=sync_detail,
    )
    rt_adjustment = float(quality_gate_detail.get("adjustment", 0.0)) if config.use_think else 0.0
    rt_effective = float(rt) + rt_adjustment
    final_tokens = float(sync_detail.get("final_think_token_count", infer_final_think_token_count(episode)))
    rt_final_judged = float(sync_detail.get("rt_final_judged", 0.0))
    rt_final_score = float(sync_detail.get("rt_final_score", 0.0))
    final_think_supports_answer = rt_final_judged <= 0.0 or rt_final_score >= 0.50
    final_short_correct_bonus = 0.0
    if (
        config.use_accuracy
        and config.final_short_correct_bonus_scale > 0.0
        and episode.is_correct()
        and float(answer_shape_detail.get("penalty", 0.0)) >= 0.0
        and final_think_supports_answer
        and float(config.final_short_correct_min_tokens) <= final_tokens <= float(config.final_short_correct_max_tokens)
        and not _is_final_meta_or_placeholder(_final_think_text(episode))
    ):
        final_short_correct_bonus = float(config.final_short_correct_bonus_scale)
        outcome += final_short_correct_bonus

    total = outcome
    if config.use_think:
        total += config.think_scale * rt_effective
    if config.use_update:
        total += config.update_scale * float(ru_detail.get("score", 0.0))

    return {
        "R_f": rf,
        "R_a": ra,
        "R_t": rt,
        "R_t_raw": rt,
        "R_t_effective": rt_effective,
        "R_t_quality_gate_adjustment": rt_adjustment,
        "R_u": float(ru_detail.get("score", 0.0)),
        "R_t_per_chunk": list(rt_per_chunk),
        "R_u_per_tick": list(ru_detail.get("per_tick_scores", [])),
        "R_t_judged_mask": list(rt_judged_mask),
        "R_t_final": float(sync_detail.get("rt_final_score", 0.0)),
        "R_t_final_judged": float(sync_detail.get("rt_final_judged", 0.0)),
        "R_c": rc,
        "R_s": rs,
        "R_p": rp,
        "think_quality_gate_quality": float(quality_gate_detail.get("quality", 0.0)),
        "think_quality_gate_quality_gap": float(quality_gate_detail.get("quality_gap", 0.0)),
        "think_quality_gate_low_quality_think_penalty": float(
            quality_gate_detail.get("low_quality_think_penalty", 0.0)
        ),
        "think_quality_gate_overthink_excess": float(quality_gate_detail.get("overthink_excess", 0.0)),
        "think_quality_gate_overthink_penalty": float(quality_gate_detail.get("overthink_penalty", 0.0)),
        "think_quality_gate_final_quality": float(quality_gate_detail.get("final_quality", 0.0)),
        "think_quality_gate_final_quality_gap": float(quality_gate_detail.get("final_quality_gap", 0.0)),
        "think_quality_gate_final_quality_penalty": float(
            quality_gate_detail.get("final_quality_penalty", 0.0)
        ),
        "think_quality_gate_final_meta_penalty": float(
            quality_gate_detail.get("final_meta_penalty", 0.0)
        ),
        "think_quality_gate_final_excess": float(quality_gate_detail.get("final_excess", 0.0)),
        "think_quality_gate_final_penalty": float(quality_gate_detail.get("final_penalty", 0.0)),
        "answer_rule_correct": float(bool(getattr(episode, "answer_rule_correct", False))),
        "answer_fallback_invoked": float(bool(getattr(episode, "answer_fallback_invoked", False))),
        "answer_fallback_rescued": float(bool(getattr(episode, "answer_fallback_rescued", False))),
        "answer_fallback_short_circuit_no_final_answer": float(
            bool(getattr(episode, "answer_fallback_short_circuit_no_final_answer", False))
        ),
        "answer_shape_penalty": float(answer_shape_penalty),
        "answer_type_mismatch": float(answer_shape_detail.get("answer_type_mismatch", 0.0)),
        "final_think_answer_type_mismatch": float(
            answer_shape_detail.get("final_think_answer_type_mismatch", 0.0)
        ),
        "question_form_answer": float(answer_shape_detail.get("question_form_answer", 0.0)),
        "option_label_only": float(answer_shape_detail.get("option_label_only", 0.0)),
        "final_short_correct_bonus": float(final_short_correct_bonus),
        "final_short_pairwise_adjustment": 0.0,
        "final_think_token_count": float(sync_detail.get("final_think_token_count", 0.0)),
        "final_think_token_penalty": float(sync_detail.get("final_think_token_penalty", 0.0)),
        "final_think_generation_wall_clock_seconds": float(
            sync_detail.get("final_think_generation_wall_clock_seconds", 0.0)
        ),
        "response_latency_token_proxy": float(
            sync_detail.get("response_latency_token_proxy", infer_response_latency_token_proxy(episode))
        ),
        "text_first_token_wall_clock_seconds": float(
            sync_detail.get("text_first_token_wall_clock_seconds", 0.0)
        ),
        "text_first_token_penalty": float(sync_detail.get("text_first_token_penalty", 0.0)),
        "effective_text_first_token_seconds": float(
            sync_detail.get("effective_text_first_token_seconds", 0.0)
        ),
        "effective_text_first_token_penalty": float(
            sync_detail.get("effective_text_first_token_penalty", 0.0)
        ),
        "text_streaming_supported": float(sync_detail.get("text_streaming_supported", 0.0)),
        "response_onset_seconds": float(sync_detail.get("response_onset_seconds", 0.0)),
        "effective_response_onset_seconds": float(
            sync_detail.get("effective_response_onset_seconds", 0.0)
        ),
        "post_eof_wall_clock_seconds": float(sync_detail.get("post_eof_wall_clock_seconds", 0.0)),
        "post_eof_wall_clock_penalty": float(sync_detail.get("post_eof_wall_clock_penalty", 0.0)),
        "effective_response_onset_penalty": float(
            sync_detail.get("effective_response_onset_penalty", 0.0)
        ),
        "answer_generation_wall_clock_seconds": float(
            sync_detail.get("answer_generation_wall_clock_seconds", 0.0)
        ),
        "controller_total_wall_clock_seconds": float(
            sync_detail.get("controller_total_wall_clock_seconds", 0.0)
        ),
        "sync_think_verbosity_penalty": float(sync_detail.get("think_verbosity_penalty", 0.0)),
        "sync_answer_length_penalty": float(sync_detail.get("answer_length_penalty", 0.0)),
        "sync_symbolic_eof_wait_penalty": float(
            sync_detail.get("symbolic_eof_wait_penalty", 0.0)
        ),
        "sync_token_latency_penalty": float(sync_detail.get("token_latency_penalty", 0.0)),
        "wait_rate": float(ru_detail.get("wait_rate", 0.0)),
        "think_rate": float(ru_detail.get("think_rate", 0.0)),
        "updates_per_episode": float(ru_detail.get("updates_per_episode", 0.0)),
        "teacher_boundary_recall": float(ru_detail.get("teacher_boundary_recall", 0.0)),
        "teacher_boundary_precision": float(ru_detail.get("teacher_boundary_precision", 0.0)),
        "teacher_boundary_f1": float(ru_detail.get("teacher_boundary_f1", 0.0)),
        "false_negative_update_rate": float(ru_detail.get("false_negative_update_rate", 0.0)),
        "false_positive_update_rate": float(ru_detail.get("false_positive_update_rate", 0.0)),
        "episodes_with_zero_think_before_eof": float(ru_detail.get("episodes_with_zero_think_before_eof", 0.0)),
        "eof_to_answer_lag": float(ru_detail.get("eof_to_answer_lag", 0.0)),
        "target_update_count": int(ru_detail.get("target_update_count", 0)),
        "predicted_update_count": int(ru_detail.get("predicted_update_count", 0)),
        "matched_update_count": int(ru_detail.get("matched_update_count", 0)),
        "predicted_to_target_ratio": float(ru_detail.get("predicted_to_target_ratio", 0.0)),
        "over_prediction_excess": float(ru_detail.get("over_prediction_excess", 0.0)),
        "under_prediction_gap": float(ru_detail.get("under_prediction_gap", 0.0)),
        "teacher_wait_rate": float(ru_detail.get("teacher_wait_rate", 0.0)),
        "scheduled_wait_target": float(ru_detail.get("scheduled_wait_target", 0.0)),
        "scheduled_wait_tolerance": float(ru_detail.get("scheduled_wait_tolerance", 0.0)),
        "wait_over_target_gap": float(ru_detail.get("wait_over_target_gap", 0.0)),
        "wait_under_target_gap": float(ru_detail.get("wait_under_target_gap", 0.0)),
        "wait_corridor_penalty": float(ru_detail.get("wait_corridor_penalty", 0.0)),
        "scheduled_true_negative_reward": float(ru_detail.get("scheduled_true_negative_reward", 0.0)),
        "scheduled_under_prediction_penalty": float(ru_detail.get("scheduled_under_prediction_penalty", 0.0)),
        "scheduled_precision_beta": float(ru_detail.get("scheduled_precision_beta", 0.0)),
        "teacher_anchor_weight": float(ru_detail.get("teacher_anchor_weight", 0.0)),
        "teacher_score": float(ru_detail.get("teacher_score", 0.0)),
        "policy_score": float(ru_detail.get("policy_score", 0.0)),
        "policy_think_density": float(ru_detail.get("policy_think_density", 0.0)),
        "policy_lag_cost": float(ru_detail.get("policy_lag_cost", 0.0)),
        "policy_sparse_target_density": float(ru_detail.get("policy_sparse_target_density", 0.0)),
        "policy_sparse_tolerance": float(ru_detail.get("policy_sparse_tolerance", 0.0)),
        "policy_overthink_gap": float(ru_detail.get("policy_overthink_gap", 0.0)),
        "policy_underthink_gap": float(ru_detail.get("policy_underthink_gap", 0.0)),
        "schedule_progress": float(ru_detail.get("schedule_progress", 0.0)),
        "R_outcome": outcome,
        "total": total,
    }


def _inject_update_metrics(results: Dict[str, float], ru_detail: Dict[str, float]) -> None:
    results["R_u"] = float(ru_detail.get("score", 0.0))
    results["R_u_per_tick"] = list(ru_detail.get("per_tick_scores", []))
    results["wait_rate"] = float(ru_detail.get("wait_rate", 0.0))
    results["think_rate"] = float(ru_detail.get("think_rate", 0.0))
    results["updates_per_episode"] = float(ru_detail.get("updates_per_episode", 0.0))
    results["teacher_boundary_recall"] = float(ru_detail.get("teacher_boundary_recall", 0.0))
    results["teacher_boundary_precision"] = float(ru_detail.get("teacher_boundary_precision", 0.0))
    results["teacher_boundary_f1"] = float(ru_detail.get("teacher_boundary_f1", 0.0))
    results["false_negative_update_rate"] = float(ru_detail.get("false_negative_update_rate", 0.0))
    results["false_positive_update_rate"] = float(ru_detail.get("false_positive_update_rate", 0.0))
    results["episodes_with_zero_think_before_eof"] = float(ru_detail.get("episodes_with_zero_think_before_eof", 0.0))
    results["eof_to_answer_lag"] = float(ru_detail.get("eof_to_answer_lag", 0.0))
    results["target_update_count"] = int(ru_detail.get("target_update_count", 0))
    results["predicted_update_count"] = int(ru_detail.get("predicted_update_count", 0))
    results["matched_update_count"] = int(ru_detail.get("matched_update_count", 0))
    results["predicted_to_target_ratio"] = float(ru_detail.get("predicted_to_target_ratio", 0.0))
    results["over_prediction_excess"] = float(ru_detail.get("over_prediction_excess", 0.0))
    results["under_prediction_gap"] = float(ru_detail.get("under_prediction_gap", 0.0))
    results["teacher_wait_rate"] = float(ru_detail.get("teacher_wait_rate", 0.0))
    results["scheduled_wait_target"] = float(ru_detail.get("scheduled_wait_target", 0.0))
    results["scheduled_wait_tolerance"] = float(ru_detail.get("scheduled_wait_tolerance", 0.0))
    results["wait_over_target_gap"] = float(ru_detail.get("wait_over_target_gap", 0.0))
    results["wait_under_target_gap"] = float(ru_detail.get("wait_under_target_gap", 0.0))
    results["wait_corridor_penalty"] = float(ru_detail.get("wait_corridor_penalty", 0.0))
    results["scheduled_true_negative_reward"] = float(ru_detail.get("scheduled_true_negative_reward", 0.0))
    results["scheduled_under_prediction_penalty"] = float(ru_detail.get("scheduled_under_prediction_penalty", 0.0))
    results["scheduled_precision_beta"] = float(ru_detail.get("scheduled_precision_beta", 0.0))
    results["teacher_anchor_weight"] = float(ru_detail.get("teacher_anchor_weight", 0.0))
    results["teacher_score"] = float(ru_detail.get("teacher_score", 0.0))
    results["policy_score"] = float(ru_detail.get("policy_score", 0.0))
    results["policy_think_density"] = float(ru_detail.get("policy_think_density", 0.0))
    results["policy_lag_cost"] = float(ru_detail.get("policy_lag_cost", 0.0))
    results["policy_sparse_target_density"] = float(ru_detail.get("policy_sparse_target_density", 0.0))
    results["policy_sparse_tolerance"] = float(ru_detail.get("policy_sparse_tolerance", 0.0))
    results["policy_overthink_gap"] = float(ru_detail.get("policy_overthink_gap", 0.0))
    results["policy_underthink_gap"] = float(ru_detail.get("policy_underthink_gap", 0.0))
    results["schedule_progress"] = float(ru_detail.get("schedule_progress", 0.0))


def _inject_answer_fallback_metrics(results: Dict[str, float], episode: PCoTEpisode) -> None:
    results["answer_rule_correct"] = float(bool(getattr(episode, "answer_rule_correct", False)))
    results["answer_fallback_invoked"] = float(bool(getattr(episode, "answer_fallback_invoked", False)))
    results["answer_fallback_rescued"] = float(bool(getattr(episode, "answer_fallback_rescued", False)))
    results["answer_fallback_short_circuit_no_final_answer"] = float(
        bool(getattr(episode, "answer_fallback_short_circuit_no_final_answer", False))
    )


def compute_rewards(
    episode: PCoTEpisode,
    config: RewardConfig,
    judge: Optional["LLMJudge"] = None,
    answer_fallback_judge: Optional[Any] = None,
    balance_context: Optional[BatchBalanceContext] = None,
) -> Dict[str, float]:
    if config.use_reference_answer_fallback:
        prepare_reference_answer_fallback_batch(
            [episode],
            judge=answer_fallback_judge,
        )
    results = _empty_reward_dict(episode, config)

    ru_detail = compute_update_timing_detail(
        episode,
        fallback=config.update_fallback,
        true_positive_reward=config.update_true_positive_reward,
        true_negative_reward=config.update_true_negative_reward,
        false_positive_penalty=config.update_false_positive_penalty,
        false_negative_penalty=config.update_false_negative_penalty,
        tolerance_ticks=config.update_tolerance_ticks,
        target_threshold=config.update_target_threshold,
        false_negative_rate_penalty=config.update_false_negative_rate_penalty,
        false_positive_rate_penalty=config.update_false_positive_rate_penalty,
        over_prediction_penalty=config.update_over_prediction_penalty,
        under_prediction_penalty=config.update_under_prediction_penalty,
        precision_beta=config.update_precision_beta,
        progress_fraction=config.progress_fraction,
        progress_power=config.update_progress_power,
        current_step=config.current_step,
        max_steps=config.max_steps,
        wait_target_start=config.update_wait_target_start,
        wait_tolerance_start=config.update_wait_tolerance_start,
        wait_tolerance_end=config.update_wait_tolerance_end,
        wait_over_target_penalty=config.update_wait_over_target_penalty,
        wait_under_target_penalty=config.update_wait_under_target_penalty,
        true_negative_reward_start=config.update_true_negative_reward_start,
        under_prediction_penalty_start=config.update_under_prediction_penalty_start,
        precision_beta_start=config.update_precision_beta_start,
        teacher_anchor_end=config.update_teacher_anchor_end,
        policy_correct_reward=config.update_policy_correct_reward,
        policy_wrong_penalty=config.update_policy_wrong_penalty,
        policy_think_density_penalty=config.update_policy_think_density_penalty,
        policy_lag_penalty=config.update_policy_lag_penalty,
        policy_lag_normalizer=config.update_policy_lag_normalizer,
        policy_sparse_target_easy=config.update_policy_sparse_target_easy,
        policy_sparse_target_hard=config.update_policy_sparse_target_hard,
        policy_sparse_tolerance=config.update_policy_sparse_tolerance,
        policy_zero_think_wrong_penalty=config.update_policy_zero_think_wrong_penalty,
        policy_zero_think_correct_penalty=config.update_policy_zero_think_correct_penalty,
        policy_medium_threshold=config.update_policy_medium_threshold,
        policy_hard_threshold=config.update_policy_hard_threshold,
        policy_zero_think_medium_multiplier=config.update_policy_zero_think_medium_multiplier,
        policy_zero_think_hard_multiplier=config.update_policy_zero_think_hard_multiplier,
        policy_recall_zero_medium_multiplier=config.update_policy_recall_zero_medium_multiplier,
        policy_recall_zero_hard_multiplier=config.update_policy_recall_zero_hard_multiplier,
        policy_target_hit_bonus_medium=config.update_policy_target_hit_bonus_medium,
        policy_target_hit_bonus_hard=config.update_policy_target_hit_bonus_hard,
    ).to_dict()
    _inject_update_metrics(results, ru_detail)

    rf = 1.0
    if config.use_format:
        rf = reward_format(episode)
        results["R_f"] = rf
        if rf <= 0:
            penalty = config.format_scale * rf
            _inject_answer_fallback_metrics(results, episode)
            results["R_outcome"] = penalty
            results["total"] = penalty
            return results

    ra = 0.0
    if config.use_accuracy:
        ra = reward_accuracy(
            episode,
            think_threshold=config.think_threshold,
            min_effective_tokens=config.min_effective_tokens,
            balance_context=balance_context,
            mode=config.accuracy_mode,
            state_floor_tokens=config.state_floor_tokens,
            depth_normalizer_tokens=config.depth_normalizer_tokens,
            difficulty_default=config.difficulty_default,
            difficulty_margin=config.difficulty_margin,
            lambda_easy=config.lambda_easy,
            lambda_hard=config.lambda_hard,
            correct_reward=config.correct_reward,
            wrong_penalty=config.wrong_penalty,
        )

    think_detail = {
        "mean": 0.0,
        "per_chunk": [0.0 for _ in range(episode.n_chunks)],
        "judged_mask": [False for _ in range(episode.n_chunks)],
        "pre_chunk_scores": [0.0 for _ in range(episode.n_chunks)],
        "pre_judged_mask": [False for _ in range(episode.n_chunks)],
        "final_score": 0.0,
        "final_judged": False,
    }
    if config.use_think:
        if judge is not None:
            think_detail = reward_think_sync_details(
                episode,
                judge,
                batch_mean_fallback=config.think_fallback,
                prompt_version=config.rt_prompt_version,
            )

    rc = config.consistency_default
    if config.use_consistency:
        if judge is not None:
            rc = reward_consistency_sync(
                episode,
                judge,
                min_tokens=config.min_effective_tokens,
                prompt_version=config.rc_prompt_version,
            )

    rs = 0.0
    sync_detail: Dict[str, float] = {}
    if config.use_sync:
        sync_detail = compute_sync_detail(
            episode,
            gpu_speed=config.gpu_speed_tps,
            alpha=config.sync_alpha,
            free_memory_tokens=config.sync_free_memory_tokens,
            eof_wait_penalty=config.sync_eof_wait_penalty,
            answer_alpha=config.sync_answer_alpha,
            free_answer_tokens=config.sync_free_answer_tokens,
            final_think_token_alpha=config.sync_final_think_token_alpha,
            free_final_think_tokens=config.sync_free_final_think_tokens,
            final_think_token_penalty_cap=config.sync_final_think_token_penalty_cap,
            latency_token_alpha=config.sync_latency_token_alpha,
            free_latency_tokens=config.sync_free_latency_tokens,
            post_eof_wall_clock_alpha=config.sync_post_eof_wall_clock_alpha,
            free_post_eof_wall_clock_seconds=config.sync_free_post_eof_wall_clock_seconds,
            text_first_token_alpha=config.sync_text_first_token_alpha,
            free_text_first_token_seconds=config.sync_free_text_first_token_seconds,
            effective_text_first_token_alpha=config.sync_effective_text_first_token_alpha,
            free_effective_text_first_token_seconds=config.sync_free_effective_text_first_token_seconds,
            effective_response_onset_alpha=config.sync_effective_response_onset_alpha,
            free_effective_response_onset_seconds=config.sync_free_effective_response_onset_seconds,
        )
        rs = float(sync_detail.get("score", 0.0))

    return _finalize_reward_dict(
        episode,
        config,
        rf=rf,
        ra=ra,
        rt=float(think_detail["mean"]) if config.use_think else 0.0,
        rt_per_chunk=list(think_detail["per_chunk"]),
        rt_judged_mask=list(think_detail["judged_mask"]),
        ru_detail=ru_detail,
        rc=rc if config.use_consistency else config.consistency_default,
        rs=rs,
        rp=0.0,
        sync_detail=dict(
            sync_detail,
            rt_final_score=float(think_detail.get("final_score", 0.0)),
            rt_final_judged=float(bool(think_detail.get("final_judged", False))),
        ),
    )


def compute_rewards_batch(
    episodes: List[PCoTEpisode],
    config: RewardConfig,
    judge: Optional["LLMJudge"] = None,
    answer_fallback_judge: Optional[Any] = None,
) -> List[Dict[str, float]]:
    if config.use_reference_answer_fallback:
        prepare_reference_answer_fallback_batch(
            episodes,
            judge=answer_fallback_judge,
        )
    high_think_flags = [
        compute_effort(ep, config.min_effective_tokens) > config.think_threshold
        for ep in episodes
    ]
    think_ratio = (
        sum(1 for flag in high_think_flags if flag) / float(len(high_think_flags))
        if high_think_flags
        else 0.5
    )
    balance_context = BatchBalanceContext(
        think_ratio=think_ratio,
        progress=config.progress_fraction,
    )

    think_details = None
    if config.use_think:
        if judge is not None:
            think_details = reward_think_sync_batch_details(
                episodes,
                judge,
                default_fallback=config.think_fallback,
                prompt_version=config.rt_prompt_version,
            )
        else:
            think_details = [
                {
                    "mean": 0.0,
                    "per_chunk": [0.0 for _ in range(episode.n_chunks)],
                    "judged_mask": [False for _ in range(episode.n_chunks)],
                    "pre_chunk_scores": [0.0 for _ in range(episode.n_chunks)],
                    "pre_judged_mask": [False for _ in range(episode.n_chunks)],
                    "final_score": 0.0,
                    "final_judged": False,
                }
                for episode in episodes
            ]

    results: List[Dict[str, float]] = []
    for idx, episode in enumerate(episodes):
        rf = reward_format(episode) if config.use_format else 1.0
        ru_detail = compute_update_timing_detail(
            episode,
            fallback=config.update_fallback,
            true_positive_reward=config.update_true_positive_reward,
            true_negative_reward=config.update_true_negative_reward,
            false_positive_penalty=config.update_false_positive_penalty,
            false_negative_penalty=config.update_false_negative_penalty,
            tolerance_ticks=config.update_tolerance_ticks,
            target_threshold=config.update_target_threshold,
            false_negative_rate_penalty=config.update_false_negative_rate_penalty,
            false_positive_rate_penalty=config.update_false_positive_rate_penalty,
            over_prediction_penalty=config.update_over_prediction_penalty,
            under_prediction_penalty=config.update_under_prediction_penalty,
            precision_beta=config.update_precision_beta,
            progress_fraction=config.progress_fraction,
            progress_power=config.update_progress_power,
            current_step=config.current_step,
            max_steps=config.max_steps,
            wait_target_start=config.update_wait_target_start,
            wait_tolerance_start=config.update_wait_tolerance_start,
            wait_tolerance_end=config.update_wait_tolerance_end,
            wait_over_target_penalty=config.update_wait_over_target_penalty,
            wait_under_target_penalty=config.update_wait_under_target_penalty,
            true_negative_reward_start=config.update_true_negative_reward_start,
            under_prediction_penalty_start=config.update_under_prediction_penalty_start,
            precision_beta_start=config.update_precision_beta_start,
            teacher_anchor_end=config.update_teacher_anchor_end,
            policy_correct_reward=config.update_policy_correct_reward,
            policy_wrong_penalty=config.update_policy_wrong_penalty,
            policy_think_density_penalty=config.update_policy_think_density_penalty,
            policy_lag_penalty=config.update_policy_lag_penalty,
            policy_lag_normalizer=config.update_policy_lag_normalizer,
            policy_sparse_target_easy=config.update_policy_sparse_target_easy,
            policy_sparse_target_hard=config.update_policy_sparse_target_hard,
            policy_sparse_tolerance=config.update_policy_sparse_tolerance,
            policy_zero_think_wrong_penalty=config.update_policy_zero_think_wrong_penalty,
            policy_zero_think_correct_penalty=config.update_policy_zero_think_correct_penalty,
            policy_medium_threshold=config.update_policy_medium_threshold,
            policy_hard_threshold=config.update_policy_hard_threshold,
            policy_zero_think_medium_multiplier=config.update_policy_zero_think_medium_multiplier,
            policy_zero_think_hard_multiplier=config.update_policy_zero_think_hard_multiplier,
            policy_recall_zero_medium_multiplier=config.update_policy_recall_zero_medium_multiplier,
            policy_recall_zero_hard_multiplier=config.update_policy_recall_zero_hard_multiplier,
            policy_target_hit_bonus_medium=config.update_policy_target_hit_bonus_medium,
            policy_target_hit_bonus_hard=config.update_policy_target_hit_bonus_hard,
        ).to_dict()
        if config.use_format and rf <= 0:
            reward_dict = _empty_reward_dict(episode, config)
            penalty = config.format_scale * rf
            reward_dict["R_f"] = rf
            _inject_answer_fallback_metrics(reward_dict, episode)
            _inject_update_metrics(reward_dict, ru_detail)
            reward_dict["R_outcome"] = penalty
            reward_dict["total"] = penalty
            results.append(reward_dict)
            continue

        ra = reward_accuracy(
            episode,
            think_threshold=config.think_threshold,
            min_effective_tokens=config.min_effective_tokens,
            balance_context=balance_context,
            mode=config.accuracy_mode,
            state_floor_tokens=config.state_floor_tokens,
            depth_normalizer_tokens=config.depth_normalizer_tokens,
            difficulty_default=config.difficulty_default,
            difficulty_margin=config.difficulty_margin,
            lambda_easy=config.lambda_easy,
            lambda_hard=config.lambda_hard,
            correct_reward=config.correct_reward,
            wrong_penalty=config.wrong_penalty,
        ) if config.use_accuracy else 0.0

        think_detail = think_details[idx] if think_details is not None else {
            "mean": 0.0,
            "per_chunk": [0.0 for _ in range(episode.n_chunks)],
            "judged_mask": [False for _ in range(episode.n_chunks)],
            "pre_chunk_scores": [0.0 for _ in range(episode.n_chunks)],
            "pre_judged_mask": [False for _ in range(episode.n_chunks)],
            "final_score": 0.0,
            "final_judged": False,
        }

        rc = config.consistency_default
        if config.use_consistency:
            if judge is not None:
                rc = reward_consistency_sync(
                    episode,
                    judge,
                    min_tokens=config.min_effective_tokens,
                    prompt_version=config.rc_prompt_version,
                )

        rs = 0.0
        sync_detail: Dict[str, float] = {}
        if config.use_sync:
            sync_detail = compute_sync_detail(
                episode,
                gpu_speed=config.gpu_speed_tps,
                alpha=config.sync_alpha,
                free_memory_tokens=config.sync_free_memory_tokens,
                eof_wait_penalty=config.sync_eof_wait_penalty,
                answer_alpha=config.sync_answer_alpha,
                free_answer_tokens=config.sync_free_answer_tokens,
                final_think_token_alpha=config.sync_final_think_token_alpha,
                free_final_think_tokens=config.sync_free_final_think_tokens,
                final_think_token_penalty_cap=config.sync_final_think_token_penalty_cap,
                latency_token_alpha=config.sync_latency_token_alpha,
                free_latency_tokens=config.sync_free_latency_tokens,
                post_eof_wall_clock_alpha=config.sync_post_eof_wall_clock_alpha,
                free_post_eof_wall_clock_seconds=config.sync_free_post_eof_wall_clock_seconds,
                text_first_token_alpha=config.sync_text_first_token_alpha,
                free_text_first_token_seconds=config.sync_free_text_first_token_seconds,
                effective_text_first_token_alpha=config.sync_effective_text_first_token_alpha,
                free_effective_text_first_token_seconds=config.sync_free_effective_text_first_token_seconds,
                effective_response_onset_alpha=config.sync_effective_response_onset_alpha,
                free_effective_response_onset_seconds=config.sync_free_effective_response_onset_seconds,
            )
            rs = float(sync_detail.get("score", 0.0))

        results.append(
            _finalize_reward_dict(
                episode,
                config,
                rf=rf,
                ra=ra,
                rt=float(think_detail["mean"]) if config.use_think else 0.0,
                rt_per_chunk=list(think_detail["per_chunk"]),
                rt_judged_mask=list(think_detail["judged_mask"]),
                ru_detail=ru_detail,
                rc=rc if config.use_consistency else config.consistency_default,
                rs=rs,
                rp=0.0,
                sync_detail=dict(
                    sync_detail,
                    rt_final_score=float(think_detail.get("final_score", 0.0)),
                    rt_final_judged=float(bool(think_detail.get("final_judged", False))),
                ),
            )
        )
    return results


def format_reward_summary(reward_dict: dict) -> str:
    parts = []
    for key in ["R_f", "R_a", "R_t", "R_u", "R_c", "R_s", "R_p"]:
        if key in reward_dict:
            parts.append(f"{key}={reward_dict[key]:+.3f}")
    if "R_outcome" in reward_dict:
        parts.append(f"R_outcome={reward_dict['R_outcome']:+.3f}")
    total = reward_dict.get("total", 0.0)
    return "  ".join(parts) + f"  →  total={total:+.3f}"
