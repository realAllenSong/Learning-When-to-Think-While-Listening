"""
R_a: Adaptive Think Accuracy Reward
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .episode import PCoTEpisode


DEFAULT_MIN_EFFECTIVE_TOKENS = 3
DEFAULT_STATE_FLOOR_TOKENS = 3
DEFAULT_EXTRA_DEPTH_NORMALIZER = 6
DEFAULT_DIFFICULTY_SCORE = 0.5
DEFAULT_DIFFICULTY_MARGIN = 0.1
DEFAULT_ACCURACY_MODE = "legacy_quadrant"
DEFAULT_QUADRANT_SCORES: Dict[tuple, float] = {
    (True, True): 1.0,
    (True, False): 0.0,
    (False, True): 2.0,
    (False, False): -1.0,
}

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "and",
        "or",
        "of",
        "for",
        "with",
        "this",
        "that",
        "be",
        "as",
        "by",
        "are",
        "was",
    }
)
_MIN_WORD_LEN = 2


def count_effective_tokens(think: str) -> int:
    if not think or not think.strip():
        return 0

    words = think.lower().split()
    effective = [
        word.strip(".,!?;:-'\"()[]{}")
        for word in words
        if len(word.strip(".,!?;:-'\"()[]{}")) >= _MIN_WORD_LEN
        and word.strip(".,!?;:-'\"()[]{}") not in _STOP_WORDS
    ]
    return len(effective)


def compute_effort(
    episode: PCoTEpisode,
    min_effective_tokens: int = DEFAULT_MIN_EFFECTIVE_TOKENS,
) -> float:
    if episode.n_chunks == 0:
        return 0.0

    thinks_to_count = episode.thinks[: episode.n_chunks]
    triggered = sum(
        1
        for think in thinks_to_count
        if count_effective_tokens(think) >= min_effective_tokens
    )
    return triggered / episode.n_chunks


def compute_extra_depth(
    episode: PCoTEpisode,
    *,
    state_floor_tokens: int = DEFAULT_STATE_FLOOR_TOKENS,
    depth_normalizer_tokens: int = DEFAULT_EXTRA_DEPTH_NORMALIZER,
) -> float:
    if episode.n_chunks == 0:
        return 0.0
    if depth_normalizer_tokens <= 0:
        raise ValueError("depth_normalizer_tokens must be positive")

    thinks_to_count = episode.thinks[: episode.n_chunks]
    total = 0.0
    for think in thinks_to_count:
        extra = max(0, count_effective_tokens(think) - state_floor_tokens)
        total += min(1.0, extra / float(depth_normalizer_tokens))
    return total / float(episode.n_chunks)


def get_required_depth(
    episode: PCoTEpisode,
    *,
    default_score: float = DEFAULT_DIFFICULTY_SCORE,
) -> float:
    metadata = getattr(episode, "difficulty_metadata", {}) or {}
    if "difficulty_score" in metadata:
        try:
            return float(min(1.0, max(0.0, float(metadata["difficulty_score"]))))
        except Exception:
            pass

    bucket = str(metadata.get("difficulty_bucket", "")).strip().lower()
    if bucket == "easy":
        return 0.2
    if bucket == "medium":
        return 0.5
    if bucket == "hard":
        return 0.8
    return float(min(1.0, max(0.0, default_score)))


@dataclass
class BatchBalanceContext:
    think_ratio: float
    progress: float = 0.0


def compute_balance_gammas(context: Optional[BatchBalanceContext]) -> tuple[float, float]:
    if context is None:
        return 1.0, 1.0

    progress = min(1.0, max(0.0, context.progress))
    lambda_think = min(1.0, max(0.0, context.think_ratio))
    anneal = 1.0 - progress
    gamma_think = math.exp(-lambda_think * anneal)
    gamma_nothink = math.exp(-(1.0 - lambda_think) * anneal)
    return gamma_think, gamma_nothink


def reward_accuracy(
    episode: PCoTEpisode,
    think_threshold: float = 0.3,
    min_effective_tokens: int = DEFAULT_MIN_EFFECTIVE_TOKENS,
    quadrant_scores: Optional[Dict] = None,
    balance_context: Optional[BatchBalanceContext] = None,
    mode: str = DEFAULT_ACCURACY_MODE,
    state_floor_tokens: int = DEFAULT_STATE_FLOOR_TOKENS,
    depth_normalizer_tokens: int = DEFAULT_EXTRA_DEPTH_NORMALIZER,
    difficulty_default: float = DEFAULT_DIFFICULTY_SCORE,
    difficulty_margin: float = DEFAULT_DIFFICULTY_MARGIN,
    lambda_easy: float = 0.5,
    lambda_hard: float = 1.0,
    correct_reward: float = 2.0,
    wrong_penalty: float = -2.0,
) -> float:
    if mode == "difficulty_aware_v1":
        r_acc = float(correct_reward if episode.is_correct() else wrong_penalty)
        effort = compute_extra_depth(
            episode,
            state_floor_tokens=state_floor_tokens,
            depth_normalizer_tokens=depth_normalizer_tokens,
        )
        required = get_required_depth(episode, default_score=difficulty_default)
        overthinking = max(0.0, effort - required - difficulty_margin)
        if episode.is_correct():
            return float(r_acc - lambda_easy * overthinking)
        underthinking = max(0.0, required - effort - difficulty_margin)
        return float(r_acc - lambda_easy * overthinking - lambda_hard * underthinking)

    if mode != DEFAULT_ACCURACY_MODE:
        raise ValueError(f"Unsupported reward_accuracy mode: {mode}")

    scores = quadrant_scores or DEFAULT_QUADRANT_SCORES

    effort = compute_effort(episode, min_effective_tokens)
    is_high_think = effort > think_threshold
    is_correct = episode.is_correct()

    if balance_context is None:
        return float(scores[(is_high_think, is_correct)])

    gamma_think, gamma_nothink = compute_balance_gammas(balance_context)

    if is_high_think and is_correct:
        return float(gamma_think * scores[(True, True)])
    if is_high_think and not is_correct:
        return float(gamma_think * scores[(True, False)] + (1.0 - gamma_think) * (-1.0))
    if not is_high_think and is_correct:
        return float(gamma_nothink * scores[(False, True)])
    return float(gamma_nothink * scores[(False, False)] + (1.0 - gamma_nothink) * (-2.0))
