"""
R_s: synchronization and latency penalty.

Training interpretation:

- softly penalize excessive think verbosity beyond a free per-update token budget
- strongly penalize extra wait / lag after AUDIO_END
- optionally penalize excessive controller token-latency before `<answer>`

This keeps latency pressure explicit without requiring LLM-judge signals.
"""

import re
from typing import Any, Dict, List, Optional

from .episode import PCoTEpisode
from .reward_update import infer_eof_to_answer_lag


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_GPU_SPEED_TPS = 50.0   # tokens per second, typical for 7B on A100
DEFAULT_ALPHA = 0.10           # stronger latency pressure for long think traces
DEFAULT_CHUNK_DURATION = 2.0   # seconds
DEFAULT_FREE_STATE_TOKENS = 12
DEFAULT_EOF_WAIT_PENALTY = 0.5
DEFAULT_ANSWER_ALPHA = 0.02
DEFAULT_FREE_ANSWER_TOKENS = 16
DEFAULT_FINAL_THINK_TOKEN_ALPHA = 0.30
DEFAULT_FREE_FINAL_THINK_TOKENS = 6
DEFAULT_FINAL_THINK_TOKEN_PENALTY_CAP = 3.0
DEFAULT_LATENCY_TOKEN_ALPHA = 0.0
DEFAULT_FREE_LATENCY_TOKENS = 0
DEFAULT_POST_EOF_WALL_CLOCK_ALPHA = 0.0
DEFAULT_FREE_POST_EOF_WALL_CLOCK_SECONDS = 0.20
DEFAULT_TEXT_FIRST_TOKEN_ALPHA = 0.0
DEFAULT_FREE_TEXT_FIRST_TOKEN_SECONDS = 0.20
DEFAULT_EFFECTIVE_TEXT_FIRST_TOKEN_ALPHA = 0.0
DEFAULT_FREE_EFFECTIVE_TEXT_FIRST_TOKEN_SECONDS = 0.20
DEFAULT_EFFECTIVE_RESPONSE_ONSET_ALPHA = 0.5
DEFAULT_FREE_EFFECTIVE_RESPONSE_ONSET_SECONDS = 0.20


# ── Token length estimation ────────────────────────────────────────────────────

def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text.

    Uses character count divided by average chars-per-token (more stable than
    word count for telegraphic / multilingual think content).

    chars_per_token ≈ 4.0 is a reasonable approximation for English with BPE.
    """
    if not text or not text.strip():
        return 0
    return max(1, round(len(text.strip()) / chars_per_token))


def infer_response_latency_token_proxy(episode: PCoTEpisode) -> int:
    """
    Approximate controller token-latency before the first `<answer>`.

    This is intentionally a lightweight training-side proxy rather than the
    exact tokenizer-based evaluation metric. The proxy counts:

    - each explicit `<wait/>` as one controller token
    - each `<think>...</think>` body by `estimate_token_count`
    - any residual plain text before `<answer>` by `estimate_token_count`

    The goal is to capture "how much controller content happened before the
    final answer" without pulling a tokenizer into the reward path.
    """
    raw_sequence = str(getattr(episode, "raw_sequence", "") or "").strip()
    if not raw_sequence:
        think_parts = [
            "<think>{}</think>".format(str(text).strip())
            if str(text or "").strip()
            else "<wait/>"
            for text in list(getattr(episode, "thinks", []) or [])
        ]
        raw_sequence = "".join(think_parts)
        answer = str(getattr(episode, "answer", "") or "").strip()
        if answer:
            raw_sequence += "<answer>{}</answer>".format(answer)
    if not raw_sequence:
        return 0

    answer_match = re.search(r"<answer>.*?</answer>", raw_sequence, flags=re.IGNORECASE | re.DOTALL)
    prefix_text = raw_sequence[: answer_match.start()] if answer_match else raw_sequence
    if not prefix_text.strip():
        return 0

    wait_count = len(re.findall(r"<wait\s*/>", prefix_text, flags=re.IGNORECASE))
    think_bodies = re.findall(r"<think>(.*?)</think>", prefix_text, flags=re.IGNORECASE | re.DOTALL)
    predict_bodies = re.findall(r"<predict>(.*?)</predict>", prefix_text, flags=re.IGNORECASE | re.DOTALL)

    content_tokens = sum(estimate_token_count(body) for body in think_bodies)
    content_tokens += sum(estimate_token_count(body) for body in predict_bodies)

    residual = re.sub(r"<wait\s*/>", " ", prefix_text, flags=re.IGNORECASE)
    residual = re.sub(r"<think>.*?</think>", " ", residual, flags=re.IGNORECASE | re.DOTALL)
    residual = re.sub(r"<predict>.*?</predict>", " ", residual, flags=re.IGNORECASE | re.DOTALL)
    residual = re.sub(r"<[^>]+>", " ", residual)
    residual_tokens = estimate_token_count(residual)

    return int(wait_count + content_tokens + residual_tokens)


def _event_is_final_think(event: Dict[str, Any]) -> bool:
    if bool(event.get("is_final_think")):
        return True
    timing = event.get("timing")
    return isinstance(timing, dict) and bool(timing.get("is_final_think"))


def infer_final_think_token_count(episode: PCoTEpisode) -> int:
    runtime_timing = extract_runtime_timing(episode)
    direct_count = _safe_float(runtime_timing.get("final_think_token_count"))
    if direct_count is not None:
        return max(0, int(round(direct_count)))

    if episode.rollout_events:
        final_think_text = ""
        for event in episode.rollout_events:
            kind = str(event.get("kind", "")).strip().lower()
            if kind != "assistant_think" or not _event_is_final_think(event):
                continue
            final_think_text = str(event.get("think") or "").strip()
            if final_think_text:
                continue
            raw_output = str(event.get("normalized_output") or event.get("raw_output") or "").strip()
            match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.IGNORECASE | re.DOTALL)
            if match is not None:
                final_think_text = str(match.group(1) or "").strip()
        if final_think_text:
            return estimate_token_count(final_think_text)

    raw_sequence = str(getattr(episode, "raw_sequence", "") or "").strip()
    if not raw_sequence:
        return 0
    answer_match = re.search(r"<answer>.*?</answer>", raw_sequence, flags=re.IGNORECASE | re.DOTALL)
    prefix_text = raw_sequence[: answer_match.start()] if answer_match else raw_sequence
    think_bodies = re.findall(r"<think>(.*?)</think>", prefix_text, flags=re.IGNORECASE | re.DOTALL)
    if not think_bodies:
        return 0
    return estimate_token_count(str(think_bodies[-1] or "").strip())


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def extract_runtime_timing(episode: PCoTEpisode) -> Dict[str, Optional[float]]:
    metadata = dict(getattr(episode, "controller_metadata", {}) or {})
    candidates: List[Dict[str, Any]] = [metadata]

    timing = metadata.get("timing")
    if isinstance(timing, dict):
        candidates.append(timing)

    policy_rollout_metadata = metadata.get("policy_rollout_metadata")
    if isinstance(policy_rollout_metadata, dict):
        candidates.append(policy_rollout_metadata)
        rollout_timing = policy_rollout_metadata.get("timing")
        if isinstance(rollout_timing, dict):
            candidates.append(rollout_timing)

    resolved: Dict[str, Optional[float]] = {}
    for key in (
        "post_eof_total_wall_clock_sec",
        "answer_generation_wall_clock_sec",
        "controller_total_wall_clock_sec",
        "final_think_generation_wall_clock_sec",
        "final_think_token_count",
        "text_first_token_wall_clock_seconds",
        "effective_text_first_token_seconds",
        "text_streaming_supported",
        "response_onset_seconds",
        "effective_response_onset_seconds",
    ):
        resolved[key] = None
        for candidate in candidates:
            value = (
                bool(candidate.get(key))
                if key == "text_streaming_supported"
                else _safe_float(candidate.get(key))
            )
            if value is not None:
                resolved[key] = value
                break
    return resolved


# ── Per-chunk penalty ──────────────────────────────────────────────────────────

def _trajectory_think_penalty(
    thinks: List[str],
    *,
    gpu_speed: float,
    alpha: float,
    free_state_tokens: int,
) -> float:
    non_empty = [think for think in thinks if estimate_token_count(think) > 0]
    if not non_empty:
        return 0.0

    token_budget = max(0, int(free_state_tokens)) * len(non_empty)
    total_tokens = sum(estimate_token_count(think) for think in non_empty)
    overflow_tokens = max(0, total_tokens - token_budget)
    if overflow_tokens <= 0:
        return 0.0

    if gpu_speed <= 0:
        overflow_seconds = float(overflow_tokens)
    else:
        overflow_seconds = float(overflow_tokens) / float(gpu_speed)
    return -float(alpha) * overflow_seconds


def _answer_length_penalty(
    answer: str,
    *,
    gpu_speed: float,
    alpha: float,
    free_answer_tokens: int,
) -> float:
    answer_tokens = estimate_token_count(answer)
    overflow_tokens = max(0, int(answer_tokens) - int(max(0, free_answer_tokens)))
    if overflow_tokens <= 0:
        return 0.0
    if gpu_speed <= 0:
        overflow_seconds = float(overflow_tokens)
    else:
        overflow_seconds = float(overflow_tokens) / float(gpu_speed)
    return -float(alpha) * overflow_seconds


def _response_latency_token_penalty(
    episode: PCoTEpisode,
    *,
    gpu_speed: float,
    alpha: float,
    free_latency_tokens: int,
) -> float:
    if alpha <= 0:
        return 0.0
    latency_tokens = infer_response_latency_token_proxy(episode)
    overflow_tokens = max(0, int(latency_tokens) - int(max(0, free_latency_tokens)))
    if overflow_tokens <= 0:
        return 0.0
    if gpu_speed <= 0:
        overflow_seconds = float(overflow_tokens)
    else:
        overflow_seconds = float(overflow_tokens) / float(gpu_speed)
    return -float(alpha) * overflow_seconds


def _final_think_token_penalty(
    final_think_token_count: int,
    *,
    gpu_speed: float,
    alpha: float,
    free_final_think_tokens: int,
    penalty_cap: float = DEFAULT_FINAL_THINK_TOKEN_PENALTY_CAP,
) -> float:
    if alpha <= 0:
        return 0.0
    overflow_tokens = max(0, int(final_think_token_count) - int(max(0, free_final_think_tokens)))
    if overflow_tokens <= 0:
        return 0.0
    penalty = -float(alpha) * float(overflow_tokens)
    cap = max(0.0, float(penalty_cap or 0.0))
    if cap > 0.0:
        penalty = max(penalty, -cap)
    return float(penalty)


def _post_eof_wall_clock_penalty(
    post_eof_total_wall_clock_sec: Optional[float],
    *,
    alpha: float,
    free_seconds: float,
) -> float:
    if alpha <= 0.0 or post_eof_total_wall_clock_sec is None:
        return 0.0
    overflow_seconds = max(
        0.0,
        float(post_eof_total_wall_clock_sec) - max(0.0, float(free_seconds)),
    )
    if overflow_seconds <= 0.0:
        return 0.0
    return -float(alpha) * overflow_seconds


def _text_first_token_penalty(
    text_first_token_wall_clock_seconds: Optional[float],
    *,
    alpha: float,
    free_seconds: float,
) -> float:
    if alpha <= 0.0 or text_first_token_wall_clock_seconds is None:
        return 0.0
    overflow_seconds = max(
        0.0,
        float(text_first_token_wall_clock_seconds) - max(0.0, float(free_seconds)),
    )
    if overflow_seconds <= 0.0:
        return 0.0
    return -float(alpha) * overflow_seconds


def _effective_text_first_token_penalty(
    effective_text_first_token_seconds: Optional[float],
    *,
    alpha: float,
    free_seconds: float,
) -> float:
    if alpha <= 0.0 or effective_text_first_token_seconds is None:
        return 0.0
    overflow_seconds = max(
        0.0,
        float(effective_text_first_token_seconds) - max(0.0, float(free_seconds)),
    )
    if overflow_seconds <= 0.0:
        return 0.0
    return -float(alpha) * overflow_seconds


def _effective_response_onset_penalty(
    effective_response_onset_seconds: Optional[float],
    *,
    alpha: float,
    free_seconds: float,
) -> float:
    if alpha <= 0.0 or effective_response_onset_seconds is None:
        return 0.0
    overflow_seconds = max(
        0.0,
        float(effective_response_onset_seconds) - max(0.0, float(free_seconds)),
    )
    if overflow_seconds <= 0.0:
        return 0.0
    return -float(alpha) * overflow_seconds


# ── Episode-level reward ───────────────────────────────────────────────────────

def compute_sync_detail(
    episode: PCoTEpisode,
    gpu_speed: float = DEFAULT_GPU_SPEED_TPS,
    alpha: float = DEFAULT_ALPHA,
    free_memory_tokens: int = DEFAULT_FREE_STATE_TOKENS,
    eof_wait_penalty: float = DEFAULT_EOF_WAIT_PENALTY,
    answer_alpha: float = DEFAULT_ANSWER_ALPHA,
    free_answer_tokens: int = DEFAULT_FREE_ANSWER_TOKENS,
    final_think_token_alpha: float = DEFAULT_FINAL_THINK_TOKEN_ALPHA,
    free_final_think_tokens: int = DEFAULT_FREE_FINAL_THINK_TOKENS,
    final_think_token_penalty_cap: float = DEFAULT_FINAL_THINK_TOKEN_PENALTY_CAP,
    latency_token_alpha: float = DEFAULT_LATENCY_TOKEN_ALPHA,
    free_latency_tokens: int = DEFAULT_FREE_LATENCY_TOKENS,
    post_eof_wall_clock_alpha: float = DEFAULT_POST_EOF_WALL_CLOCK_ALPHA,
    free_post_eof_wall_clock_seconds: float = DEFAULT_FREE_POST_EOF_WALL_CLOCK_SECONDS,
    text_first_token_alpha: float = DEFAULT_TEXT_FIRST_TOKEN_ALPHA,
    free_text_first_token_seconds: float = DEFAULT_FREE_TEXT_FIRST_TOKEN_SECONDS,
    effective_text_first_token_alpha: float = DEFAULT_EFFECTIVE_TEXT_FIRST_TOKEN_ALPHA,
    free_effective_text_first_token_seconds: float = DEFAULT_FREE_EFFECTIVE_TEXT_FIRST_TOKEN_SECONDS,
    effective_response_onset_alpha: float = DEFAULT_EFFECTIVE_RESPONSE_ONSET_ALPHA,
    free_effective_response_onset_seconds: float = DEFAULT_FREE_EFFECTIVE_RESPONSE_ONSET_SECONDS,
) -> Dict[str, float]:
    """
    Compute aggregate sync / latency penalty R_s for a controller episode.

    The reward uses a latency-oriented penalty stack:

    1. think verbosity overflow relative to a free token budget per think update
    2. explicit post-EOF wait / answer lag penalty
    3. optional pre-answer controller token-latency proxy penalty

    Args:
        episode:    The controller episode.
        gpu_speed:  Inference speed in tokens/second (measure this on your cluster!).
        alpha:      Penalty scale factor.

    Returns:
        Float ≤ 0.0.
    """
    think_penalty = _trajectory_think_penalty(
        list(episode.thinks or []),
        gpu_speed=gpu_speed,
        alpha=alpha,
        free_state_tokens=free_memory_tokens,
    )
    answer_penalty = _answer_length_penalty(
        str(getattr(episode, "answer", "") or ""),
        gpu_speed=gpu_speed,
        alpha=answer_alpha,
        free_answer_tokens=free_answer_tokens,
    )
    runtime_timing = extract_runtime_timing(episode)
    final_think_token_count = infer_final_think_token_count(episode)
    final_think_token_penalty = _final_think_token_penalty(
        final_think_token_count,
        gpu_speed=gpu_speed,
        alpha=final_think_token_alpha,
        free_final_think_tokens=free_final_think_tokens,
        penalty_cap=final_think_token_penalty_cap,
    )
    eof_to_answer_lag = float(infer_eof_to_answer_lag(episode, max(0, int(episode.n_chunks))))
    symbolic_eof_wait_penalty = 0.0
    if eof_to_answer_lag > 0.0:
        symbolic_eof_wait_penalty = -float(eof_wait_penalty) * eof_to_answer_lag
    response_latency_token_proxy = infer_response_latency_token_proxy(episode)
    token_latency_penalty = _response_latency_token_penalty(
        episode,
        gpu_speed=gpu_speed,
        alpha=latency_token_alpha,
        free_latency_tokens=free_latency_tokens,
    )
    post_eof_wall_clock_penalty = _post_eof_wall_clock_penalty(
        runtime_timing.get("post_eof_total_wall_clock_sec"),
        alpha=post_eof_wall_clock_alpha,
        free_seconds=free_post_eof_wall_clock_seconds,
    )
    text_first_token_penalty = _text_first_token_penalty(
        runtime_timing.get("text_first_token_wall_clock_seconds"),
        alpha=text_first_token_alpha,
        free_seconds=free_text_first_token_seconds,
    )
    effective_text_first_token_penalty = _effective_text_first_token_penalty(
        runtime_timing.get("effective_text_first_token_seconds"),
        alpha=effective_text_first_token_alpha,
        free_seconds=free_effective_text_first_token_seconds,
    )
    effective_response_onset_penalty = _effective_response_onset_penalty(
        runtime_timing.get("effective_response_onset_seconds"),
        alpha=effective_response_onset_alpha,
        free_seconds=free_effective_response_onset_seconds,
    )
    score = (
        think_penalty
        + answer_penalty
        + symbolic_eof_wait_penalty
        + final_think_token_penalty
        + token_latency_penalty
        + post_eof_wall_clock_penalty
        + text_first_token_penalty
        + effective_text_first_token_penalty
        + effective_response_onset_penalty
    )
    return {
        "score": float(score),
        "think_verbosity_penalty": float(think_penalty),
        "answer_length_penalty": float(answer_penalty),
        "symbolic_eof_wait_penalty": float(symbolic_eof_wait_penalty),
        "final_think_token_penalty": float(final_think_token_penalty),
        "token_latency_penalty": float(token_latency_penalty),
        "post_eof_wall_clock_penalty": float(post_eof_wall_clock_penalty),
        "text_first_token_penalty": float(text_first_token_penalty),
        "effective_text_first_token_penalty": float(effective_text_first_token_penalty),
        "effective_response_onset_penalty": float(effective_response_onset_penalty),
        "final_think_token_count": float(final_think_token_count),
        "response_latency_token_proxy": float(response_latency_token_proxy),
        "eof_to_answer_lag": float(eof_to_answer_lag),
        "post_eof_wall_clock_seconds": float(runtime_timing.get("post_eof_total_wall_clock_sec") or 0.0),
        "final_think_generation_wall_clock_seconds": float(
            runtime_timing.get("final_think_generation_wall_clock_sec") or 0.0
        ),
        "text_first_token_wall_clock_seconds": float(
            runtime_timing.get("text_first_token_wall_clock_seconds") or 0.0
        ),
        "effective_text_first_token_seconds": float(
            runtime_timing.get("effective_text_first_token_seconds") or 0.0
        ),
        "text_streaming_supported": 1.0 if runtime_timing.get("text_streaming_supported") else 0.0,
        "answer_generation_wall_clock_seconds": float(
            runtime_timing.get("answer_generation_wall_clock_sec") or 0.0
        ),
        "controller_total_wall_clock_seconds": float(
            runtime_timing.get("controller_total_wall_clock_sec") or 0.0
        ),
        "response_onset_seconds": float(runtime_timing.get("response_onset_seconds") or 0.0),
        "effective_response_onset_seconds": float(
            runtime_timing.get("effective_response_onset_seconds") or 0.0
        ),
    }


def reward_sync(
    episode: PCoTEpisode,
    gpu_speed: float = DEFAULT_GPU_SPEED_TPS,
    alpha: float = DEFAULT_ALPHA,
    free_memory_tokens: int = DEFAULT_FREE_STATE_TOKENS,
    eof_wait_penalty: float = DEFAULT_EOF_WAIT_PENALTY,
    answer_alpha: float = DEFAULT_ANSWER_ALPHA,
    free_answer_tokens: int = DEFAULT_FREE_ANSWER_TOKENS,
    final_think_token_alpha: float = DEFAULT_FINAL_THINK_TOKEN_ALPHA,
    free_final_think_tokens: int = DEFAULT_FREE_FINAL_THINK_TOKENS,
    final_think_token_penalty_cap: float = DEFAULT_FINAL_THINK_TOKEN_PENALTY_CAP,
    latency_token_alpha: float = DEFAULT_LATENCY_TOKEN_ALPHA,
    free_latency_tokens: int = DEFAULT_FREE_LATENCY_TOKENS,
    post_eof_wall_clock_alpha: float = DEFAULT_POST_EOF_WALL_CLOCK_ALPHA,
    free_post_eof_wall_clock_seconds: float = DEFAULT_FREE_POST_EOF_WALL_CLOCK_SECONDS,
    text_first_token_alpha: float = DEFAULT_TEXT_FIRST_TOKEN_ALPHA,
    free_text_first_token_seconds: float = DEFAULT_FREE_TEXT_FIRST_TOKEN_SECONDS,
    effective_text_first_token_alpha: float = DEFAULT_EFFECTIVE_TEXT_FIRST_TOKEN_ALPHA,
    free_effective_text_first_token_seconds: float = DEFAULT_FREE_EFFECTIVE_TEXT_FIRST_TOKEN_SECONDS,
    effective_response_onset_alpha: float = DEFAULT_EFFECTIVE_RESPONSE_ONSET_ALPHA,
    free_effective_response_onset_seconds: float = DEFAULT_FREE_EFFECTIVE_RESPONSE_ONSET_SECONDS,
) -> float:
    detail = compute_sync_detail(
        episode,
        gpu_speed=gpu_speed,
        alpha=alpha,
        free_memory_tokens=free_memory_tokens,
        eof_wait_penalty=eof_wait_penalty,
        answer_alpha=answer_alpha,
        free_answer_tokens=free_answer_tokens,
        final_think_token_alpha=final_think_token_alpha,
        free_final_think_tokens=free_final_think_tokens,
        final_think_token_penalty_cap=final_think_token_penalty_cap,
        latency_token_alpha=latency_token_alpha,
        free_latency_tokens=free_latency_tokens,
        post_eof_wall_clock_alpha=post_eof_wall_clock_alpha,
        free_post_eof_wall_clock_seconds=free_post_eof_wall_clock_seconds,
        text_first_token_alpha=text_first_token_alpha,
        free_text_first_token_seconds=free_text_first_token_seconds,
        effective_text_first_token_alpha=effective_text_first_token_alpha,
        free_effective_text_first_token_seconds=free_effective_text_first_token_seconds,
        effective_response_onset_alpha=effective_response_onset_alpha,
        free_effective_response_onset_seconds=free_effective_response_onset_seconds,
    )
    return float(detail["score"])


# ── Utility: measure GPU speed ─────────────────────────────────────────────────

def measure_gpu_speed(
    model,
    tokenizer,
    n_tokens: int = 50,
    n_trials: int = 5,
) -> float:
    """
    Measure actual token generation speed on the current GPU.

    Call this once at the start of training to calibrate R_s.

    Returns:
        Measured speed in tokens per second.
    """
    import time
    import torch

    prompt_ids = tokenizer("The quick brown fox", return_tensors="pt").input_ids.to(model.device)
    speeds = []

    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt_ids, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        speeds.append(n_tokens / (t1 - t0))

    measured = sum(speeds) / len(speeds)
    print(f"[R_s] Measured GPU speed: {measured:.1f} tokens/sec")
    return measured
