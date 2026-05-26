"""
R_c: Consistency reward over the visible think-chain and final answer.
"""

import os
from typing import TYPE_CHECKING

from prompts.reward_prompts import CHAIN_CONSISTENCY_JUDGE_PROMPT, get_rc_prompt_template

from .episode import PCoTEpisode
from .reward_accuracy import count_effective_tokens, DEFAULT_MIN_EFFECTIVE_TOKENS
from .reward_sync import estimate_token_count

if TYPE_CHECKING:
    from .judge import LLMJudge


DEFAULT_STATE_CHAIN_TOKEN_BUDGET = 2400
DEFAULT_UPDATE_TOKEN_BUDGET = 96


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    text = str(text or "").strip()
    if max_tokens <= 0 or not text:
        return ""
    if estimate_token_count(text) <= max_tokens:
        return text

    marker = " ... [truncated for judge context] ... "
    max_chars = max(24, int(max_tokens * 4))
    if max_chars <= len(marker) + 16:
        return text[: max(0, max_chars - 4)].rstrip() + " ..."

    keep_chars = max_chars - len(marker)
    head_chars = max(8, int(keep_chars * 0.65))
    tail_chars = max(8, keep_chars - head_chars)
    return "{}{}{}".format(
        text[:head_chars].rstrip(),
        marker,
        text[-tail_chars:].lstrip(),
    )


def _fit_state_chain_to_budget(rows: list[str], max_tokens: int) -> str:
    if not rows:
        return "(no intermediate state)"
    if max_tokens <= 0:
        return "\n".join(rows)

    full_chain = "\n".join(rows)
    if estimate_token_count(full_chain) <= max_tokens:
        return full_chain

    marker_template = "  ... omitted {} earlier updates to fit judge context ..."
    marker_tokens = estimate_token_count(marker_template.format(len(rows)))
    selected_reversed: list[str] = []
    used_tokens = marker_tokens
    omitted = len(rows)

    for row in reversed(rows):
        remaining = max_tokens - used_tokens
        if remaining <= 8:
            break
        row_text = row
        row_tokens = estimate_token_count(row_text)
        if row_tokens > remaining:
            row_text = _truncate_text_to_token_budget(row_text, remaining)
            row_tokens = estimate_token_count(row_text)
        if row_tokens <= remaining:
            selected_reversed.append(row_text)
            used_tokens += row_tokens
            omitted -= 1
        if used_tokens >= max_tokens:
            break

    if not selected_reversed:
        selected_reversed.append(
            _truncate_text_to_token_budget(rows[-1], max(8, max_tokens - marker_tokens))
        )
        omitted = max(0, len(rows) - 1)

    marker = marker_template.format(max(0, omitted))
    return "\n".join([marker] + list(reversed(selected_reversed)))


def _is_context_length_judge_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "judge http 400" in text
        and (
            "maximum context length" in text
            or "input_tokens" in text
            or "reduce the length of the input prompt" in text
        )
    )


def build_consistency_prompt(
    episode: PCoTEpisode,
    *,
    prompt_version: str = CHAIN_CONSISTENCY_JUDGE_PROMPT,
    max_state_chain_tokens: int | None = None,
    max_update_tokens: int | None = None,
) -> str:
    if not episode.thinks:
        state_chain = "(no intermediate state)"
    else:
        metadata = dict(getattr(episode, "controller_metadata", {}) or {})
        teacher_segments = metadata.get("teacher_segments")
        rows = []
        update_budget = (
            _env_int("RC_UPDATE_TOKEN_BUDGET", DEFAULT_UPDATE_TOKEN_BUDGET)
            if max_update_tokens is None
            else int(max_update_tokens)
        )
        for i in range(len(episode.thinks)):
            think = episode.thinks[i].strip() if i < len(episode.thinks) else ""
            think = _truncate_text_to_token_budget(think, update_budget)
            span_text = ""
            if isinstance(teacher_segments, list) and i < len(teacher_segments) and isinstance(teacher_segments[i], dict):
                try:
                    start = float(teacher_segments[i].get("start"))
                    end = float(teacher_segments[i].get("end"))
                    span_text = " span={:.2f}s-{:.2f}s".format(start, end)
                except Exception:
                    span_text = ""
            rows.append(
                "  Update {}{}: think={}".format(
                    i + 1,
                    span_text,
                    think if think else "(empty think)",
                )
            )
        chain_budget = (
            _env_int("RC_STATE_CHAIN_TOKEN_BUDGET", DEFAULT_STATE_CHAIN_TOKEN_BUDGET)
            if max_state_chain_tokens is None
            else int(max_state_chain_tokens)
        )
        state_chain = _fit_state_chain_to_budget(rows, chain_budget)
    return get_rc_prompt_template(prompt_version).format(
        question=episode.question.strip() or "(no question)",
        state_chain=state_chain,
        answer=episode.answer.strip() or "(no answer produced)",
        reference_answer=episode.gt_answer.strip() or "(no reference answer available)",
    )


def _has_substantive_reasoning_state(
    episode: PCoTEpisode,
    min_tokens: int = DEFAULT_MIN_EFFECTIVE_TOKENS,
) -> bool:
    """Return True if the rollout has any non-trivial intermediate state."""
    return any(
        count_effective_tokens(t) >= min_tokens
        for t in episode.thinks
    )


# ── Episode-level reward ───────────────────────────────────────────────────────

def reward_consistency_sync(
    episode: PCoTEpisode,
    judge: "LLMJudge",
    min_tokens: int = DEFAULT_MIN_EFFECTIVE_TOKENS,
    prompt_version: str = CHAIN_CONSISTENCY_JUDGE_PROMPT,
) -> float:
    """
    Compute consistency reward R_c synchronously.

    Returns:
        1.0  — consistent and supported
        0.5  — weak / incomplete but not contradicted
        0.0  — contradicted / unsupported
    """
    if not episode.answer.strip():
        return 0.0  # no answer produced → cannot be consistent

    if not _has_substantive_reasoning_state(episode, min_tokens=min_tokens):
        return 0.0

    prompt = build_consistency_prompt(episode, prompt_version=prompt_version)
    try:
        score = judge.score_sync(prompt)
    except RuntimeError as exc:
        if _is_context_length_judge_error(exc):
            return 0.0
        raise
    return score


def reward_consistency_async(
    episode: PCoTEpisode,
    judge: "LLMJudge",
    episode_id: str,
    reward_buffer: dict,
    min_tokens: int = DEFAULT_MIN_EFFECTIVE_TOKENS,
    prompt_version: str = CHAIN_CONSISTENCY_JUDGE_PROMPT,
) -> None:
    """
    Submit consistency judge request asynchronously.

    Args:
        episode:       The controller episode.
        judge:         The async LLMJudge instance.
        episode_id:    Unique key for this episode in reward_buffer.
        reward_buffer: Shared dict where the score is written on completion.
    """
    if not episode.answer.strip():
        reward_buffer[episode_id] = 0.0
        return

    # No visible chain should not earn a consistency bonus.
    if not _has_substantive_reasoning_state(episode, min_tokens=min_tokens):
        reward_buffer[episode_id] = 0.0
        return

    prompt = build_consistency_prompt(episode, prompt_version=prompt_version)

    def callback(_, score: float) -> None:
        reward_buffer[episode_id] = score

    judge.submit(episode_id=episode_id, prompt=prompt, callback=callback)
