"""
R_t: LLM-as-judge reward over actual visible `<think>` actions.

The active design judges only real emitted think actions:
- pre-EOF `<think>` updates
- the final-think emitted immediately before the answer

`wait` actions and chunks without a true `<think>` are not judged by R_t.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from prompts.reward_prompts import THOUGHT_QUALITY_JUDGE_PROMPT, get_rt_prompt_template

from .episode import PCoTEpisode

if TYPE_CHECKING:
    from .judge import LLMJudge


_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class _ThinkJudgeItem:
    chunk_index: int
    think: str
    previous_think: str
    earlier_state_chain: List[str]
    caption: str
    segment_span: str
    is_final_think: bool
    forced_score: Optional[float] = None


def build_think_prompt(
    *,
    question: str,
    previous_think: str,
    earlier_state_chain: str = "",
    think: str,
    caption: str,
    final_answer: str = "",
    reference_answer: str = "",
    prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT,
    segment_span: str = "",
    is_final_think: bool = False,
) -> str:
    template = get_rt_prompt_template(prompt_version)
    think_kind_label = (
        "final-think written after the full audio was heard"
        if is_final_think
        else "pre-EOF local reasoning update"
    )
    think_kind_expectation = (
        "This is the last reasoning state before the answer. It should be short, compressed, "
        "answer-supporting, and still contain reasoning rather than a bare answer or placeholder."
        if is_final_think
        else "This should be a grounded local reasoning update that adds a real analytical step "
        "beyond simple state summary."
    )
    answer_leak_rule = (
        "for final-think: bare answer-only text, `<answer>...</answer>` leakage, or a naked final choice "
        "with no supporting reasoning"
        if is_final_think
        else (
            "for pre-EOF think: bare final-answer commitment, `<answer>...</answer>` leakage, "
            "or answer-only text with no evidence/provisional state. A grounded candidate answer "
            "is allowed when it is supported by the current evidence and remains a state update."
        )
    )
    return template.format(
        question=question.strip() or "(no question available)",
        segment_span=segment_span.strip() or "(unknown span)",
        previous_state=previous_think.strip() or "(empty)",
        earlier_state_chain=earlier_state_chain.strip() or "(empty)",
        caption=caption.strip() or "(no caption available)",
        think=think.strip() or "(empty)",
        think_kind_label=think_kind_label,
        think_kind_expectation=think_kind_expectation,
        answer_leak_rule=answer_leak_rule,
        final_answer=str(final_answer or "").strip() or "(no final answer produced)",
        reference_answer=str(reference_answer or "").strip() or "(no reference answer available)",
    )


def _teacher_segment_span(episode: PCoTEpisode, chunk_idx: int) -> str:
    metadata = dict(getattr(episode, "controller_metadata", {}) or {})
    segments = metadata.get("teacher_segments")
    if not isinstance(segments, list) or not (0 <= chunk_idx < len(segments)):
        return ""
    segment = segments[chunk_idx]
    if not isinstance(segment, dict):
        return ""
    try:
        start = float(segment.get("start"))
        end = float(segment.get("end"))
    except Exception:
        return ""
    return f"{start:.2f}s-{end:.2f}s"


def _mean(values: List[float], fallback: float = 0.0) -> float:
    if not values:
        return float(fallback)
    return float(sum(values) / len(values))


def _extract_think_text(event: Dict[str, Any]) -> str:
    direct = str(event.get("think") or "").strip()
    if direct:
        return direct
    raw = str(event.get("normalized_output") or event.get("raw_output") or "").strip()
    if not raw:
        return ""
    match = _THINK_TAG_PATTERN.search(raw)
    if match:
        return str(match.group(1) or "").strip()
    return ""


def _iter_judge_items(episode: PCoTEpisode) -> List[_ThinkJudgeItem]:
    items: List[_ThinkJudgeItem] = []
    previous_chain: List[str] = []

    if episode.rollout_events:
        for event in episode.rollout_events:
            if str(event.get("kind", "")).strip() != "assistant_think":
                continue
            think = _extract_think_text(event)
            if not think:
                continue
            chunk_index = int(event.get("chunk_index", max(0, episode.n_chunks - 1)))
            is_final = bool(
                event.get("is_final_think")
                or bool(dict(event.get("timing") or {}).get("is_final_think"))
            )
            timing = dict(event.get("timing") or {})
            forced_score = 0.0 if bool(is_final and timing.get("final_think_fallback_used")) else None
            if is_final:
                caption = "(full audio already heard; judge the final compressed reasoning state)"
                segment_span = "AUDIO_END (full audio heard)"
            else:
                caption = (
                    episode.chunk_captions[chunk_index]
                    if 0 <= chunk_index < len(episode.chunk_captions)
                    else ""
                )
                segment_span = _teacher_segment_span(episode, chunk_index)
            previous_think = previous_chain[-1] if previous_chain else ""
            items.append(
                _ThinkJudgeItem(
                    chunk_index=chunk_index,
                    think=think,
                    previous_think=previous_think,
                    earlier_state_chain=list(previous_chain),
                    caption=caption,
                    segment_span=segment_span,
                    is_final_think=is_final,
                    forced_score=forced_score,
                )
            )
            previous_chain.append(think)
        if items:
            return items

    # Fallback path for unit tests / legacy episodes without rollout_events.
    thinks = [str(item or "").strip() for item in list(episode.thinks or []) if str(item or "").strip()]
    if not thinks:
        return []
    has_explicit_final = len(thinks) > int(episode.n_chunks)
    for idx, think in enumerate(thinks):
        is_final = bool(has_explicit_final and idx == len(thinks) - 1)
        chunk_index = max(0, min(idx, max(0, episode.n_chunks - 1)))
        if is_final:
            caption = "(full audio already heard; judge the final compressed reasoning state)"
            segment_span = "AUDIO_END (full audio heard)"
        else:
            caption = (
                episode.chunk_captions[chunk_index]
                if 0 <= chunk_index < len(episode.chunk_captions)
                else ""
            )
            segment_span = _teacher_segment_span(episode, chunk_index)
        previous_think = previous_chain[-1] if previous_chain else ""
        items.append(
            _ThinkJudgeItem(
                chunk_index=chunk_index,
                think=think,
                previous_think=previous_think,
                earlier_state_chain=list(previous_chain),
                caption=caption,
                segment_span=segment_span,
                is_final_think=is_final,
                forced_score=None,
            )
        )
        previous_chain.append(think)
    return items


def _build_episode_detail(
    episode: PCoTEpisode,
    *,
    item_specs: List[tuple[int, _ThinkJudgeItem]],
    by_key: Dict[tuple[int, int], float],
) -> Dict[str, Any]:
    pre_scores = [0.0 for _ in range(episode.n_chunks)]
    pre_mask = [False for _ in range(episode.n_chunks)]
    final_score = 0.0
    final_judged = False
    judged_scores: List[float] = []

    for item_idx, item in item_specs:
        score = float(item.forced_score) if item.forced_score is not None else float(by_key[(item.chunk_index, item_idx)])
        judged_scores.append(score)
        if item.is_final_think:
            final_score = score
            final_judged = True
            continue
        if 0 <= item.chunk_index < len(pre_scores):
            pre_scores[item.chunk_index] = score
            pre_mask[item.chunk_index] = True

    return {
        "mean": _mean(judged_scores, 0.0),
        "per_chunk": list(pre_scores),
        "judged_mask": list(pre_mask),
        "pre_chunk_scores": list(pre_scores),
        "pre_judged_mask": list(pre_mask),
        "final_score": float(final_score),
        "final_judged": bool(final_judged),
        "n_judged_pre": int(sum(1 for flag in pre_mask if flag)),
        "n_judged_total": int(len(judged_scores)),
    }


def reward_think_sync_details(
    episode: PCoTEpisode,
    judge: "LLMJudge",
    batch_mean_fallback: float = 0.0,
    prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT,
) -> Dict[str, Any]:
    item_specs = _iter_judge_items(episode)
    if not item_specs:
        return _build_episode_detail(episode, item_specs=[], by_key={})

    prompt_specs = []
    forced_scores: Dict[tuple[int, int], float] = {}
    for item_idx, item in enumerate(item_specs):
        if item.forced_score is not None:
            forced_scores[(item.chunk_index, item_idx)] = float(item.forced_score)
            continue
        prompt_specs.append(
            (
                item.chunk_index,
                item_idx,
                build_think_prompt(
                    question=episode.question,
                    previous_think=item.previous_think,
                    earlier_state_chain="\n".join(
                        "  - {}".format(state) for state in item.earlier_state_chain
                    ),
                    think=item.think,
                    caption=item.caption,
                    final_answer=episode.answer,
                    reference_answer=episode.gt_answer,
                    prompt_version=prompt_version,
                    segment_span=item.segment_span,
                    is_final_think=item.is_final_think,
                ),
            )
        )
    scores = judge.score_batch([spec[2] for spec in prompt_specs]) if prompt_specs else []
    neutral_fallback = _mean([float(score) for score in scores], batch_mean_fallback)
    by_key = dict(forced_scores)
    by_key.update({
        (chunk_index, item_idx): float(score)
        for (chunk_index, item_idx, _), score in zip(prompt_specs, scores)
    })
    if not by_key:
        return {
            "mean": float(neutral_fallback),
            "per_chunk": [0.0 for _ in range(episode.n_chunks)],
            "judged_mask": [False for _ in range(episode.n_chunks)],
            "pre_chunk_scores": [0.0 for _ in range(episode.n_chunks)],
            "pre_judged_mask": [False for _ in range(episode.n_chunks)],
            "final_score": 0.0,
            "final_judged": False,
            "n_judged_pre": 0,
            "n_judged_total": 0,
        }
    return _build_episode_detail(
        episode,
        item_specs=list(enumerate(item_specs)),
        by_key=by_key,
    )


def reward_think_sync(
    episode: PCoTEpisode,
    judge: "LLMJudge",
    batch_mean_fallback: float = 0.0,
    prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT,
) -> float:
    return float(
        reward_think_sync_details(
            episode,
            judge,
            batch_mean_fallback=batch_mean_fallback,
            prompt_version=prompt_version,
        )["mean"]
    )


def reward_think_sync_batch_details(
    episodes: List[PCoTEpisode],
    judge: "LLMJudge",
    default_fallback: float = 0.0,
    prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT,
) -> List[Dict[str, Any]]:
    prompt_specs = []
    forced_scores: Dict[tuple[int, int, int], float] = {}
    episode_items: List[List[_ThinkJudgeItem]] = []
    for ep_idx, episode in enumerate(episodes):
        items = _iter_judge_items(episode)
        episode_items.append(items)
        for item_idx, item in enumerate(items):
            if item.forced_score is not None:
                forced_scores[(ep_idx, item.chunk_index, item_idx)] = float(item.forced_score)
                continue
            prompt_specs.append(
                (
                    ep_idx,
                    item.chunk_index,
                    item_idx,
                    build_think_prompt(
                        question=episode.question,
                        previous_think=item.previous_think,
                        earlier_state_chain="\n".join(
                            "  - {}".format(state) for state in item.earlier_state_chain
                        ),
                        think=item.think,
                        caption=item.caption,
                        final_answer=episode.answer,
                        reference_answer=episode.gt_answer,
                        prompt_version=prompt_version,
                        segment_span=item.segment_span,
                        is_final_think=item.is_final_think,
                    ),
                )
            )

    scores = judge.score_batch([spec[3] for spec in prompt_specs]) if prompt_specs else []
    neutral_fallback = _mean([float(score) for score in scores], default_fallback)
    by_key: Dict[tuple[int, int, int], float] = dict(forced_scores)
    for (ep_idx, chunk_index, item_idx, _), score in zip(prompt_specs, scores):
        by_key[(ep_idx, chunk_index, item_idx)] = float(score)

    details: List[Dict[str, Any]] = []
    for ep_idx, episode in enumerate(episodes):
        items = episode_items[ep_idx]
        if not items:
            details.append(
                {
                    "mean": 0.0,
                    "per_chunk": [0.0 for _ in range(episode.n_chunks)],
                    "judged_mask": [False for _ in range(episode.n_chunks)],
                    "pre_chunk_scores": [0.0 for _ in range(episode.n_chunks)],
                    "pre_judged_mask": [False for _ in range(episode.n_chunks)],
                    "final_score": 0.0,
                    "final_judged": False,
                    "n_judged_pre": 0,
                    "n_judged_total": 0,
                }
            )
            continue
        episode_by_key = {
            (chunk_index, item_idx): score
            for (batch_ep_idx, chunk_index, item_idx), score in by_key.items()
            if batch_ep_idx == ep_idx
        }
        if not episode_by_key:
            details.append(
                {
                    "mean": float(neutral_fallback),
                    "per_chunk": [0.0 for _ in range(episode.n_chunks)],
                    "judged_mask": [False for _ in range(episode.n_chunks)],
                    "pre_chunk_scores": [0.0 for _ in range(episode.n_chunks)],
                    "pre_judged_mask": [False for _ in range(episode.n_chunks)],
                    "final_score": 0.0,
                    "final_judged": False,
                    "n_judged_pre": 0,
                    "n_judged_total": 0,
                }
            )
            continue
        details.append(
            _build_episode_detail(
                episode,
                item_specs=list(enumerate(items)),
                by_key=episode_by_key,
            )
        )
    return details


def reward_think_sync_batch(
    episodes: List[PCoTEpisode],
    judge: "LLMJudge",
    default_fallback: float = 0.0,
    prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT,
) -> List[float]:
    details = reward_think_sync_batch_details(
        episodes,
        judge,
        default_fallback=default_fallback,
        prompt_version=prompt_version,
    )
    return [float(item["mean"]) for item in details]


def reward_think_async(
    episode: PCoTEpisode,
    judge: "LLMJudge",
    episode_id: str,
    reward_buffer: dict,
    batch_mean_fallback: float = 0.0,
    prompt_version: str = THOUGHT_QUALITY_JUDGE_PROMPT,
) -> None:
    item_specs = _iter_judge_items(episode)
    if not item_specs:
        reward_buffer[episode_id] = 0.0
        return

    pending_scores: Dict[int, float] = {}

    def make_callback(item_idx: int):
        def callback(_, score: float) -> None:
            pending_scores[item_idx] = float(score)
            if len(pending_scores) == len(item_specs):
                reward_buffer[episode_id] = _mean(list(pending_scores.values()), batch_mean_fallback)

        return callback

    for item_idx, item in enumerate(item_specs):
        prompt = build_think_prompt(
            question=episode.question,
            previous_think=item.previous_think,
            earlier_state_chain="\n".join("  - {}".format(state) for state in item.earlier_state_chain),
            think=item.think,
            caption=item.caption,
            final_answer=episode.answer,
            reference_answer=episode.gt_answer,
            prompt_version=prompt_version,
            segment_span=item.segment_span,
            is_final_think=item.is_final_think,
        )
        judge.submit(
            episode_id="{}::think_{}".format(episode_id, item_idx),
            prompt=prompt,
            callback=make_callback(item_idx),
        )
