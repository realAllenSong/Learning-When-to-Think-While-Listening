"""
R_f: Adaptive Think Format Reward for interleaved streaming rollout.

The controller format is:

- `<wait/>`
- `<think>...</think>`
- `<answer>...</answer>`
"""

import re

from .episode import PCoTEpisode


_UNIFIED_THINK_PATTERN = re.compile(r"\s*<think>(.*?)</think>\s*", re.DOTALL)
_WAIT_PATTERN = re.compile(r"\s*<wait\s*/>\s*", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"\s*<answer>(.*?)</answer>\s*", re.DOTALL)
_NESTED_TAG_PATTERN = re.compile(r"</?(think|predict|answer|wait)>", re.IGNORECASE)


def _is_valid_unified_think_action(text: str) -> bool:
    match = _UNIFIED_THINK_PATTERN.fullmatch(text.strip())
    if match is None:
        return False
    think_body = match.group(1)
    if "<answer>" in text or "</answer>" in text or "<wait" in text:
        return False
    if _NESTED_TAG_PATTERN.search(think_body or ""):
        return False
    return bool((think_body or "").strip())


def _is_valid_chunk_update(text: str) -> bool:
    return _is_valid_unified_think_action(text)


def _event_action_text(event: dict) -> str:
    normalized_output = str(event.get("normalized_output", "")).strip()
    if normalized_output:
        return normalized_output
    return str(event.get("raw_output", "")).strip()


def _event_model_action_text(event: dict) -> str:
    model_raw = str(event.get("model_raw_output", "")).strip()
    if model_raw:
        return model_raw
    return str(event.get("raw_output", "")).strip()


def _consume_visible_actions(seq: str) -> tuple[list[str], int]:
    actions: list[str] = []
    pos = 0
    length = len(seq)
    while pos < length:
        think_match = _UNIFIED_THINK_PATTERN.match(seq, pos)
        if think_match is not None:
            matched_text = think_match.group(0)
            if not _is_valid_unified_think_action(matched_text):
                break
            actions.append("think")
            pos = think_match.end()
            continue

        wait_match = _WAIT_PATTERN.match(seq, pos)
        if wait_match is not None:
            actions.append("wait")
            pos = wait_match.end()
            continue

        break
    return actions, pos


def _validate_interleaved_events(episode: PCoTEpisode) -> float:
    events = episode.rollout_events
    if not events:
        return 0.0

    heard_chunks = 0
    expected_chunk = 0
    answer_seen = False

    for event in events:
        kind = str(event.get("kind", "")).strip()

        if kind == "user_audio":
            if answer_seen:
                return 0.0
            chunk_index = int(event.get("chunk_index", -1))
            if chunk_index != expected_chunk:
                return 0.0
            heard_chunks += 1
            expected_chunk += 1
            continue

        if kind == "assistant_think":
            if answer_seen:
                return 0.0
            chunk_index = int(event.get("chunk_index", -1))
            if chunk_index < 0 or chunk_index >= heard_chunks:
                return 0.0
            action_text = _event_action_text(event)
            model_action_text = _event_model_action_text(event)
            if "<answer>" in action_text or "<answer>" in model_action_text:
                return -1.0
            timing = dict(event.get("timing") or {})
            if bool(event.get("is_final_think")) or bool(timing.get("is_final_think")):
                if not _is_valid_chunk_update(model_action_text):
                    return -1.0
            if not _is_valid_chunk_update(action_text):
                return 0.0
            continue

        if kind == "assistant_wait":
            if answer_seen:
                return 0.0
            chunk_index = int(event.get("chunk_index", -1))
            if chunk_index < 0 or chunk_index >= heard_chunks:
                return 0.0
            action_text = _event_action_text(event)
            model_action_text = _event_model_action_text(event)
            if "<answer>" in model_action_text:
                return -1.0
            if _WAIT_PATTERN.fullmatch(action_text) is None:
                return 0.0
            continue

        if kind == "assistant_answer":
            action_text = _event_action_text(event)
            if heard_chunks < episode.n_chunks:
                return -1.0
            if answer_seen:
                return 0.0
            answer_match = _ANSWER_PATTERN.fullmatch(action_text)
            if answer_match is None:
                return 0.0
            if "<think>" in action_text:
                return 0.0
            answer_seen = True
            continue

        return 0.0

    if heard_chunks != episode.n_chunks:
        return 0.0
    if not answer_seen:
        return 0.0
    return 1.0


def reward_format(episode: PCoTEpisode) -> float:
    if episode.rollout_events:
        return _validate_interleaved_events(episode)

    seq = episode.raw_sequence
    if not seq or not seq.strip():
        return 0.0

    if "<answer>" not in seq or "</answer>" not in seq:
        return 0.0

    actions, pos = _consume_visible_actions(seq)

    remaining = seq[pos:]
    answer_match = _ANSWER_PATTERN.match(remaining)
    if answer_match is None:
        searched_answer = _ANSWER_PATTERN.search(remaining)
        if (
            searched_answer is not None
            and not remaining[: searched_answer.start()].strip()
            and episode.n_chunks
            and len(actions) < episode.n_chunks
        ):
            return -1.0
        return 0.0

    if answer_match.end() != len(remaining):
        return 0.0

    if episode.n_chunks:
        if len(actions) == episode.n_chunks:
            return 1.0
        if len(actions) == episode.n_chunks + 1 and actions and actions[-1] == "think":
            return 1.0
        return -1.0 if len(actions) < episode.n_chunks else 0.0
    return 1.0
