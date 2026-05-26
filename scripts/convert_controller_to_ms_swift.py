#!/usr/bin/env python3
"""Convert local controller JSONL into MS-Swift-friendly SFT or DAPO rows.

- `mode=sft`: emits `messages + assistant` rows for `swift sft`.
- `mode=grpo`: emits prompt-only rows plus metadata used by the DAPO trainer.

The emitted rows follow the standard `ms-swift` dataset shape:
- `messages`
- optional `audios`
- pass-through extra fields for custom GRPO reward functions
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SFT_SYSTEM_PROMPT = (
    "You are a streaming audio reasoning model. "
    "Audio evidence arrives on a fixed tick grid. "
    "For supervision, emit only grounded incremental <think>...</think> updates "
    "at informative boundaries, then finish with a final <answer>...</answer>. "
    "Non-informative pauses remain implicit and should not be verbalized as explicit wait tokens."
)


DEFAULT_GRPO_SYSTEM_PROMPT = (
    "You are a streaming audio reasoning controller. "
    "Audio evidence arrives on a fixed tick grid. "
    "After listening, produce a sequence of XML actions using only "
    "<wait/>, <think>...</think>, and a final <answer>...</answer>. "
    "Use <wait/> when the current tick does not change the running state. "
    "Use <think>...</think> only for grounded incremental updates. "
    "Only emit <answer> after all audio evidence has been heard."
)


def strip_pause_anchors(text: str) -> str:
    """Remove simple pause markers while preserving the spoken text."""
    value = str(text or "")
    value = re.sub(r"\[(?:pause|silence)[^\]]*\]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<(?:pause|silence)[^>]*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\{(?:pause|silence)[^}]*\}", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def update_tick_indices_from_times(
    times_sec: Iterable[float],
    *,
    tick_seconds: float,
    total_duration: float,
) -> List[int]:
    """Map update boundary times to zero-based controller tick indices."""
    tick = float(tick_seconds)
    if tick <= 0:
        raise ValueError("tick_seconds must be positive")
    n_ticks = max(1, int(math.ceil(max(0.0, float(total_duration)) / tick)))
    indices = set()
    for value in times_sec:
        time_value = min(max(0.0, float(value)), float(total_duration))
        boundary = min(float(math.ceil(time_value / tick)) * tick, float(total_duration))
        idx = max(0, int(round(boundary / tick)) - 1)
        indices.add(min(idx, n_ticks - 1))
    return sorted(indices)


def boundary_after_tick(n_ticks: int, update_tick_indices: Iterable[int]) -> List[bool]:
    """Return a boolean mask indicating which ticks should emit an update."""
    n = max(0, int(n_ticks))
    mask = [False for _ in range(n)]
    for idx in update_tick_indices:
        value = int(idx)
        if 0 <= value < n:
            mask[value] = True
    return mask


def thinks_by_tick(n_ticks: int, update_tick_indices: Iterable[int], thinks: Iterable[str]) -> List[str]:
    """Place sparse teacher thoughts onto a dense controller tick grid."""
    dense = ["" for _ in range(max(0, int(n_ticks)))]
    for idx, think in zip(update_tick_indices, thinks):
        tick_idx = int(idx)
        if 0 <= tick_idx < len(dense):
            dense[tick_idx] = str(think or "").strip()
    return dense


def _resolve_audio_path(path_str: str, project_root: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_text_list(value: Any) -> List[str]:
    return [str(item).strip() for item in _ensure_list(value)]


def _resolve_audio_chunks(record: Dict[str, Any], project_root: Path) -> List[str]:
    for key in ("audio_chunks", "audio_chunk_paths", "audios"):
        if key in record:
            return [_resolve_audio_path(path, project_root) for path in _normalize_text_list(record.get(key))]
    return []


def _resolve_question(record: Dict[str, Any]) -> str:
    for key in ("question", "question_text"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raw_text = str(record.get("raw_text_with_anchors", "")).strip()
    if raw_text:
        return strip_pause_anchors(raw_text)
    return ""


def _resolve_answer(record: Dict[str, Any]) -> str:
    for key in ("gt_answer", "solution", "final_answer"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def _resolve_teacher_thinks(record: Dict[str, Any]) -> List[str]:
    for key in ("think_annotations", "teacher_thinks", "pcot_thoughts"):
        value = _normalize_text_list(record.get(key))
        if value:
            return value
    return []


def _build_controller_metadata(record: Dict[str, Any], n_chunks: int, tick_seconds: float) -> Dict[str, Any]:
    controller_metadata = dict(record.get("controller_metadata") or {})
    controller_metadata.setdefault("n_ticks", n_chunks)
    controller_metadata.setdefault("ingest_grid_seconds", float(tick_seconds))

    update_tick_indices = controller_metadata.get("update_tick_indices")
    if not isinstance(update_tick_indices, list):
        update_tick_indices = []

    if not update_tick_indices:
        quantized_times = record.get("quantized_update_times_sec") or record.get("quantized_pause_boundaries_sec") or []
        if quantized_times:
            try:
                total_duration = float(record.get("total_duration_sec", n_chunks * tick_seconds))
            except Exception:
                total_duration = float(n_chunks) * float(tick_seconds)
            update_tick_indices = update_tick_indices_from_times(
                quantized_times,
                tick_seconds=tick_seconds,
                total_duration=total_duration,
            )
            controller_metadata["update_tick_indices"] = update_tick_indices

    if "boundary_after_tick_t" not in controller_metadata:
        controller_metadata["boundary_after_tick_t"] = boundary_after_tick(n_chunks, update_tick_indices)

    return controller_metadata


def _build_user_prompt(
    question: str,
    n_audios: int,
    *,
    mode: str,
    question_visible_from_text: bool = False,
) -> str:
    audio_slots = "<audio>" * max(1, n_audios)
    if mode == "sft":
        instruction = (
            "Listen to the full pre-chunked audio stream. "
            "Emit only grounded <think>...</think> updates at informative boundaries, "
            "then finish with <answer>...</answer>. "
            "Do not write explicit <wait/> markers in the supervised target."
        )
    else:
        instruction = (
            "Return a streaming control trace over the audio ticks using "
            "<wait/> or <think>...</think>, then finish with <answer>...</answer>."
        )
    question_block = (
        "Question: {}\n".format(question.strip())
        if question_visible_from_text and str(question or "").strip()
        else "The spoken question and any answer options were provided only through the audio.\n"
    )
    return f"{audio_slots}\n{question_block}{instruction}"


def _build_raw_teacher_trace(teacher_thinks: List[str]) -> List[str]:
    actions: List[str] = []
    for think in teacher_thinks:
        text = str(think or "").strip()
        if text:
            actions.append(f"<think>{text}</think>")
        else:
            actions.append("<wait/>")
    return actions


def _build_teacher_sequence(teacher_thinks: List[str], answer: str, *, style: str) -> str:
    actions = _build_raw_teacher_trace(teacher_thinks)
    if style == "compact_no_wait":
        prefix = "".join(action for action in actions if action != "<wait/>")
    elif style == "raw_trace":
        prefix = "".join(actions)
    else:
        raise ValueError(f"Unsupported SFT target style: {style}")
    return f"{prefix}<answer>{answer.strip()}</answer>"


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def _build_swift_row(
    record: Dict[str, Any],
    *,
    project_root: Path,
    mode: str,
    tick_seconds: float,
    system_prompt: str,
    sft_target_style: str,
    question_visible_from_text: bool,
) -> Dict[str, Any]:
    audio_chunks = _resolve_audio_chunks(record, project_root)
    question = _resolve_question(record)
    gt_answer = _resolve_answer(record)
    audio_id = str(record.get("audio_id", "")).strip() or "unknown_audio"
    chunk_captions = _normalize_text_list(record.get("chunk_captions"))
    teacher_thinks = _resolve_teacher_thinks(record)
    n_chunks = int(record.get("n_chunks", len(audio_chunks) or len(chunk_captions) or len(teacher_thinks) or 0))

    if not audio_chunks:
        raise ValueError(f"{audio_id}: missing audio_chunks")
    if not question:
        raise ValueError(f"{audio_id}: missing question")
    if not gt_answer:
        raise ValueError(f"{audio_id}: missing gt_answer")

    controller_metadata = _build_controller_metadata(record, n_chunks=n_chunks, tick_seconds=tick_seconds)
    update_tick_indices = list(controller_metadata.get("update_tick_indices") or [])
    if not teacher_thinks:
        teacher_thinks = ["" for _ in range(n_chunks)]
    elif len(teacher_thinks) != n_chunks:
        if update_tick_indices and len(update_tick_indices) == len(teacher_thinks):
            teacher_thinks = thinks_by_tick(n_chunks, update_tick_indices, teacher_thinks)
        else:
            raise ValueError(
                f"{audio_id}: len(think_annotations)={len(teacher_thinks)} does not match n_chunks={n_chunks}"
            )
    difficulty_metadata = dict(record.get("difficulty_metadata") or {})
    chunk_durations = list(record.get("chunk_durations") or [float(tick_seconds) for _ in range(n_chunks)])

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_user_prompt(
                question,
                len(audio_chunks),
                mode=mode,
                question_visible_from_text=question_visible_from_text,
            ),
        },
    ]

    teacher_action_trace = _build_raw_teacher_trace(teacher_thinks)
    teacher_sequence_raw = _build_teacher_sequence(teacher_thinks, gt_answer, style="raw_trace")
    teacher_sequence_compact = _build_teacher_sequence(teacher_thinks, gt_answer, style="compact_no_wait")

    row: Dict[str, Any] = {
        "messages": messages,
        "audios": audio_chunks,
        "audio_id": audio_id,
        "question": question,
        "gt_answer": gt_answer,
        "solution": gt_answer,
        "question_visible_from_text": bool(question_visible_from_text),
        "task_spec_mode": "audio_only",
        "audio_chunk_paths": audio_chunks,
        "chunk_captions": chunk_captions,
        "teacher_thinks": teacher_thinks,
        "teacher_action_trace": teacher_action_trace,
        "teacher_sequence_raw": teacher_sequence_raw,
        "teacher_sequence_compact": teacher_sequence_compact,
        "chunk_durations": chunk_durations,
        "controller_metadata": {
            **controller_metadata,
            "question_visible_from_text": bool(question_visible_from_text),
        },
        "difficulty_metadata": difficulty_metadata,
    }
    for optional_key in (
        "raw_text_with_anchors",
        "tts_text",
        "pcot_thoughts",
        "final_answer",
        "alignment_end_times_sec",
        "quantized_update_times_sec",
        "augmentation_metadata",
    ):
        if optional_key in record:
            row[optional_key] = record[optional_key]

    if mode == "sft":
        target_sequence = _build_teacher_sequence(teacher_thinks, gt_answer, style=sft_target_style)
        row["messages"] = messages + [
            {"role": "assistant", "content": target_sequence}
        ]
        row["sft_target_style"] = sft_target_style

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert controller JSONL into MS-Swift SFT or DAPO JSONL")
    parser.add_argument("--input", required=True, help="Controller JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--mode", required=True, choices=["sft", "grpo"])
    parser.add_argument("--project-root", default=".", help="Project root for resolving relative audio chunk paths")
    parser.add_argument("--tick-seconds", type=float, default=0.5)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument(
        "--sft-target-style",
        default="compact_no_wait",
        choices=["compact_no_wait", "raw_trace"],
        help="How SFT assistant targets should be rendered. The paper setting is compact_no_wait.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in _iter_rows(input_path):
            row = _build_swift_row(
                record,
                project_root=project_root,
                mode=str(args.mode),
                tick_seconds=float(args.tick_seconds),
                system_prompt=(
                    str(args.system_prompt)
                    if str(args.system_prompt).strip()
                    else (DEFAULT_SFT_SYSTEM_PROMPT if args.mode == "sft" else DEFAULT_GRPO_SYSTEM_PROMPT)
                ),
                sft_target_style=str(args.sft_target_style),
                question_visible_from_text=False,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"output": str(output_path), "rows": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
