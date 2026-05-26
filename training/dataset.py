"""
Dataset adapter for wait-think-answer DAPO training.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from rewards.episode import PCoTEpisode

from .caption_utils import clean_chunk_caption, clean_chunk_caption_qwen3


@dataclass
class PCoTTrainSample:
    """
    One training-ready streaming controller episode.
    """

    audio_id: str
    audio_chunk_paths: List[str]
    question: str
    gt_answer: str
    chunk_captions_raw: List[str]
    chunk_captions: List[str]
    teacher_thinks: List[str] = field(default_factory=list)
    teacher_action_trace: List[str] = field(default_factory=list)
    teacher_sequence_compact: str = ""
    audio_class: str = ""
    n_chunks: int = 0
    difficulty_metadata: dict = field(default_factory=dict)
    controller_metadata: dict = field(default_factory=dict)

    def to_ground_truth_episode(self) -> PCoTEpisode:
        return PCoTEpisode(
            audio_chunk_paths=self.audio_chunk_paths,
            question=self.question,
            gt_answer=self.gt_answer,
            chunk_captions=self.chunk_captions,
            audio_id=self.audio_id,
            chunk_durations=[2.0] * self.n_chunks,
            difficulty_metadata=dict(self.difficulty_metadata),
            controller_metadata=dict(self.controller_metadata),
        )


def _resolve_chunk_path(path_str: str, project_root: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def _strict_list(record: dict, key: str) -> List[str]:
    value = record.get(key, [])
    if not isinstance(value, list):
        raise ValueError("{} must be a list".format(key))
    return [str(item) for item in value]


def _build_sample(
    record: dict,
    project_root: Path,
    clean_captions: bool,
    require_teacher: bool,
    check_paths: bool,
    difficulty_by_id: Optional[dict],
    controller_by_id: Optional[dict],
) -> PCoTTrainSample:
    audio_id = str(record["audio_id"]).strip()
    audio_chunks = [_resolve_chunk_path(path_str, project_root) for path_str in _strict_list(record, "audio_chunks")]
    stored_chunk_captions = _strict_list(record, "chunk_captions")
    chunk_captions_raw = (
        _strict_list(record, "chunk_captions_raw")
        if "chunk_captions_raw" in record
        else list(stored_chunk_captions)
    )
    teacher_thinks = _strict_list(record, "think_annotations") if "think_annotations" in record else []
    teacher_action_trace = _strict_list(record, "teacher_action_trace") if "teacher_action_trace" in record else []
    teacher_sequence_compact = str(record.get("teacher_sequence_compact", "") or "").strip()

    if len(audio_chunks) != len(chunk_captions_raw):
        raise ValueError(
            "{}: len(audio_chunks)={} != len(chunk_captions)={}".format(
                audio_id, len(audio_chunks), len(chunk_captions_raw)
            )
        )

    if require_teacher and not teacher_thinks:
        raise ValueError("{}: teacher think annotations required".format(audio_id))

    if teacher_thinks and len(teacher_thinks) != len(audio_chunks):
        raise ValueError(
            "{}: len(think_annotations)={} != len(audio_chunks)={}".format(
                audio_id, len(teacher_thinks), len(audio_chunks)
            )
        )

    if check_paths:
        for path_str in audio_chunks:
            if not Path(path_str).exists():
                raise FileNotFoundError("{}: missing chunk path {}".format(audio_id, path_str))

    caption_cleaner = str(record.get("caption_cleaner", "")).strip().lower()
    captions_are_precleaned = "chunk_captions_raw" in record
    if clean_captions:
        if captions_are_precleaned:
            chunk_captions = list(stored_chunk_captions)
        elif caption_cleaner == "qwen3":
            chunk_captions = [clean_chunk_caption_qwen3(text) for text in chunk_captions_raw]
        else:
            chunk_captions = [clean_chunk_caption(text) for text in chunk_captions_raw]
    else:
        chunk_captions = list(stored_chunk_captions if captions_are_precleaned else chunk_captions_raw)

    question = str(record["question"]).strip()
    gt_answer = str(record["gt_answer"]).strip()
    n_chunks = int(record.get("n_chunks", len(audio_chunks)))
    difficulty_metadata = dict(difficulty_by_id.get(audio_id, {})) if difficulty_by_id else {}
    if "difficulty_metadata" in record and isinstance(record["difficulty_metadata"], dict):
        difficulty_metadata.update(record["difficulty_metadata"])
    controller_metadata = dict(controller_by_id.get(audio_id, {})) if controller_by_id else {}
    if "controller_metadata" in record and isinstance(record["controller_metadata"], dict):
        controller_metadata.update(record["controller_metadata"])

    if n_chunks != len(audio_chunks):
        raise ValueError(
            "{}: n_chunks={} != len(audio_chunks)={}".format(
                audio_id, n_chunks, len(audio_chunks)
            )
        )

    return PCoTTrainSample(
        audio_id=audio_id,
        audio_chunk_paths=audio_chunks,
        question=question,
        gt_answer=gt_answer,
        chunk_captions_raw=chunk_captions_raw,
        chunk_captions=chunk_captions,
        teacher_thinks=teacher_thinks,
        teacher_action_trace=teacher_action_trace,
        teacher_sequence_compact=teacher_sequence_compact,
        audio_class=str(record.get("audio_class", "")).strip(),
        n_chunks=n_chunks,
        difficulty_metadata=difficulty_metadata,
        controller_metadata=controller_metadata,
    )


def _load_metadata_sidecar(path: Optional[str]) -> dict:
    if not path:
        return {}
    sidecar = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            audio_id = str(record.get("audio_id", "")).strip()
            if not audio_id:
                continue
            sidecar[audio_id] = record
    return sidecar


def apply_question_visibility_override(
    samples: List[PCoTTrainSample],
    question_visible_from_text: Optional[bool],
) -> List[PCoTTrainSample]:
    """
    Force one controller-visibility contract across all loaded RL samples.

    This is the safest place to align rollout prompts, actor-side turn
    reconstruction, rollout-event rendering, and saved episode metadata with the
    intended runtime protocol.
    """
    if question_visible_from_text is None:
        return samples

    normalized = bool(question_visible_from_text)
    task_spec_mode = "audio_only"
    for sample in samples:
        metadata = dict(sample.controller_metadata or {})
        metadata["question_visible_from_text"] = normalized
        metadata["question_visible_from_chunk_1"] = normalized
        metadata["task_spec_mode"] = task_spec_mode
        sample.controller_metadata = metadata
    return samples


def load_pcot_dataset(
    path: str,
    limit: int = 0,
    project_root: Optional[str] = None,
    clean_captions: bool = True,
    require_teacher: bool = False,
    check_paths: bool = True,
    difficulty_metadata_path: Optional[str] = None,
    controller_metadata_path: Optional[str] = None,
) -> List[PCoTTrainSample]:
    """
    Load controller training records from JSONL.
    """
    project_root_path = Path(project_root or ".").resolve()
    samples: List[PCoTTrainSample] = []
    difficulty_by_id = _load_metadata_sidecar(difficulty_metadata_path)
    controller_by_id = _load_metadata_sidecar(controller_metadata_path)

    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(
                _build_sample(
                    record=record,
                    project_root=project_root_path,
                    clean_captions=clean_captions,
                    require_teacher=require_teacher,
                    check_paths=check_paths,
                    difficulty_by_id=difficulty_by_id,
                    controller_by_id=controller_by_id,
                )
            )
            if limit and len(samples) >= limit:
                break

    if not samples:
        raise ValueError("No usable samples loaded from {}".format(path))

    return samples
