"""
PCoTEpisode: core data structure for one streaming controller episode.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .answer_scoring import answers_match

@dataclass
class PCoTEpisode:
    # === Ground truth ===
    audio_chunk_paths: List[str]
    question: str
    gt_answer: str
    chunk_captions: List[str]
    audio_id: str = ""

    # === Generated rollout ===
    thinks: List[str] = field(default_factory=list)
    answer: str = ""
    raw_sequence: str = ""
    rollout_events: List[Dict[str, Any]] = field(default_factory=list)
    difficulty_metadata: Dict[str, Any] = field(default_factory=dict)
    controller_metadata: Dict[str, Any] = field(default_factory=dict)
    answer_rule_correct: Optional[bool] = None
    answer_correct_override: Optional[bool] = None
    answer_fallback_invoked: bool = False
    answer_fallback_rescued: bool = False
    answer_fallback_short_circuit_no_final_answer: bool = False
    answer_fallback_source: str = ""
    answer_fallback_judge_result: str = ""
    answer_fallback_judge_answer: str = ""

    # === Timing ===
    chunk_durations: List[float] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.audio_chunk_paths)

    @property
    def is_complete(self) -> bool:
        return bool(self.answer.strip())

    @classmethod
    def from_rollout(
        cls,
        raw_sequence: str,
        audio_chunk_paths: List[str],
        question: str,
        gt_answer: str,
        chunk_captions: List[str],
        audio_id: str = "",
        chunk_durations: Optional[List[float]] = None,
    ) -> "PCoTEpisode":
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        thinks = [m.group(1).strip() for m in think_pattern.finditer(raw_sequence)]
        answer_matches = answer_pattern.findall(raw_sequence)
        answer = answer_matches[-1].strip() if answer_matches else ""
        durations = chunk_durations or [2.0] * len(audio_chunk_paths)

        return cls(
            audio_chunk_paths=audio_chunk_paths,
            question=question,
            gt_answer=gt_answer,
            chunk_captions=chunk_captions,
            audio_id=audio_id,
            thinks=thinks,
            answer=answer,
            raw_sequence=raw_sequence,
            chunk_durations=durations,
        )

    def is_correct(self, normalize: bool = True) -> bool:
        if normalize:
            if self.answer_correct_override is not None:
                return bool(self.answer_correct_override)
            return answers_match(
                answer_text=self.answer,
                gt_answer=self.gt_answer,
                difficulty_metadata=self.difficulty_metadata,
                controller_metadata=self.controller_metadata,
            )
        return self.answer == self.gt_answer

    def answer_appears_before_last_chunk(self) -> bool:
        if self.rollout_events:
            heard_chunks = 0
            for event in self.rollout_events:
                kind = str(event.get("kind", "")).strip()
                if kind == "user_audio":
                    heard_chunks += 1
                if kind == "assistant_answer":
                    return heard_chunks < self.n_chunks
            return False

        if not self.raw_sequence:
            return False

        answer_pos = self.raw_sequence.find("<answer>")
        if answer_pos == -1:
            return False

        pre_answer_text = self.raw_sequence[:answer_pos]
        closed_thinks = pre_answer_text.count("</think>")
        return closed_thinks < max(0, self.n_chunks - 1)

    def get_chunk_duration(self, chunk_idx: int) -> float:
        if self.chunk_durations and chunk_idx < len(self.chunk_durations):
            return self.chunk_durations[chunk_idx]
        return 2.0

    def has_substantive_caption(self, chunk_idx: int, min_chars: int = 10) -> bool:
        if chunk_idx >= len(self.chunk_captions):
            return False
        caption = self.chunk_captions[chunk_idx].strip().lower()
        if len(caption) < min_chars:
            return False
        silence_keywords = {
            "silence",
            "silent",
            "quiet",
            "no sound",
            "background noise",
            "noise only",
        }
        return not any(keyword in caption for keyword in silence_keywords)

    def iter_assistant_events(self) -> List[Dict[str, Any]]:
        return [event for event in self.rollout_events if str(event.get("kind", "")).startswith("assistant_")]

    def __repr__(self) -> str:
        return (
            f"PCoTEpisode(audio_id={self.audio_id!r}, n_chunks={self.n_chunks}, "
            f"n_thinks={len(self.thinks)}, "
            f"answer={self.answer!r}, gt_answer={self.gt_answer!r}, "
            f"difficulty_bucket={self.difficulty_metadata.get('difficulty_bucket', '')!r}, "
            f"has_controller_metadata={bool(self.controller_metadata)})"
        )
