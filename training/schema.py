"""
Streaming rollout schema for wait-think-answer DAPO training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from rewards.episode import PCoTEpisode


@dataclass
class StreamingStep:
    chunk_index: int
    turn_type: str = "think"  # think | wait | answer
    audio_chunk_path: str = ""
    audio_window_paths: List[str] = field(default_factory=list)
    audio_window_span: Dict[str, Any] = field(default_factory=dict)
    prompt_text: str = ""
    think: str = ""
    predict: str = ""
    answer: str = ""
    raw_output: str = ""
    normalized_output: str = ""
    timing: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "turn_type": self.turn_type,
            "audio_chunk_path": self.audio_chunk_path,
            "audio_window_paths": list(self.audio_window_paths),
            "audio_window_span": dict(self.audio_window_span),
            "prompt_text": self.prompt_text,
            "think": self.think,
            "predict": self.predict,
            "answer": self.answer,
            "raw_output": self.raw_output,
            "normalized_output": self.normalized_output,
            "timing": dict(self.timing),
        }


@dataclass
class StreamingRollout:
    """
    One complete sampled trajectory for a single audio episode.

    Phase 1:
      - think turns per chunk
      - one final answer turn
    Phase 3:
      - think turns may additionally carry a nested <predict> segment
    """

    audio_id: str
    question: str
    thinks: List[str]
    answer: str
    predicts: List[str] = field(default_factory=list)
    steps: List[StreamingStep] = field(default_factory=list)
    raw_sequence: str = ""
    backend_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _controller_output_for_step(step: StreamingStep) -> str:
        normalized = str(step.normalized_output or "").strip()
        if normalized:
            return normalized
        return str(step.raw_output or "").strip()

    def validate_phase(self, phase: int) -> None:
        if phase < 3 and any(p.strip() for p in self.predicts):
            raise ValueError("Phase {} rollout must not contain predict slots".format(phase))
        if self.predicts and len(self.predicts) != len(self.thinks):
            raise ValueError("predict list length must match think list length")

    def build_raw_sequence(self) -> str:
        if self.steps:
            parts = []
            for step in self.steps:
                if step.turn_type not in {"think", "wait", "answer"}:
                    continue
                rendered = self._controller_output_for_step(step)
                if rendered:
                    parts.append(rendered)
            return "".join(parts)

        parts = []
        n_updates = max(len(self.thinks), len(self.predicts))
        for idx in range(n_updates):
            think = self.thinks[idx].strip() if idx < len(self.thinks) else ""
            predict = self.predicts[idx].strip() if idx < len(self.predicts) else ""
            if think:
                parts.append("<think>{}</think>".format(think))
            else:
                parts.append("<wait/>")
            if predict.strip():
                parts.append("<predict>{}</predict>".format(predict.strip()))
        parts.append("<answer>{}</answer>".format(self.answer.strip()))
        return "".join(parts)

    def ensure_raw_sequence(self) -> None:
        if not self.raw_sequence:
            self.raw_sequence = self.build_raw_sequence()

    def _assistant_event_for_step(self, step: StreamingStep) -> Dict[str, Any]:
        controller_output = self._controller_output_for_step(step)
        event: Dict[str, Any] = {
            "chunk_index": int(step.chunk_index),
            "raw_output": controller_output,
            "model_raw_output": step.raw_output,
            "normalized_output": step.normalized_output,
        }
        timing = dict(step.timing or {})
        if timing:
            event["timing"] = timing

        if step.turn_type == "think":
            event["kind"] = "assistant_think"
            event["think"] = step.think
            event["predict"] = step.predict
            if bool(timing.get("is_final_think")):
                event["is_final_think"] = True
        elif step.turn_type == "wait":
            event["kind"] = "assistant_wait"
        elif step.turn_type == "answer":
            event["kind"] = "assistant_answer"
            event["answer"] = step.answer or self.answer
        return event

    def build_rollout_events(self, sample) -> List[Dict[str, Any]]:
        rollout_metadata = dict(self.metadata or {})
        sample_metadata = dict(getattr(sample, "controller_metadata", {}) or {})
        if "question_visible_from_text" in rollout_metadata:
            question_visible_from_text = bool(rollout_metadata.get("question_visible_from_text"))
        elif "question_visible_from_text" in sample_metadata:
            question_visible_from_text = bool(sample_metadata.get("question_visible_from_text"))
        elif "question_visible_from_chunk_1" in rollout_metadata:
            question_visible_from_text = bool(rollout_metadata.get("question_visible_from_chunk_1"))
        elif "question_visible_from_chunk_1" in sample_metadata:
            question_visible_from_text = bool(sample_metadata.get("question_visible_from_chunk_1"))
        else:
            question_visible_from_text = False
        total_chunks = len(sample.audio_chunk_paths)

        def _user_audio_event(chunk_index: int) -> Dict[str, Any]:
            return {
                "kind": "user_audio",
                "chunk_index": int(chunk_index),
                "audio_chunk_path": sample.audio_chunk_paths[chunk_index],
                "question_visible": bool(question_visible_from_text and chunk_index == 0),
            }

        if self.steps:
            events: List[Dict[str, Any]] = []
            next_user_chunk = 0
            answer_seen = False
            last_valid_chunk = max(0, total_chunks - 1)

            for step in self.steps:
                target_chunk = min(max(int(step.chunk_index), 0), last_valid_chunk)
                while next_user_chunk <= target_chunk and next_user_chunk < total_chunks:
                    events.append(_user_audio_event(next_user_chunk))
                    next_user_chunk += 1

                assistant_event = self._assistant_event_for_step(step)
                if assistant_event:
                    events.append(assistant_event)
                    if str(assistant_event.get("kind", "")).strip().lower() == "assistant_answer":
                        answer_seen = True
                        break

            if not answer_seen:
                while next_user_chunk < total_chunks:
                    events.append(_user_audio_event(next_user_chunk))
                    next_user_chunk += 1
                if self.answer.strip():
                    events.append(
                        {
                            "kind": "assistant_answer",
                            "chunk_index": max(0, total_chunks - 1),
                            "raw_output": "<answer>{}</answer>".format(self.answer.strip()),
                            "normalized_output": "<answer>{}</answer>".format(self.answer.strip()),
                            "answer": self.answer,
                        }
                    )
            return events

        think_steps = {}
        wait_steps = {}
        answer_steps = []
        for step in self.steps:
            if step.turn_type == "think":
                think_steps[step.chunk_index] = step
            elif step.turn_type == "wait":
                wait_steps[step.chunk_index] = step
            elif step.turn_type == "answer":
                answer_steps.append(step)

        events: List[Dict[str, Any]] = []
        for idx, chunk_path in enumerate(sample.audio_chunk_paths):
            events.append(
                {
                    "kind": "user_audio",
                    "chunk_index": idx,
                    "audio_chunk_path": chunk_path,
                    "question_visible": bool(question_visible_from_text and idx == 0),
                }
            )
            step = think_steps.get(idx)
            if step is not None:
                events.append(self._assistant_event_for_step(step))
                continue
            wait_step = wait_steps.get(idx)
            if wait_step is not None:
                events.append(self._assistant_event_for_step(wait_step))

        if answer_steps:
            events.append(self._assistant_event_for_step(answer_steps[-1]))
        elif self.answer.strip():
            events.append(
                {
                    "kind": "assistant_answer",
                    "chunk_index": sample.n_chunks - 1,
                    "raw_output": "<answer>{}</answer>".format(self.answer.strip()),
                    "normalized_output": "<answer>{}</answer>".format(self.answer.strip()),
                    "answer": self.answer,
                }
            )

        return events

    def to_episode(self, sample) -> PCoTEpisode:
        self.ensure_raw_sequence()
        tick_seconds = 2.0
        metadata = dict(getattr(sample, "controller_metadata", {}) or {})
        rollout_metadata = dict(self.metadata or {})
        try:
            tick_seconds = float(metadata.get("ingest_grid_seconds", tick_seconds))
        except Exception:
            tick_seconds = 2.0
        controller_metadata = dict(metadata)
        if rollout_metadata:
            controller_metadata["policy_rollout_metadata"] = rollout_metadata
            rollout_timing = dict(rollout_metadata.get("timing") or {})
            if not rollout_timing:
                for step in reversed(self.steps):
                    if dict(step.timing or {}):
                        rollout_timing.update(dict(step.timing or {}))
                        break
            if isinstance(rollout_timing, dict):
                merged_timing = dict(controller_metadata.get("timing", {}) or {})
                merged_timing.update(rollout_timing)
                controller_metadata["timing"] = merged_timing
                for key in (
                    "controller_total_wall_clock_sec",
                    "answer_generation_wall_clock_sec",
                    "post_eof_total_wall_clock_sec",
                    "final_think_generation_wall_clock_sec",
                    "final_think_token_count",
                    "text_first_token_wall_clock_seconds",
                    "effective_text_first_token_seconds",
                    "text_streaming_supported",
                    "response_onset_seconds",
                    "effective_response_onset_seconds",
                ):
                    if key in rollout_timing and key not in controller_metadata:
                        controller_metadata[key] = rollout_timing[key]
        return PCoTEpisode(
            audio_chunk_paths=sample.audio_chunk_paths,
            question=sample.question,
            gt_answer=sample.gt_answer,
            chunk_captions=sample.chunk_captions,
            audio_id=sample.audio_id,
            thinks=list(self.thinks),
            answer=self.answer,
            raw_sequence=self.raw_sequence,
            rollout_events=self.build_rollout_events(sample),
            chunk_durations=[tick_seconds] * sample.n_chunks,
            difficulty_metadata=dict(getattr(sample, "difficulty_metadata", {}) or {}),
            controller_metadata=controller_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        metadata = dict(self.metadata)
        if "timing" not in metadata:
            for step in reversed(self.steps):
                if dict(step.timing or {}):
                    metadata["timing"] = dict(step.timing)
                    break
        return {
            "audio_id": self.audio_id,
            "question": self.question,
            "thinks": list(self.thinks),
            "predicts": list(self.predicts),
            "answer": self.answer,
            "raw_sequence": self.raw_sequence or self.build_raw_sequence(),
            "backend_name": self.backend_name,
            "metadata": metadata,
            "timing": dict(metadata.get("timing") or {}),
            "steps": [step.to_dict() for step in self.steps],
        }
