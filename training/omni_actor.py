"""
Training-side helpers for building Omni turn-level supervision examples.

These utilities mirror the rollout interaction contract but stay independent
from the external vLLM server. They are the bridge between sampled streaming
trajectories and a future trainable Qwen2.5-Omni actor.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .audio_io import load_mono_audio
from .policy import (
    build_omni_answer_prompt,
    build_omni_chunk_prompt,
    build_omni_final_answer_after_think_prompt,
    build_omni_system_prompt,
)

SAMPLE_RATE = 16000
MIN_QWEN_AUDIO_WINDOW_SEC = 2.0
_QWEN_SYSTEM_PROMPT_WARNING_PREFIX = "System prompt modified, audio output may not work as expected."


@dataclass
class OmniTurnExample:
    turn_index: int
    turn_type: str
    messages: List[Dict[str, Any]]
    chunk_index: int = -1
    is_final_think: bool = False
    prompt_index: int = 0
    rollout_index: int = 0
    audio_paths: List[str] = field(default_factory=list)
    audio_array: Optional[Any] = None
    target_text: str = ""

    def load_audio_arrays(self, sampling_rate: int = 16000) -> List[Any]:
        if self.audio_array is not None:
            return [self.audio_array]
        return load_audio_arrays(self.audio_paths, sampling_rate=sampling_rate)


def load_audio_arrays(
    audio_paths: Sequence[str],
    sampling_rate: int = 16000,
    cache: Optional[Dict[Tuple[str, int], Any]] = None,
) -> List[Any]:
    audio_arrays: List[Any] = []
    for path_str in audio_paths:
        cache_key = (str(path_str), int(sampling_rate))
        if cache is not None and cache_key in cache:
            audio_arrays.append(cache[cache_key])
            continue
        audio = load_mono_audio(path_str, sampling_rate=sampling_rate)
        if cache is not None:
            cache[cache_key] = audio
        audio_arrays.append(audio)
    return audio_arrays


def _concat_audio_window_arrays(
    audio_arrays: Sequence[Any],
    *,
    sampling_rate: int = SAMPLE_RATE,
    min_duration_sec: float = MIN_QWEN_AUDIO_WINDOW_SEC,
) -> Any:
    if not audio_arrays:
        target_samples = max(1, int(round(float(min_duration_sec) * float(sampling_rate))))
        return np.zeros((target_samples,), dtype=np.float32)

    merged = np.concatenate([np.asarray(array, dtype=np.float32) for array in audio_arrays], axis=0)
    target_samples = max(1, int(round(float(min_duration_sec) * float(sampling_rate))))
    if int(merged.shape[0]) < target_samples:
        pad = target_samples - int(merged.shape[0])
        merged = np.pad(merged, (pad, 0), mode="constant")
    return np.asarray(merged, dtype=np.float32)


def _system_message(question_visible: bool = False) -> Dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": build_omni_system_prompt(question_visible=question_visible)}],
    }


def _question_visible_from_rollout_or_sample(sample, rollout) -> bool:
    return False


def _user_chunk_message(prompt_text: str, chunk_path: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": chunk_path,
                    "format": "wav",
                },
            },
        ],
    }


def _user_answer_message(prompt_text: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": prompt_text}],
    }


def _assistant_message(target_text: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": target_text}],
    }


def _full_prefix_audio_paths_for_step(sample, step) -> List[str]:
    explicit_paths = list(getattr(step, "audio_window_paths", []) or [])
    if explicit_paths:
        return explicit_paths

    sample_paths = list(getattr(sample, "audio_chunk_paths", []) or [])
    try:
        chunk_index = int(getattr(step, "chunk_index", -1))
    except Exception:
        chunk_index = -1
    if sample_paths and chunk_index >= 0:
        end = min(chunk_index + 1, len(sample_paths))
        if end > 0:
            return list(sample_paths[:end])

    audio_chunk_path = str(getattr(step, "audio_chunk_path", "") or "").strip()
    if audio_chunk_path:
        return [audio_chunk_path]
    return []


def _append_running_observations(prompt_text: str, previous_thinks: Sequence[str]) -> str:
    text = str(prompt_text or "")
    if "Running observations so far:" in text:
        return text
    visible = [str(think or "").strip() for think in previous_thinks if str(think or "").strip()]
    if visible:
        observations = "\n".join(
            "[{}] <think>{}</think>".format(index + 1, think)
            for index, think in enumerate(visible)
        )
    else:
        observations = "(none)"
    return "{}\n\nRunning observations so far:\n{}".format(text, observations)


def _normalize_chat_template_text(rendered: Any) -> str:
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, list):
        if len(rendered) != 1:
            raise ValueError("Expected a single rendered chat template item, got {}".format(len(rendered)))
        return str(rendered[0])
    return str(rendered)


class _SuppressQwenSystemPromptWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.getMessage()).startswith(_QWEN_SYSTEM_PROMPT_WARNING_PREFIX)


@contextmanager
def _suppress_qwen_system_prompt_warning():
    root_logger = logging.getLogger()
    warning_filter = _SuppressQwenSystemPromptWarning()
    root_logger.addFilter(warning_filter)
    try:
        yield
    finally:
        root_logger.removeFilter(warning_filter)


def _apply_chat_template(processor, messages, *, tokenize: bool, add_generation_prompt: bool):
    # Qwen2.5-Omni emits a root-level warning whenever the system prompt differs
    # from its TTS default prompt. That warning is expected for our controller
    # prompt and otherwise floods smoke/full training logs.
    with _suppress_qwen_system_prompt_warning():
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


def build_turn_training_examples(
    sample,
    rollout,
    *,
    prompt_index: int = 0,
    rollout_index: int = 0,
) -> List[OmniTurnExample]:
    """
    Build per-turn training examples that mirror the rollout dialogue history.
    """
    examples: List[OmniTurnExample] = []
    question_visible = _question_visible_from_rollout_or_sample(sample, rollout)
    previous_visible_thinks: List[str] = []

    controller_steps = [step for step in rollout.steps if step.turn_type in {"think", "wait"}]
    for turn_index, step in enumerate(controller_steps):
        prompt_text = step.prompt_text or _append_running_observations(
            build_omni_chunk_prompt(
                sample.question,
                chunk_index=step.chunk_index,
                total_chunks=sample.n_chunks,
                question_visible=question_visible,
            ),
            previous_visible_thinks,
        )
        user_message = _user_chunk_message(prompt_text, step.audio_chunk_path)
        window_paths = _full_prefix_audio_paths_for_step(sample, step)
        audio_array = _concat_audio_window_arrays(
            load_audio_arrays(window_paths, sampling_rate=SAMPLE_RATE) if window_paths else [],
            sampling_rate=SAMPLE_RATE,
        )
        if step.turn_type == "wait":
            target_text = step.normalized_output or "<wait/>"
        else:
            timing = dict(step.timing or {})
            if bool(timing.get("is_final_think")) and bool(timing.get("final_think_fallback_used")):
                # Keep the sampled raw protocol violation as the policy target.
                # The fallback text is only context for continuing the rollout;
                # training on it would teach the model to emit placeholders.
                target_text = step.raw_output or step.normalized_output or "<think>{}</think>".format(step.think)
            else:
                target_text = step.normalized_output or "<think>{}</think>".format(step.think)
        messages = [_system_message(question_visible=question_visible), user_message, _assistant_message(target_text)]
        examples.append(
            OmniTurnExample(
                prompt_index=prompt_index,
                rollout_index=rollout_index,
                turn_index=turn_index,
                turn_type=step.turn_type,
                chunk_index=step.chunk_index,
                is_final_think=bool(dict(step.timing or {}).get("is_final_think")),
                messages=messages,
                audio_paths=window_paths,
                audio_array=audio_array,
                target_text=target_text,
            )
        )
        if step.turn_type == "think" and str(step.think or "").strip():
            previous_visible_thinks.append(str(step.think).strip())

    answer_steps = [step for step in rollout.steps if step.turn_type == "answer"]
    if answer_steps:
        answer_step = answer_steps[-1]
        has_final_think = any(
            step.turn_type == "think" and bool(dict(step.timing or {}).get("is_final_think"))
            for step in rollout.steps
        )
        prompt_text = answer_step.prompt_text or (
            build_omni_final_answer_after_think_prompt(
                sample.question,
                sample.n_chunks,
                question_visible=question_visible,
            )
            if has_final_think
            else build_omni_answer_prompt(
                sample.question,
                sample.n_chunks,
                question_visible=question_visible,
            )
        )
        user_message = _user_chunk_message(prompt_text, answer_step.audio_chunk_path)
        window_paths = _full_prefix_audio_paths_for_step(sample, answer_step)
        audio_array = _concat_audio_window_arrays(
            load_audio_arrays(window_paths, sampling_rate=SAMPLE_RATE) if window_paths else [],
            sampling_rate=SAMPLE_RATE,
        )
        target_text = answer_step.normalized_output or "<answer>{}</answer>".format(rollout.answer)
        examples.append(
            OmniTurnExample(
                prompt_index=prompt_index,
                rollout_index=rollout_index,
                turn_index=len(examples),
                turn_type="answer",
                chunk_index=max(0, sample.n_chunks - 1),
                messages=[
                    _system_message(question_visible=question_visible),
                    user_message,
                    _assistant_message(target_text),
                ],
                audio_paths=window_paths,
                audio_array=audio_array,
                target_text=target_text,
            )
        )

    return examples


def prepare_turn_model_inputs(
    processor,
    turn_example: OmniTurnExample,
    sampling_rate: int = 16000,
    audio_cache: Optional[Dict[Tuple[str, int], Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert one turn example into model inputs with labels that supervise only
    the final assistant message.
    """
    prompt_messages = turn_example.messages[:-1]
    full_messages = turn_example.messages
    if turn_example.audio_array is not None:
        audio_arrays = [turn_example.audio_array]
    else:
        audio_arrays = load_audio_arrays(
            turn_example.audio_paths,
            sampling_rate=sampling_rate,
            cache=audio_cache,
        )

    prompt_text = _normalize_chat_template_text(
        _apply_chat_template(
            processor,
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    full_text = _normalize_chat_template_text(
        _apply_chat_template(
            processor,
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    prompt_inputs = processor(
        text=prompt_text,
        audio=audio_arrays if audio_arrays else None,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=full_text,
        audio=audio_arrays if audio_arrays else None,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_len = int(full_inputs["input_ids"].shape[1])
    if prompt_len >= full_len:
        raise ValueError(
            "Prompt length {} must be smaller than full length {} for turn {}".format(
                prompt_len,
                full_len,
                turn_example.turn_index,
            )
        )

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        labels = labels.masked_fill(full_inputs["attention_mask"].eq(0), -100)
    full_inputs["labels"] = labels
    return full_inputs


def prepare_turn_model_inputs_batch(
    processor,
    turn_examples: Sequence[OmniTurnExample],
    sampling_rate: int = 16000,
    audio_cache: Optional[Dict[Tuple[str, int], Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a batch of turn examples into model inputs with labels that
    supervise only the final assistant message of each example.
    """
    if not turn_examples:
        raise ValueError("turn_examples must be non-empty")

    prompt_texts: List[str] = []
    full_texts: List[str] = []
    prompt_audios: List[Any] = []
    full_audios: List[Any] = []

    for turn_example in turn_examples:
        prompt_messages = turn_example.messages[:-1]
        full_messages = turn_example.messages
        if turn_example.audio_array is not None:
            audio_arrays = [turn_example.audio_array]
        else:
            audio_arrays = load_audio_arrays(
                turn_example.audio_paths,
                sampling_rate=sampling_rate,
                cache=audio_cache,
            )

        prompt_texts.append(
            _normalize_chat_template_text(
                _apply_chat_template(
                    processor,
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        )
        full_texts.append(
            _normalize_chat_template_text(
                _apply_chat_template(
                    processor,
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        )
        prompt_audios.extend(audio_arrays)
        full_audios.extend(audio_arrays)

    prompt_inputs = processor(
        text=prompt_texts,
        audio=prompt_audios if prompt_audios else None,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    full_inputs = processor(
        text=full_texts,
        audio=full_audios if full_audios else None,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    if "attention_mask" in prompt_inputs:
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    else:
        prompt_lens = [int(prompt_inputs["input_ids"].shape[1])] * len(turn_examples)

    if "attention_mask" in full_inputs:
        full_lens = full_inputs["attention_mask"].sum(dim=1).tolist()
    else:
        full_lens = [int(full_inputs["input_ids"].shape[1])] * len(turn_examples)

    labels = full_inputs["input_ids"].clone()
    if "attention_mask" in full_inputs:
        labels = labels.masked_fill(full_inputs["attention_mask"].eq(0), -100)

    for row_idx, (prompt_len, full_len) in enumerate(zip(prompt_lens, full_lens)):
        if int(prompt_len) >= int(full_len):
            raise ValueError(
                "Prompt length {} must be smaller than full length {} for batched turn {}".format(
                    prompt_len,
                    full_len,
                    row_idx,
                )
            )
        labels[row_idx, : int(prompt_len)] = -100

    full_inputs["labels"] = labels
    return full_inputs
