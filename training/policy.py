"""
Policy backend abstractions for wait-think-answer rollout.
"""

import abc
import base64
import concurrent.futures
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from prompts.policy_prompts import (
    DEFAULT_FINAL_POLICY_PROMPT_VERSION,
    DEFAULT_POLICY_PROMPT_VERSION,
    build_omni_answer_instruction as _build_policy_answer_instruction,
    build_omni_answer_prompt as _build_policy_answer_prompt,
    build_omni_chunk_prompt as _build_policy_chunk_prompt,
    build_omni_final_answer_after_think_prompt as _build_policy_final_answer_after_think_prompt,
    build_omni_final_think_prompt as _build_policy_final_think_prompt,
    build_omni_system_prompt as _build_policy_system_prompt,
)
from rewards.reward_sync import estimate_token_count

from .checkpointing import CheckpointArtifact
from .audio_io import load_mono_audio
from .schema import StreamingRollout, StreamingStep
from .windowing import (
    MIN_QWEN_AUDIO_WINDOW_SEC,
    WINDOW_MODE_FULL_PREFIX,
    WINDOW_MODE_SINCE_LAST_THINK,
    build_controller_window_spec,
)

SAMPLE_RATE = 16000
QWEN_OUTPUT_AUDIO_SAMPLE_RATE = 24000
DEFAULT_AUDIO_RESPONSE_ONSET_PRIOR_SECONDS = 0.30


def _parse_mcq_options(question: str) -> List[str]:
    if "Options:" in question:
        option_text = question.split("Options:", 1)[1]
        parts = re.split(r"\s+[A-Z]\)\s*", " " + option_text.strip())
        return [part.strip(" ,.;'\"") for part in parts if part.strip(" ,.;'\"")]

    bracket_match = re.search(r"following options:\s*(\[[^\]]+\])", question, flags=re.IGNORECASE)
    if bracket_match:
        raw = bracket_match.group(1)
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item).strip(" ,.;'\"") for item in parsed if str(item).strip(" ,.;'\"")]
        except json.JSONDecodeError:
            pass

    return []


def _infer_answer_prompt_metadata(question: str) -> tuple[str, List[str]]:
    choices = _parse_mcq_options(question)
    evaluation_type = "multiple_choice" if choices else ""
    return evaluation_type, choices


def _question_stem_for_think(question: str) -> str:
    stem = str(question or "").strip()
    stem = re.sub(r"Please choose the answer from the following options:.*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"Options:.*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"Output the final answer in\s*<answer>.*", "", stem, flags=re.IGNORECASE)
    return stem.strip(" \n\t.:")


def _normalize_option_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip().lower())
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def _looks_like_option_guess(text: str, question: str) -> bool:
    normalized = _normalize_option_text(text)
    if not normalized:
        return False
    if normalized in {"...", "answer", "unknown"}:
        return True
    for option in _parse_mcq_options(question):
        if normalized == _normalize_option_text(option):
            return True
    return False


def _mask_answer_leak(text: str, answer: str) -> str:
    if not text or not answer:
        return text

    masked = text
    for token in sorted(set(answer.lower().split()), key=len, reverse=True):
        if len(token) < 3:
            continue
        masked = re.sub(re.escape(token), "source", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\bsource source\b", "source", masked, flags=re.IGNORECASE)
    return masked.strip()


def _truncate_for_think(text: str, max_words: int = 8) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words]).strip(" ,.;:-")


def _collapse_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_text_from_delta(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
                continue
            if "text" in item:
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _decode_audio_blob_string(raw: Any) -> Optional[bytes]:
    if not isinstance(raw, str):
        return None
    payload = raw.strip()
    if not payload:
        return None
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1].strip()
    try:
        return base64.b64decode(payload)
    except Exception:
        return None


def _extract_audio_blob_from_mapping(mapping: Any) -> Optional[bytes]:
    if not isinstance(mapping, dict):
        return None
    for key in ("data", "b64_json", "b64", "audio_data"):
        blob = _decode_audio_blob_string(mapping.get(key))
        if blob:
            return blob
    audio_obj = mapping.get("audio")
    if isinstance(audio_obj, dict):
        for key in ("data", "b64_json", "b64", "audio_data"):
            blob = _decode_audio_blob_string(audio_obj.get(key))
            if blob:
                return blob
    return None


def _extract_response_audio_blob_deep(node: Any, depth: int = 0) -> Optional[bytes]:
    if depth > 6 or node is None:
        return None
    if isinstance(node, dict):
        blob = _extract_audio_blob_from_mapping(node)
        if blob:
            return blob
        for key in (
            "audio",
            "output_audio",
            "message",
            "content",
            "choices",
            "multimodal_output",
            "response",
            "delta",
        ):
            if key in node:
                blob = _extract_response_audio_blob_deep(node.get(key), depth + 1)
                if blob:
                    return blob
        for value in node.values():
            blob = _extract_response_audio_blob_deep(value, depth + 1)
            if blob:
                return blob
        return None
    if isinstance(node, list):
        for item in node:
            blob = _extract_response_audio_blob_deep(item, depth + 1)
            if blob:
                return blob
    return None


def _extract_response_audio_blob(data: Dict[str, Any]) -> Optional[bytes]:
    choices = list(data.get("choices") or [])
    for choice in choices:
        blob = _extract_audio_blob_from_mapping(choice)
        if blob:
            return blob
        message = dict(choice.get("message") or {})
        blob = _extract_audio_blob_from_mapping(message)
        if blob:
            return blob
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                blob = _extract_audio_blob_from_mapping(item)
                if blob:
                    return blob
    for key in ("audio", "message", "multimodal_output"):
        blob = _extract_audio_blob_from_mapping(data.get(key))
        if blob:
            return blob
    return _extract_response_audio_blob_deep(data)


def _maybe_dump_missing_audio_response(payload: Dict[str, Any]) -> None:
    debug_dir = str(os.environ.get("WTA_AUDIO_RESPONSE_DEBUG_DIR") or "").strip()
    if not debug_dir:
        return
    try:
        root = Path(debug_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / "missing_audio_response_{}.json".format(int(time.time() * 1000))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


def _decode_audio_blob(audio_blob: bytes) -> tuple[Optional[np.ndarray], Optional[int]]:
    if not audio_blob:
        return None, None
    try:
        import soundfile as sf

        array, sample_rate = sf.read(io.BytesIO(audio_blob), dtype="float32")
        waveform = np.asarray(array, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform, int(sample_rate)
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(audio_blob), "rb") as handle:
            sample_rate = int(handle.getframerate())
            n_channels = int(handle.getnchannels())
            sample_width = int(handle.getsampwidth())
            frames = handle.readframes(handle.getnframes())
        if sample_width == 2:
            waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            waveform = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None, None
        if n_channels > 1:
            waveform = waveform.reshape(-1, n_channels).mean(axis=1)
        return waveform.astype(np.float32), sample_rate
    except Exception:
        return None, None


def _estimate_response_onset_seconds(
    waveform: Optional[np.ndarray],
    sample_rate: Optional[int],
) -> Optional[float]:
    if waveform is None or sample_rate is None or int(sample_rate) <= 0:
        return None
    if waveform.size == 0:
        return None

    try:
        import torch
        import torchaudio
    except Exception:
        torch = None
        torchaudio = None

    waveform_16k = None
    if torch is not None and torchaudio is not None:
        try:
            tensor = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
            if int(sample_rate) != SAMPLE_RATE:
                tensor = torchaudio.functional.resample(
                    tensor,
                    orig_freq=int(sample_rate),
                    new_freq=SAMPLE_RATE,
                )
            waveform_16k = tensor
        except Exception:
            waveform_16k = None

    if waveform_16k is not None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            vad_model = load_silero_vad()
            timestamps = get_speech_timestamps(
                waveform_16k,
                vad_model,
                sampling_rate=SAMPLE_RATE,
                return_seconds=False,
            )
            if timestamps:
                return float(int(timestamps[0]["start"])) / float(SAMPLE_RATE)
        except Exception:
            pass

        try:
            array = waveform_16k.detach().cpu().numpy().astype(np.float32)
        except Exception:
            array = None
        if array is not None and array.size > 0:
            max_abs = float(np.max(np.abs(array)))
            if max_abs <= 1e-8:
                return None
            threshold = max(0.015, 0.08 * max_abs)
            hit = np.flatnonzero(np.abs(array) >= threshold)
            if hit.size == 0:
                return None
            return float(int(hit[0])) / float(SAMPLE_RATE)

    raw = np.asarray(waveform, dtype=np.float32)
    max_abs = float(np.max(np.abs(raw)))
    if max_abs <= 1e-8:
        return None
    threshold = max(0.015, 0.08 * max_abs)
    hit = np.flatnonzero(np.abs(raw) >= threshold)
    if hit.size == 0:
        return None
    return float(int(hit[0])) / float(int(sample_rate))


def _resolve_chat_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def _extract_tag(raw_text: str, tag: str) -> List[str]:
    pattern = re.compile(r"<{0}>(.*?)</{0}>".format(tag), re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(raw_text)]


_WAIT_TAG_PATTERN = re.compile(r"<wait\s*/>", re.IGNORECASE)


def _should_force_wait_for_prefix(window_span: Dict[str, Any], force_wait_before_sec: float) -> bool:
    try:
        threshold = float(force_wait_before_sec)
    except Exception:
        return False
    if threshold <= 0.0:
        return False
    try:
        end_sec = float((window_span or {}).get("end_sec"))
    except Exception:
        return False
    return end_sec < threshold


def _normalize_controller_step_output(
    raw_text: str,
    *,
    question: str,
    allow_predict: bool = False,
) -> tuple[str, str, str, str]:
    text = str(raw_text or "").strip()
    if _WAIT_TAG_PATTERN.fullmatch(text):
        return "", "", "wait", "<wait/>"

    think_matches = _extract_tag(text, "think")
    think_body = think_matches[0] if think_matches else ""

    predict_matches = _extract_tag(think_body, "predict")
    predict = predict_matches[0] if (allow_predict and predict_matches) else ""

    body_without_predict = re.sub(r"<predict>.*?</predict>", "", think_body, flags=re.DOTALL).strip()
    if body_without_predict:
        body_without_predict = re.sub(r"\s+", " ", body_without_predict).strip()

    if not body_without_predict and not think_matches:
        text_without_xml = re.sub(r"<answer>.*?</answer>", " ", text, flags=re.DOTALL)
        text_without_xml = re.sub(r"<wait\s*/>", " ", text_without_xml, flags=re.IGNORECASE)
        text_without_xml = re.sub(r"</?(think|memory|predict|answer)>", " ", text_without_xml, flags=re.IGNORECASE)
        text_without_xml = re.sub(r"\s+", " ", text_without_xml).strip()
        if text_without_xml:
            body_without_predict = text_without_xml

    think = sanitize_streaming_think(body_without_predict, question=question)
    if think:
        return think, predict.strip(), "think", "<think>{}</think>".format(think)
    return "", "", "wait", "<wait/>"


def _normalize_think_output(raw_text: str, allow_predict: bool) -> tuple[str, str, str]:
    think, predict, _turn_type, normalized = _normalize_controller_step_output(
        raw_text,
        question="",
        allow_predict=allow_predict,
    )
    return think, predict, normalized


def _normalize_answer_output(raw_text: str) -> tuple[str, str]:
    text = str(raw_text or "").strip()
    answer_matches = _extract_tag(text, "answer")
    answer = answer_matches[0] if answer_matches else text
    answer = re.sub(r"\s+", " ", answer.strip())
    return answer, "<answer>{}</answer>".format(answer)


def build_omni_system_prompt(
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
    question_visible: bool = False,
) -> str:
    return _build_policy_system_prompt(version, question_visible=question_visible)


def _avoid_deprecated_memory_xml_tag(prompt: str) -> str:
    return str(prompt or "").replace("<memory>", "memory tag")


def build_omni_chunk_prompt(
    question: str,
    chunk_index: int,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
) -> str:
    return _avoid_deprecated_memory_xml_tag(
        _build_policy_chunk_prompt(
            question,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            question_visible=question_visible,
            version=version,
        )
    )


def build_omni_answer_prompt(
    question: str,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
    evaluation_type: str = "",
    choices: Optional[List[str]] = None,
    xml_answer_format: bool = True,
) -> str:
    return _avoid_deprecated_memory_xml_tag(
        _build_policy_answer_prompt(
            question,
            total_chunks=total_chunks,
            question_visible=question_visible,
            version=version,
            evaluation_type=evaluation_type,
            choices=choices,
            xml_answer_format=xml_answer_format,
        )
    )


def build_omni_final_think_prompt(
    question: str,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
) -> str:
    return _avoid_deprecated_memory_xml_tag(
        _build_policy_final_think_prompt(
            question,
            total_chunks=total_chunks,
            question_visible=question_visible,
            version=version,
        )
    )


def build_omni_final_answer_after_think_prompt(
    question: str,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
    evaluation_type: str = "",
    choices: Optional[List[str]] = None,
    xml_answer_format: bool = True,
) -> str:
    return _avoid_deprecated_memory_xml_tag(
        _build_policy_final_answer_after_think_prompt(
            question,
            total_chunks=total_chunks,
            question_visible=question_visible,
            version=version,
            evaluation_type=evaluation_type,
            choices=choices,
            xml_answer_format=xml_answer_format,
        )
    )


def _question_visible_from_text(sample, default: bool = False) -> bool:
    return False


def _tick_seconds(sample) -> float:
    metadata = dict(getattr(sample, "controller_metadata", {}) or {})
    try:
        return float(metadata.get("ingest_grid_seconds", 2.0))
    except Exception:
        return 2.0


def _full_prefix_window_span(sample) -> Dict[str, float]:
    last_chunk_index = max(0, int(getattr(sample, "n_chunks", 0)) - 1)
    end_sec = float(last_chunk_index + 1) * float(_tick_seconds(sample))
    return {
        "mode": WINDOW_MODE_FULL_PREFIX,
        "start_index": 0,
        "end_index": last_chunk_index,
        "start_sec": 0.0,
        "end_sec": float(end_sec),
    }


def _render_observations(previous_thinks: List[str]) -> str:
    if not previous_thinks:
        return "(none)"
    return "\n".join(
        "[{}] <think>{}</think>".format(idx + 1, str(think).strip())
        for idx, think in enumerate(previous_thinks)
    )


def _build_window_prompt(
    *,
    question: str,
    chunk_index: int,
    total_chunks: int,
    previous_thinks: List[str],
    window_span: Dict[str, float],
    audio_window_mode: str,
    version: str,
    question_visible_from_text: bool = False,
) -> str:
    base_prompt = build_omni_chunk_prompt(
        question,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        question_visible=question_visible_from_text,
        version=version,
    )
    observations = _render_observations(previous_thinks)
    if audio_window_mode == WINDOW_MODE_SINCE_LAST_THINK:
        window_note = (
            "This audio window contains the speech since the previous reasoning update, "
            "plus a short overlap for continuity."
        )
    else:
        window_note = (
            "This audio is the full speech prefix from the start up to the current boundary.\n"
            "Even though the attached audio is the full prefix, decide whether to think based on "
            "whether the newly extended prefix adds a new answer-supporting reasoning update beyond "
            "the previous visible thinks. Do not restate old prefix evidence."
        )
    insert_block = (
        "Running observations so far:\n"
        "{}\n"
        "Current audio window: {:.2f}s to {:.2f}s.\n"
        "{}"
    ).format(
        observations,
        float(window_span.get("start_sec", 0.0)),
        float(window_span.get("end_sec", 0.0)),
        window_note,
    )
    first_newline = base_prompt.find("\n")
    if first_newline < 0:
        return "{}\n{}".format(base_prompt, insert_block)
    return "{}\n{}\n{}".format(base_prompt[:first_newline], insert_block, base_prompt[first_newline + 1 :])


def _build_window_answer_prompt(
    *,
    question: str,
    total_chunks: int,
    previous_thinks: List[str],
    window_span: Dict[str, float],
    audio_window_mode: str,
    question_visible_from_text: bool = False,
    evaluation_type: str = "",
    choices: Optional[List[str]] = None,
) -> str:
    observations = _render_observations(previous_thinks)
    if audio_window_mode == WINDOW_MODE_SINCE_LAST_THINK:
        window_note = (
            "This final audio window contains the speech since the previous reasoning update, "
            "plus a short overlap for continuity."
        )
    else:
        window_note = "This final audio is the full speech prefix from the start up to AUDIO_END."
    question_block = (
        "{}\n".format(question.strip())
        if question_visible_from_text and str(question or "").strip()
        else "The spoken question and any answer options were provided only through the audio.\n"
    )
    answer_instruction = _build_policy_answer_instruction(
        xml_answer_format=True,
        evaluation_type=evaluation_type,
        choices=choices,
    )
    return (
        "AUDIO_END.\n"
        "All {} local evidence units have been heard.\n"
        "Use all answer-relevant information from all local evidence units heard so far, not only the final audio window.\n"
        "Use the earlier visible <think> updates, if any, as running memory.\n"
        "Running observations so far:\n"
        "{}\n"
        "Current audio window: {:.2f}s to {:.2f}s.\n"
        "{}\n"
        "{}"
        "Answer the spoken question from that full audio evidence and the running observations.\n"
        "{}\n"
        "Use the real answer content from the audio.\n"
        "Do not repeat the spoken question.\n"
        "Do not output <wait/> or <think>."
    ).format(
        total_chunks,
        observations,
        float(window_span.get("start_sec", 0.0)),
        float(window_span.get("end_sec", 0.0)),
        window_note,
        question_block,
        answer_instruction,
    )


def _build_window_final_think_prompt(
    *,
    question: str,
    total_chunks: int,
    previous_thinks: List[str],
    window_span: Dict[str, float],
    question_visible_from_text: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
) -> str:
    base_prompt = build_omni_final_think_prompt(
        question,
        total_chunks=total_chunks,
        question_visible=question_visible_from_text,
        version=version,
    )
    observations = _render_observations(previous_thinks)
    insert_block = (
        "Running observations so far:\n"
        "{}\n"
        "Current audio window: {:.2f}s to {:.2f}s.\n"
        "This final audio is the full speech prefix from the start up to AUDIO_END."
    ).format(
        observations,
        float(window_span.get("start_sec", 0.0)),
        float(window_span.get("end_sec", 0.0)),
    )
    first_newline = base_prompt.find("\n")
    if first_newline < 0:
        return "{}\n{}".format(base_prompt, insert_block)
    return "{}\n{}\n{}".format(base_prompt[:first_newline], insert_block, base_prompt[first_newline + 1 :])


def _build_window_final_answer_after_think_prompt(
    *,
    question: str,
    total_chunks: int,
    previous_thinks: List[str],
    window_span: Dict[str, float],
    question_visible_from_text: bool = False,
    evaluation_type: str = "",
    choices: Optional[List[str]] = None,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
) -> str:
    base_prompt = build_omni_final_answer_after_think_prompt(
        question,
        total_chunks=total_chunks,
        question_visible=question_visible_from_text,
        version=version,
        evaluation_type=evaluation_type,
        choices=choices,
        xml_answer_format=True,
    )
    observations = _render_observations(previous_thinks)
    insert_block = (
        "Running observations so far:\n"
        "{}\n"
        "Current audio window: {:.2f}s to {:.2f}s.\n"
        "This final audio is the full speech prefix from the start up to AUDIO_END."
    ).format(
        observations,
        float(window_span.get("start_sec", 0.0)),
        float(window_span.get("end_sec", 0.0)),
    )
    first_newline = base_prompt.find("\n")
    if first_newline < 0:
        return "{}\n{}".format(base_prompt, insert_block)
    return "{}\n{}\n{}".format(base_prompt[:first_newline], insert_block, base_prompt[first_newline + 1 :])


def _load_audio_array(path_str: str, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    return load_mono_audio(str(path_str), sampling_rate=sampling_rate)


def _concat_audio_arrays(audio_arrays: List[np.ndarray]) -> np.ndarray:
    if not audio_arrays:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate([np.asarray(audio, dtype=np.float32) for audio in audio_arrays], axis=0)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    q = min(max(float(q), 0.0), 1.0)
    position = q * float(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - float(lower)
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _ensure_min_audio_window(audio_array: np.ndarray, *, sampling_rate: int = SAMPLE_RATE, min_duration_sec: float = MIN_QWEN_AUDIO_WINDOW_SEC) -> np.ndarray:
    target_samples = max(1, int(round(float(min_duration_sec) * float(sampling_rate))))
    audio_array = np.asarray(audio_array, dtype=np.float32)
    if int(audio_array.shape[0]) >= target_samples:
        return audio_array
    pad = target_samples - int(audio_array.shape[0])
    return np.pad(audio_array, (pad, 0), mode="constant")


def _encode_audio_array(audio_array: np.ndarray, *, sampling_rate: int = SAMPLE_RATE) -> str:
    buffer = io.BytesIO()
    audio_array = np.asarray(audio_array, dtype=np.float32)
    clipped = np.clip(audio_array, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sampling_rate))
        handle.writeframes(pcm16.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def sanitize_streaming_think(think: str, question: str, max_words: int = 12) -> str:
    think = re.sub(r"\s+", " ", str(think or "").strip())
    if not think:
        return ""
    if _looks_like_option_guess(think, question):
        return ""
    return _truncate_for_think(think, max_words=max_words)


def sanitize_final_think_fallback(raw_text: str, question: str, max_words: int = 12) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""
    if re.search(r"<answer>.*?</answer>", text, flags=re.IGNORECASE | re.DOTALL):
        return ""
    text = re.sub(r"</?(think|predict|memory|wait|answer)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return sanitize_streaming_think(text, question=question, max_words=max_words)


def _synthesize_final_think(
    *,
    question: str,
    candidate_updates: List[str],
    fallback_texts: Optional[List[str]] = None,
) -> str:
    for candidate in reversed(list(candidate_updates or [])):
        cleaned = sanitize_streaming_think(candidate, question=question)
        if cleaned:
            return cleaned
    for fallback in reversed(list(fallback_texts or [])):
        cleaned = sanitize_streaming_think(fallback, question=question)
        if cleaned:
            return cleaned
    return "final reasoning state"


def _append_final_think_and_answer_steps(
    *,
    sample,
    steps: List[StreamingStep],
    visible_thinks: List[str],
    answer: str,
    question_visible: bool,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
) -> Dict[str, float]:
    final_window_paths = list(sample.audio_chunk_paths)
    final_window_span = _full_prefix_window_span(sample)
    evaluation_type, choices = _infer_answer_prompt_metadata(sample.question)
    fallback_updates = [
        _mask_answer_leak(_truncate_for_think(caption, max_words=12), sample.gt_answer)
        for caption in list(getattr(sample, "chunk_captions", []) or [])
        if str(caption or "").strip()
    ]
    final_think = _synthesize_final_think(
        question=sample.question,
        candidate_updates=[think for think in visible_thinks if str(think or "").strip()],
        fallback_texts=fallback_updates,
    )
    final_think_prompt = _build_window_final_think_prompt(
        question=sample.question,
        total_chunks=sample.n_chunks,
        previous_thinks=[think for think in visible_thinks if str(think or "").strip()],
        window_span=final_window_span,
        question_visible_from_text=question_visible,
        version=version,
    )
    final_think_normalized = "<think>{}</think>".format(final_think)
    final_think_token_count = int(estimate_token_count(final_think))
    final_timing = {
        "generation_wall_clock_sec": 0.0,
        "final_think_generation_wall_clock_sec": 0.0,
        "final_think_token_count": float(final_think_token_count),
        "is_final_think": True,
    }
    steps.append(
        StreamingStep(
            chunk_index=max(0, sample.n_chunks - 1),
            turn_type="think",
            audio_chunk_path=sample.audio_chunk_paths[max(0, sample.n_chunks - 1)],
            audio_window_paths=final_window_paths,
            audio_window_span=final_window_span,
            prompt_text=final_think_prompt,
            think=final_think,
            raw_output=final_think_normalized,
            normalized_output=final_think_normalized,
            timing=dict(final_timing),
        )
    )
    answer_prompt = _build_window_final_answer_after_think_prompt(
        question=sample.question,
        total_chunks=sample.n_chunks,
        previous_thinks=[think for think in visible_thinks if str(think or "").strip()] + [final_think],
        window_span=final_window_span,
        question_visible_from_text=question_visible,
        evaluation_type=evaluation_type,
        choices=choices,
        version=version,
    )
    answer_block = "<answer>{}</answer>".format(answer)
    answer_timing = {
        "generation_wall_clock_sec": 0.0,
        "answer_generation_wall_clock_sec": 0.0,
        "post_eof_total_wall_clock_sec": 0.0,
        "final_think_generation_wall_clock_sec": 0.0,
        "final_think_token_count": float(final_think_token_count),
    }
    steps.append(
        StreamingStep(
            chunk_index=max(0, sample.n_chunks - 1),
            turn_type="answer",
            audio_chunk_path=sample.audio_chunk_paths[max(0, sample.n_chunks - 1)],
            audio_window_paths=final_window_paths,
            audio_window_span=final_window_span,
            prompt_text=answer_prompt,
            answer=answer,
            raw_output=answer_block,
            normalized_output=answer_block,
            timing=dict(answer_timing),
        )
    )
    return {
        "final_think_generation_wall_clock_sec": 0.0,
        "final_think_token_count": float(final_think_token_count),
        "answer_generation_wall_clock_sec": 0.0,
        "post_eof_total_wall_clock_sec": 0.0,
    }


def _merge_update_infos(infos: List[Dict[str, Any]], backend_name: str) -> Dict[str, Any]:
    if not infos:
        return {"mode": "dry-run", "backend": backend_name, "n_prompt_groups": 0}
    if len(infos) == 1:
        merged = dict(infos[0])
        merged.setdefault("n_prompt_groups", 1)
        return merged

    merged: Dict[str, Any] = {
        "mode": infos[0].get("mode", "mixed"),
        "backend": backend_name,
        "n_prompt_groups": len(infos),
    }
    count_sum_keys = {"n_rollouts", "n_turn_examples", "n_forward_batches", "supervised_tokens"}
    metric_mean_keys = {"mean_loss", "mean_advantage", "mean_reward", "grad_norm"}

    for key in count_sum_keys:
        values = [info.get(key) for info in infos if isinstance(info.get(key), (int, float)) and not isinstance(info.get(key), bool)]
        if values:
            merged[key] = sum(values)

    for key in metric_mean_keys:
        values = [float(info[key]) for info in infos if isinstance(info.get(key), (int, float)) and not isinstance(info.get(key), bool)]
        if values:
            merged[key] = sum(values) / float(len(values))

    for key in ("actor_model", "credit_assignment", "kl_beta", "resume_checkpoint"):
        for info in infos:
            value = info.get(key)
            if value not in (None, "", []):
                merged[key] = value
                break

    return merged


class PolicyBackend(abc.ABC):
    name = "abstract"

    def start(self) -> Dict[str, Any]:
        return {"status": "noop"}

    def stop(self) -> Dict[str, Any]:
        return {"status": "noop"}

    @abc.abstractmethod
    def rollout(self, sample, rng: random.Random, phase: int = 1) -> StreamingRollout:
        raise NotImplementedError

    def rollout_group(self, sample, group_size: int, seed: int, phase: int = 1) -> List[StreamingRollout]:
        return [
            self.rollout(sample, rng=random.Random(seed + idx), phase=phase)
            for idx in range(group_size)
        ]

    def update_group(self, sample, rollouts, episodes, rewards, advantages, step_index: Optional[int] = None) -> Dict[str, Any]:
        return {"mode": "dry-run", "backend": self.name}

    def update_step(self, group_batches: List[Dict[str, Any]], step_index: Optional[int] = None) -> Dict[str, Any]:
        infos = []
        for group_batch in group_batches:
            infos.append(
                self.update_group(
                    group_batch["sample"],
                    group_batch["rollouts"],
                    group_batch.get("episodes", []),
                    group_batch["rewards"],
                    group_batch["advantages"],
                    step_index=step_index,
                )
            )
        return _merge_update_infos(infos, backend_name=self.name)

    def save_checkpoint(self, path: str, step: int, checkpoint_mode: Optional[str] = None) -> CheckpointArtifact:
        Path(path).write_text(
            '{"backend": "%s", "step": %d}\n' % (self.name, step),
            encoding="utf-8",
        )
        checkpoint_path = Path(path)
        checkpoint_dir = checkpoint_path.with_suffix("") if checkpoint_path.suffix else checkpoint_path
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {"backend": self.name, "step": step, "checkpoint_mode": checkpoint_mode or "metadata-only"},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return CheckpointArtifact(
            step=step,
            mode=str(checkpoint_mode or "metadata-only"),
            checkpoint_path=str(checkpoint_path),
            checkpoint_dir=str(checkpoint_dir),
            metadata_path=str(metadata_path),
        )

    def reload_from_checkpoint(self, artifact: CheckpointArtifact) -> Dict[str, Any]:
        return {"status": "noop", "reloadable": artifact.is_reloadable}


class TeacherPolicyBackend(PolicyBackend):
    name = "teacher"

    def rollout(self, sample, rng: random.Random, phase: int = 1) -> StreamingRollout:
        question_visible = _question_visible_from_text(sample)
        teacher_thinks = list(sample.teacher_thinks) if sample.teacher_thinks else [""] * sample.n_chunks
        thinks: List[str] = []
        steps = []
        for idx, teacher_think in enumerate(teacher_thinks):
            think = sanitize_streaming_think(teacher_think, question=sample.question)
            if not think:
                fallback = _truncate_for_think(sample.chunk_captions[idx], max_words=8)
                if idx < sample.n_chunks - 1:
                    fallback = _mask_answer_leak(fallback, sample.gt_answer)
                think = sanitize_streaming_think(fallback, question=sample.question) or "evidence update"
            normalized = "<think>{}</think>".format(think.strip())
            steps.append(
                StreamingStep(
                    chunk_index=idx,
                    turn_type="think",
                    audio_chunk_path=sample.audio_chunk_paths[idx],
                    prompt_text=build_omni_chunk_prompt(
                        sample.question,
                        chunk_index=idx,
                        total_chunks=sample.n_chunks,
                        question_visible=question_visible,
                    ),
                    think=think,
                    raw_output=normalized,
                    normalized_output=normalized,
                )
            )
            thinks.append(think)
        final_timing = _append_final_think_and_answer_steps(
            sample=sample,
            steps=steps,
            visible_thinks=thinks,
            answer=sample.gt_answer,
            question_visible=question_visible,
        )
        rollout = StreamingRollout(
            audio_id=sample.audio_id,
            question=sample.question,
            thinks=thinks,
            answer=sample.gt_answer,
            steps=steps,
            backend_name=self.name,
            metadata={
                "question_visible_from_chunk_1": bool(question_visible),
                "question_visible_from_text": bool(question_visible),
                "task_spec_mode": "audio_only",
                "timing": dict(final_timing),
            },
        )
        rollout.validate_phase(phase)
        rollout.ensure_raw_sequence()
        return rollout


class HeuristicCaptionPolicyBackend(PolicyBackend):
    name = "heuristic"

    def __init__(
        self,
        no_think_prob: float = 0.35,
        correct_answer_prob: float = 0.55,
        max_think_words: int = 8,
    ):
        self.no_think_prob = no_think_prob
        self.correct_answer_prob = correct_answer_prob
        self.max_think_words = max_think_words

    def _should_emit_no_think(self, caption: str, previous_caption: str, rng: random.Random) -> bool:
        if not caption.strip():
            return True
        if rng.random() < self.no_think_prob:
            return True
        if previous_caption and caption == previous_caption:
            return True
        return False

    def _sample_answer(self, sample, rng: random.Random) -> str:
        if rng.random() < self.correct_answer_prob:
            return sample.gt_answer

        options = _parse_mcq_options(sample.question)
        wrong_options = [option for option in options if option.lower() != sample.gt_answer.lower()]
        if wrong_options:
            return rng.choice(wrong_options)
        return "unknown"

    def rollout(self, sample, rng: random.Random, phase: int = 1) -> StreamingRollout:
        question_visible = _question_visible_from_text(sample)
        thinks: List[str] = []
        steps: List[StreamingStep] = []
        previous_caption = ""

        for idx, caption in enumerate(sample.chunk_captions):
            if self._should_emit_no_think(caption, previous_caption, rng):
                think = ""
                turn_type = "wait"
                normalized = "<wait/>"
            else:
                think = _truncate_for_think(caption, max_words=self.max_think_words)
                if idx < sample.n_chunks - 1:
                    think = _mask_answer_leak(think, sample.gt_answer)
                turn_type = "think"
                normalized = "<think>{}</think>".format(think)

            previous_caption = caption
            thinks.append(think)
            steps.append(
                StreamingStep(
                    chunk_index=idx,
                    turn_type=turn_type,
                    audio_chunk_path=sample.audio_chunk_paths[idx],
                    prompt_text=build_omni_chunk_prompt(
                        sample.question,
                        chunk_index=idx,
                        total_chunks=sample.n_chunks,
                        question_visible=question_visible,
                    ),
                    think=think,
                    raw_output=normalized,
                    normalized_output=normalized,
                )
            )

        answer = self._sample_answer(sample, rng)
        final_timing = _append_final_think_and_answer_steps(
            sample=sample,
            steps=steps,
            visible_thinks=[think for think in thinks if think.strip()],
            answer=answer,
            question_visible=question_visible,
        )

        rollout = StreamingRollout(
            audio_id=sample.audio_id,
            question=sample.question,
            thinks=thinks,
            answer=answer,
            steps=steps,
            backend_name=self.name,
            metadata={
                "question_visible_from_chunk_1": bool(question_visible),
                "question_visible_from_text": bool(question_visible),
                "task_spec_mode": "audio_only",
                "timing": dict(final_timing),
            },
        )
        rollout.validate_phase(phase)
        rollout.ensure_raw_sequence()
        return rollout


class OmniVLLMPolicyBackend(PolicyBackend):
    """
    Real audio-conditioned rollout backend backed by a local OpenAI-compatible server.

    The server is expected to expose Qwen2.5-Omni-7B over a chat-completions endpoint.
    """

    name = "omni-vllm"
    SYSTEM_PROMPT = build_omni_system_prompt(DEFAULT_POLICY_PROMPT_VERSION, question_visible=False)

    def __init__(
        self,
        endpoint: str,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        api_key: str = "EMPTY",
        timeout_sec: float = 180.0,
        max_think_tokens: int = 64,
        max_answer_tokens: int = 24,
        think_temperature: float = 0.8,
        answer_temperature: float = 0.2,
        think_top_p: float = 0.95,
        answer_top_p: float = 0.9,
        temperature_jitter: float = 0.15,
        rollout_workers: int = 1,
        prompt_version: str = DEFAULT_POLICY_PROMPT_VERSION,
        final_think_prompt_version: str = "",
        final_answer_prompt_version: str = "",
        audio_window_mode: str = WINDOW_MODE_FULL_PREFIX,
        overlap_chunks: int = 1,
        min_audio_window_sec: float = MIN_QWEN_AUDIO_WINDOW_SEC,
        force_wait_before_sec: float = 0.0,
        question_visible_from_text: bool = False,
        answer_audio_output: bool = True,
        answer_audio_speaker: str = "Chelsie",
        answer_audio_onset_prior_seconds: float = DEFAULT_AUDIO_RESPONSE_ONSET_PRIOR_SECONDS,
        transport: Optional[Callable[..., str]] = None,
        updater: Optional[Any] = None,
        service_controller: Optional[Any] = None,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_think_tokens = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self.think_temperature = think_temperature
        self.answer_temperature = answer_temperature
        self.think_top_p = think_top_p
        self.answer_top_p = answer_top_p
        self.temperature_jitter = temperature_jitter
        self.rollout_workers = max(1, int(rollout_workers))
        self.prompt_version = str(prompt_version or DEFAULT_POLICY_PROMPT_VERSION)
        self.final_think_prompt_version = str(
            final_think_prompt_version or DEFAULT_FINAL_POLICY_PROMPT_VERSION
        )
        self.final_answer_prompt_version = str(
            final_answer_prompt_version or self.final_think_prompt_version
        )
        self.question_visible_from_text = False
        self.system_prompt = build_omni_system_prompt(
            self.prompt_version,
            question_visible=self.question_visible_from_text,
        )
        self.audio_window_mode = str(audio_window_mode or WINDOW_MODE_FULL_PREFIX)
        self.overlap_chunks = max(1, int(overlap_chunks))
        self.min_audio_window_sec = float(min_audio_window_sec)
        self.force_wait_before_sec = max(0.0, float(force_wait_before_sec or 0.0))
        self.answer_audio_output = bool(answer_audio_output)
        self.answer_audio_speaker = str(answer_audio_speaker or "Chelsie")
        self.answer_audio_onset_prior_seconds = max(0.0, float(answer_audio_onset_prior_seconds))
        self.transport = transport
        self.updater = updater
        self.service_controller = service_controller

    def start(self) -> Dict[str, Any]:
        if self.service_controller is None:
            return super().start()
        return self.service_controller.start()

    def stop(self) -> Dict[str, Any]:
        if self.service_controller is None:
            return super().stop()
        return self.service_controller.stop()

    def _encode_audio_chunk(self, path_str: str) -> str:
        data = Path(path_str).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _post_chat_response(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float,
        modalities: Optional[List[str]] = None,
        speaker: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.transport is not None:
            try:
                response = self.transport(messages, max_tokens, temperature, seed, top_p, modalities, speaker)
            except TypeError:
                try:
                    response = self.transport(messages, max_tokens, temperature, seed, top_p)
                except TypeError:
                    response = self.transport(messages, max_tokens, temperature, seed)
            if isinstance(response, dict):
                return {
                    "raw_text": str(response.get("raw_text", "")).strip(),
                    "audio_waveform": response.get("audio_waveform"),
                    "audio_sample_rate": response.get("audio_sample_rate"),
                    "response_onset_seconds": response.get("response_onset_seconds"),
                }
            return {
                "raw_text": str(response).strip(),
                "audio_waveform": None,
                "audio_sample_rate": None,
                "response_onset_seconds": None,
            }

        url = _resolve_chat_url(self.endpoint)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.api_key),
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if modalities:
            payload["modalities"] = list(modalities)
        if speaker and modalities and "audio" in {str(item) for item in modalities}:
            # OpenAI-compatible chat audio output expects a root-level `audio`
            # object rather than a bare `speaker` field. We still keep the old
            # `speaker` key for backward compatibility with local forks, but the
            # `audio` payload is the portable path that can unlock answer audio
            # in stock vLLM/OpenAI-compatible handlers.
            payload["audio"] = {
                "voice": str(speaker),
                "format": "wav",
            }
            payload["speaker"] = str(speaker)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only in live mode
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("policy HTTP {}: {}".format(exc.code, body)) from exc

        raw_text = ""
        for choice in list(data.get("choices") or []):
            message = dict(choice.get("message") or {})
            content_text = _collapse_message_content(message.get("content", ""))
            if content_text:
                raw_text = content_text
                break

        audio_waveform = None
        audio_sample_rate = None
        response_onset_seconds = None
        audio_blob = _extract_response_audio_blob(data)
        if audio_blob:
            audio_waveform, audio_sample_rate = _decode_audio_blob(audio_blob)
            response_onset_seconds = _estimate_response_onset_seconds(audio_waveform, audio_sample_rate)
        elif modalities and "audio" in {str(item) for item in modalities}:
            _maybe_dump_missing_audio_response(
                {
                    "request_payload": payload,
                    "response_payload": data,
                }
            )
            if raw_text:
                response_onset_seconds = float(self.answer_audio_onset_prior_seconds)

        return {
            "raw_text": raw_text,
            "audio_waveform": audio_waveform,
            "audio_sample_rate": audio_sample_rate,
            "response_onset_seconds": response_onset_seconds,
        }

    def _post_chat_text_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float,
    ) -> Dict[str, Any]:
        if self.transport is not None:
            try:
                response = self.transport(messages, max_tokens, temperature, seed, top_p, ["text"], None)
            except TypeError:
                try:
                    response = self.transport(messages, max_tokens, temperature, seed, top_p)
                except TypeError:
                    response = self.transport(messages, max_tokens, temperature, seed)
            if isinstance(response, dict):
                raw_text = str(response.get("raw_text", "")).strip()
                ttft = response.get("text_first_token_wall_clock_seconds", response.get("ttft_sec"))
                latency = response.get("text_generation_wall_clock_seconds", response.get("latency_sec"))
                return {
                    "raw_text": raw_text,
                    "text_first_token_wall_clock_seconds": None if ttft is None else float(ttft),
                    "text_generation_wall_clock_seconds": None if latency is None else float(latency),
                    "text_streaming_supported": bool(response.get("text_streaming_supported", ttft is not None)),
                }
            raw_text = str(response).strip()
            zero = 0.0 if raw_text else None
            return {
                "raw_text": raw_text,
                "text_first_token_wall_clock_seconds": zero,
                "text_generation_wall_clock_seconds": zero,
                "text_streaming_supported": False,
            }

        url = _resolve_chat_url(self.endpoint)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.api_key),
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": True,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started_at = time.perf_counter()
        first_token_at: Optional[float] = None
        text_parts: List[str] = []

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                while True:
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_text = line[len("data:") :].strip()
                    if payload_text == "[DONE]":
                        break
                    event = json.loads(payload_text)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = _extract_text_from_delta(delta.get("content"))
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_parts.append(piece)
        except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only in live mode
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("policy streaming HTTP {}: {}".format(exc.code, body)) from exc

        finished_at = time.perf_counter()
        raw_text = "".join(text_parts).strip()
        if first_token_at is not None:
            return {
                "raw_text": raw_text,
                "text_first_token_wall_clock_seconds": float(first_token_at - started_at),
                "text_generation_wall_clock_seconds": float(finished_at - started_at),
                "text_streaming_supported": True,
            }

        fallback_started_at = time.perf_counter()
        fallback = self._post_chat_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            modalities=["text"],
        )
        fallback_finished_at = time.perf_counter()
        fallback_text = str(fallback.get("raw_text", "")).strip()
        fallback_latency = float(fallback_finished_at - fallback_started_at)
        return {
            "raw_text": fallback_text,
            "text_first_token_wall_clock_seconds": None if not fallback_text else fallback_latency,
            "text_generation_wall_clock_seconds": None if not fallback_text else fallback_latency,
            "text_streaming_supported": False,
        }

    def _post_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float,
        modalities: Optional[List[str]] = None,
        speaker: Optional[str] = None,
    ) -> str:
        return str(
            self._post_chat_response(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                modalities=modalities,
                speaker=speaker,
            ).get("raw_text", "")
        ).strip()

    def _build_chunk_prompt(self, sample, chunk_index: int) -> str:
        return build_omni_chunk_prompt(
            sample.question,
            chunk_index=chunk_index,
            total_chunks=sample.n_chunks,
            question_visible=self.question_visible_from_text,
            version=self.prompt_version,
        )

    def _build_final_think_prompt(self, sample, previous_thinks: List[str]) -> str:
        return _build_window_final_think_prompt(
            question=sample.question,
            total_chunks=sample.n_chunks,
            previous_thinks=previous_thinks,
            window_span=_full_prefix_window_span(sample),
            question_visible_from_text=self.question_visible_from_text,
            version=self.final_think_prompt_version,
        )

    def _build_final_answer_after_think_prompt(self, sample, previous_thinks: List[str]) -> str:
        evaluation_type, choices = _infer_answer_prompt_metadata(sample.question)
        return _build_window_final_answer_after_think_prompt(
            question=sample.question,
            total_chunks=sample.n_chunks,
            previous_thinks=previous_thinks,
            window_span=_full_prefix_window_span(sample),
            question_visible_from_text=self.question_visible_from_text,
            version=self.final_answer_prompt_version,
            evaluation_type=evaluation_type,
            choices=choices,
        )

    def _sample_turn_temperature(self, base: float, rng: random.Random) -> float:
        if self.temperature_jitter <= 0:
            return max(0.0, base)
        return max(0.0, base + rng.uniform(-self.temperature_jitter, self.temperature_jitter))

    def rollout(self, sample, rng: random.Random, phase: int = 1) -> StreamingRollout:
        thinks: List[str] = []
        predicts: List[str] = []
        steps: List[StreamingStep] = []
        visible_thinks: List[str] = []
        previous_boundary_tick_idx: Optional[int] = None
        tick_seconds = _tick_seconds(sample)
        overall_start = time.perf_counter()
        step_generation_wall_clock_secs: List[float] = []

        for idx, chunk_path in enumerate(sample.audio_chunk_paths):
            window_spec = build_controller_window_spec(
                current_tick_idx=idx,
                previous_boundary_tick_idx=previous_boundary_tick_idx,
                overlap_chunks=self.overlap_chunks,
                audio_window_mode=self.audio_window_mode,
                unit_start_sec=lambda tick_idx: float(tick_idx) * float(tick_seconds),
                unit_end_sec=lambda tick_idx: float(tick_idx + 1) * float(tick_seconds),
            )
            window_paths = list(sample.audio_chunk_paths[window_spec.start_index : window_spec.end_index + 1])
            window_span = window_spec.to_dict()
            prompt_text = _build_window_prompt(
                question=sample.question,
                chunk_index=idx,
                total_chunks=sample.n_chunks,
                previous_thinks=visible_thinks,
                window_span=window_span,
                audio_window_mode=self.audio_window_mode,
                version=self.prompt_version,
                question_visible_from_text=self.question_visible_from_text,
            )
            force_wait = _should_force_wait_for_prefix(window_span, self.force_wait_before_sec)
            if force_wait:
                raw_output = "<wait/>"
                think, predict, turn_type, normalized = "", "", "wait", "<wait/>"
                step_generation_wall_clock_sec = 0.0
            else:
                window_audio = _ensure_min_audio_window(
                    _concat_audio_arrays([_load_audio_array(path) for path in window_paths]),
                    min_duration_sec=self.min_audio_window_sec,
                )
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": _encode_audio_array(window_audio),
                                "format": "wav",
                            },
                        },
                    ],
                }
                step_generation_start = time.perf_counter()
                raw_output = self._post_chat(
                    messages=[{"role": "system", "content": self.system_prompt}, user_message],
                    max_tokens=self.max_think_tokens,
                    temperature=self._sample_turn_temperature(self.think_temperature, rng),
                    seed=rng.randint(0, 2**31 - 1),
                    top_p=self.think_top_p,
                    modalities=["text"],
                )
                step_generation_wall_clock_sec = time.perf_counter() - step_generation_start
                step_generation_wall_clock_secs.append(step_generation_wall_clock_sec)
                think, predict, turn_type, normalized = _normalize_controller_step_output(
                    raw_output,
                    question=sample.question,
                    allow_predict=phase >= 3,
                )
            if phase >= 3 and predict:
                predict = _truncate_for_think(predict, max_words=12)
            thinks.append(think)
            if phase >= 3:
                predicts.append(predict)

            step = StreamingStep(
                chunk_index=idx,
                turn_type=turn_type,
                audio_chunk_path=chunk_path,
                audio_window_paths=window_paths,
                audio_window_span=window_span,
                prompt_text=prompt_text,
                think=think,
                predict=predict if phase >= 3 else "",
                raw_output=raw_output,
                normalized_output=normalized,
                timing={
                    "generation_wall_clock_sec": float(step_generation_wall_clock_sec),
                    "forced_wait": bool(force_wait),
                    "force_wait_before_sec": float(self.force_wait_before_sec),
                },
            )
            steps.append(step)
            if turn_type == "think":
                visible_thinks.append(think)
                previous_boundary_tick_idx = idx

        answer_phase_start = time.perf_counter()
        final_window_spec = build_controller_window_spec(
            current_tick_idx=max(0, sample.n_chunks - 1),
            previous_boundary_tick_idx=previous_boundary_tick_idx,
            overlap_chunks=self.overlap_chunks,
            audio_window_mode=WINDOW_MODE_FULL_PREFIX,
            unit_start_sec=lambda tick_idx: float(tick_idx) * float(tick_seconds),
            unit_end_sec=lambda tick_idx: float(tick_idx + 1) * float(tick_seconds),
        )
        final_window_paths = list(sample.audio_chunk_paths[final_window_spec.start_index : final_window_spec.end_index + 1])
        final_window_audio = _ensure_min_audio_window(
            _concat_audio_arrays([_load_audio_array(path) for path in final_window_paths]),
            min_duration_sec=self.min_audio_window_sec,
        )
        final_window_span = final_window_spec.to_dict()
        evaluation_type, choices = _infer_answer_prompt_metadata(sample.question)
        final_think_prompt = _build_window_final_think_prompt(
            question=sample.question,
            total_chunks=sample.n_chunks,
            previous_thinks=visible_thinks,
            window_span=final_window_span,
            question_visible_from_text=self.question_visible_from_text,
            version=self.final_think_prompt_version,
        )
        final_think_user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": final_think_prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": _encode_audio_array(final_window_audio),
                        "format": "wav",
                    },
                },
            ],
        }
        final_think_messages = [{"role": "system", "content": self.system_prompt}, final_think_user_message]
        final_think_generation_start = time.perf_counter()
        final_think_raw_output = self._post_chat(
            messages=final_think_messages,
            max_tokens=self.max_think_tokens,
            temperature=self._sample_turn_temperature(self.think_temperature, rng),
            seed=rng.randint(0, 2**31 - 1),
            top_p=self.think_top_p,
            modalities=["text"],
        )
        final_think_generation_wall_clock_sec = time.perf_counter() - final_think_generation_start
        step_generation_wall_clock_secs.append(final_think_generation_wall_clock_sec)
        final_think, _final_predict, final_think_turn_type, final_think_normalized = _normalize_controller_step_output(
            final_think_raw_output,
            question=sample.question,
            allow_predict=False,
        )
        final_think_raw_valid = bool(final_think_turn_type == "think" and final_think.strip())
        final_think_fallback_used = False
        if not final_think_raw_valid:
            final_think_fallback_used = True
            fallback_final_think = sanitize_final_think_fallback(final_think_raw_output, question=sample.question)
            if not fallback_final_think:
                fallback_final_think = "final reasoning state"
            final_think = fallback_final_think
            final_think_normalized = "<think>{}</think>".format(final_think)
        final_think_token_count = int(estimate_token_count(final_think))
        steps.append(
            StreamingStep(
                chunk_index=max(0, sample.n_chunks - 1),
                turn_type="think",
                audio_chunk_path=sample.audio_chunk_paths[max(0, sample.n_chunks - 1)],
                audio_window_paths=final_window_paths,
                audio_window_span=final_window_span,
                prompt_text=final_think_prompt,
                think=final_think,
                raw_output=final_think_raw_output,
                normalized_output=final_think_normalized,
                timing={
                    "generation_wall_clock_sec": float(final_think_generation_wall_clock_sec),
                    "final_think_generation_wall_clock_sec": float(final_think_generation_wall_clock_sec),
                    "final_think_token_count": float(final_think_token_count),
                    "is_final_think": True,
                    "final_think_raw_valid": bool(final_think_raw_valid),
                    "final_think_fallback_used": bool(final_think_fallback_used),
                    "final_think_raw_turn_type": str(final_think_turn_type or ""),
                },
            )
        )
        final_visible_thinks = list(visible_thinks) + [final_think]
        answer_prompt = _build_window_final_answer_after_think_prompt(
            question=sample.question,
            total_chunks=sample.n_chunks,
            previous_thinks=final_visible_thinks,
            window_span=final_window_span,
            question_visible_from_text=self.question_visible_from_text,
            evaluation_type=evaluation_type,
            choices=choices,
            version=self.final_answer_prompt_version,
        )
        answer_user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": answer_prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": _encode_audio_array(final_window_audio),
                        "format": "wav",
                    },
                },
            ],
        }
        answer_messages = [{"role": "system", "content": self.system_prompt}, answer_user_message]
        text_first_token_wall_clock_seconds = None
        effective_text_first_token_seconds = None
        text_streaming_supported = False
        if self.answer_audio_output:
            answer_generation_start = time.perf_counter()
            answer_generation = self._post_chat_response(
                messages=answer_messages,
                max_tokens=self.max_answer_tokens,
                temperature=self.answer_temperature,
                seed=rng.randint(0, 2**31 - 1),
                top_p=self.answer_top_p,
                modalities=["text", "audio"],
                speaker=self.answer_audio_speaker,
            )
            answer_generation_wall_clock_sec = time.perf_counter() - answer_generation_start
        else:
            answer_generation = self._post_chat_text_streaming_response(
                messages=answer_messages,
                max_tokens=self.max_answer_tokens,
                temperature=self.answer_temperature,
                seed=rng.randint(0, 2**31 - 1),
                top_p=self.answer_top_p,
            )
            answer_generation_wall_clock_sec = float(
                answer_generation.get("text_generation_wall_clock_seconds") or 0.0
            )
            if answer_generation.get("text_first_token_wall_clock_seconds") is not None:
                text_first_token_wall_clock_seconds = float(
                    answer_generation.get("text_first_token_wall_clock_seconds")
                )
                effective_text_first_token_seconds = float(final_think_generation_wall_clock_sec) + float(
                    text_first_token_wall_clock_seconds
                )
            text_streaming_supported = bool(answer_generation.get("text_streaming_supported"))
        post_eof_total_wall_clock_sec = time.perf_counter() - answer_phase_start
        step_generation_wall_clock_secs.append(answer_generation_wall_clock_sec)
        raw_answer = str(answer_generation.get("raw_text", "")).strip()
        response_onset_seconds = answer_generation.get("response_onset_seconds")
        effective_response_onset_seconds = None
        if response_onset_seconds is not None:
            effective_response_onset_seconds = float(final_think_generation_wall_clock_sec) + float(
                answer_generation_wall_clock_sec
            ) + float(response_onset_seconds)
        answer, normalized_answer = _normalize_answer_output(raw_answer)
        steps.append(
            StreamingStep(
                chunk_index=max(0, sample.n_chunks - 1),
                turn_type="answer",
                audio_chunk_path=sample.audio_chunk_paths[max(0, sample.n_chunks - 1)],
                audio_window_paths=final_window_paths,
                audio_window_span=final_window_span,
                prompt_text=answer_prompt,
                answer=answer,
                raw_output=raw_answer,
                normalized_output=normalized_answer,
                timing={
                    "generation_wall_clock_sec": float(answer_generation_wall_clock_sec),
                    "answer_generation_wall_clock_sec": float(answer_generation_wall_clock_sec),
                    "post_eof_total_wall_clock_sec": float(post_eof_total_wall_clock_sec),
                    "final_think_generation_wall_clock_sec": float(final_think_generation_wall_clock_sec),
                    "final_think_token_count": float(final_think_token_count),
                    "text_first_token_wall_clock_seconds": None
                    if text_first_token_wall_clock_seconds is None
                    else float(text_first_token_wall_clock_seconds),
                    "effective_text_first_token_seconds": None
                    if effective_text_first_token_seconds is None
                    else float(effective_text_first_token_seconds),
                    "text_streaming_supported": bool(text_streaming_supported),
                    "response_onset_seconds": None if response_onset_seconds is None else float(response_onset_seconds),
                    "effective_response_onset_seconds": None
                    if effective_response_onset_seconds is None
                    else float(effective_response_onset_seconds),
                },
            )
        )
        decision_generation_wall_clock_secs = step_generation_wall_clock_secs[:-1]
        controller_total_wall_clock_sec = time.perf_counter() - overall_start
        timing = {
            "controller_total_wall_clock_sec": float(controller_total_wall_clock_sec),
            "controller_setup_wall_clock_sec": float(
                max(0.0, controller_total_wall_clock_sec - float(sum(step_generation_wall_clock_secs)))
            ),
            "decision_generation_wall_clock_sec_total": float(sum(decision_generation_wall_clock_secs)),
            "decision_generation_wall_clock_sec_mean": (
                float(sum(decision_generation_wall_clock_secs)) / float(len(decision_generation_wall_clock_secs))
                if decision_generation_wall_clock_secs
                else None
            ),
            "decision_generation_wall_clock_sec_p95": _percentile(decision_generation_wall_clock_secs, 0.95),
            "answer_generation_wall_clock_sec": float(answer_generation_wall_clock_sec),
            "post_eof_total_wall_clock_sec": float(post_eof_total_wall_clock_sec),
            "final_think_generation_wall_clock_sec": float(final_think_generation_wall_clock_sec),
            "final_think_token_count": float(final_think_token_count),
            "text_first_token_wall_clock_seconds": None
            if text_first_token_wall_clock_seconds is None
            else float(text_first_token_wall_clock_seconds),
            "effective_text_first_token_seconds": None
            if effective_text_first_token_seconds is None
            else float(effective_text_first_token_seconds),
            "text_streaming_supported": bool(text_streaming_supported),
            "response_onset_seconds": None if response_onset_seconds is None else float(response_onset_seconds),
            "effective_response_onset_seconds": None
            if effective_response_onset_seconds is None
            else float(effective_response_onset_seconds),
            "step_generation_wall_clock_sec_mean": (
                float(sum(step_generation_wall_clock_secs)) / float(len(step_generation_wall_clock_secs))
                if step_generation_wall_clock_secs
                else None
            ),
            "step_generation_wall_clock_sec_p95": _percentile(step_generation_wall_clock_secs, 0.95),
        }

        rollout = StreamingRollout(
            audio_id=sample.audio_id,
            question=sample.question,
            thinks=thinks,
            predicts=predicts,
            answer=answer,
            steps=steps,
            backend_name=self.name,
            metadata={
                "question_visible_from_chunk_1": bool(self.question_visible_from_text),
                "question_visible_from_text": bool(self.question_visible_from_text),
                "task_spec_mode": "audio_only",
                "policy_model": self.model_name,
                "policy_endpoint": self.endpoint,
                "audio_window_mode": self.audio_window_mode,
                "overlap_chunks": self.overlap_chunks,
                "force_wait_before_sec": self.force_wait_before_sec,
                "timing": timing,
                "controller_total_wall_clock_sec": timing["controller_total_wall_clock_sec"],
                "answer_generation_wall_clock_sec": timing["answer_generation_wall_clock_sec"],
                "post_eof_total_wall_clock_sec": timing["post_eof_total_wall_clock_sec"],
                "text_first_token_wall_clock_seconds": timing.get("text_first_token_wall_clock_seconds"),
                "effective_text_first_token_seconds": timing.get("effective_text_first_token_seconds"),
                "text_streaming_supported": timing.get("text_streaming_supported"),
                "response_onset_seconds": timing.get("response_onset_seconds"),
                "effective_response_onset_seconds": timing.get("effective_response_onset_seconds"),
            },
        )
        rollout.validate_phase(phase)
        rollout.ensure_raw_sequence()
        return rollout

    def rollout_group(self, sample, group_size: int, seed: int, phase: int = 1) -> List[StreamingRollout]:
        if self.rollout_workers <= 1 or group_size <= 1:
            return super().rollout_group(sample=sample, group_size=group_size, seed=seed, phase=phase)

        rollouts: List[Optional[StreamingRollout]] = [None] * group_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(group_size, self.rollout_workers)) as executor:
            future_to_index = {
                executor.submit(self.rollout, sample, random.Random(seed + idx), phase): idx for idx in range(group_size)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                rollouts[idx] = future.result()
        return [rollout for rollout in rollouts if rollout is not None]

    def update_group(self, sample, rollouts, episodes, rewards, advantages, step_index: Optional[int] = None) -> Dict[str, Any]:
        if self.updater is None:
            return super().update_group(sample, rollouts, episodes, rewards, advantages, step_index=step_index)
        return self.updater.update_group(
            sample=sample,
            rollouts=rollouts,
            rewards=rewards,
            advantages=advantages,
            step_index=step_index,
        )

    def update_step(self, group_batches: List[Dict[str, Any]], step_index: Optional[int] = None) -> Dict[str, Any]:
        if self.updater is None:
            return super().update_step(group_batches, step_index=step_index)
        return self.updater.update_groups(group_batches, step_index=step_index)

    def save_checkpoint(self, path: str, step: int, checkpoint_mode: Optional[str] = None) -> CheckpointArtifact:
        if self.updater is None:
            return super().save_checkpoint(path, step, checkpoint_mode=checkpoint_mode)
        return self.updater.save_checkpoint(path, step, checkpoint_mode=checkpoint_mode)

    def reload_from_checkpoint(self, artifact: CheckpointArtifact) -> Dict[str, Any]:
        if self.service_controller is None:
            return super().reload_from_checkpoint(artifact)
        if not artifact.is_reloadable:
            return {"status": "skipped", "reason": "checkpoint-not-reloadable", "mode": artifact.mode}
        info = self.service_controller.reload(artifact.reloadable_model_path)
        info["mode"] = artifact.mode
        info["step"] = artifact.step
        return info
