"""
Lightweight audio I/O helpers that only rely on the standard library + numpy.

The DAPO launcher uses the core runtime environment, which
does not ship with librosa/soundfile/scipy by default. Training
controller code therefore use WAV-native loading plus a simple numpy resampler.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)
    duration = float(audio.shape[0]) / float(source_rate)
    target_length = max(1, int(round(duration * float(target_rate))))
    source_positions = np.linspace(0.0, float(audio.shape[0] - 1), num=int(audio.shape[0]), dtype=np.float32)
    target_positions = np.linspace(0.0, float(audio.shape[0] - 1), num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


def load_mono_audio(path: str, *, sampling_rate: int = 16000) -> np.ndarray:
    path = str(Path(path))
    with wave.open(path, "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        source_rate = int(handle.getframerate())
        n_frames = int(handle.getnframes())
        pcm = handle.readframes(n_frames)

    if sample_width == 1:
        audio = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width: {}".format(sample_width))

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    audio = _resample_linear(audio, source_rate=source_rate, target_rate=int(sampling_rate))
    return np.asarray(audio, dtype=np.float32)
