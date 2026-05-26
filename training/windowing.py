"""
Shared controller window construction for training-time rollout.

This module is the single source of truth for how we expose audio to the
controller at decision tick `i`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


MIN_QWEN_AUDIO_WINDOW_SEC = 2.0
WINDOW_MODE_FULL_PREFIX = "full_prefix"
WINDOW_MODE_SINCE_LAST_THINK = "since_last_think"


@dataclass(frozen=True)
class ControllerWindowSpec:
    start_index: int
    end_index: int
    start_sec: float
    end_sec: float
    num_units: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "start_tick_index": float(self.start_index),
            "end_tick_index": float(self.end_index),
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "num_chunks": float(self.num_units),
        }


def controller_window_start_index(
    *,
    previous_boundary_tick_idx: Optional[int],
    overlap_chunks: int,
    audio_window_mode: str,
) -> int:
    if audio_window_mode == WINDOW_MODE_FULL_PREFIX or previous_boundary_tick_idx is None:
        return 0
    return max(0, int(previous_boundary_tick_idx) - max(1, int(overlap_chunks)) + 1)


def build_controller_window_spec(
    *,
    current_tick_idx: int,
    previous_boundary_tick_idx: Optional[int],
    overlap_chunks: int,
    audio_window_mode: str,
    unit_start_sec: Callable[[int], float],
    unit_end_sec: Callable[[int], float],
) -> ControllerWindowSpec:
    start_idx = controller_window_start_index(
        previous_boundary_tick_idx=previous_boundary_tick_idx,
        overlap_chunks=overlap_chunks,
        audio_window_mode=audio_window_mode,
    )
    end_idx = int(current_tick_idx)
    if end_idx < start_idx:
        raise ValueError(
            "controller window end index {} cannot be smaller than start index {}".format(
                end_idx,
                start_idx,
            )
        )
    return ControllerWindowSpec(
        start_index=start_idx,
        end_index=end_idx,
        start_sec=float(unit_start_sec(start_idx)),
        end_sec=float(unit_end_sec(end_idx)),
        num_units=int(end_idx - start_idx + 1),
    )
