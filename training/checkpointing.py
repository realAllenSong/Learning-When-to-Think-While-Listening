"""
Checkpoint helpers for online controller training.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CheckpointArtifact:
    step: int
    mode: str
    checkpoint_path: str
    checkpoint_dir: str
    metadata_path: str = ""
    optimizer_path: str = ""
    reloadable_model_path: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def is_reloadable(self) -> bool:
        return bool(self.reloadable_model_path)


def _step_key(path: Path) -> int:
    match = re.search(r"step_(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return -1


def prune_step_checkpoints(root_dir: str, keep: int) -> List[str]:
    """
    Delete older `step_XXXXXX` directories under `root_dir`, keeping the newest `keep`.
    """
    if keep <= 0:
        return []

    root = Path(root_dir)
    if not root.exists():
        return []

    step_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    step_dirs.sort(key=_step_key, reverse=True)

    removed: List[str] = []
    for path in step_dirs[keep:]:
        shutil.rmtree(path, ignore_errors=False)
        removed.append(str(path))
    return removed


def _load_checkpoint_score(checkpoint_dir: Path, score_key: str) -> Optional[float]:
    sidecar_path = checkpoint_dir / "trainer_metrics.json"
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get(str(score_key))
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ranked_step_checkpoints(
    root_dir: str,
    *,
    score_key: str = "mean_total",
    maximize: bool = True,
) -> List[Dict[str, object]]:
    root = Path(root_dir)
    if not root.exists():
        return []

    step_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    scored = []
    for path in step_dirs:
        score = _load_checkpoint_score(path, score_key)
        if score is None:
            continue
        scored.append(
            {
                "step": _step_key(path),
                "score": float(score),
                "checkpoint_dir": str(path),
                "trainer_metrics_path": str(path / "trainer_metrics.json"),
            }
        )
    scored.sort(
        key=lambda item: (float(item["score"]), int(item["step"])),
        reverse=bool(maximize),
    )
    return scored


def checkpoint_would_enter_top_k(
    root_dir: str,
    *,
    step: int,
    score: float,
    keep_best: int,
    score_key: str = "mean_total",
    maximize: bool = True,
) -> bool:
    if keep_best <= 0:
        return False

    ranked = ranked_step_checkpoints(root_dir, score_key=score_key, maximize=maximize)
    if any(int(item["step"]) == int(step) for item in ranked):
        return False
    if len(ranked) < int(keep_best):
        return True

    cutoff = ranked[int(keep_best) - 1]
    cutoff_score = float(cutoff["score"])
    cutoff_step = int(cutoff["step"])
    if maximize:
        return float(score) > cutoff_score or (
            float(score) == cutoff_score and int(step) > cutoff_step
        )
    return float(score) < cutoff_score or (
        float(score) == cutoff_score and int(step) > cutoff_step
    )


def _bucket_id(step: int, bucket_size: int) -> int:
    if bucket_size <= 0:
        return 0
    return max(0, (int(step) - 1) // int(bucket_size))


def checkpoint_would_enter_bucket_top_k(
    root_dir: str,
    *,
    step: int,
    score: float,
    bucket_size: int,
    keep_best_per_bucket: int,
    score_key: str = "mean_total",
    maximize: bool = True,
) -> bool:
    if bucket_size <= 0 or keep_best_per_bucket <= 0:
        return False

    root = Path(root_dir)
    if not root.exists():
        return True

    target_bucket = _bucket_id(int(step), int(bucket_size))
    step_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    scored: List[Tuple[float, int, Path]] = []
    for path in step_dirs:
        step_key = _step_key(path)
        if _bucket_id(step_key, int(bucket_size)) != target_bucket:
            continue
        existing_score = _load_checkpoint_score(path, score_key)
        if existing_score is None:
            continue
        if step_key == int(step):
            return False
        scored.append((float(existing_score), step_key, path))

    if len(scored) < int(keep_best_per_bucket):
        return True

    scored.sort(key=lambda item: (item[0], item[1]), reverse=bool(maximize))
    cutoff_score, cutoff_step, _ = scored[int(keep_best_per_bucket) - 1]
    if maximize:
        return float(score) > cutoff_score or (float(score) == cutoff_score and int(step) > cutoff_step)
    return float(score) < cutoff_score or (float(score) == cutoff_score and int(step) > cutoff_step)


def prune_step_checkpoints_recent_and_best(
    root_dir: str,
    *,
    keep_recent: int,
    keep_best: int = 0,
    alt_score_key: str = "",
    alt_keep_best: int = 0,
    alt_maximize: bool = True,
    bucket_size: int = 0,
    keep_best_per_bucket: int = 0,
    alt_bucket_size: int = 0,
    alt_keep_best_per_bucket: int = 0,
    user_goal_score_key: str = "",
    user_goal_keep_best: int = 0,
    user_goal_maximize: bool = True,
    user_goal_bucket_size: int = 0,
    user_goal_keep_best_per_bucket: int = 0,
    score_key: str = "mean_total",
    maximize: bool = True,
) -> List[str]:
    """
    Delete older `step_XXXXXX` directories under `root_dir`, keeping a union of:

    - the newest `keep_recent` checkpoints
    - the best `keep_best` checkpoints ranked by `score_key` from `trainer_metrics.json`
    - the best `keep_best_per_bucket` checkpoints inside each `bucket_size` step bucket
    - the best `alt_keep_best` checkpoints ranked by `alt_score_key`
    - the best `alt_keep_best_per_bucket` checkpoints inside each `alt_bucket_size`
      step bucket ranked by `alt_score_key`
    - the best user-goal checkpoints ranked by `user_goal_score_key`, both globally
      and within `user_goal_bucket_size` buckets
    """
    if (
        keep_recent <= 0
        and keep_best <= 0
        and alt_keep_best <= 0
        and user_goal_keep_best <= 0
        and (bucket_size <= 0 or keep_best_per_bucket <= 0)
        and (alt_bucket_size <= 0 or alt_keep_best_per_bucket <= 0)
        and (user_goal_bucket_size <= 0 or user_goal_keep_best_per_bucket <= 0)
    ):
        return []

    root = Path(root_dir)
    if not root.exists():
        return []

    step_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    step_dirs.sort(key=_step_key, reverse=True)

    protected: Set[Path] = set()
    if keep_recent > 0:
        protected.update(step_dirs[:keep_recent])

    if keep_best > 0:
        scored = []
        for path in step_dirs:
            score = _load_checkpoint_score(path, score_key)
            if score is None:
                continue
            scored.append((score, _step_key(path), path))
        scored.sort(
            key=lambda item: (item[0], item[1]),
            reverse=bool(maximize),
        )
        protected.update(path for _, _, path in scored[:keep_best])

    if alt_score_key and alt_keep_best > 0:
        alt_scored = []
        for path in step_dirs:
            score = _load_checkpoint_score(path, alt_score_key)
            if score is None:
                continue
            alt_scored.append((score, _step_key(path), path))
        alt_scored.sort(
            key=lambda item: (item[0], item[1]),
            reverse=bool(alt_maximize),
        )
        protected.update(path for _, _, path in alt_scored[:alt_keep_best])

    if user_goal_score_key and user_goal_keep_best > 0:
        user_goal_scored = []
        for path in step_dirs:
            score = _load_checkpoint_score(path, user_goal_score_key)
            if score is None:
                continue
            user_goal_scored.append((score, _step_key(path), path))
        user_goal_scored.sort(
            key=lambda item: (item[0], item[1]),
            reverse=bool(user_goal_maximize),
        )
        protected.update(path for _, _, path in user_goal_scored[:user_goal_keep_best])

    if bucket_size > 0 and keep_best_per_bucket > 0:
        bucketed: Dict[int, List[Tuple[float, int, Path]]] = {}
        for path in step_dirs:
            score = _load_checkpoint_score(path, score_key)
            if score is None:
                continue
            step_key = _step_key(path)
            bucketed.setdefault(_bucket_id(step_key, int(bucket_size)), []).append((float(score), step_key, path))
        for entries in bucketed.values():
            entries.sort(key=lambda item: (item[0], item[1]), reverse=bool(maximize))
            protected.update(path for _, _, path in entries[: int(keep_best_per_bucket)])

    if alt_score_key and alt_bucket_size > 0 and alt_keep_best_per_bucket > 0:
        alt_bucketed: Dict[int, List[Tuple[float, int, Path]]] = {}
        for path in step_dirs:
            score = _load_checkpoint_score(path, alt_score_key)
            if score is None:
                continue
            step_key = _step_key(path)
            alt_bucketed.setdefault(_bucket_id(step_key, int(alt_bucket_size)), []).append(
                (float(score), step_key, path)
            )
        for entries in alt_bucketed.values():
            entries.sort(key=lambda item: (item[0], item[1]), reverse=bool(alt_maximize))
            protected.update(path for _, _, path in entries[: int(alt_keep_best_per_bucket)])

    if user_goal_score_key and user_goal_bucket_size > 0 and user_goal_keep_best_per_bucket > 0:
        user_goal_bucketed: Dict[int, List[Tuple[float, int, Path]]] = {}
        for path in step_dirs:
            score = _load_checkpoint_score(path, user_goal_score_key)
            if score is None:
                continue
            step_key = _step_key(path)
            user_goal_bucketed.setdefault(_bucket_id(step_key, int(user_goal_bucket_size)), []).append(
                (float(score), step_key, path)
            )
        for entries in user_goal_bucketed.values():
            entries.sort(key=lambda item: (item[0], item[1]), reverse=bool(user_goal_maximize))
            protected.update(path for _, _, path in entries[: int(user_goal_keep_best_per_bucket)])

    removed: List[str] = []
    for path in step_dirs:
        if path in protected:
            continue
        shutil.rmtree(path, ignore_errors=False)
        removed.append(str(path))
    return removed
