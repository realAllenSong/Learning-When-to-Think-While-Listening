"""
Sample ordering helpers for streaming controller training.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


def _normalized(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _metadata_bucket_key(metadata: Dict[str, Any], bucket_keys: Sequence[str]) -> Tuple[str, ...]:
    return tuple(_normalized(metadata.get(key)) for key in bucket_keys)


def reorder_items(
    items: Sequence[Any],
    *,
    mode: str = "sequential",
    metadata_getter: Callable[[Any], Dict[str, Any]] | None = None,
    bucket_keys: Sequence[str] = ("topic", "difficulty"),
    seed: int = 7,
) -> List[Any]:
    """
    Reorder items deterministically for training.

    Supported modes:

    - `sequential`: preserve input order
    - `shuffle`: one deterministic global shuffle
    - `balanced_interleave`: shuffle inside each bucket and round-robin across
      buckets so early prefixes see a wider mix of topic / difficulty
    """
    items_list = list(items)
    normalized_mode = str(mode or "sequential").strip().lower()
    rng = random.Random(int(seed))

    if normalized_mode in {"", "sequential", "none"}:
        return items_list

    if normalized_mode == "shuffle":
        indices = list(range(len(items_list)))
        rng.shuffle(indices)
        return [items_list[idx] for idx in indices]

    if normalized_mode != "balanced_interleave":
        raise ValueError("Unsupported sample order mode: {}".format(mode))

    getter = metadata_getter or (lambda item: dict(getattr(item, "difficulty_metadata", {}) or {}))
    buckets: Dict[Tuple[str, ...], List[Any]] = defaultdict(list)
    for item in items_list:
        buckets[_metadata_bucket_key(dict(getter(item) or {}), bucket_keys)].append(item)

    ordered_bucket_keys = sorted(buckets)
    rng.shuffle(ordered_bucket_keys)
    for key in ordered_bucket_keys:
        rng.shuffle(buckets[key])

    output: List[Any] = []
    remaining = True
    while remaining:
        remaining = False
        for key in ordered_bucket_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            remaining = True
            output.append(bucket.pop())
    return output


def prefix_bucket_coverage(
    items: Sequence[Any],
    *,
    metadata_getter: Callable[[Any], Dict[str, Any]] | None = None,
    bucket_keys: Sequence[str] = ("topic", "difficulty"),
    prefix_sizes: Iterable[int] = (100, 500, 1000, 2000),
) -> Dict[int, int]:
    getter = metadata_getter or (lambda item: dict(getattr(item, "difficulty_metadata", {}) or {}))
    items_list = list(items)
    coverage: Dict[int, int] = {}
    for prefix_size in prefix_sizes:
        buckets = {
            _metadata_bucket_key(dict(getter(item) or {}), bucket_keys)
            for item in items_list[: max(0, int(prefix_size))]
        }
        coverage[int(prefix_size)] = len(buckets)
    return coverage
