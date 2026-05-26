"""
Training-side answer extraction and normalization.

This module follows the task-side scoring conventions used during training:
- extract the final answer span first
- normalize per task type
- use strict comparison on the normalized answer

It is still cheaper and more deterministic than a judge-backed reward.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_PREFIX_PATTERN = re.compile(r"^(answer|final answer|the answer)\s*[:：\-]\s*", flags=re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^\w\s]")
_MULTISPACE_PATTERN = re.compile(r"\s+")
_DIGIT_TOKEN_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
_MCQ_TASK_NAMES = {
    "arc-e",
    "arc-c",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "siqa",
}
_NUMERIC_TASK_NAMES = {
    "gsm8k",
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1000,
}
_UNIT_NORMALIZATION = {
    "second": "second",
    "seconds": "second",
    "sec": "second",
    "secs": "second",
    "minute": "minute",
    "minutes": "minute",
    "min": "minute",
    "mins": "minute",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "liter": "liter",
    "liters": "liter",
    "litre": "liter",
    "litres": "liter",
    "milliliter": "milliliter",
    "milliliters": "milliliter",
    "millilitre": "milliliter",
    "millilitres": "milliliter",
    "gallon": "gallon",
    "gallons": "gallon",
    "meter": "meter",
    "meters": "meter",
    "metre": "meter",
    "metres": "meter",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "kilometre": "kilometer",
    "kilometres": "kilometer",
    "centimeter": "centimeter",
    "centimeters": "centimeter",
    "centimetre": "centimeter",
    "centimetres": "centimeter",
    "millimeter": "millimeter",
    "millimeters": "millimeter",
    "millimetre": "millimeter",
    "millimetres": "millimeter",
    "inch": "inch",
    "inches": "inch",
    "foot": "foot",
    "feet": "foot",
    "yard": "yard",
    "yards": "yard",
    "mile": "mile",
    "miles": "mile",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "gram": "gram",
    "grams": "gram",
    "pound": "pound",
    "pounds": "pound",
    "lb": "pound",
    "lbs": "pound",
    "dollar": "dollar",
    "dollars": "dollar",
    "cent": "cent",
    "cents": "cent",
    "percent": "percent",
    "percentage": "percent",
}


def _merge_metadata(*payloads: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for payload in payloads:
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def _coerce_choices(raw: Any) -> List[str]:
    if isinstance(raw, dict):
        maybe_text = raw.get("text")
        if isinstance(maybe_text, list):
            return [str(item).strip() for item in maybe_text if str(item).strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _normalize_choice(choice: str) -> str:
    return _MULTISPACE_PATTERN.sub(" ", str(choice or "").strip().lower())


def _extract_answer_span(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    first_line = value.splitlines()[0].strip()
    first_line = _PREFIX_PATTERN.sub("", first_line)
    return first_line.strip().strip("\"'`")


def _normalize_short_answer_open(text: str) -> str:
    value = _extract_answer_span(text).lower()
    if not value:
        return ""
    value = re.sub(r"^[\"'`\s]+|[\"'`\s]+$", "", value)
    value = re.sub(r"[\.,!?;:]+$", "", value)
    value = re.sub(r"[_/\-]+", " ", value)
    value = _NON_WORD_PATTERN.sub(" ", value)
    return _MULTISPACE_PATTERN.sub(" ", value).strip()


def _normalize_free_form_numeric(text: str) -> str:
    value = _extract_answer_span(text)
    matches = NUMBER_PATTERN.findall(value)
    if not matches:
        return _normalize_short_answer_open(value)
    return matches[-1].replace(",", "")


def _normalize_numeric_string(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    rendered = "{:.12f}".format(float(value)).rstrip("0").rstrip(".")
    return rendered or "0"


def _parse_quantity_prefix(tokens: Sequence[str]) -> tuple[str | None, int]:
    if not tokens:
        return None, 0

    first = str(tokens[0] or "").strip().lower()
    first = first.replace(",", "")
    if _DIGIT_TOKEN_PATTERN.fullmatch(first):
        return _normalize_numeric_string(float(first)), 1

    sign = 1
    index = 0
    if first in {"minus", "negative"}:
        sign = -1
        index = 1

    total = 0
    current = 0
    consumed = False
    saw_point = False
    decimal_digits: List[str] = []

    while index < len(tokens):
        token = str(tokens[index] or "").strip().lower().replace(",", "")
        if token == "and":
            index += 1
            continue
        if token == "point":
            saw_point = consumed
            index += 1
            break
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            consumed = True
            index += 1
            continue
        if token in _SCALE_WORDS:
            scale = _SCALE_WORDS[token]
            if current == 0:
                current = 1
            if scale >= 1000:
                total += current * scale
                current = 0
            else:
                current *= scale
            consumed = True
            index += 1
            continue
        break

    if saw_point:
        while index < len(tokens):
            token = str(tokens[index] or "").strip().lower().replace(",", "")
            if token in _NUMBER_WORDS and _NUMBER_WORDS[token] < 10:
                decimal_digits.append(str(_NUMBER_WORDS[token]))
                index += 1
                continue
            if _DIGIT_TOKEN_PATTERN.fullmatch(token) and "." not in token and len(token) == 1:
                decimal_digits.append(token)
                index += 1
                continue
            break
        if not decimal_digits:
            return None, 0

    if not consumed:
        return None, 0

    value = sign * (total + current)
    normalized = _normalize_numeric_string(float(value))
    if decimal_digits:
        normalized = "{}.{}".format(normalized, "".join(decimal_digits))
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized, index


def _normalize_quantity_unit_phrase(text: str) -> str:
    normalized = _normalize_short_answer_open(text)
    if not normalized:
        return ""
    tokens = normalized.split()
    quantity, consumed = _parse_quantity_prefix(tokens)
    if quantity is None or consumed <= 0:
        return ""
    suffix_tokens = tokens[consumed:]
    if not suffix_tokens:
        return quantity
    normalized_suffix: List[str] = []
    for token in suffix_tokens:
        unit = _UNIT_NORMALIZATION.get(token)
        if unit is None:
            return ""
        normalized_suffix.append(unit)
    return "{} {}".format(quantity, " ".join(normalized_suffix)).strip()


def _canonicalize_multiple_choice(text: str, choices: Sequence[str]) -> str:
    answer_text = _extract_answer_span(text)
    if not answer_text:
        return ""

    normalized_text = _normalize_choice(answer_text)
    normalized_choices = [_normalize_choice(choice) for choice in choices]
    lookup = {normalized: original for normalized, original in zip(normalized_choices, choices)}
    if normalized_text in lookup:
        return lookup[normalized_text]

    label_match = re.match(r"^\(?([A-Da-d1-4])[\)\.\:\-\s].*$|^([A-Da-d1-4])$", answer_text)
    if label_match:
        label = (label_match.group(1) or label_match.group(2)).upper()
        if label.isdigit():
            index = int(label) - 1
        else:
            index = ord(label) - ord("A")
        if 0 <= index < len(choices):
            return str(choices[index])

    option_only_match = re.match(r"^option\s+([A-Da-d1-4])$", answer_text.strip(), flags=re.IGNORECASE)
    if option_only_match:
        label = option_only_match.group(1).upper()
        if label.isdigit():
            index = int(label) - 1
        else:
            index = ord(label) - ord("A")
        if 0 <= index < len(choices):
            return str(choices[index])

    label_plus_option = re.match(r"^(?:option\s+)?([A-Da-d1-4])[\)\.\:\-\s]+(.+)$", answer_text, flags=re.IGNORECASE)
    if label_plus_option:
        option_text = label_plus_option.group(2).strip()
        normalized_option_text = _normalize_choice(option_text)
        if normalized_option_text in lookup:
            return lookup[normalized_option_text]

    matched = [choice for choice in choices if _normalize_choice(choice) in normalized_text]
    if len(matched) == 1:
        return str(matched[0])

    return answer_text.strip()


def _infer_scoring_type(metadata: Dict[str, Any], gt_answer: str, choices: Sequence[str]) -> str:
    evaluation_type = str(metadata.get("evaluation_type", "")).strip().lower()
    if evaluation_type == "multiple_choice":
        return "multiple_choice"
    if evaluation_type == "free_form_numeric":
        return "free_form_numeric"

    benchmark_task = str(
        metadata.get("benchmark_task")
        or metadata.get("task")
        or metadata.get("source_dataset")
        or ""
    ).strip().lower()
    if any(task in benchmark_task for task in _NUMERIC_TASK_NAMES):
        return "free_form_numeric"
    if choices or any(task in benchmark_task for task in _MCQ_TASK_NAMES):
        return "multiple_choice"

    normalized_gt = _normalize_short_answer_open(gt_answer)
    if normalized_gt and NUMBER_PATTERN.fullmatch(normalized_gt):
        return "free_form_numeric"
    return "short_answer_open"


def normalized_answer_pair(
    *,
    answer_text: str,
    gt_answer: str,
    difficulty_metadata: Dict[str, Any] | None = None,
    controller_metadata: Dict[str, Any] | None = None,
) -> Tuple[str, str, str]:
    metadata = _merge_metadata(difficulty_metadata or {}, controller_metadata or {})
    choices = _coerce_choices(metadata.get("choices"))
    scoring_type = _infer_scoring_type(metadata, gt_answer, choices)

    if scoring_type == "multiple_choice":
        normalized_prediction = _canonicalize_multiple_choice(answer_text, choices) if choices else _normalize_short_answer_open(answer_text)
        normalized_target = str(gt_answer or "").strip()
        if choices:
            normalized_target = _canonicalize_multiple_choice(gt_answer, choices)
        else:
            normalized_target = _normalize_short_answer_open(gt_answer)
        return str(normalized_prediction), str(normalized_target), scoring_type

    if scoring_type == "free_form_numeric":
        return (
            _normalize_free_form_numeric(answer_text),
            _normalize_free_form_numeric(gt_answer),
            scoring_type,
        )

    return (
        _normalize_short_answer_open(answer_text),
        _normalize_short_answer_open(gt_answer),
        scoring_type,
    )


def answers_match(
    *,
    answer_text: str,
    gt_answer: str,
    difficulty_metadata: Dict[str, Any] | None = None,
    controller_metadata: Dict[str, Any] | None = None,
) -> bool:
    normalized_prediction, normalized_target, scoring_type = normalized_answer_pair(
        answer_text=answer_text,
        gt_answer=gt_answer,
        difficulty_metadata=difficulty_metadata,
        controller_metadata=controller_metadata,
    )
    if not normalized_prediction or not normalized_target:
        return False
    if normalized_prediction == normalized_target:
        return True
    if scoring_type == "short_answer_open":
        normalized_prediction_quantity = _normalize_quantity_unit_phrase(normalized_prediction)
        normalized_target_quantity = _normalize_quantity_unit_phrase(normalized_target)
        if (
            normalized_prediction_quantity
            and normalized_target_quantity
            and normalized_prediction_quantity == normalized_target_quantity
        ):
            return True
        return normalized_prediction.startswith(normalized_target + " ")
    return False
