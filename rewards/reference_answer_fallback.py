"""
Reference-model fallback judging for R_a correctness.

This module keeps the training-side fallback narrow and auditable:
- deterministic rule-based matching runs first
- only deterministic-fail cases with an extractable final answer are eligible
  for rescue
- cases with no extractable final answer are never sent to the model

The actual model inference is delegated to a caller-supplied object that
implements `judge_answer_equivalence_batch(requests)`.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, Iterable, List, Sequence

from .answer_scoring import answers_match


REFERENCE_ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are a strict training-time answer equivalence judge. "
    "Do not output chain-of-thought, explanations, markdown, bullet points, "
    "or code fences. Return exactly one minified JSON object."
)

REFERENCE_ANSWER_JUDGE_PROMPT = """Return EXACTLY one minified JSON object and nothing else.
The JSON schema is:
{{"model_final_answer":"...","judge_result":"<correct>|<incorrect>|<no_final_answer>"}}

Rules:
- Do not output any prose before or after the JSON.
- Do not output 'Thinking Process'.
- Do not output markdown or code fences.
- Use only the information in the question, choices (if provided), ground-truth answer, and model response.
- Ignore politeness, filler, and other non-factual wording.
- Treat spoken paraphrases as equivalent when they clearly refer to the same answer.
- Treat digit-vs-number-word variants as equivalent when they denote the same quantity.
- Treat equivalent quantity+unit variants as correct when both quantity and unit match, such as "30 liters" vs "thirty liters" or "45 seconds" vs "forty-five seconds".
- If choices are provided, judge equivalence relative to those choices.
- This fallback is used only after deterministic exact-match/rule scoring says the sample is incorrect.
- Rescue semantically equivalent final answers even when the surface form differs.
- Treat compatible quantity conversions as correct when they denote the same amount, such as "90 minutes" vs "one and a half hours" and "50 cents" vs "half a dollar".
- Do not mark answers correct when the quantity differs or when the unit meaning changes.
- If the model response does not contain a final answer, use "<no_final_answer>" for both fields.

Question:
{question}

Choices:
{choices_block}

Ground-truth answer:
{gt_answer}

Model response:
{model_output}

Output JSON only."""

REFERENCE_ANSWER_JUDGE_RETRY_PROMPT = """Your previous reply was invalid because it was not a single JSON object.
Repair the scoring now.

Return EXACTLY one minified JSON object and nothing else:
{{"model_final_answer":"...","judge_result":"<correct>|<incorrect>|<no_final_answer>"}}

No prose. No thinking. No markdown. No code fences.

Question:
{question}

Choices:
{choices_block}

Ground-truth answer:
{gt_answer}

Model response:
{model_output}

Your previous invalid reply:
{prior_raw}

Output JSON only."""

_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> Dict[str, object] | None:
    payload = str(text or "").strip()
    if not payload:
        return None
    match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_judge_label(text: object) -> str:
    value = str(text or "").strip().lower()
    if "no_final_answer" in value:
        return "<no_final_answer>"
    if "correct" in value and "incorrect" not in value:
        return "<correct>"
    if "incorrect" in value:
        return "<incorrect>"
    return "<incorrect>"


def _extract_explicit_judge_label(text: object) -> str:
    value = str(text or "").strip().lower()
    if not value:
        return ""
    if value in {"<correct>", "<incorrect>", "<no_final_answer>", "correct", "incorrect", "no_final_answer"}:
        return _normalize_judge_label(value)
    matches = re.findall(
        r"(?m)^\s*(<correct>|<incorrect>|<no_final_answer>|correct|incorrect|no_final_answer)\s*$",
        value,
    )
    return _normalize_judge_label(matches[-1]) if matches else ""


def resolve_judge_decision(
    *,
    parsed: Dict[str, object] | None,
    raw: str,
    gt_answer: str,
    model_output: str,
    evaluation_type: str = "",
    choices: Sequence[str] | None = None,
) -> Dict[str, object]:
    del evaluation_type
    parsed = dict(parsed or {})
    model_final_answer = str(parsed.get("model_final_answer", "")).strip()
    if not model_final_answer:
        model_final_answer = _extract_final_answer_span(model_output)
    if parsed:
        judge_result = _normalize_judge_label(parsed.get("judge_result", ""))
    else:
        judge_result = _extract_explicit_judge_label(raw)
    if not model_final_answer:
        model_final_answer = "<no_final_answer>"
        judge_result = "<no_final_answer>"
    if judge_result not in {"<no_final_answer>", "<correct>", "<incorrect>"}:
        judge_result = "<incorrect>"
    return {
        "judge_model_final_answer": model_final_answer,
        "judge_result": judge_result,
        "judge_correct": judge_result == "<correct>",
    }


def _merged_metadata(*payloads: Dict[str, Any]) -> Dict[str, Any]:
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


def _format_choice_block(choices: Sequence[str] | None) -> str:
    normalized = [str(choice).strip() for choice in (choices or []) if str(choice).strip()]
    if not normalized:
        return "(none)"
    return "\n".join(f"- {choice}" for choice in normalized)


def build_reference_answer_judge_prompt(
    *,
    question: str,
    gt_answer: str,
    model_output: str,
    choices: Sequence[str] | None = None,
) -> str:
    return REFERENCE_ANSWER_JUDGE_PROMPT.format(
        question=str(question or "").strip(),
        choices_block=_format_choice_block(choices),
        gt_answer=str(gt_answer or "").strip(),
        model_output=str(model_output or "").strip(),
    )


def build_reference_answer_judge_retry_prompt(
    *,
    question: str,
    gt_answer: str,
    model_output: str,
    prior_raw: str,
    choices: Sequence[str] | None = None,
) -> str:
    return REFERENCE_ANSWER_JUDGE_RETRY_PROMPT.format(
        question=str(question or "").strip(),
        choices_block=_format_choice_block(choices),
        gt_answer=str(gt_answer or "").strip(),
        model_output=str(model_output or "").strip(),
        prior_raw=str(prior_raw or "").strip(),
    )


def resolve_reference_answer_judge_output(
    *,
    raw: str,
    gt_answer: str,
    model_output: str,
    choices: Sequence[str] | None = None,
) -> Dict[str, Any]:
    parsed = extract_json_object(raw) or {}
    return resolve_judge_decision(
        parsed=parsed,
        raw=raw,
        gt_answer=str(gt_answer or ""),
        model_output=str(model_output or ""),
        evaluation_type="",
        choices=list(choices or []),
    )


def has_reference_answer_judge_json(raw: str) -> bool:
    return extract_json_object(raw) is not None


def _extract_final_answer_span(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    matches = _ANSWER_TAG_PATTERN.findall(value)
    if matches:
        return str(matches[-1] or "").strip()
    return value


def _reset_episode_answer_cache(episode: Any) -> None:
    episode.answer_rule_correct = None
    episode.answer_correct_override = None
    episode.answer_fallback_invoked = False
    episode.answer_fallback_rescued = False
    episode.answer_fallback_short_circuit_no_final_answer = False
    episode.answer_fallback_source = ""
    episode.answer_fallback_judge_result = ""
    episode.answer_fallback_judge_answer = ""


def prepare_reference_answer_fallback_batch(
    episodes: Sequence[Any],
    judge: Any | None = None,
) -> None:
    pending_requests: List[Dict[str, Any]] = []
    pending_episodes: List[Any] = []

    for index, episode in enumerate(episodes):
        _reset_episode_answer_cache(episode)
        metadata = _merged_metadata(
            getattr(episode, "difficulty_metadata", {}) or {},
            getattr(episode, "controller_metadata", {}) or {},
        )
        choices = _coerce_choices(metadata.get("choices"))
        rule_correct = answers_match(
            answer_text=str(getattr(episode, "answer", "") or ""),
            gt_answer=str(getattr(episode, "gt_answer", "") or ""),
            difficulty_metadata=getattr(episode, "difficulty_metadata", {}) or {},
            controller_metadata=getattr(episode, "controller_metadata", {}) or {},
        )
        episode.answer_rule_correct = bool(rule_correct)
        episode.answer_correct_override = bool(rule_correct)
        episode.answer_fallback_source = "rule_correct" if rule_correct else "rule_incorrect"

        if rule_correct:
            continue

        final_answer_span = _extract_final_answer_span(str(getattr(episode, "answer", "") or ""))
        if not final_answer_span:
            episode.answer_fallback_short_circuit_no_final_answer = True
            episode.answer_fallback_source = "no_final_answer"
            episode.answer_correct_override = False
            continue

        if judge is None:
            episode.answer_fallback_source = "rule_incorrect_no_judge"
            episode.answer_correct_override = False
            continue

        pending_requests.append(
            {
                "episode_index": int(index),
                "question": str(getattr(episode, "question", "") or ""),
                "gt_answer": str(getattr(episode, "gt_answer", "") or ""),
                "model_output": f"<answer>{final_answer_span}</answer>",
                "choices": list(choices),
            }
        )
        pending_episodes.append(episode)

    if not pending_requests:
        return

    judge_method = getattr(judge, "judge_answer_equivalence_batch", None)
    if not callable(judge_method):
        raise TypeError("answer_fallback_judge must implement judge_answer_equivalence_batch(requests)")

    decisions = list(judge_method(pending_requests))
    if len(decisions) != len(pending_requests):
        raise RuntimeError(
            "answer_fallback_judge returned {} decisions for {} requests".format(
                len(decisions), len(pending_requests)
            )
        )

    for episode, decision in zip(pending_episodes, decisions):
        judge_result = str(decision.get("judge_result") or "").strip() or "<incorrect>"
        judge_correct = bool(decision.get("judge_correct")) or judge_result == "<correct>"
        episode.answer_fallback_invoked = True
        episode.answer_fallback_rescued = bool(judge_correct)
        episode.answer_fallback_judge_result = judge_result
        episode.answer_fallback_judge_answer = str(decision.get("judge_model_final_answer") or "").strip()
        episode.answer_correct_override = bool(judge_correct)
        episode.answer_fallback_source = (
            "reference_fallback_rescued" if judge_correct else "reference_fallback_rejected"
        )
