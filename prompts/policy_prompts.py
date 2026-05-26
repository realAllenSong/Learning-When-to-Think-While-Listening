"""Controller prompt templates used by the public training release."""

from __future__ import annotations

import re
from typing import Sequence


CONTROLLER_POLICY_PROMPT = "controller_policy_prompt"
FINAL_CONTROLLER_PROMPT = "final_controller_prompt"

DEFAULT_POLICY_PROMPT_VERSION = CONTROLLER_POLICY_PROMPT
DEFAULT_FINAL_POLICY_PROMPT_VERSION = FINAL_CONTROLLER_PROMPT
SUPPORTED_POLICY_PROMPT_VERSIONS = (
    CONTROLLER_POLICY_PROMPT,
    FINAL_CONTROLLER_PROMPT,
)


def _validate_version(version: str) -> str:
    version = str(version or DEFAULT_POLICY_PROMPT_VERSION)
    if version not in SUPPORTED_POLICY_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported controller prompt version: {version}")
    return version


def _question_stem_for_think(question: str) -> str:
    stem = str(question or "").strip()
    stem = re.sub(r"Please choose the answer from the following options:.*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"Options:.*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"Output the final answer in\s*<answer>.*", "", stem, flags=re.IGNORECASE)
    return stem.strip(" \n\t.:")


def _question_block(question: str, question_visible: bool) -> str:
    if question_visible and str(question or "").strip():
        return f"Text copy of the spoken prompt, for reference only:\n{question.strip()}\n"
    return "The question and any choices are provided through audio only.\n"


def _format_choices(choices: Sequence[str] | None) -> str:
    normalized = [str(choice).strip() for choice in (choices or []) if str(choice).strip()]
    if not normalized:
        return ""
    return "Allowed choices:\n" + "\n".join(f"- {choice}" for choice in normalized) + "\n"


def build_omni_system_prompt(
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
    question_visible: bool = False,
) -> str:
    version = _validate_version(version)
    visibility_rule = (
        "You may use the text prompt as a reference, but the audio prefix remains the source of timing evidence."
        if question_visible
        else "Do not assume a text transcript is available; reason from the audio prefix and visible state memory."
    )
    if version == FINAL_CONTROLLER_PROMPT:
        final_style = (
            "Final reasoning should be a compact answer cue: a direct candidate answer plus the shortest "
            "supporting evidence or computation needed for the answer."
        )
    else:
        final_style = (
            "Intermediate reasoning should preserve concrete facts, eliminations, relations, quantities, "
            "corrections, comparisons, or subtotals that help the later final answer."
        )
    return (
        "You are a streaming audio reasoning controller. At each call, choose exactly one valid action: "
        "<wait/>, <think>...</think>, or <answer>...</answer>.\n"
        "Before AUDIO_END, only <wait/> and <think>...</think> are valid. Use <wait/> when the current "
        "audio adds no useful answer-relevant state. Use <think>...</think> only for a short concrete "
        "state update that later calls can reuse.\n"
        "After AUDIO_END, emit one final <think>...</think> and then one <answer>...</answer>.\n"
        f"{visibility_rule}\n"
        f"{final_style}\n"
        "Do not write about speaker tone, pauses, phrasing, confidence, answer shape, or the phrase "
        "'final reasoning state'. Do not guess a final answer before AUDIO_END."
    )


def build_omni_chunk_prompt(
    question: str,
    chunk_index: int,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
) -> str:
    _validate_version(version)
    stem = _question_stem_for_think(question)
    return (
        f"Controller call for audio unit {chunk_index + 1} of {total_chunks}.\n"
        f"{_question_block(question, question_visible)}"
        f"Question focus: {stem or '(audio only)'}\n"
        "Decide whether this prefix adds a useful reasoning state.\n"
        "Output exactly one action. If there is no new concrete answer-relevant state, output <wait/>. "
        "If there is new useful state, output one short <think>...</think>."
    )


def build_omni_answer_instruction(
    *,
    evaluation_type: str = "",
    choices: Sequence[str] | None = None,
    xml_answer_format: bool = True,
) -> str:
    choice_block = _format_choices(choices)
    type_rule = ""
    if str(evaluation_type or "").lower() == "multiple_choice" and choices:
        type_rule = "The final answer must match one of the allowed choices.\n"
    elif str(evaluation_type or "").lower() == "free_form_numeric":
        type_rule = "For numeric questions, output the final requested number or numeric expression.\n"
    wrapper = "<answer>...</answer>" if xml_answer_format else "plain text"
    return (
        f"{choice_block}"
        f"{type_rule}"
        f"Output only the final short answer in {wrapper} format. Do not include reasoning text."
    )


def build_omni_answer_prompt(
    question: str,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
    evaluation_type: str = "",
    choices: Sequence[str] | None = None,
    xml_answer_format: bool = True,
) -> str:
    _validate_version(version)
    return (
        "AUDIO_END.\n"
        f"All {total_chunks} audio units have been heard.\n"
        f"{_question_block(question, question_visible)}"
        "Use the complete audio evidence and visible reasoning memory.\n"
        + build_omni_answer_instruction(
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
    version = _validate_version(version)
    if version == FINAL_CONTROLLER_PROMPT:
        guidance = (
            "Write a compact final <think>: direct candidate answer plus the shortest useful evidence "
            "or computation. Avoid generic task framing."
        )
    else:
        guidance = (
            "Write one compact final <think> that verifies the answer from the full audio and visible state memory."
        )
    return (
        "AUDIO_END.\n"
        f"All {total_chunks} audio units have been heard.\n"
        f"{_question_block(question, question_visible)}"
        f"{guidance}\n"
        "Output exactly one final <think>...</think>."
    )


def build_omni_final_answer_after_think_prompt(
    question: str,
    total_chunks: int,
    question_visible: bool = False,
    version: str = DEFAULT_POLICY_PROMPT_VERSION,
    evaluation_type: str = "",
    choices: Sequence[str] | None = None,
    xml_answer_format: bool = True,
) -> str:
    _validate_version(version)
    return (
        "AUDIO_END.\n"
        f"All {total_chunks} audio units have been heard, and the final <think> state is visible.\n"
        f"{_question_block(question, question_visible)}"
        "Use the final <think> as the primary answer cue, while checking it against the full audio evidence.\n"
        + build_omni_answer_instruction(
            evaluation_type=evaluation_type,
            choices=choices,
            xml_answer_format=xml_answer_format,
        )
    )
