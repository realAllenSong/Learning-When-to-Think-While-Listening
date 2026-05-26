"""Reward-judge prompt templates used by the public DAPO training path."""

from __future__ import annotations


THOUGHT_QUALITY_JUDGE_PROMPT = "thought_quality_judge_prompt"
CHAIN_CONSISTENCY_JUDGE_PROMPT = "chain_consistency_judge_prompt"


THOUGHT_QUALITY_JUDGE_PROMPTS = {
    THOUGHT_QUALITY_JUDGE_PROMPT: """\
You are scoring a single visible <think> state emitted by a streaming audio reasoning controller.

Return exactly one minified JSON object:
{{"score":0.0,"reason":"..."}}

Scoring target:
- 1.0: short, concrete, and useful for the final answer.
- 0.5: partly useful but too vague, too long, redundant, or weakly grounded.
- 0.0: generic, meta, unsupported, answer-only leakage, malformed, or unrelated.

The score should reward compact semantic state, not polished prose. A good state records a concrete fact,
elimination, relation, quantity, correction, comparison, or subtotal that helps a later answer. Penalize
states that talk about the speaker, tone, pause, uncertainty, wording, answer shape, or the phrase
"final reasoning state".

Question:
{question}

Think kind:
{think_kind_label}

Expectation:
{think_kind_expectation}

Answer-leak rule:
{answer_leak_rule}

Audio/caption evidence for this step:
{caption}

Segment span:
{segment_span}

Earlier visible state chain:
{earlier_state_chain}

Previous state:
{previous_state}

Candidate <think>:
{think}

Final answer:
{final_answer}

Reference answer:
{reference_answer}

Output JSON only.""",
}


CHAIN_CONSISTENCY_JUDGE_PROMPTS = {
    CHAIN_CONSISTENCY_JUDGE_PROMPT: """\
You are scoring whether the visible reasoning chain supports the final answer for a streaming audio
reasoning controller.

Return exactly one minified JSON object:
{{"score":0.0,"reason":"..."}}

Scoring target:
- 1.0: the chain contains concrete states that support the final answer and are consistent with the
  reference answer.
- 0.5: the chain is partly useful but incomplete, redundant, or weakly connected to the answer.
- 0.0: the chain supports a wrong answer, contradicts the final answer, relies on meta commentary, or
  fails to provide useful state.

Do not reward a fluent chain that reaches the wrong answer. The chain should justify the answer with
content from the audio rather than delivery cues or task-format comments.

Question:
{question}

Visible state chain:
{state_chain}

Final model answer:
{answer}

Reference answer:
{reference_answer}

Output JSON only.""",
}


def get_rc_prompt_template(version: str) -> str:
    if version not in CHAIN_CONSISTENCY_JUDGE_PROMPTS:
        raise ValueError(f"Unsupported chain-consistency judge prompt version: {version}")
    return CHAIN_CONSISTENCY_JUDGE_PROMPTS[version]


def get_rt_prompt_template(version: str) -> str:
    if version not in THOUGHT_QUALITY_JUDGE_PROMPTS:
        raise ValueError(f"Unsupported thought-quality judge prompt version: {version}")
    return THOUGHT_QUALITY_JUDGE_PROMPTS[version]
