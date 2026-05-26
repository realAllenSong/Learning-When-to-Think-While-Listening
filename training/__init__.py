"""Training utilities for wait-think-answer DAPO.

This package intentionally uses lazy imports so lightweight tools such as
data converters can reuse small schema/prompt helpers without importing the
full trainable stack and its heavy dependencies.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "clean_chunk_caption": ("training.caption_utils", "clean_chunk_caption"),
    "clean_chunk_caption_qwen3": ("training.caption_utils", "clean_chunk_caption_qwen3"),
    "resolve_caption_cleaner": ("training.caption_utils", "resolve_caption_cleaner"),
    "CheckpointArtifact": ("training.checkpointing", "CheckpointArtifact"),
    "prune_step_checkpoints": ("training.checkpointing", "prune_step_checkpoints"),
    "PCoTTrainSample": ("training.dataset", "PCoTTrainSample"),
    "apply_question_visibility_override": ("training.dataset", "apply_question_visibility_override"),
    "load_pcot_dataset": ("training.dataset", "load_pcot_dataset"),
    "masked_mean": ("training.grpo_loss", "masked_mean"),
    "clipped_dapo_token_objective": ("training.dapo_loss", "clipped_dapo_token_objective"),
    "sequence_logprob_from_token_logprobs": ("training.grpo_loss", "sequence_logprob_from_token_logprobs"),
    "grpo_objective": ("training.grpo_loss", "grpo_objective"),
    "approximate_reverse_kl": ("training.grpo_loss", "approximate_reverse_kl"),
    "grpo_loss": ("training.grpo_loss", "grpo_loss"),
    "OmniTurnExample": ("training.omni_actor", "OmniTurnExample"),
    "build_turn_training_examples": ("training.omni_actor", "build_turn_training_examples"),
    "load_audio_arrays": ("training.omni_actor", "load_audio_arrays"),
    "prepare_turn_model_inputs": ("training.omni_actor", "prepare_turn_model_inputs"),
    "prepare_turn_model_inputs_batch": ("training.omni_actor", "prepare_turn_model_inputs_batch"),
    "OmniActorUpdateConfig": ("training.omni_updater", "OmniActorUpdateConfig"),
    "TrainableOmniActorUpdater": ("training.omni_updater", "TrainableOmniActorUpdater"),
    "PolicyBackend": ("training.policy", "PolicyBackend"),
    "TeacherPolicyBackend": ("training.policy", "TeacherPolicyBackend"),
    "HeuristicCaptionPolicyBackend": ("training.policy", "HeuristicCaptionPolicyBackend"),
    "OmniVLLMPolicyBackend": ("training.policy", "OmniVLLMPolicyBackend"),
    "LocalVLLMServiceConfig": ("training.vllm_service", "LocalVLLMServiceConfig"),
    "LocalVLLMServiceController": ("training.vllm_service", "LocalVLLMServiceController"),
    "StreamingRollout": ("training.schema", "StreamingRollout"),
    "StreamingStep": ("training.schema", "StreamingStep"),
    "TrainerConfig": ("training.trainer", "TrainerConfig"),
    "StreamingGRPOTrainer": ("training.trainer", "StreamingGRPOTrainer"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'training' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
