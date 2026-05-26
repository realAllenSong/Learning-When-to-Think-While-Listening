"""
Core GRPO loss math utilities.

This module is rollout-backend agnostic and can be reused by a future
trainable Qwen2.5-Omni actor updater.
"""

from typing import Optional

import torch


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    mask = mask.to(values.dtype)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (values * mask).sum(dim=dim) / denom


def sequence_logprob_from_token_logprobs(
    token_logprobs: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if token_mask is None:
        return token_logprobs.sum(dim=-1)
    return (token_logprobs * token_mask.to(token_logprobs.dtype)).sum(dim=-1)


def normalized_sequence_logprob_from_token_logprobs(
    token_logprobs: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return the mean log-probability per supervised token.

    For long language-model completions, using raw summed sequence log-probs
    inside PPO-style ratios makes the ratio scale exponentially with sequence
    length. The normalized variant keeps the ratio sensitive to average
    token-level drift instead of turn length.
    """
    if token_mask is None:
        return token_logprobs.mean(dim=-1)
    return masked_mean(token_logprobs, mask=token_mask, dim=-1)


def grpo_objective(
    policy_token_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sequence-level policy-gradient style objective used inside GRPO.
    """
    seq_logprob = sequence_logprob_from_token_logprobs(policy_token_logprobs, token_mask=token_mask)
    return advantages.to(seq_logprob.dtype) * seq_logprob


def clipped_grpo_objective(
    policy_token_logprobs: torch.Tensor,
    behavior_sequence_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> torch.Tensor:
    """
    PPO-style clipped sequence objective for GRPO-style updates.

    The rollout policy is treated as a frozen behavior policy for the current
    update. This becomes meaningful once we do more than one optimization pass
    over the same sampled trajectories.
    """
    seq_logprob = normalized_sequence_logprob_from_token_logprobs(
        policy_token_logprobs,
        token_mask=token_mask,
    )
    behavior_sequence_logprobs = behavior_sequence_logprobs.to(seq_logprob.dtype)
    advantages = advantages.to(seq_logprob.dtype)

    log_ratio = (seq_logprob - behavior_sequence_logprobs).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(epsilon_low), 1.0 + float(epsilon_high))

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    return torch.where(advantages >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))


def clip_fraction_from_sequence_logprobs(
    policy_sequence_logprobs: torch.Tensor,
    behavior_sequence_logprobs: torch.Tensor,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> torch.Tensor:
    """
    Fraction of sequences whose PPO-style ratio falls outside the clip window.
    """
    log_ratio = (
        policy_sequence_logprobs - behavior_sequence_logprobs.to(policy_sequence_logprobs.dtype)
    ).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(epsilon_low), 1.0 + float(epsilon_high))
    return (ratio - clipped_ratio).abs().gt(1e-8).to(policy_sequence_logprobs.dtype).mean()


def approximate_observed_kl_from_sequence_logprobs(
    policy_sequence_logprobs: torch.Tensor,
    behavior_sequence_logprobs: torch.Tensor,
) -> torch.Tensor:
    """
    PPO-style observed KL proxy between the current policy and the frozen
    behavior policy that generated the rollout.
    """
    log_ratio = (
        policy_sequence_logprobs - behavior_sequence_logprobs.to(policy_sequence_logprobs.dtype)
    ).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    return ((ratio - 1.0) - log_ratio).mean()


def approximate_reverse_kl(
    policy_token_logprobs: torch.Tensor,
    reference_token_logprobs: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    A stable non-negative reverse-KL proxy on sampled tokens.
    """
    log_ratio = (policy_token_logprobs - reference_token_logprobs).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    per_token_proxy = (ratio - 1.0) - log_ratio
    return masked_mean(per_token_proxy, mask=token_mask, dim=-1)


def grpo_loss(
    policy_token_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    reference_token_logprobs: Optional[torch.Tensor] = None,
    behavior_sequence_logprobs: Optional[torch.Tensor] = None,
    token_mask: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> torch.Tensor:
    """
    Return a scalar loss to minimize.
    """
    if behavior_sequence_logprobs is not None:
        objective = clipped_grpo_objective(
            policy_token_logprobs=policy_token_logprobs,
            behavior_sequence_logprobs=behavior_sequence_logprobs,
            advantages=advantages,
            token_mask=token_mask,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
        )
    else:
        objective = grpo_objective(
            policy_token_logprobs=policy_token_logprobs,
            advantages=advantages,
            token_mask=token_mask,
        )
    if reference_token_logprobs is not None and beta > 0:
        objective = objective - beta * approximate_reverse_kl(
            policy_token_logprobs=policy_token_logprobs,
            reference_token_logprobs=reference_token_logprobs,
            token_mask=token_mask,
        )
    return -objective.mean()
