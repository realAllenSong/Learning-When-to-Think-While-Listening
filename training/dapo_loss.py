"""
Token-level DAPO-style loss helpers.

These utilities keep the controller trainer structure intact while
swapping the actor update objective from sequence-level clipped GRPO/PPO-style
loss to a token-level clipped objective. They also provide the extra shaping
pieces we need for a closer DAPO-style setup, including explicit overlong
penalties instead of relying only on decode caps.
"""

from __future__ import annotations

from typing import Optional

import torch

from .grpo_loss import masked_mean


def clipped_dapo_token_objective(
    policy_token_logprobs: torch.Tensor,
    behavior_token_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> torch.Tensor:
    policy_token_logprobs = policy_token_logprobs.to(dtype=torch.float32)
    behavior_token_logprobs = behavior_token_logprobs.to(policy_token_logprobs.dtype)
    advantages = advantages.to(policy_token_logprobs.dtype).unsqueeze(-1)

    if token_mask is None:
        token_mask = torch.ones_like(policy_token_logprobs, dtype=torch.bool)
    mask = token_mask.to(policy_token_logprobs.dtype)

    log_ratio = (policy_token_logprobs - behavior_token_logprobs).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(epsilon_low), 1.0 + float(epsilon_high))

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    per_token = torch.where(advantages >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))
    per_token = per_token * mask
    return masked_mean(per_token, mask=token_mask, dim=-1)


def clip_fraction_from_token_logprobs(
    policy_token_logprobs: torch.Tensor,
    behavior_token_logprobs: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> torch.Tensor:
    if token_mask is None:
        token_mask = torch.ones_like(policy_token_logprobs, dtype=torch.bool)
    log_ratio = (policy_token_logprobs - behavior_token_logprobs.to(policy_token_logprobs.dtype)).clamp(
        min=-20.0,
        max=20.0,
    )
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(epsilon_low), 1.0 + float(epsilon_high))
    outside = (ratio - clipped_ratio).abs().gt(1e-8).to(policy_token_logprobs.dtype)
    return masked_mean(outside, mask=token_mask, dim=-1).mean()


def approximate_observed_kl_from_token_logprobs(
    policy_token_logprobs: torch.Tensor,
    behavior_token_logprobs: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if token_mask is None:
        token_mask = torch.ones_like(policy_token_logprobs, dtype=torch.bool)
    log_ratio = (policy_token_logprobs - behavior_token_logprobs.to(policy_token_logprobs.dtype)).clamp(
        min=-20.0,
        max=20.0,
    )
    ratio = torch.exp(log_ratio)
    per_token = (ratio - 1.0) - log_ratio
    return masked_mean(per_token, mask=token_mask, dim=-1).mean()


def overlong_shaping_penalty(
    token_mask: torch.Tensor,
    *,
    threshold_tokens: int = 32,
    penalty_slope: float = 0.03,
    penalty_cap: float = 1.0,
) -> torch.Tensor:
    """
    Linear per-sequence overlong shaping used by the DAPO token-level line.

    The penalty stays at zero until the effective decoded length exceeds the
    threshold, then increases linearly with each extra token and is capped to
    avoid dominating the policy objective.
    """

    lengths = token_mask.to(dtype=torch.float32).sum(dim=-1)
    if threshold_tokens <= 0 or penalty_slope <= 0:
        return torch.zeros_like(lengths)
    excess = (lengths - float(threshold_tokens)).clamp(min=0.0)
    penalty = excess * float(penalty_slope)
    if penalty_cap > 0:
        penalty = penalty.clamp(max=float(penalty_cap))
    return penalty
