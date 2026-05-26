"""
Reward functions for wait-think-answer controller training.

Quick import:
    from rewards import PCoTEpisode, RewardConfig, compute_rewards, LLMJudge
"""

from .episode import PCoTEpisode
from .combined import RewardConfig, compute_rewards, compute_rewards_batch, format_reward_summary
from .reward_accuracy import BatchBalanceContext

__all__ = [
    "PCoTEpisode",
    "RewardConfig",
    "compute_rewards",
    "compute_rewards_batch",
    "format_reward_summary",
    "LLMJudge",
    "BatchBalanceContext",
]


def __getattr__(name):
    if name == "LLMJudge":
        from .judge import LLMJudge

        return LLMJudge
    raise AttributeError("module 'rewards' has no attribute {!r}".format(name))
