"""Hammer learning module — deterministic strategy selection and reward-driven adaptation.

When enabled via configs/learning.toml, Hammer adapts strategy selection across runs
using epsilon-greedy policy over strategy scores updated by a deterministic reward function.
All learning updates are logged to .hammer/learning/audit.log.
"""
from __future__ import annotations

from .strategies import Strategy
from .strategy_selector import StrategySelector
from .reward import RunResult, compute_reward
from .experience_store import ExperienceStore
from .state_encoder import encode_repo_state

__all__ = [
    "Strategy",
    "StrategySelector",
    "RunResult",
    "compute_reward",
    "ExperienceStore",
    "encode_repo_state",
]
