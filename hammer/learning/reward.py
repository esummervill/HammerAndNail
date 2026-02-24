"""Deterministic reward function. Pure function, no external calls, no LLM involvement."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunResult:
    """Structured result for reward computation."""

    tests_passed: bool
    validation_passed: bool
    runtime_errors: int
    lint_errors: int
    retries: int


def compute_reward(run_result: RunResult) -> float:
    """Compute deterministic reward from run result. Pure function."""
    reward = 0.0
    if run_result.tests_passed:
        reward += 10.0
    if run_result.validation_passed:
        reward += 5.0
    reward -= 5.0 * run_result.runtime_errors
    reward -= 2.0 * run_result.lint_errors
    reward -= 1.0 * run_result.retries
    return reward
