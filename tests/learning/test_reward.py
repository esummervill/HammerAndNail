"""Tests for reward computation."""
import pytest

from hammer.learning.reward import RunResult, compute_reward


def test_reward_tests_and_validation_passed():
    r = RunResult(
        tests_passed=True,
        validation_passed=True,
        runtime_errors=0,
        lint_errors=0,
        retries=0,
    )
    assert compute_reward(r) == 15.0


def test_reward_tests_passed_only():
    r = RunResult(
        tests_passed=True,
        validation_passed=False,
        runtime_errors=0,
        lint_errors=0,
        retries=0,
    )
    assert compute_reward(r) == 10.0


def test_reward_validation_passed_only():
    r = RunResult(
        tests_passed=False,
        validation_passed=True,
        runtime_errors=0,
        lint_errors=0,
        retries=0,
    )
    assert compute_reward(r) == 5.0


def test_reward_penalties():
    r = RunResult(
        tests_passed=False,
        validation_passed=False,
        runtime_errors=2,
        lint_errors=1,
        retries=3,
    )
    # -10 (runtime) - 2 (lint) - 3 (retries) = -15
    assert compute_reward(r) == -15.0


def test_reward_pure_function_no_side_effects():
    """Reward is deterministic and has no external calls."""
    r = RunResult(
        tests_passed=True,
        validation_passed=True,
        runtime_errors=0,
        lint_errors=0,
        retries=1,
    )
    assert compute_reward(r) == 14.0
    assert compute_reward(r) == 14.0
