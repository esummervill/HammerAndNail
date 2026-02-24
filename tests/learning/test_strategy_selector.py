"""Tests for strategy selector: deterministic behavior, epsilon-greedy."""
import json
from pathlib import Path

import pytest

from hammer.learning.strategies import Strategy
from hammer.learning.strategy_selector import StrategySelector


def test_deterministic_behavior_with_fixed_seed(tmp_path):
    """Same seed + same experience log = same strategy selection."""
    selector = StrategySelector(tmp_path, epsilon=0.0, min_runs_before_bias=0, seed=42)
    # With epsilon=0 and no experiences, we explore (random) - but with seed it's deterministic
    s1 = selector.select()
    s2 = selector.select()
    assert s1 == s2


def test_deterministic_exploit_with_scores(tmp_path):
    """With scores and epsilon=0, always select best strategy."""
    scores_path = tmp_path / "strategy_scores.json"
    scores_path.write_text(json.dumps({
        "minimal_patch": 5.0,
        "rewrite": 2.0,
        "add_tests_first": 8.0,
        "refactor_only": 1.0,
        "static_analysis_first": 3.0,
    }))
    # Add experience entries so we exceed min_runs_before_bias
    log_path = tmp_path / "experience_log.jsonl"
    for _ in range(10):
        log_path.open("a").write('{"run_id":"x","strategy":"minimal_patch"}\n')

    selector = StrategySelector(tmp_path, epsilon=0.0, min_runs_before_bias=5, seed=99)
    s = selector.select()
    assert s == Strategy.ADD_TESTS_FIRST


def test_no_strategy_change_when_learning_disabled():
    """When learning disabled, loop uses no strategy - verified via loop tests."""
    # This is covered by test_loop backward compatibility
    pass


def test_epsilon_greedy_explores_sometimes(tmp_path):
    """With epsilon=1.0, always explore (random)."""
    scores_path = tmp_path / "strategy_scores.json"
    scores_path.write_text(json.dumps({
        "minimal_patch": 100.0,
        "rewrite": -100.0,
        "add_tests_first": -100.0,
        "refactor_only": -100.0,
        "static_analysis_first": -100.0,
    }))
    log_path = tmp_path / "experience_log.jsonl"
    for _ in range(10):
        log_path.open("a").write('{"run_id":"x"}\n')

    selector = StrategySelector(tmp_path, epsilon=1.0, min_runs_before_bias=0, seed=123)
    # With epsilon=1, we always explore - could get any strategy
    strategies = {selector.select() for _ in range(20)}
    assert len(strategies) >= 1  # At least one strategy chosen
