"""Tests for loop + learning: backward compatibility, no learning when disabled."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hammer.core.loop import LoopConfig, run_loop, StabilityReason


def _make_config(tmp_path: Path) -> LoopConfig:
    return LoopConfig(
        repo=tmp_path,
        directive="fix failing tests",
        model="test-model",
        provider="ollama",
        max_iterations=3,
        test_command="pytest",
    )


def test_backward_compatibility_no_learning_config(tmp_path):
    """When no learning config exists, loop runs identically to pre-learning version."""
    # Ensure no learning config
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    # No learning.toml = learning disabled

    config = _make_config(tmp_path)

    with (
        patch("hammer.core.loop.create_engineer_branch", return_value="EngineerExternal/ts"),
        patch("hammer.core.loop.get_provider") as mock_provider,
        patch("hammer.core.loop.extract_diff") as mock_diff,
        patch("hammer.core.loop.validate_patch"),
        patch("hammer.core.loop._apply_patch", return_value="abc123"),
        patch("hammer.core.loop._run_tests", return_value=MagicMock(passed=True)),
        patch("hammer.core.loop.save_state"),
        patch("hammer.core.loop._write_pr_summary"),
    ):
        mock_provider.return_value.generate.return_value = "some response"
        mock_diff.return_value = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"
        result = run_loop(config)

    assert result.reason == StabilityReason.TESTS_PASS
    # No .hammer/learning directory created when learning disabled
    learning_dir = tmp_path / ".hammer" / "learning"
    assert not (learning_dir / "experience_log.jsonl").exists()


def test_learning_enabled_records_experience(tmp_path):
    """When learning enabled, experience is recorded."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "learning.toml").write_text("""
[learning]
enabled = true
epsilon = 0.1
alpha = 0.1
min_runs_before_bias = 0
seed = 42
""")

    config = _make_config(tmp_path)

    with (
        patch("hammer.core.loop.create_engineer_branch", return_value="EngineerExternal/ts"),
        patch("hammer.core.loop.get_provider") as mock_provider,
        patch("hammer.core.loop.extract_diff") as mock_diff,
        patch("hammer.core.loop.validate_patch"),
        patch("hammer.core.loop._apply_patch", return_value="abc123"),
        patch("hammer.core.loop._run_tests", return_value=MagicMock(passed=True)),
        patch("hammer.core.loop.save_state"),
        patch("hammer.core.loop._write_pr_summary"),
    ):
        mock_provider.return_value.generate.return_value = "some response"
        mock_diff.return_value = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"
        result = run_loop(config)

    assert result.reason == StabilityReason.TESTS_PASS
    learning_dir = tmp_path / ".hammer" / "learning"
    assert (learning_dir / "experience_log.jsonl").exists()
    assert (learning_dir / "strategy_scores.json").exists()
    assert (learning_dir / "audit.log").exists()
