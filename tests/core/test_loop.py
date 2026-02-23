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


def test_loop_exits_on_max_iterations(tmp_path):
    config = _make_config(tmp_path)

    with (
        patch("hammer.core.loop.create_engineer_branch", return_value="EngineerExternal/ts"),
        patch("hammer.core.loop.get_provider") as mock_provider,
        patch("hammer.core.loop.extract_diff") as mock_diff,
        patch("hammer.core.loop.validate_patch"),
        patch("hammer.core.loop._apply_patch", return_value="abc123"),
        patch("hammer.core.loop._run_tests", return_value=MagicMock(passed=False)),
        patch("hammer.core.loop.save_state"),
    ):
        mock_provider.return_value.generate.return_value = "some response"
        mock_diff.return_value = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"
        result = run_loop(config)

    assert result.reason == StabilityReason.MAX_ITERATIONS


def test_loop_exits_when_no_diff(tmp_path):
    config = _make_config(tmp_path)

    with (
        patch("hammer.core.loop.create_engineer_branch", return_value="EngineerExternal/ts"),
        patch("hammer.core.loop.get_provider") as mock_provider,
        patch("hammer.core.loop.extract_diff") as mock_diff,
        patch("hammer.core.loop.save_state"),
    ):
        from hammer.core.diff_manager import DiffValidationError
        mock_provider.return_value.generate.return_value = "No changes needed."
        mock_diff.side_effect = DiffValidationError("no unified diff")
        result = run_loop(config)

    assert result.reason == StabilityReason.NO_DIFF


def test_loop_exits_when_tests_pass(tmp_path):
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
