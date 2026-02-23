import subprocess
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from hammer.core.branch_manager import (
    BranchViolation,
    is_engineer_branch,
    create_engineer_branch,
    current_branch,
    BRANCH_PREFIX,
)


def test_branch_prefix():
    assert BRANCH_PREFIX == "EngineerExternal/"


def test_is_engineer_branch_true():
    assert is_engineer_branch("EngineerExternal/20260223-120000") is True


def test_is_engineer_branch_false():
    assert is_engineer_branch("main") is False
    assert is_engineer_branch("feature/my-feature") is False


def test_create_branch_returns_prefixed_name(tmp_path):
    with patch("hammer.core.branch_manager._run_git") as mock_git:
        mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
        branch = create_engineer_branch(tmp_path)
    assert branch.startswith(BRANCH_PREFIX)


def test_create_branch_raises_on_checkout_failure(tmp_path):
    with patch("hammer.core.branch_manager._run_git") as mock_git:
        mock_git.return_value = MagicMock(
            returncode=1, stdout="", stderr="not a git repo"
        )
        with pytest.raises(BranchViolation, match="Cannot checkout main"):
            create_engineer_branch(tmp_path)
