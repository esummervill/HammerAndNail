"""Git branch isolation enforcement for HammerAndNail.

All engineer work happens inside EngineerExternal/<timestamp> branches.
Never operates on main. Never auto-merges.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("hammer.core.branch_manager")

BRANCH_PREFIX = "EngineerExternal/"


class BranchViolation(Exception):
    """Raised when engineer attempts a disallowed branch operation."""


def _run_git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def current_branch(repo: Path) -> str:
    result = _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def is_engineer_branch(branch: str) -> bool:
    return branch.startswith(BRANCH_PREFIX)


def create_engineer_branch(repo: Path) -> str:
    """Checkout main, pull, and create EngineerExternal/<timestamp>.

    Returns the new branch name.
    Raises BranchViolation on any git failure.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch_name = f"{BRANCH_PREFIX}{timestamp}"

    result = _run_git("checkout", "main", cwd=repo)
    if result.returncode != 0:
        raise BranchViolation(f"Cannot checkout main: {result.stderr.strip()}")

    result = _run_git("pull", "--ff-only", cwd=repo)
    if result.returncode != 0:
        logger.warning("git pull failed (continuing): %s", result.stderr.strip())

    result = _run_git("checkout", "-b", branch_name, cwd=repo)
    if result.returncode != 0:
        raise BranchViolation(f"Cannot create branch {branch_name}: {result.stderr.strip()}")

    logger.info("Created branch: %s", branch_name)
    return branch_name


def assert_on_engineer_branch(repo: Path) -> None:
    """Raise BranchViolation if not on an EngineerExternal branch."""
    branch = current_branch(repo)
    if not is_engineer_branch(branch):
        raise BranchViolation(
            f"Not on an engineer branch: '{branch}'. "
            f"Expected prefix '{BRANCH_PREFIX}'."
        )
