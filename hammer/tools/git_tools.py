"""Git tool implementations for the tool registry.

All commands are whitelisted and run as subprocesses.
No shell=True. No arbitrary commands.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("hammer.tools.git")


def _git(*args: str, repo: Path, timeout: int = 15) -> dict:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def run_git_status(repo: str, **_) -> dict:
    return _git("status", "--short", repo=Path(repo))


def run_git_diff(repo: str, **_) -> dict:
    return _git("diff", repo=Path(repo))


def run_git_apply(repo: str, patch_file: str, **_) -> dict:
    return _git("apply", patch_file, repo=Path(repo))


def run_git_apply_check(repo: str, patch_file: str, **_) -> dict:
    return _git("apply", "--check", patch_file, repo=Path(repo))


def run_git_log(repo: str, n: int = 5, **_) -> dict:
    return _git("log", "--oneline", f"-{n}", repo=Path(repo))
