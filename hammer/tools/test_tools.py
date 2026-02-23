"""Test runner tools for the tool registry."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("hammer.tools.test")


def _run(cmd: list[str], repo: Path, timeout: int = 120) -> dict:
    result = subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-2000:],
        "success": result.returncode == 0,
    }


def run_pytest(repo: str, args: list[str] | None = None, **_) -> dict:
    cmd = ["python", "-m", "pytest"] + (args or ["-v", "--tb=short"])
    return _run(cmd, Path(repo))


def run_compileall(repo: str, **_) -> dict:
    return _run(["python", "-m", "compileall", "."], Path(repo))


def run_npm_build(repo: str, **_) -> dict:
    return _run(["npm", "run", "build"], Path(repo))


def run_lint(repo: str, **_) -> dict:
    return _run(["python", "-m", "flake8", "."], Path(repo))
