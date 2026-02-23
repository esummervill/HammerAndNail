"""Docker Compose tools for the tool registry."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("hammer.tools.docker")


def _compose(subcmd: list[str], repo: Path, timeout: int = 60) -> dict:
    result = subprocess.run(
        ["docker", "compose"] + subcmd,
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


def run_docker_compose_ps(repo: str, **_) -> dict:
    return _compose(["ps"], Path(repo))


def run_docker_compose_logs(repo: str, service: str = "", **_) -> dict:
    cmd = ["logs", "--tail=50"]
    if service:
        cmd.append(service)
    return _compose(cmd, Path(repo))


def run_docker_compose_up(repo: str, **_) -> dict:
    return _compose(["up", "-d", "--build"], Path(repo), timeout=300)


def run_docker_compose_down(repo: str, **_) -> dict:
    return _compose(["down"], Path(repo))
