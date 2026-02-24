"""Shared runner entrypoint used by the conversational plan and existing CLI."""
from __future__ import annotations

import sys

from pathlib import Path

from . import __version__
from .core.loop import LoopConfig, StabilityReason, run_loop


def run_engineering_loop(
    repo_path: Path,
    directive_text: str,
    model: str,
    provider: str,
    max_iterations: int,
    test_command: str,
    max_diff_lines: int,
    initial_diff: str | None = None,
    stop_after_initial_diff: bool = False,
) -> None:
    print(f"HammerAndNail v{__version__}")
    print(f"  Repo:      {repo_path}")
    print(f"  Model:     {model} via {provider}")
    print(f"  Max iter:  {max_iterations}")
    print(f"  Tests:     {test_command}")
    print("")

    config = LoopConfig(
        repo=repo_path,
        directive=directive_text,
        model=model,
        provider=provider,
        max_iterations=max_iterations,
        test_command=test_command,
        max_diff_lines=max_diff_lines,
        initial_diff=initial_diff,
        stop_after_initial_diff=stop_after_initial_diff,
    )

    result = run_loop(config)

    print("")
    print(f"Loop complete — {result.reason.value}")
    print(f"  Run ID:     {result.run_id}")
    print(f"  Branch:     {result.branch}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Patches:    {len(result.state.patches)}")

    if result.reason == StabilityReason.TESTS_PASS:
        sys.exit(0)
    else:
        sys.exit(1)
