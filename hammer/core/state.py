"""Run state management for HammerAndNail.

State is stored as JSON files in <repo>/.hammer/runs/<run_id>.json.
No database required. State is rebuilt from disk each iteration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PatchRecord:
    iteration: int
    patch_hash: str
    commit_sha: str
    lines_changed: int
    timestamp: str


@dataclass
class TestResult:
    iteration: int
    command: str
    exit_code: int
    stdout: str
    stderr: str
    passed: bool
    timestamp: str


@dataclass
class RunState:
    run_id: str
    repo_path: str
    directive: str
    iteration: int
    branch: str
    patches: list[PatchRecord] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    stable: bool = False
    failure_reason: str = ""


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _runs_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_state(state: RunState, runs_dir: Path) -> Path:
    _runs_dir(runs_dir)
    path = runs_dir / f"{state.run_id}.json"

    data = {
        "run_id": state.run_id,
        "repo_path": state.repo_path,
        "directive": state.directive,
        "iteration": state.iteration,
        "branch": state.branch,
        "patches": [asdict(p) for p in state.patches],
        "test_results": [asdict(t) for t in state.test_results],
        "stable": state.stable,
        "failure_reason": state.failure_reason,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_state(run_id: str, runs_dir: Path) -> RunState:
    path = runs_dir / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No state file for run_id={run_id}: {path}")
    data = json.loads(path.read_text())
    data["patches"] = [PatchRecord(**p) for p in data.get("patches", [])]
    data["test_results"] = [TestResult(**t) for t in data.get("test_results", [])]
    return RunState(**data)
