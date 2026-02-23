"""Core 7-phase pipeline loop for HammerAndNail.

The LLM never owns the loop. The runtime drives all phase transitions.
State is loaded fresh each iteration. No conversation memory.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .branch_manager import create_engineer_branch
from .diff_manager import DiffValidationError, extract_diff
from .state import PatchRecord, RunState, TestResult, new_run_id, save_state
from .validator import PatchValidationError, validate_patch
from ..llm.model_router import DEFAULT_PROVIDER, get_provider
from ..tools.registry import ToolRegistry

logger = logging.getLogger("hammer.core.loop")

FORBIDDEN_PREFIXES = (".venv/", ".hammer/", ".pytest_cache/", "__pycache__/")
FORBIDDEN_SUFFIXES = (".pyc", ".egg-info")


class StabilityReason(str, Enum):
    TESTS_PASS = "tests_pass"
    NO_DIFF = "no_diff"
    MAX_ITERATIONS = "max_iterations"
    DUPLICATE_PATCH = "duplicate_patch"
    ERROR = "error"


@dataclass
class LoopResult:
    run_id: str
    branch: str
    iterations: int
    reason: StabilityReason
    state: RunState


@dataclass
class LoopConfig:
    repo: Path
    directive: str
    model: str
    provider: str = DEFAULT_PROVIDER
    max_iterations: int = 10
    test_command: str = "pytest"
    max_diff_lines: int = 500
    temperature: float = 0.2
    max_tokens: int = 8192
    initial_diff: str | None = None


def run_loop(config: LoopConfig) -> LoopResult:
    """Execute the 7-phase pipeline loop against the target repo."""
    run_id = new_run_id()
    runs_dir = config.repo / ".hammer" / "runs"
    provider = get_provider(config.provider)
    registry = ToolRegistry()

    # Phase 0: Branch Safety
    branch = create_engineer_branch(config.repo)
    logger.info("Run %s starting on branch %s", run_id, branch)

    state = RunState(
        run_id=run_id,
        repo_path=str(config.repo),
        directive=config.directive,
        iteration=0,
        branch=branch,
    )

    seen_patch_hashes: set[str] = set()

    initial_diff_consumed = False
    for iteration in range(config.max_iterations):
        state.iteration = iteration
        logger.info("--- Iteration %d ---", iteration)

        # Phase 1: Read State
        save_state(state, runs_dir)

        # Phase 2: Generate
        if config.initial_diff and not initial_diff_consumed:
            diff_source = "initial_diff"
            raw_response = config.initial_diff
            initial_diff_consumed = True
        else:
            prompt = _build_prompt(config, state, registry)
            raw_response = provider.generate(
                prompt=prompt,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            diff_source = "llm"

        # Phase 3: Extract Diff
        try:
            diff_text = extract_diff(raw_response, max_lines=config.max_diff_lines)
        except DiffValidationError as exc:
            logger.info("No valid diff from %s: %s", diff_source, exc)
            state.stable = True
            state.failure_reason = str(exc)
            save_state(state, runs_dir)
            return LoopResult(run_id, branch, iteration, StabilityReason.NO_DIFF, state)

        # Duplicate patch detection — skip iteration rather than aborting
        patch_hash = hashlib.sha256(diff_text.encode()).hexdigest()[:16]
        if patch_hash in seen_patch_hashes:
            logger.warning("Duplicate patch detected on iteration %d — skipping", iteration)
            state.failure_reason = "duplicate_patch"
            save_state(state, runs_dir)
            continue
        seen_patch_hashes.add(patch_hash)

        # Phase 4: Validate
        try:
            validate_patch(diff_text, config.repo)
        except PatchValidationError as exc:
            logger.warning("Patch validation failed: %s — skipping iteration", exc)
            state.failure_reason = f"patch_validation: {exc}"
            save_state(state, runs_dir)
            continue

        # Phase 5: Apply
        commit_sha = _apply_patch(diff_text, config.repo, iteration)
        state.patches.append(
            PatchRecord(
                iteration=iteration,
                patch_hash=patch_hash,
                commit_sha=commit_sha,
                lines_changed=len(diff_text.splitlines()),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

        # Phase 6: Test
        test_result = _run_tests(config, iteration)
        state.test_results.append(test_result)

        # Phase 7: Update State
        save_state(state, runs_dir)

        if test_result.passed:
            logger.info("Tests passed on iteration %d", iteration)
            state.stable = True
            save_state(state, runs_dir)
            _write_pr_summary(state, runs_dir)
            return LoopResult(run_id, branch, iteration + 1, StabilityReason.TESTS_PASS, state)

    logger.info("Max iterations reached")
    _write_pr_summary(state, runs_dir)
    return LoopResult(run_id, branch, config.max_iterations, StabilityReason.MAX_ITERATIONS, state)


def _build_prompt(config: LoopConfig, state: RunState, registry: ToolRegistry) -> str:
    try:
        git_diff = registry.call("git_diff", {"repo": str(config.repo)})
        diff_output = git_diff.get("stdout", "") or "(clean)"
    except Exception:
        diff_output = "(git diff unavailable)"

    last_test = state.test_results[-1] if state.test_results else None

    sections = [
        "You are a precise code engineer. Output ONLY a unified diff. No explanation.",
        "",
        f"DIRECTIVE:\n{config.directive}",
        "",
        f"ITERATION: {state.iteration}",
        "",
        "CURRENT GIT DIFF (working tree):",
        diff_output,
        "",
    ]

    if last_test:
        sections += [
            "LAST TEST RUN:",
            f"Exit code: {last_test.exit_code}",
            f"Output (last 2000 chars):\n{last_test.stdout[-2000:]}",
            "",
        ]

    sections += [
        "OUTPUT FORMAT: A single unified diff block only.",
        "If no changes are needed, output exactly: NO_CHANGES",
    ]

    return "\n".join(sections)


def _extract_modified_files(diff_text: str) -> list[str]:
    files: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            raw_path = line[6:].split("\t", 1)[0]
            if not raw_path or raw_path == "/dev/null":
                continue

            if raw_path.startswith(("a/", "b/")):
                raw_path = raw_path[2:]
            normalized = raw_path.lstrip("./")
            if normalized:
                files.add(normalized)
    return sorted(files)


def _assert_safe_file(path: str) -> None:
    normalized = path.replace("\\", "/").lstrip("./")
    for prefix in FORBIDDEN_PREFIXES:
        if normalized.startswith(prefix):
            raise RuntimeError(f"Unsafe file staging detected: {path}")
    for suffix in FORBIDDEN_SUFFIXES:
        if normalized.endswith(suffix):
            raise RuntimeError(f"Unsafe file staging detected: {path}")


def _parse_status_paths(status_output: str) -> set[str]:
    paths: set[str] = set()
    for line in status_output.splitlines():
        if not line:
            continue
        candidate = line[3:].strip()
        if "->" in candidate:
            candidate = candidate.split("->", 1)[1].strip()
        paths.add(candidate)
    return paths


def _validate_staged_files(repo: Path, expected_files: list[str]) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    entries = [line for line in result.stdout.splitlines() if line.strip()]
    if len(entries) > len(expected_files):
        raise RuntimeError("Unexpected additional files staged before commit.")
    staged_paths = _parse_status_paths(result.stdout)
    unexpected = staged_paths - set(expected_files)
    if unexpected:
        raise RuntimeError(f"Unsafe file staging detected: {sorted(unexpected)}")


def _apply_patch(diff_text: str, repo: Path, iteration: int) -> str:
    modified_files = _extract_modified_files(diff_text)
    if not modified_files:
        raise RuntimeError("Patch did not modify any files.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
        tmp.write(diff_text)
        patch_path = Path(tmp.name)

    try:
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )

        for file in modified_files:
            _assert_safe_file(file)
            subprocess.run(
                ["git", "add", "--", file],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )

        _validate_staged_files(repo, modified_files)

        subprocess.run(
            ["git", "commit", "-m", f"hammer: iteration {iteration}"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )

        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return sha_result.stdout.strip()
    finally:
        patch_path.unlink(missing_ok=True)


def _run_tests(config: LoopConfig, iteration: int) -> TestResult:
    parts = config.test_command.split()
    result = subprocess.run(
        parts,
        cwd=config.repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    return TestResult(
        iteration=iteration,
        command=config.test_command,
        exit_code=result.returncode,
        stdout=result.stdout[-8000:],
        stderr=result.stderr[-2000:],
        passed=result.returncode == 0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _write_pr_summary(state: RunState, runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / f"{state.run_id}_PR_SUMMARY.md"
    patches = "\n".join(
        f"- Iteration {p.iteration}: commit `{p.commit_sha}` ({p.lines_changed} lines)"
        for p in state.patches
    ) or "No patches applied"

    test_status = "PASS" if state.stable else "FAIL"
    last_test = state.test_results[-1] if state.test_results else None

    summary = f"""# PR Summary — {state.run_id}

**Branch:** `{state.branch}`
**Directive:** {state.directive}
**Iterations:** {state.iteration}
**Final Status:** {test_status}

## Patches Applied
{patches}

## Test Results
Exit code: {last_test.exit_code if last_test else 'N/A'}

## Risk Assessment
{'Low — all tests pass' if state.stable else 'Review required — tests did not pass'}
"""
    summary_path.write_text(summary)
    logger.info("PR summary written to %s", summary_path)
