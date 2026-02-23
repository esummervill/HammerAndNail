# HammerAndNail Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build HammerAndNail — a modular, open-source autonomous coding runtime that uses a deterministic 7-phase pipeline loop to apply LLM-generated diffs to any local git repo.

**Architecture:** Layered phase pipeline (read_state → generate → extract_diff → validate → apply → test → update_state). LLM is a swappable provider abstracted behind an ABC. Tools are registered with JSON contracts and a whitelist enforced at runtime. State is JSON files in `.hammer/runs/`. No LLM conversation memory — every prompt is rebuilt from state + git.

**Tech Stack:** Python 3.11+, Click (CLI), httpx (Ollama HTTP), pytest (tests), tomllib (config), pyproject.toml (packaging). No database dependencies.

---

## Task 1: Repository Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `hammer/__init__.py`
- Create: `hammer/core/__init__.py`
- Create: `hammer/llm/__init__.py`
- Create: `hammer/tools/__init__.py`
- Create: `hammer/plugins/__init__.py`
- Create: `configs/default.toml`
- Create: `examples/directive.md`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/tools/__init__.py`
- Create: `.gitignore`
- Create: `LICENSE`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "hammernail"
version = "0.1.0"
description = "Modular autonomous coding runtime"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "httpx>=0.27",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
    "respx>=0.21",
]

[project.scripts]
hammer = "hammer.cli:main"
EngineerExternal = "hammer.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["hammer*"]
```

**Step 2: Create all `__init__.py` files (empty)**

```bash
mkdir -p hammer/core hammer/llm hammer/tools hammer/plugins
mkdir -p tests/core tests/llm tests/tools
mkdir -p configs examples docs/plans
touch hammer/__init__.py hammer/core/__init__.py hammer/llm/__init__.py
touch hammer/tools/__init__.py hammer/plugins/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/llm/__init__.py tests/tools/__init__.py
```

**Step 3: Create `configs/default.toml`**

```toml
[model]
provider = "ollama"
model = "qwen3-coder:30b"
url = "http://localhost:11434"
temperature = 0.2
max_tokens = 4096
timeout = 120

[loop]
max_iterations = 10
max_diff_lines = 500

[tools]
test_command = "pytest"
```

**Step 4: Create `examples/directive.md`**

```markdown
# Example Directive

Fix all failing tests in this repository.

## Constraints
- Do not modify test files
- Prefer minimal changes
- Keep changes under 200 lines per patch
```

**Step 5: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
.hammer/
*.json
!configs/
.env
```

**Step 6: Create `LICENSE` (Apache 2.0 header)**

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright 2026 HammerAndNail Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

**Step 7: Install dev dependencies**

```bash
cd /Users/ethansummervill/Projects/EngineerRuntime
pip install -e ".[dev]"
```

**Step 8: Commit scaffold**

```bash
git init
git add .
git commit -m "chore: scaffold HammerAndNail repository structure"
```

---

## Task 2: State Manager

**Files:**
- Create: `hammer/core/state.py`
- Create: `tests/core/test_state.py`

**Step 1: Write failing tests**

```python
# tests/core/test_state.py
import json
from pathlib import Path
import pytest
from hammer.core.state import RunState, load_state, save_state, new_run_id


def test_new_run_id_is_timestamp_string():
    run_id = new_run_id()
    assert isinstance(run_id, str)
    assert len(run_id) == 15  # YYYYmmdd-HHMMSS


def test_save_and_load_roundtrip(tmp_path):
    state = RunState(
        run_id="20260223-120000",
        repo_path=str(tmp_path),
        directive="fix tests",
        iteration=0,
        branch="EngineerExternal/20260223-120000",
        patches=[],
        test_results=[],
        stable=False,
    )
    save_state(state, tmp_path / ".hammer" / "runs")
    loaded = load_state("20260223-120000", tmp_path / ".hammer" / "runs")
    assert loaded.run_id == state.run_id
    assert loaded.directive == state.directive
    assert loaded.stable is False


def test_load_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_state("nonexistent", tmp_path / ".hammer" / "runs")
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/core/test_state.py -v
```
Expected: `ModuleNotFoundError: No module named 'hammer.core.state'`

**Step 3: Implement `hammer/core/state.py`**

```python
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

    def _serialize(obj):
        if isinstance(obj, (PatchRecord, TestResult)):
            return asdict(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

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
```

**Step 4: Run tests to verify PASS**

```bash
pytest tests/core/test_state.py -v
```
Expected: all 3 tests PASS

**Step 5: Commit**

```bash
git add hammer/core/state.py tests/core/test_state.py
git commit -m "feat: add JSON run state manager"
```

---

## Task 3: Branch Manager

**Files:**
- Create: `hammer/core/branch_manager.py`
- Create: `tests/core/test_branch_manager.py`

**Step 1: Write failing tests**

```python
# tests/core/test_branch_manager.py
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
    """create_engineer_branch should return a name starting with BRANCH_PREFIX."""
    # We mock git calls to avoid needing a real repo
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
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/core/test_branch_manager.py -v
```

**Step 3: Implement `hammer/core/branch_manager.py`**

```python
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
```

**Step 4: Run tests to verify PASS**

```bash
pytest tests/core/test_branch_manager.py -v
```

**Step 5: Commit**

```bash
git add hammer/core/branch_manager.py tests/core/test_branch_manager.py
git commit -m "feat: add branch isolation manager"
```

---

## Task 4: Diff Manager

**Files:**
- Create: `hammer/core/diff_manager.py`
- Create: `tests/core/test_diff_manager.py`

**Step 1: Write failing tests**

```python
# tests/core/test_diff_manager.py
import pytest
from hammer.core.diff_manager import extract_diff, DiffValidationError, count_diff_lines

SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
-    pass
+    return "hello"
+
"""

FENCED_DIFF = f"Some text\n```diff\n{SAMPLE_DIFF}```\nMore text"


def test_extract_raw_diff():
    result = extract_diff(SAMPLE_DIFF)
    assert result == SAMPLE_DIFF


def test_extract_fenced_diff():
    result = extract_diff(FENCED_DIFF)
    assert "--- a/foo.py" in result
    assert "+++ b/foo.py" in result


def test_extract_empty_raises():
    with pytest.raises(DiffValidationError, match="no unified diff"):
        extract_diff("Just some text with no diff content")


def test_extract_too_large_raises():
    huge_diff = SAMPLE_DIFF + ("+line\n" * 600)
    with pytest.raises(DiffValidationError, match="exceeds maximum"):
        extract_diff(huge_diff, max_lines=500)


def test_count_diff_lines():
    assert count_diff_lines(SAMPLE_DIFF) > 0
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/core/test_diff_manager.py -v
```

**Step 3: Implement `hammer/core/diff_manager.py`**

```python
"""Unified diff extraction and validation.

Parses LLM response text to extract a valid unified diff.
Rejects diffs that are empty, malformed, or exceed size limits.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("hammer.core.diff_manager")

# Match ```diff ... ``` fenced blocks
_FENCED_DIFF_RE = re.compile(r"```(?:diff)?\s*\n(.*?)```", re.DOTALL)
# Match raw unified diff headers
_RAW_DIFF_RE = re.compile(r"(^---\s+\S.*?$)", re.MULTILINE)


class DiffValidationError(ValueError):
    """Raised when a diff cannot be extracted or is invalid."""


def extract_diff(text: str, max_lines: int = 500) -> str:
    """Extract unified diff from LLM response text.

    Tries fenced ```diff blocks first, then raw unified diff headers.
    Raises DiffValidationError if no valid diff is found or size limit exceeded.
    """
    diff_text = _try_fenced(text) or _try_raw(text)

    if not diff_text:
        raise DiffValidationError("no unified diff found in LLM response")

    diff_text = diff_text.strip()
    lines = diff_text.splitlines()

    if len(lines) > max_lines:
        raise DiffValidationError(
            f"diff exceeds maximum of {max_lines} lines (got {len(lines)})"
        )

    # Must have at least one hunk header
    if not any(line.startswith("@@") for line in lines):
        raise DiffValidationError("diff has no hunk headers (not a valid unified diff)")

    return diff_text + "\n"


def count_diff_lines(diff_text: str) -> int:
    return len(diff_text.splitlines())


def _try_fenced(text: str) -> str | None:
    match = _FENCED_DIFF_RE.search(text)
    if match:
        logger.debug("Extracted fenced diff block")
        return match.group(1)
    return None


def _try_raw(text: str) -> str | None:
    match = _RAW_DIFF_RE.search(text)
    if match:
        start = match.start()
        logger.debug("Extracted raw diff starting at char %d", start)
        return text[start:]
    return None
```

**Step 4: Run tests to verify PASS**

```bash
pytest tests/core/test_diff_manager.py -v
```

**Step 5: Commit**

```bash
git add hammer/core/diff_manager.py tests/core/test_diff_manager.py
git commit -m "feat: add unified diff extractor and validator"
```

---

## Task 5: Validator (git apply --check)

**Files:**
- Create: `hammer/core/validator.py`
- Create: `tests/core/test_validator.py`

**Step 1: Write failing tests**

```python
# tests/core/test_validator.py
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from hammer.core.validator import validate_patch, PatchValidationError

SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,3 @@
 def hello():
-    pass
+    return "hello"
"""


def test_validate_patch_success(tmp_path):
    with patch("hammer.core.validator._run_git_apply_check") as mock_check:
        mock_check.return_value = MagicMock(returncode=0, stderr="")
        # Should not raise
        validate_patch(SAMPLE_DIFF, tmp_path)


def test_validate_patch_failure_raises(tmp_path):
    with patch("hammer.core.validator._run_git_apply_check") as mock_check:
        mock_check.return_value = MagicMock(
            returncode=1, stderr="error: patch does not apply"
        )
        with pytest.raises(PatchValidationError, match="patch does not apply"):
            validate_patch(SAMPLE_DIFF, tmp_path)
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/core/test_validator.py -v
```

**Step 3: Implement `hammer/core/validator.py`**

```python
"""Patch validation via git apply --check (dry-run).

Runs git apply --check before attempting to apply any patch.
This prevents partial application and keeps the working tree clean.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("hammer.core.validator")


class PatchValidationError(ValueError):
    """Raised when a patch fails git apply --check."""


def _run_git_apply_check(patch_path: Path, repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def validate_patch(diff_text: str, repo: Path) -> None:
    """Dry-run a unified diff against the repo using git apply --check.

    Raises PatchValidationError if the patch would not apply cleanly.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False
    ) as tmp:
        tmp.write(diff_text)
        patch_path = Path(tmp.name)

    try:
        result = _run_git_apply_check(patch_path, repo)
        if result.returncode != 0:
            msg = result.stderr.strip() or "git apply --check failed"
            logger.warning("Patch validation failed: %s", msg)
            raise PatchValidationError(msg)
        logger.debug("Patch validated successfully")
    finally:
        patch_path.unlink(missing_ok=True)
```

**Step 4: Run tests to verify PASS**

```bash
pytest tests/core/test_validator.py -v
```

**Step 5: Commit**

```bash
git add hammer/core/validator.py tests/core/test_validator.py
git commit -m "feat: add patch validator (git apply --check)"
```

---

## Task 6: LLM Abstraction Layer

**Files:**
- Create: `hammer/llm/base.py`
- Create: `hammer/llm/ollama_provider.py`
- Create: `hammer/llm/model_router.py`
- Create: `tests/llm/test_ollama_provider.py`
- Create: `tests/llm/test_model_router.py`

**Step 1: Write failing tests**

```python
# tests/llm/test_ollama_provider.py
import pytest
import respx
import httpx
from hammer.llm.ollama_provider import OllamaProvider

BASE_URL = "http://localhost:11434"


def test_generate_returns_response_text():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/api/generate").mock(
            return_value=httpx.Response(
                200, json={"response": "def hello():\n    return 'hi'\n"}
            )
        )
        provider = OllamaProvider(base_url=BASE_URL)
        result = provider.generate(
            prompt="fix this", model="qwen3-coder:30b", max_tokens=512, temperature=0.2
        )
    assert "def hello" in result


def test_generate_raises_on_http_error():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/api/generate").mock(return_value=httpx.Response(500))
        provider = OllamaProvider(base_url=BASE_URL)
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            provider.generate("prompt", "model", 512, 0.2)


# tests/llm/test_model_router.py
import os
import pytest
from hammer.llm.model_router import get_provider
from hammer.llm.ollama_provider import OllamaProvider


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("HAMMER_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, OllamaProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("HAMMER_PROVIDER", "nonexistent")
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider()
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/llm/ -v
```

**Step 3: Implement `hammer/llm/base.py`**

```python
"""LLM provider abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for all LLM provider implementations."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        """Generate a completion for the given prompt.

        Returns the raw text response from the model.
        Raises RuntimeError on provider failures.
        """
```

**Step 4: Implement `hammer/llm/ollama_provider.py`**

```python
"""Ollama LLM provider for HammerAndNail.

Uses POST /api/generate with stream=False.
Compatible with any Ollama-served model.
"""
from __future__ import annotations

import logging
import os

import httpx

from .base import LLMProvider

logger = logging.getLogger("hammer.llm.ollama")

DEFAULT_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("HAMMER_LLM_URL", DEFAULT_URL)).rstrip("/")
        self.timeout = float(os.getenv("HAMMER_LLM_TIMEOUT", "120"))

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            logger.debug("Ollama response length: %d chars", len(text))
            return text
        except httpx.HTTPStatusError as exc:
            logger.exception("Ollama HTTP error: %s", exc.response.status_code)
            raise RuntimeError(f"Ollama request failed: {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.exception("Ollama connection error")
            raise RuntimeError(f"Ollama connection failed: {exc}") from exc
```

**Step 5: Implement `hammer/llm/model_router.py`**

```python
"""Environment-driven LLM provider router.

Reads HAMMER_PROVIDER env var to select the active provider.
All provider-specific configuration is handled inside each provider class.

To add a new provider:
  1. Subclass LLMProvider
  2. Add to _PROVIDERS dict below
"""
from __future__ import annotations

import os

from .base import LLMProvider
from .ollama_provider import OllamaProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
}

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3-coder:30b"


def get_provider() -> LLMProvider:
    """Return a configured LLM provider based on HAMMER_PROVIDER env var."""
    name = os.getenv("HAMMER_PROVIDER", DEFAULT_PROVIDER).lower()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(_PROVIDERS.keys())}"
        )
    return cls()


def get_model() -> str:
    """Return the configured model name from HAMMER_MODEL env var."""
    return os.getenv("HAMMER_MODEL", DEFAULT_MODEL)
```

**Step 6: Run tests to verify PASS**

```bash
pytest tests/llm/ -v
```

**Step 7: Commit**

```bash
git add hammer/llm/ tests/llm/
git commit -m "feat: add LLM abstraction layer (Ollama provider + env router)"
```

---

## Task 7: Tool Registry

**Files:**
- Create: `hammer/tools/registry.py`
- Create: `hammer/tools/git_tools.py`
- Create: `hammer/tools/test_tools.py`
- Create: `hammer/tools/docker_tools.py`
- Create: `tests/tools/test_registry.py`

**Step 1: Write failing tests**

```python
# tests/tools/test_registry.py
import pytest
from unittest.mock import patch, MagicMock
from hammer.tools.registry import ToolRegistry, ToolNotAllowed


def test_registry_has_git_tools():
    registry = ToolRegistry()
    assert registry.has("git_status")
    assert registry.has("git_diff")


def test_registry_blocks_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(ToolNotAllowed):
        registry.call("rm_rf", {})


def test_registry_call_git_status(tmp_path):
    registry = ToolRegistry()
    with patch("hammer.tools.git_tools.run_git_status") as mock_fn:
        mock_fn.return_value = {"status": "clean"}
        result = registry.call("git_status", {"repo": str(tmp_path)})
    assert result["status"] == "clean"
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/tools/test_registry.py -v
```

**Step 3: Implement `hammer/tools/git_tools.py`**

```python
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
    return _git("log", f"--oneline", f"-{n}", repo=Path(repo))
```

**Step 4: Implement `hammer/tools/test_tools.py`**

```python
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
        "stdout": result.stdout[-8000:],  # Truncate to last 8000 chars
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
```

**Step 5: Implement `hammer/tools/docker_tools.py`**

```python
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
```

**Step 6: Implement `hammer/tools/registry.py`**

```python
"""Tool registry for HammerAndNail.

Tools are registered with JSON contracts and invoked by name.
Unknown or disallowed tools raise ToolNotAllowed.

To add a new tool:
  1. Implement the function in the appropriate tools module
  2. Add an entry to _TOOL_CATALOG below
  3. Plugin modules can call registry.register() at startup
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .git_tools import (
    run_git_apply,
    run_git_apply_check,
    run_git_diff,
    run_git_log,
    run_git_status,
)
from .test_tools import run_compileall, run_lint, run_npm_build, run_pytest
from .docker_tools import (
    run_docker_compose_down,
    run_docker_compose_logs,
    run_docker_compose_ps,
    run_docker_compose_up,
)

logger = logging.getLogger("hammer.tools.registry")


class ToolNotAllowed(Exception):
    """Raised when an unknown or disallowed tool is invoked."""


@dataclass
class ToolDefinition:
    name: str
    description: str
    fn: Callable[..., dict]
    allowed: bool = True


_TOOL_CATALOG: list[ToolDefinition] = [
    ToolDefinition("git_status", "Show working tree status", run_git_status),
    ToolDefinition("git_diff", "Show unstaged changes", run_git_diff),
    ToolDefinition("git_apply", "Apply a patch file", run_git_apply),
    ToolDefinition("git_apply_check", "Dry-run patch check", run_git_apply_check),
    ToolDefinition("git_log", "Show recent commits", run_git_log),
    ToolDefinition("pytest", "Run pytest test suite", run_pytest),
    ToolDefinition("compileall", "Python compile check", run_compileall),
    ToolDefinition("npm_build", "Run npm build", run_npm_build),
    ToolDefinition("lint", "Run flake8 lint", run_lint),
    ToolDefinition("docker_compose_ps", "Show container status", run_docker_compose_ps),
    ToolDefinition("docker_compose_logs", "Tail container logs", run_docker_compose_logs),
    ToolDefinition("docker_compose_up", "Start services", run_docker_compose_up),
    ToolDefinition("docker_compose_down", "Stop services", run_docker_compose_down),
]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {
            t.name: t for t in _TOOL_CATALOG if t.allowed
        }

    def has(self, name: str) -> bool:
        return name in self._tools

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool (used by plugins)."""
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def call(self, name: str, args: dict[str, Any]) -> dict:
        if name not in self._tools:
            raise ToolNotAllowed(f"Tool '{name}' is not registered or not allowed")
        tool = self._tools[name]
        logger.debug("Calling tool: %s args=%s", name, args)
        return tool.fn(**args)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "allowed": t.allowed}
            for t in self._tools.values()
        ]
```

**Step 7: Run tests to verify PASS**

```bash
pytest tests/tools/ -v
```

**Step 8: Commit**

```bash
git add hammer/tools/ tests/tools/
git commit -m "feat: add tool registry with git/test/docker tools"
```

---

## Task 8: Plugin Interface

**Files:**
- Modify: `hammer/plugins/__init__.py`

**Step 1: Implement plugin discovery**

```python
# hammer/plugins/__init__.py
"""Plugin discovery interface for HammerAndNail.

Plugins are Python modules that expose a `register(registry)` function.
Place plugin modules in this package or install them as separate packages
with the entry point group 'hammer.plugins'.

Example plugin:
    # my_plugin/hammer_plugin.py
    from hammer.tools.registry import ToolDefinition

    def register(registry):
        registry.register(ToolDefinition(
            name="my_tool",
            description="Does something custom",
            fn=my_tool_fn,
        ))
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger("hammer.plugins")


def load_plugins(registry) -> int:
    """Discover and load all plugins in this package.

    Returns the number of plugins loaded.
    """
    loaded = 0
    package_path = Path(__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        try:
            module = importlib.import_module(f"hammer.plugins.{module_name}")
            if hasattr(module, "register"):
                module.register(registry)
                logger.info("Loaded plugin: %s", module_name)
                loaded += 1
        except Exception:
            logger.exception("Failed to load plugin: %s", module_name)

    return loaded
```

**Step 2: Commit**

```bash
git add hammer/plugins/__init__.py
git commit -m "feat: add plugin discovery interface"
```

---

## Task 9: Core Loop

**Files:**
- Create: `hammer/core/loop.py`
- Create: `tests/core/test_loop.py`

**Step 1: Write failing tests**

```python
# tests/core/test_loop.py
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from hammer.core.loop import LoopConfig, run_loop, StabilityReason
from hammer.core.state import RunState


def _make_config(tmp_path: Path) -> LoopConfig:
    return LoopConfig(
        repo=tmp_path,
        directive="fix failing tests",
        model="test-model",
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
        mock_provider.return_value.generate.return_value = "```diff\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n```"
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
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/core/test_loop.py -v
```

**Step 3: Implement `hammer/core/loop.py`**

```python
"""Core 7-phase pipeline loop for HammerAndNail.

The LLM never owns the loop. The runtime drives all phase transitions.
State is loaded fresh each iteration. No conversation memory.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .branch_manager import create_engineer_branch
from .diff_manager import DiffValidationError, extract_diff
from .state import PatchRecord, RunState, TestResult, new_run_id, save_state
from .validator import PatchValidationError, validate_patch
from ..llm.model_router import get_provider
from ..tools.registry import ToolRegistry

logger = logging.getLogger("hammer.core.loop")


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
    max_iterations: int = 10
    test_command: str = "pytest"
    max_diff_lines: int = 500
    temperature: float = 0.2
    max_tokens: int = 4096


def run_loop(config: LoopConfig) -> LoopResult:
    """Execute the 7-phase pipeline loop against the target repo."""
    run_id = new_run_id()
    runs_dir = config.repo / ".hammer" / "runs"
    provider = get_provider()
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

    for iteration in range(config.max_iterations):
        state.iteration = iteration
        logger.info("--- Iteration %d ---", iteration)

        # Phase 1: Read State (already in memory; reload for robustness)
        save_state(state, runs_dir)

        # Phase 2: Generate
        prompt = _build_prompt(config, state, registry)
        raw_response = provider.generate(
            prompt=prompt,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        # Phase 3: Extract Diff
        try:
            diff_text = extract_diff(raw_response, max_lines=config.max_diff_lines)
        except DiffValidationError as exc:
            logger.info("No valid diff in response: %s", exc)
            state.stable = True
            state.failure_reason = str(exc)
            save_state(state, runs_dir)
            return LoopResult(run_id, branch, iteration, StabilityReason.NO_DIFF, state)

        # Duplicate patch detection
        patch_hash = hashlib.sha256(diff_text.encode()).hexdigest()[:16]
        if patch_hash in seen_patch_hashes:
            logger.warning("Duplicate patch detected — loop escape")
            state.failure_reason = "duplicate_patch"
            save_state(state, runs_dir)
            return LoopResult(run_id, branch, iteration, StabilityReason.DUPLICATE_PATCH, state)
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
    git_diff = registry.call("git_diff", {"repo": str(config.repo)})
    last_test = state.test_results[-1] if state.test_results else None

    sections = [
        "You are a precise code engineer. Output ONLY a unified diff. No explanation.",
        "",
        f"DIRECTIVE:\n{config.directive}",
        "",
        f"ITERATION: {state.iteration}",
        "",
        "CURRENT GIT DIFF (working tree):",
        git_diff.get("stdout", "(clean)") or "(clean)",
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


def _apply_patch(diff_text: str, repo: Path, iteration: int) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
        tmp.write(diff_text)
        patch_path = Path(tmp.name)

    try:
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=repo, check=True, capture_output=True, text=True, timeout=15
        )
        result = subprocess.run(
            ["git", "commit", "-am", f"hammer: iteration {iteration}"],
            cwd=repo, capture_output=True, text=True, timeout=15
        )
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=5
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

```
{last_test.stdout[-2000:] if last_test else '(no tests run)'}
```

## Risk Assessment
{'Low — all tests pass' if state.stable else 'Review required — tests did not pass'}
"""
    summary_path.write_text(summary)
    logger.info("PR summary written to %s", summary_path)
```

**Step 4: Run tests to verify PASS**

```bash
pytest tests/core/test_loop.py -v
```

**Step 5: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all tests PASS

**Step 6: Commit**

```bash
git add hammer/core/loop.py tests/core/test_loop.py
git commit -m "feat: implement 7-phase pipeline loop"
```

---

## Task 10: CLI

**Files:**
- Create: `hammer/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
from click.testing import CliRunner
from hammer.cli import main


def test_cli_run_requires_repo():
    runner = CliRunner()
    result = runner.invoke(main, ["run"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_cli_tools_list():
    runner = CliRunner()
    result = runner.invoke(main, ["tools", "list"])
    assert result.exit_code == 0
    assert "git_status" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
```

**Step 2: Run to verify FAIL**

```bash
pytest tests/test_cli.py -v
```

**Step 3: Implement `hammer/cli.py`**

```python
"""HammerAndNail CLI entry point.

Usage:
    hammer run --repo /path/to/repo --directive directive.md
    hammer tools list
    hammer --version

Both `hammer` and `EngineerExternal` are registered as entry points.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .core.loop import LoopConfig, StabilityReason, run_loop
from .llm.model_router import DEFAULT_MODEL, DEFAULT_PROVIDER
from .tools.registry import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

__version__ = "0.1.0"


@click.group()
@click.version_option(__version__, prog_name="hammer")
def main():
    """HammerAndNail — modular autonomous coding runtime."""


@main.command()
@click.option("--repo", required=True, type=click.Path(exists=True, file_okay=False), help="Target repository path")
@click.option("--directive", required=True, type=click.Path(exists=True, dir_okay=False), help="Directive markdown file")
@click.option("--model", default=DEFAULT_MODEL, envvar="HAMMER_MODEL", show_default=True, help="LLM model name")
@click.option("--provider", default=DEFAULT_PROVIDER, envvar="HAMMER_PROVIDER", show_default=True, help="LLM provider (ollama)")
@click.option("--max-iterations", default=10, show_default=True, help="Maximum loop iterations")
@click.option("--test-command", default="pytest", show_default=True, help="Test command to run after each patch")
@click.option("--max-diff-lines", default=500, show_default=True, help="Maximum lines in a single diff")
def run(repo, directive, model, provider, max_iterations, test_command, max_diff_lines):
    """Run the engineering loop against a repository."""
    repo_path = Path(repo).resolve()
    directive_text = Path(directive).read_text()

    click.echo(f"HammerAndNail v{__version__}")
    click.echo(f"  Repo:      {repo_path}")
    click.echo(f"  Model:     {model} via {provider}")
    click.echo(f"  Max iter:  {max_iterations}")
    click.echo(f"  Tests:     {test_command}")
    click.echo("")

    config = LoopConfig(
        repo=repo_path,
        directive=directive_text,
        model=model,
        max_iterations=max_iterations,
        test_command=test_command,
        max_diff_lines=max_diff_lines,
    )

    result = run_loop(config)

    click.echo("")
    click.echo(f"Loop complete — {result.reason.value}")
    click.echo(f"  Run ID:     {result.run_id}")
    click.echo(f"  Branch:     {result.branch}")
    click.echo(f"  Iterations: {result.iterations}")
    click.echo(f"  Patches:    {len(result.state.patches)}")

    if result.reason == StabilityReason.TESTS_PASS:
        click.secho("  Status: STABLE — all tests pass", fg="green")
        sys.exit(0)
    else:
        click.secho(f"  Status: UNSTABLE — {result.reason.value}", fg="yellow")
        sys.exit(1)


@main.group()
def tools():
    """Tool registry commands."""


@tools.command("list")
def tools_list():
    """List all registered tools."""
    registry = ToolRegistry()
    click.echo("Registered tools:")
    for tool in registry.list_tools():
        status = "✓" if tool["allowed"] else "✗"
        click.echo(f"  {status}  {tool['name']:<25} {tool['description']}")
```

**Step 4: Run tests to verify PASS**

```bash
pytest tests/test_cli.py -v
```

**Step 5: Run full test suite**

```bash
pytest tests/ -v
```

**Step 6: Commit**

```bash
git add hammer/cli.py tests/test_cli.py
git commit -m "feat: add CLI entry points (hammer run + tools list)"
```

---

## Task 11: README

**Files:**
- Create: `README.md`

**Step 1: Write README**

```markdown
# HammerAndNail

**Modular autonomous coding runtime.**

HammerAndNail runs a deterministic agent loop against any git repository,
using an LLM to generate patches, validate them with `git apply --check`,
apply them, run tests, and iterate to stability.

The intelligence is in the architecture, not the token count.

## Architecture

```
hammer run
    │
    ├─ Phase 0: Branch Safety (EngineerExternal/<timestamp>)
    ├─ Phase 1: Read State (.hammer/runs/<run_id>.json)
    ├─ Phase 2: Generate (LLM prompt → raw response)
    ├─ Phase 3: Extract Diff (unified diff from response)
    ├─ Phase 4: Validate (git apply --check)
    ├─ Phase 5: Apply (git apply + commit)
    ├─ Phase 6: Test (pytest / compileall / npm build)
    └─ Phase 7: Update State → repeat
```

State is never in LLM memory. Every prompt is rebuilt from JSON state + git.

## Quick Start

```bash
pip install hammernail

hammer run \
  --repo /path/to/your/repo \
  --directive directive.md
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- `qwen3-coder:30b` pulled (`ollama pull qwen3-coder:30b`)

## Configuration

```bash
export HAMMER_MODEL=qwen3-coder:30b   # default
export HAMMER_PROVIDER=ollama          # default
export HAMMER_LLM_URL=http://localhost:11434  # default
```

Or use a `configs/default.toml` (see `configs/` directory).

## Branch Safety

HammerAndNail **never touches main**. Every run:
1. Checks out `main`
2. Pulls latest
3. Creates `EngineerExternal/<timestamp>`
4. Commits all patches to this branch
5. Writes `PR_SUMMARY.md` on completion

## Plugin System

Drop a module into `hammer/plugins/` with a `register(registry)` function
to add custom tools:

```python
# hammer/plugins/my_tool.py
from hammer.tools.registry import ToolDefinition

def register(registry):
    registry.register(ToolDefinition(
        name="my_tool",
        description="Does something custom",
        fn=my_tool_fn,
    ))
```

## Model Upgrade Path

Switch providers via env var — no code changes needed:

```bash
HAMMER_PROVIDER=ollama HAMMER_MODEL=qwen3-coder:30b hammer run ...
# Future:
HAMMER_PROVIDER=vllm HAMMER_MODEL=deepseek-coder-v2 hammer run ...
```

## License

Apache 2.0
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with architecture overview"
```

---

## Task 12: Git Remote + Push

**Step 1: Add remote and push**

```bash
cd /Users/ethansummervill/Projects/EngineerRuntime
git remote add origin https://github.com/esummervill/HammerAndNail.git
git branch -M main
git push -u origin main
```

**Step 2: Verify push succeeded**

```bash
git log --oneline -10
```

Expected: all commits visible, `origin/main` tracking.

---

## Full Test Run (Final Verification)

```bash
cd /Users/ethansummervill/Projects/EngineerRuntime
pip install -e ".[dev]"
pytest tests/ -v --tb=short
hammer --version
hammer tools list
```

Expected:
- All tests PASS
- `hammer 0.1.0` printed
- Tool list printed
