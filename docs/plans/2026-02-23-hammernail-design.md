# HammerAndNail — Design Document

**Date:** 2026-02-23
**Project:** HammerAndNail (`hammer` package)
**Repo:** https://github.com/esummervill/HammerAndNail.git
**Status:** Approved for implementation

---

## Overview

HammerAndNail is a modular, open-source autonomous coding runtime. It operates on any
local git repository, using a deterministic phase pipeline loop to apply LLM-generated
diffs, validate patches, run tests, and iterate to stability.

The intelligence is in the architecture, not the token count.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Public name | HammerAndNail | Branding; CLI = `hammer` |
| State backend | JSON files | Portable, matches Strios convention |
| Loop architecture | Layered phase pipeline | Deterministic, testable, extensible |
| Default LLM | qwen3-coder:30b via Ollama | Local, no cloud dependency |
| LLM interface | ABC-based provider | Swap-friendly for vLLM / GPU upgrade |

---

## Repository Structure

```
EngineerRuntime/
  hammer/
    core/
      loop.py             # Phase pipeline orchestrator
      state.py            # JSON run state manager (.hammer/runs/<run_id>.json)
      branch_manager.py   # Git branch safety (EngineerExternal/<timestamp>)
      diff_manager.py     # Unified diff extraction and size validation
      validator.py        # git apply --check dry-run before apply
    llm/
      base.py             # LLMProvider ABC
      ollama_provider.py  # Ollama /api/generate (stream=False)
      model_router.py     # Env-driven provider selection
    tools/
      registry.py         # JSON-contract tool registry with whitelist
      git_tools.py        # git status, diff, apply, checkout-branch
      test_tools.py       # pytest, compileall
      docker_tools.py     # docker compose up/down/logs/ps
    plugins/
      __init__.py         # Plugin discovery interface
    cli.py                # Click CLI: `hammer run`
  configs/
    default.toml          # Default model/provider/timeout config
  examples/
    directive.md          # Sample directive file
  docs/plans/             # Design documents
  README.md
  pyproject.toml          # package: hammer, entry: hammer=hammer.cli:main
```

---

## Core Loop — 7-Phase Pipeline

```
Loop entry: hammer run --repo <path> --directive <file> [--max-iterations N]

Phase 0 — Branch Safety
  git checkout main
  git pull
  git checkout -b EngineerExternal/<timestamp>

Phase 1 — Read State
  load .hammer/runs/<run_id>.json
  create if first iteration (run_id = timestamp)

Phase 2 — Generate
  build scoped prompt:
    - directive content
    - current git diff (working tree)
    - test output from previous iteration
    - iteration count + stability signal
  call LLM provider → raw text response

Phase 3 — Extract Diff
  parse unified diff block from response (```diff ... ``` or raw unified diff)
  reject if: empty, >500 lines, malformed header

Phase 4 — Validate
  git apply --check (dry-run)
  fail fast on patch errors → write failure to state, continue loop

Phase 5 — Apply
  git apply
  git commit -m "hammer: iteration N - <one-line summary from diff>"

Phase 6 — Test
  run configured test suite (pytest / compileall / npm build / lint)
  capture stdout + stderr + exit code

Phase 7 — Update State
  write to run JSON:
    patch_hash, test_result, test_output, iteration, timestamp, model_used

Stability check:
  - EXIT if: all tests pass AND LLM returns no diff
  - EXIT if: max_iterations reached (default 10)
  - EXIT if: same patch hash seen twice (loop detected)
  - CONTINUE otherwise
```

**State is never in LLM memory. Each prompt is rebuilt from JSON + git state.**

---

## LLM Abstraction Layer

```python
class LLMProvider(ABC):
    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str: ...

class OllamaProvider(LLMProvider):
    # POST http://localhost:11434/api/generate
    # stream=False, returns completion text

# model_router.py reads env vars:
# HAMMER_MODEL=qwen3-coder:30b  (default)
# HAMMER_PROVIDER=ollama         (default)
# HAMMER_LLM_URL=http://localhost:11434
```

**Upgrade path:** Add `VLLMProvider`, `OpenAICompatProvider` by subclassing `LLMProvider`.
No core loop changes required.

---

## Tool Registry

Tools registered with JSON contracts:
```json
{
  "name": "git_status",
  "description": "Show working tree status",
  "args": [],
  "allowed": true
}
```

Whitelist enforced at registry level. Initial tools:
- `git_status`, `git_diff`, `git_apply_check`, `git_apply`, `git_checkout_branch`
- `pytest`, `compileall`
- `docker_compose_up`, `docker_compose_down`, `docker_compose_logs`, `docker_compose_ps`
- `npm_build`, `lint`

Plugin interface: `hammer.plugins` scanned at startup for additional tool modules.

---

## Branch Safety Rules

1. Never operate on main
2. Always start from: `git checkout main && git pull`
3. Create: `EngineerExternal/<UTC-timestamp>`
4. All commits go to this branch
5. Never auto-merge
6. On completion: write `PR_SUMMARY.md` to run directory

---

## Configuration

`configs/default.toml`:
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

All overridable via env vars (`HAMMER_*`) or CLI flags.

---

## Public Open Source Requirements

- No PHI logic
- No Strios internal secrets
- No proprietary dependencies
- Apache 2.0 license
- Hardware-agnostic (Ollama runs anywhere)
- Single `pip install hammer` entry point

---

## PR Summary Generation

On loop exit, write `PR_SUMMARY.md` containing:
- Branch name
- Directive summary
- Iterations run
- Patches applied (list of commit hashes)
- Final test status
- Risk assessment (pass/fail/partial)
