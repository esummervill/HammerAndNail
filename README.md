# HammerAndNail

**Modular autonomous coding runtime.**

HammerAndNail runs a deterministic agent loop against any git repository,
using an LLM to generate patches, validate them with `git apply --check`,
apply them, run tests, and iterate to stability.

The intelligence is in the architecture, not the token count.

---

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

---

## Quick Start

### Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- `qwen3-coder:30b` pulled: `ollama pull qwen3-coder:30b`

### Install

```bash
pip install hammernail
```

### Run

```bash
hammer run \
  --repo /path/to/your/repo \
  --directive directive.md
```

---

## Configuration

All settings are overridable via environment variables:

```bash
export HAMMER_MODEL=qwen3-coder:30b        # default
export HAMMER_PROVIDER=ollama              # default
export HAMMER_LLM_URL=http://localhost:11434  # default
export HAMMER_LLM_TIMEOUT=120             # seconds
```

Or edit `configs/default.toml`.

---

## Branch Safety

HammerAndNail **never touches main**. Every run:

1. Checks out `main`
2. Pulls latest
3. Creates `EngineerExternal/<UTC-timestamp>`
4. All commits go to this branch only
5. Writes `PR_SUMMARY.md` on completion — never auto-merges

---

## Plugin System

Drop a Python module into `hammer/plugins/` with a `register(registry)` function:

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

Plugins are auto-discovered and loaded at runtime.

---

## Model Upgrade Path

Switch providers and models via env var — no code changes required:

```bash
# Current (local Ollama)
HAMMER_PROVIDER=ollama HAMMER_MODEL=qwen3-coder:30b hammer run ...

# Future (vLLM on GPU)
HAMMER_PROVIDER=vllm HAMMER_MODEL=deepseek-coder-v2 hammer run ...

# Future (any OpenAI-compatible endpoint)
HAMMER_PROVIDER=openai_compat HAMMER_MODEL=cogito hammer run ...
```

To add a provider: subclass `LLMProvider` in `hammer/llm/`, add it to `_PROVIDERS` in `model_router.py`.

---

## CLI Reference

```
hammer run          Run the engineering loop against a repository
hammer tools list   List all registered tools
hammer --version    Show version
EngineerExternal    Alias for hammer (same entry point)
```

---

## Project Structure

```
hammer/
  core/
    loop.py           7-phase pipeline orchestrator
    state.py          JSON run state (.hammer/runs/<id>.json)
    branch_manager.py Git branch safety
    diff_manager.py   Unified diff extraction
    validator.py      git apply --check
  llm/
    base.py           LLMProvider ABC
    ollama_provider.py Ollama /api/generate
    model_router.py   Env-driven provider selection
  tools/
    registry.py       JSON-contract tool registry
    git_tools.py      Git subprocess wrappers
    test_tools.py     pytest / compileall / npm / lint
    docker_tools.py   docker compose wrappers
  plugins/
    __init__.py       Auto-discovery interface
  cli.py              Click CLI entry point
configs/
  default.toml        Default configuration
examples/
  directive.md        Sample directive file
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
