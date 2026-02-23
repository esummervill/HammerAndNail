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

### Zero-friction terminal mode

After installing `hammernail` in editable mode (`pip install -e .`) and making the package entrypoint available on your `PATH`, typing `hammer` from anywhere performs all bootstrap steps automatically:

1. Detects the current repository root and ensures `.venv` exists.
2. Upgrades `pip`, `setuptools`, `wheel` and installs the local `hammer` package into `.venv`.
3. Verifies Ollama is reachable and pulls `qwen3-coder:30b` (or your configured model) if it is missing.
4. Launches an interactive guided session that keeps rolling context in `.hammer/session.json`, asks for your goal and constraints, proposes a plan, and requests confirmation before starting the deterministic loop.

When the guided session runs, Hammer stages only the files touched by the generated patch, filters them through the forbidden prefixes/suffixes list, validates that `git status --porcelain` reported no extra entries, and refuses to commit when stray artifacts would otherwise be staged.

```
cd /path/to/target/repo
hammer
```

The experience prints:

```
Hammer Engineer Ready.
Repository detected: <name>
Model: qwen3-coder:30b
Mode: Guided
What would you like to build or improve?
```

When you start `hammer` you choose either the guided engineering session or the developer chat. The developer chat is a free-form conversation preserved under `.hammer/chat_session.json`, never mutates Git unless you explicitly run `/execute`, and runs the LLM with a larger token budget so it can reason longer without summary truncation. New chat commands include:

```
/plan           # summarize the current goal + constraints into a plan
/constraint ... # add a constraint to the next plan
/constraints    # list the recorded constraints
/execute        # confirm and run the guided engineering loop with the latest plan
```

Use `/plan` after describing your goal, review the proposed steps, adjust constraints via `/constraint`, and when you say “yes” run `/execute` to transition directly into the guided mode with the generated directive. If you say “no” to the plan, keep chatting to refine the goal before rerunning `/plan`.

Describe what to build, add constraints when prompted, confirm the plan, and Hammer will create `EngineerExternal/<timestamp>`, drive the 7-phase loop, and write `PR_SUMMARY.md`. All context is persisted so restarting `hammer` resumes your conversation history.

### One-command helper

If you keep `HammerAndNail` checked out somewhere central, `scripts/hammer.sh` takes care of venv creation, installs the package in editable mode, and then proxies command arguments to `hammer`. Run it from any path and point it at the repository you want to work on:

```bash
cd /where/you/want/to/run
/Users/ethansummervill/Projects/EngineerRuntime/scripts/hammer.sh \
  --repo /path/to/target/repo \
  --directive /path/to/directive.md
```

The helper silently upgrades `pip`, `setuptools`, and `wheel`, so rerunning it ensures the environment stays current without manual steps.

On Windows you can invoke `hammer` directly (it resolves to `hammer.bat` in this repo root) after adding the checkout directory to your `PATH` or running the batch file from that directory. The batch file mirrors the same bootstrap steps—venv creation, editable install, and execution—so typing `hammer --repo ... --directive ...` is all that’s needed once the folder is on your command path.

### Describe and run (interactive)

If you prefer to tell Hammer what you need in plain English, run `scripts/hammer_prompt.sh`. It asks for a goal plus optional constraints, writes a temporary directive (printed to the console), and then launches `hammer run` for you.

```bash
/Users/ethansummervill/Projects/EngineerRuntime/scripts/hammer_prompt.sh \
  --repo /path/to/target/repo
```

Any additional flags (e.g., `--run-id`) added after the helper are forwarded directly to `hammer run`. If you skip `--repo`, the script uses your current working directory.

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
