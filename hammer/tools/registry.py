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

import hammer.tools.git_tools as _git_mod
import hammer.tools.test_tools as _test_mod
import hammer.tools.docker_tools as _docker_mod

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
    _module: object = None
    _fn_name: str = ""


_TOOL_CATALOG: list[ToolDefinition] = [
    ToolDefinition("git_status", "Show working tree status", run_git_status, _module=_git_mod, _fn_name="run_git_status"),
    ToolDefinition("git_diff", "Show unstaged changes", run_git_diff, _module=_git_mod, _fn_name="run_git_diff"),
    ToolDefinition("git_apply", "Apply a patch file", run_git_apply, _module=_git_mod, _fn_name="run_git_apply"),
    ToolDefinition("git_apply_check", "Dry-run patch check", run_git_apply_check, _module=_git_mod, _fn_name="run_git_apply_check"),
    ToolDefinition("git_log", "Show recent commits", run_git_log, _module=_git_mod, _fn_name="run_git_log"),
    ToolDefinition("pytest", "Run pytest test suite", run_pytest, _module=_test_mod, _fn_name="run_pytest"),
    ToolDefinition("compileall", "Python compile check", run_compileall, _module=_test_mod, _fn_name="run_compileall"),
    ToolDefinition("npm_build", "Run npm build", run_npm_build, _module=_test_mod, _fn_name="run_npm_build"),
    ToolDefinition("lint", "Run flake8 lint", run_lint, _module=_test_mod, _fn_name="run_lint"),
    ToolDefinition("docker_compose_ps", "Show container status", run_docker_compose_ps, _module=_docker_mod, _fn_name="run_docker_compose_ps"),
    ToolDefinition("docker_compose_logs", "Tail container logs", run_docker_compose_logs, _module=_docker_mod, _fn_name="run_docker_compose_logs"),
    ToolDefinition("docker_compose_up", "Start services", run_docker_compose_up, _module=_docker_mod, _fn_name="run_docker_compose_up"),
    ToolDefinition("docker_compose_down", "Stop services", run_docker_compose_down, _module=_docker_mod, _fn_name="run_docker_compose_down"),
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
        # Look up the function through the module reference at call time so that
        # unittest.mock.patch on the module-level name is respected.
        if tool._module is not None and tool._fn_name:
            fn = getattr(tool._module, tool._fn_name)
        else:
            fn = tool.fn
        return fn(**args)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "allowed": t.allowed}
            for t in self._tools.values()
        ]
