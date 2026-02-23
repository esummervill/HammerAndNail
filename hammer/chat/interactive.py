"""Interactive guided experience that builds directives from conversation."""
from __future__ import annotations

import click
from pathlib import Path

from ..llm.model_router import DEFAULT_MODEL, DEFAULT_PROVIDER
from ..runner import run_engineering_loop
from .directive import build_directive
from .session import ChatSession


class InteractiveGuidedSession:
    """Manages a guided chat loop and builds directives for the core engine."""

    def __init__(
        self,
        repo_path: Path,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        max_iterations: int = 10,
        test_command: str = "pytest",
        max_diff_lines: int = 500,
    ) -> None:
        self.repo_path = repo_path
        self.model = model
        self.provider = provider
        self.max_iterations = max_iterations
        self.test_command = test_command
        self.max_diff_lines = max_diff_lines
        self.session = ChatSession(repo_path)

    def run(self) -> None:
        self._print_header()
        goal = self._prompt_goal()
        self.session.add_message("user", goal)
        constraints = self._prompt_constraints()
        plan_text = self._build_plan(goal, constraints)
        self.session.add_message("assistant", plan_text)
        click.echo("\nPlan:")
        click.secho(plan_text, fg="cyan")
        if not click.confirm("Proceed with branch creation and execution?", default=True):
            click.echo("Okay, adjust the goal or constraints and rerun `hammer` when ready.")
            return
        directive_text = self._build_directive(goal, constraints)
        run_engineering_loop(
            self.repo_path,
            directive_text,
            self.model,
            self.provider,
            self.max_iterations,
            self.test_command,
            self.max_diff_lines,
        )

    def _print_header(self) -> None:
        click.secho("Hammer Engineer Ready.", fg="green")
        click.echo(f"Repository detected: {self.repo_path.name}")
        click.echo(f"Model: {self.model}")
        click.echo("Mode: Guided")
        click.echo("What would you like to build or improve?")

    def _prompt_goal(self) -> str:
        return click.prompt("Describe the goal", type=str)

    def _prompt_constraints(self) -> list[str]:
        constraints: list[str] = []
        while True:
            constraint = click.prompt("Constraint (leave empty to finish)", default="", show_default=False)
            if not constraint.strip():
                break
            constraints.append(constraint.strip())
        return constraints

    def _build_directive(self, goal: str, constraints: list[str]) -> str:
        return build_directive(goal, constraints)

    def _build_plan(self, goal: str, constraints: list[str]) -> str:
        structure = ", ".join(p.name for p in sorted(self.repo_path.iterdir()) if p.is_dir())
        plan_lines = [
            f"1. Review repository structure ({structure or 'no subdirectories'}) and existing state.",
            f"2. Address goal: {goal.strip()}",
            f"3. Respect constraints: {', '.join(constraints) if constraints else 'none specified'}.",
            f"4. Run `{self.test_command}` after each patch and guard for regressions.",
            "5. Commit work to EngineerExternal/<timestamp> and write PR_SUMMARY.md.",
        ]
        return "\n".join(plan_lines)
