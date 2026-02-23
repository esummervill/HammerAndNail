"""Interactive guided experience that builds directives from conversation."""
from __future__ import annotations

import click
from pathlib import Path

from ..llm.model_router import DEFAULT_MODEL, DEFAULT_PROVIDER
from ..runner import run_engineering_loop
from .directive import build_directive
from .session import ChatSession


class InteractiveGuidedSession:
    """Conversation-driven guided session that submits plans before running the loop."""

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
        goal = ""
        while True:
            if not goal.strip():
                goal = self._prompt_goal()
            self.session.add_message("user", goal)

            constraints = self._prompt_constraints()
            plan_steps, plan_text = self._build_plan(goal, constraints)
            self.session.add_message("assistant", plan_text)

            click.echo("\nPlan:")
            click.secho(plan_text, fg="cyan")

            if click.confirm("Proceed with this plan and run the engineering loop?", default=True):
                directive_text = build_directive(goal, constraints, plan_steps)
                run_engineering_loop(
                    self.repo_path,
                    directive_text,
                    self.model,
                    self.provider,
                    self.max_iterations,
                    self.test_command,
                    self.max_diff_lines,
                )
                break

            click.echo("Plan declined. Let's refine it.")
            goal = click.prompt("Refine the goal (leave empty to keep prior goal)", default="", show_default=False) or goal

    def _print_header(self) -> None:
        click.secho("Hammer Engineer Ready.", fg="green")
        click.echo(f"Repository detected: {self.repo_path.name}")
        click.echo(f"Model: {self.model}")
        click.echo("Mode: Conversational Plan")
        click.echo("Describe what you want to build, review the plan, and confirm execution.")

    def _prompt_goal(self) -> str:
        return click.prompt("Goal", type=str)

    def _prompt_constraints(self) -> list[str]:
        constraints: list[str] = []
        while True:
            constraint = click.prompt("Constraint (leave empty to finish)", default="", show_default=False)
            if not constraint.strip():
                break
            constraints.append(constraint.strip())
        return constraints

    def _build_plan(self, goal: str, constraints: list[str]) -> tuple[list[str], str]:
        structure = ", ".join(p.name for p in sorted(self.repo_path.iterdir()) if p.is_dir())
        steps = [
            f"Review repository structure ({structure or 'no subdirectories'}) and current state.",
            f"Address goal: {goal.strip()}.",
            f"Respect constraints: {', '.join(constraints) if constraints else 'none specified'}.",
            f"Run `{self.test_command}` after each patch to guard regressions.",
            "Commit work to EngineerExternal/<timestamp> and produce PR_SUMMARY.md.",
        ]
        plan_lines = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps))
        return steps, plan_lines
