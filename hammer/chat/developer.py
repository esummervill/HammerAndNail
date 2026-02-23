"""Developer conversation mode for zero-friction sessions."""
from __future__ import annotations

import click
from pathlib import Path

from ..bootstrap import ensure_ollama_access, ensure_repo_env
from ..llm.model_router import DEFAULT_MODEL, DEFAULT_PROVIDER, get_provider
from ..runner import run_engineering_loop
from .directive import build_directive
from .interactive import InteractiveGuidedSession
from .session import ChatSession


def _offers_guided_transition(message: str) -> bool:
    lower = message.lower()
    triggers = ("implement", "execute", "run", "apply", "fix", "deploy")
    return any(trigger in lower for trigger in triggers)


class DeveloperChatSession:
    """Free-form developer conversation with persistent context."""

    def __init__(
        self,
        repo_path: Path,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        max_tokens: int = 2048,
        max_iterations: int = 10,
        test_command: str = "pytest",
        max_diff_lines: int = 500,
    ) -> None:
        self.repo_path = repo_path
        self.model = model
        self.provider_name = provider
        self.max_iterations = max_iterations
        self.test_command = test_command
        self.max_diff_lines = max_diff_lines
        self.max_tokens = max_tokens
        self.session = ChatSession(repo_path, session_name="chat_session.json")
        self.provider = get_provider(self.provider_name)
        self.current_goal = ""
        self.constraints: list[str] = []
        self.plan_steps: list[str] = []

    def run(self) -> None:
        click.secho("Hammer Dev Mode Active.", fg="green")
        click.echo(f"Repo: {self.repo_path.name}")
        click.echo(f"Model: {self.model}")
        click.echo("Mode: Developer Chat")
        click.echo("What are we working on?")

        while True:
            try:
                user_input = click.prompt("> ", default="", show_default=False)
            except (EOFError, KeyboardInterrupt):
                click.echo("\nExiting developer chat.")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                click.echo("Ending developer chat.")
                break

            self.session.add_message("user", user_input)
            if self._handle_command(user_input):
                continue

            response = self._prompt_llm()
            self.session.add_message("assistant", response)
            click.echo(response)

    def _prompt_llm(self) -> str:
        prompt = self._build_prompt()
        return self.provider.generate(
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

    def _build_prompt(self) -> str:
        history = []
        for message in self.session.data.get("messages", []):
            role = message.get("role", "assistant").capitalize()
            history.append(f"{role}: {message.get('content', '')}")
        lines = [
            f"You are a senior developer having a free-form conversation about {self.repo_path.name}.",
            "Keep the context of the repo in mind and continue the discussion fluidly.",
            "",
            *history,
            "",
            "Assistant:",
        ]
        return "\n".join(lines)

    def _handle_command(self, user_input: str) -> bool:
        if not user_input.startswith("/"):
            self.current_goal = user_input
            return False

        parts = user_input.split(None, 1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/plan":
            plan = self._build_plan()
            click.echo(plan)
            return True
        if command == "/constraint" and arg:
            self.constraints.append(arg)
            click.echo("Constraint noted.")
            return True
        if command == "/constraints":
            if self.constraints:
                click.echo("Current constraints:")
                for constraint in self.constraints:
                    click.echo(f"- {constraint}")
            else:
                click.echo("No constraints recorded.")
            return True
        if command in {"/execute", "/run"}:
            self._execute_plan()
            return True
        return False

    def _build_plan(self) -> str:
        goal = self.current_goal or "Review the repository and recommend actionable improvements."
        steps = [
            "Assess the existing structure and key components for clarity and modularity.",
            f"Interpret the stated goal: {goal}",
            "Translate the findings into a concrete implementation plan with clear checkpoints.",
            f"Guard each proposal with the constraints mentioned ({', '.join(self.constraints) or 'none'}) and test expectations.",
            "Prepare to apply the first step once the plan is approved.",
        ]
        self.plan_steps = steps
        lines = [f"Plan for: {goal}", ""]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")
        return "\n".join(lines)

    def _build_directive_from_plan(self) -> str:
        return build_directive(self.current_goal or "Review conversation", self.constraints, self.plan_steps)

    def _execute_plan(self) -> None:
        if not self.plan_steps:
            self._build_plan()
        click.echo("Executing plan:")
        click.echo("\n".join(f"{idx+1}. {step}" for idx, step in enumerate(self.plan_steps)))
        if not click.confirm("Proceed with Guided Execution Mode using this plan?", default=True):
            click.echo("Plan execution cancelled. Update the goal or constraints, then run /plan again.")
            return
        ensure_repo_env(self.repo_path)
        ensure_ollama_access(self.model, self.provider_name)
        directive_text = self._build_directive_from_plan()
        run_engineering_loop(
            self.repo_path,
            directive_text,
            self.model,
            self.provider_name,
            self.max_iterations,
            self.test_command,
            self.max_diff_lines,
        )
