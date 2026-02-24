"""HammerAndNail CLI entry point.

Usage:
    cd /your/repo && hammer              # zero-friction guided flow
    hammer run                           # explicit run with directives
    hammer tools list
    hammer --version

Both `hammer` and `EngineerExternal` are registered as entry points.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from .bootstrap import ensure_ollama_access, ensure_repo_env
from .chat.interactive import InteractiveGuidedSession
from .llm.model_router import DEFAULT_MODEL, DEFAULT_PROVIDER
from .runner import run_engineering_loop
from .tools.registry import ToolRegistry
from . import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Directive filenames checked in order in the repo root
_DIRECTIVE_FILENAMES = ["directive.md", "DIRECTIVE.md", ".directive.md"]


def _find_directive(repo: Path) -> str | None:
    """Return content of directive file if found in repo root, else None."""
    for name in _DIRECTIVE_FILENAMES:
        path = repo / name
        if path.exists():
            return path.read_text()
    return None


def _resolve_directive(repo_path: Path, directive_file: str | None) -> str:
    """Resolve directive text from file, auto-detect, or interactive prompt."""
    if directive_file:
        return Path(directive_file).read_text()

    found = _find_directive(repo_path)
    if found:
        click.echo(f"  Directive: directive.md")
        return found

    click.echo("No directive.md found in current directory.")
    return click.prompt("What should the engineer do?")


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="hammer")
@click.option("--model", default=DEFAULT_MODEL, envvar="HAMMER_MODEL", show_default=True, help="LLM model name")
@click.option("--provider", default=DEFAULT_PROVIDER, envvar="HAMMER_PROVIDER", show_default=True, help="LLM provider (ollama)")
@click.option("--max-iterations", default=10, show_default=True, help="Maximum loop iterations")
@click.option("--test-command", default="pytest", show_default=True, help="Test command to run after each patch")
@click.option("--max-diff-lines", default=500, show_default=True, help="Maximum lines in a single diff")
@click.pass_context
def main(ctx, model, provider, max_iterations, test_command, max_diff_lines):
    """HammerAndNail — autonomous coding runtime.

    \b
    Run from inside any git repo:

        cd /your/repo
        hammer

    Launches the Guided mode that performs the bootstrapping, interactive planning, and deterministic loop.
    """
    if ctx.invoked_subcommand is None:
        repo_path = Path.cwd()
        ensure_repo_env(repo_path)
        ensure_ollama_access(model, provider)
        InteractiveGuidedSession(
            repo_path,
            model=model,
            provider=provider,
            max_iterations=max_iterations,
            test_command=test_command,
            max_diff_lines=max_diff_lines,
        ).run()


@main.command()
@click.option("--repo", default=None, type=click.Path(file_okay=False), help="Target repo path (default: cwd)")
@click.option("--directive", default=None, type=click.Path(dir_okay=False), help="Directive file (default: directive.md or prompt)")
@click.option("--model", default=DEFAULT_MODEL, envvar="HAMMER_MODEL", show_default=True, help="LLM model name")
@click.option("--provider", default=DEFAULT_PROVIDER, envvar="HAMMER_PROVIDER", show_default=True, help="LLM provider (ollama)")
@click.option("--max-iterations", default=10, show_default=True, help="Maximum loop iterations")
@click.option("--test-command", default="pytest", show_default=True, help="Test command to run after each patch")
@click.option("--max-diff-lines", default=500, show_default=True, help="Maximum lines in a single diff")
def run(repo, directive, model, provider, max_iterations, test_command, max_diff_lines):
    """Run the engineering loop against a repository."""
    repo_path = Path(repo).resolve() if repo else Path.cwd()
    ensure_repo_env(repo_path)
    ensure_ollama_access(model, provider)
    directive_text = _resolve_directive(repo_path, directive)
    run_engineering_loop(
        repo_path,
        directive_text,
        model,
        provider,
        max_iterations,
        test_command,
        max_diff_lines,
    )


@main.group()
def tools():
    """Tool registry commands."""


@tools.command("list")
def tools_list():
    """List all registered tools."""
    registry = ToolRegistry()
    click.echo("Registered tools:")
    for tool in registry.list_tools():
        status = "+" if tool["allowed"] else "-"
        click.echo(f"  {status}  {tool['name']:<25} {tool['description']}")
