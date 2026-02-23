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
        status = "+" if tool["allowed"] else "-"
        click.echo(f"  {status}  {tool['name']:<25} {tool['description']}")
