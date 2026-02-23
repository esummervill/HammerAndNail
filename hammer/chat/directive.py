"""Utility helpers for building directives from conversational goals."""
from __future__ import annotations

from typing import Iterable


def build_directive(goal: str, constraints: Iterable[str] | None = None, plan_steps: Iterable[str] | None = None) -> str:
    """Return a directive text for the guided loop."""
    clean_goal = goal.strip()
    lines = ["# Directive generated from conversation", ""]
    lines.append(clean_goal or "No specific goal provided.")

    constraints_list = list(constraints or [])
    if constraints_list:
        lines.extend(["", "## Constraints"])
        lines.extend(f"- {constraint}" for constraint in constraints_list)

    plan_steps_list = list(plan_steps or [])
    if plan_steps_list:
        lines.extend(["", "## Plan"])
        for idx, step in enumerate(plan_steps_list, start=1):
            lines.append(f"{idx}. {step}")

    return "\n".join(lines)
