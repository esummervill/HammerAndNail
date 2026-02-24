"""Strategy enum for prompt scaffolding. Strategies modify prompt construction only, not execution logic."""
from __future__ import annotations

from enum import Enum


class Strategy(str, Enum):
    """Strategy modes that influence prompt scaffolding. Execution remains deterministic."""

    MINIMAL_PATCH = "minimal_patch"
    REWRITE = "rewrite"
    ADD_TESTS_FIRST = "add_tests_first"
    REFACTOR_ONLY = "refactor_only"
    STATIC_ANALYSIS_FIRST = "static_analysis_first"
