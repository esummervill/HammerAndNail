"""Unified diff extraction and validation.

Parses LLM response text to extract a valid unified diff.
Rejects diffs that are empty, malformed, or exceed size limits.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("hammer.core.diff_manager")

# Match ```diff ... ``` fenced blocks
_FENCED_DIFF_RE = re.compile(r"```(?:diff)?\s*\n(.*?)```", re.DOTALL)
# Match raw unified diff headers
_RAW_DIFF_RE = re.compile(r"(^---\s+\S.*?$)", re.MULTILINE)


class DiffValidationError(ValueError):
    """Raised when a diff cannot be extracted or is invalid."""


def extract_diff(text: str, max_lines: int = 500) -> str:
    """Extract unified diff from LLM response text.

    Tries fenced ```diff blocks first, then raw unified diff headers.
    Raises DiffValidationError if no valid diff is found or size limit exceeded.
    """
    diff_text = _try_fenced(text) or _try_raw(text)

    if not diff_text:
        raise DiffValidationError("no unified diff found in LLM response")

    diff_text = diff_text.strip()
    lines = diff_text.splitlines()

    if len(lines) > max_lines:
        raise DiffValidationError(
            f"diff exceeds maximum of {max_lines} lines (got {len(lines)})"
        )

    # Must have at least one hunk header
    if not any(line.startswith("@@") for line in lines):
        raise DiffValidationError("diff has no hunk headers (not a valid unified diff)")

    return diff_text + "\n"


def count_diff_lines(diff_text: str) -> int:
    return len(diff_text.splitlines())


def _try_fenced(text: str) -> str | None:
    match = _FENCED_DIFF_RE.search(text)
    if match:
        logger.debug("Extracted fenced diff block")
        return match.group(1)
    return None


def _try_raw(text: str) -> str | None:
    match = _RAW_DIFF_RE.search(text)
    if match:
        start = match.start()
        logger.debug("Extracted raw diff starting at char %d", start)
        return text[start:]
    return None
