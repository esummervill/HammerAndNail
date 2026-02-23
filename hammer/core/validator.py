"""Patch validation via git apply --check (dry-run).

Runs git apply --check before attempting to apply any patch.
This prevents partial application and keeps the working tree clean.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("hammer.core.validator")


class PatchValidationError(ValueError):
    """Raised when a patch fails git apply --check."""


def _run_git_apply_check(patch_path: Path, repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def validate_patch(diff_text: str, repo: Path) -> None:
    """Dry-run a unified diff against the repo using git apply --check.

    Raises PatchValidationError if the patch would not apply cleanly.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False
    ) as tmp:
        tmp.write(diff_text)
        patch_path = Path(tmp.name)

    try:
        result = _run_git_apply_check(patch_path, repo)
        if result.returncode != 0:
            msg = result.stderr.strip() or "git apply --check failed"
            logger.warning("Patch validation failed: %s", msg)
            raise PatchValidationError(msg)
        logger.debug("Patch validated successfully")
    finally:
        patch_path.unlink(missing_ok=True)
