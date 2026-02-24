"""Lightweight repo state encoding for strategy selection. Used by strategy_selector."""
from __future__ import annotations

from pathlib import Path


def encode_repo_state(repo_path: Path) -> dict:
    """Return lightweight state for strategy selection. Initial version uses global Q."""
    try:
        files = list(repo_path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        py_count = sum(1 for f in files if f.suffix == ".py")
        js_count = sum(1 for f in files if f.suffix in (".js", ".ts", ".tsx", ".jsx"))
        has_tests = any(
            "test" in p.name.lower() or "spec" in p.name.lower()
            for p in files
            if p.is_file()
        )

        if py_count >= js_count and py_count > 0:
            language = "python"
        elif js_count > py_count:
            language = "node"
        else:
            language = "unknown"

        if file_count < 50:
            repo_size_bucket = "small"
        elif file_count < 500:
            repo_size_bucket = "medium"
        else:
            repo_size_bucket = "large"

        return {
            "file_count": file_count,
            "language": language,
            "has_tests": has_tests,
            "repo_size_bucket": repo_size_bucket,
        }
    except Exception:
        return {
            "file_count": 0,
            "language": "unknown",
            "has_tests": False,
            "repo_size_bucket": "small",
        }
