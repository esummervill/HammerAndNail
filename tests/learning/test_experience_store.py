"""Tests for experience store: log append, score update, audit."""
import json
from pathlib import Path

import pytest

from hammer.learning.experience_store import ExperienceStore
from hammer.learning.strategies import Strategy


def test_experience_log_appends(tmp_path):
    store = ExperienceStore(tmp_path)
    store.record(
        run_id="run1",
        strategy="minimal_patch",
        reward=12.0,
        diff_size=50,
        tests_passed=True,
        repo_hash="abc",
    )
    log_path = tmp_path / "experience_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["run_id"] == "run1"
    assert entry["strategy"] == "minimal_patch"
    assert entry["reward"] == 12.0
    assert entry["tests_passed"] is True


def test_strategy_scores_update_correctly(tmp_path):
    store = ExperienceStore(tmp_path, alpha=0.1)
    store.update_strategy_score("minimal_patch", 10.0)
    scores_path = tmp_path / "strategy_scores.json"
    assert scores_path.exists()
    data = json.loads(scores_path.read_text())
    assert data["minimal_patch"] == 1.0  # 0 + 0.1 * (10 - 0)

    store.update_strategy_score("minimal_patch", 12.0)
    data = json.loads(scores_path.read_text())
    # 1.0 + 0.1 * (12 - 1.0) = 1.0 + 1.1 = 2.1
    assert abs(data["minimal_patch"] - 2.1) < 0.001


def test_audit_log_written(tmp_path):
    store = ExperienceStore(tmp_path)
    store.update_strategy_score("rewrite", 5.0)
    audit_path = tmp_path / "audit.log"
    assert audit_path.exists()
    content = audit_path.read_text()
    assert "strategy_score_update" in content
    assert "rewrite" in content
    assert "5.0" in content
