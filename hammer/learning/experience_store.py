"""Experience log and strategy score persistence. All writes go to .hammer/learning/."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .strategies import Strategy

logger = logging.getLogger("hammer.learning.experience_store")

DEFAULT_ALPHA = 0.1


class ExperienceStore:
    """Append experience entries and maintain strategy scores. Audit-logged."""

    def __init__(self, learning_dir: Path, alpha: float = DEFAULT_ALPHA) -> None:
        self.learning_dir = Path(learning_dir)
        self.alpha = alpha
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.learning_dir.mkdir(parents=True, exist_ok=True)

    def _audit_log(self, message: str, **kwargs: object) -> None:
        """Write to audit.log. No silent updates."""
        audit_path = self.learning_dir / "audit.log"
        self._ensure_dir()
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {message}"
        if kwargs:
            line += " " + json.dumps(kwargs)
        line += "\n"
        audit_path.open("a").write(line)

    def record(
        self,
        run_id: str,
        strategy: str,
        reward: float,
        diff_size: int,
        tests_passed: bool,
        repo_hash: str = "",
    ) -> None:
        """Append structured entry to experience_log.jsonl."""
        self._ensure_dir()
        log_path = self.learning_dir / "experience_log.jsonl"
        entry = {
            "run_id": run_id,
            "strategy": strategy,
            "reward": reward,
            "diff_size": diff_size,
            "tests_passed": tests_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repo_hash": repo_hash,
        }
        log_path.open("a").write(json.dumps(entry) + "\n")

    def _load_scores(self) -> dict[str, float]:
        """Load strategy scores from disk."""
        scores_path = self.learning_dir / "strategy_scores.json"
        if not scores_path.exists():
            return {s.value: 0.0 for s in Strategy}
        try:
            data = json.loads(scores_path.read_text())
            result = {s.value: 0.0 for s in Strategy}
            for k, v in data.items():
                if k in result:
                    result[k] = float(v)
            return result
        except (json.JSONDecodeError, OSError):
            return {s.value: 0.0 for s in Strategy}

    def update_strategy_score(self, strategy: str, reward: float) -> None:
        """Incremental mean update. Logged to audit."""
        scores = self._load_scores()
        current = scores.get(strategy, 0.0)
        new_score = current + self.alpha * (reward - current)
        scores[strategy] = new_score

        self._audit_log(
            "strategy_score_update",
            strategy=strategy,
            previous_score=current,
            new_score=new_score,
            reward=reward,
        )

        scores_path = self.learning_dir / "strategy_scores.json"
        scores_path.write_text(json.dumps(scores, indent=2))
