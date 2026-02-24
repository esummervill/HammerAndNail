"""Epsilon-greedy strategy selector with seeded RNG for reproducibility."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .strategies import Strategy

logger = logging.getLogger("hammer.learning.strategy_selector")

DEFAULT_EPSILON = 0.1
DEFAULT_MIN_RUNS_BEFORE_BIAS = 5


class StrategySelector:
    """Select strategy using epsilon-greedy. Fully reproducible with fixed seed."""

    def __init__(
        self,
        learning_dir: Path,
        epsilon: float = DEFAULT_EPSILON,
        min_runs_before_bias: int = DEFAULT_MIN_RUNS_BEFORE_BIAS,
        seed: int | None = None,
    ) -> None:
        self.learning_dir = Path(learning_dir)
        self.epsilon = epsilon
        self.min_runs_before_bias = min_runs_before_bias
        self._seed = seed
        self._rng = None

    def _get_rng(self):
        """Lazy-init RNG with seed for reproducibility."""
        if self._rng is None:
            import random
            self._rng = random.Random(self._seed)
        return self._rng

    def _load_scores(self) -> dict[str, float]:
        """Load strategy scores from persistent storage."""
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
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load strategy scores: %s", exc)
            return {s.value: 0.0 for s in Strategy}

    def _count_experiences(self) -> int:
        """Count entries in experience log."""
        log_path = self.learning_dir / "experience_log.jsonl"
        if not log_path.exists():
            return 0
        try:
            return sum(1 for _ in log_path.open() if _.strip())
        except OSError:
            return 0

    def select(self, state: dict | None = None) -> Strategy:
        """
        Select strategy using epsilon-greedy. Deterministic given same seed + experience log.
        state is optional; initial version ignores it and uses global Q.
        """
        scores = self._load_scores()
        experience_count = self._count_experiences()
        rng = self._get_rng()

        # Explore: random strategy
        if experience_count < self.min_runs_before_bias or rng.random() < self.epsilon:
            return rng.choice(list(Strategy))

        # Exploit: best scoring strategy
        best_strategy = max(
            Strategy,
            key=lambda s: scores.get(s.value, 0.0),
        )
        return best_strategy
