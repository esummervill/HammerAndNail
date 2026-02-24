"""Learning module configuration. Toggleable; disabled = identical to current Hammer."""
from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

DEFAULT_LEARNING_DIR = Path(".hammer") / "learning"
DEFAULT_EPSILON = 0.1
DEFAULT_ALPHA = 0.1
DEFAULT_MIN_RUNS_BEFORE_BIAS = 5


def load_learning_config(repo_path: Path) -> dict:
    """Load learning config from configs/learning.toml and env vars. Safe defaults if missing."""
    config: dict = {
        "learning_enabled": False,
        "epsilon": DEFAULT_EPSILON,
        "alpha": DEFAULT_ALPHA,
        "min_runs_before_bias": DEFAULT_MIN_RUNS_BEFORE_BIAS,
        "seed": None,
    }

    config_path = repo_path / "configs" / "learning.toml"
    if config_path.exists():
        try:
            data = tomllib.loads(config_path.read_text())
            if "learning" in data:
                cfg = data["learning"]
                config["learning_enabled"] = cfg.get("enabled", False)
                config["epsilon"] = float(cfg.get("epsilon", DEFAULT_EPSILON))
                config["alpha"] = float(cfg.get("alpha", DEFAULT_ALPHA))
                config["min_runs_before_bias"] = int(
                    cfg.get("min_runs_before_bias", DEFAULT_MIN_RUNS_BEFORE_BIAS)
                )
                if "seed" in cfg and cfg["seed"] is not None:
                    config["seed"] = int(cfg["seed"])
        except (tomllib.TOMLDecodeError, OSError, ValueError):
            pass

    # Env overrides
    if os.getenv("HAMMER_LEARNING_ENABLED", "").lower() in ("1", "true", "yes"):
        config["learning_enabled"] = True
    if "HAMMER_LEARNING_EPSILON" in os.environ:
        try:
            config["epsilon"] = float(os.environ["HAMMER_LEARNING_EPSILON"])
        except ValueError:
            pass
    if "HAMMER_LEARNING_SEED" in os.environ:
        try:
            config["seed"] = int(os.environ["HAMMER_LEARNING_SEED"])
        except ValueError:
            pass

    return config
