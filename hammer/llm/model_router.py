"""Environment-driven LLM provider router.

Reads HAMMER_PROVIDER env var to select the active provider.
All provider-specific configuration is handled inside each provider class.

To add a new provider:
  1. Subclass LLMProvider
  2. Add to _PROVIDERS dict below
"""
from __future__ import annotations

import os

from .base import LLMProvider
from .ollama_provider import OllamaProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
}

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3-coder:30b"


def get_provider(name: str | None = None) -> LLMProvider:
    """Return a configured LLM provider based on the requested name or the HAMMER_PROVIDER env var."""
    request = (name or os.getenv("HAMMER_PROVIDER", DEFAULT_PROVIDER)).lower()
    cls = _PROVIDERS.get(request)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{request}'. Available: {list(_PROVIDERS.keys())}"
        )
    return cls()


def get_model() -> str:
    """Return the configured model name from HAMMER_MODEL env var."""
    return os.getenv("HAMMER_MODEL", DEFAULT_MODEL)
