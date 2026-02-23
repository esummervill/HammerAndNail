import os
import pytest
from hammer.llm.model_router import get_provider
from hammer.llm.ollama_provider import OllamaProvider


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("HAMMER_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, OllamaProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("HAMMER_PROVIDER", "nonexistent")
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider()
