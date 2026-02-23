"""LLM provider abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for all LLM provider implementations."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        """Generate a completion for the given prompt.

        Returns the raw text response from the model.
        Raises RuntimeError on provider failures.
        """
