"""Ollama LLM provider for HammerAndNail.

Uses POST /api/generate with stream=False.
Compatible with any Ollama-served model.
"""
from __future__ import annotations

import logging
import os

import httpx

from .base import LLMProvider

logger = logging.getLogger("hammer.llm.ollama")

DEFAULT_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("HAMMER_LLM_URL", DEFAULT_URL)).rstrip("/")
        self.timeout = float(os.getenv("HAMMER_LLM_TIMEOUT", "120"))

    def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            logger.debug("Ollama response length: %d chars", len(text))
            return text
        except httpx.HTTPStatusError as exc:
            logger.exception("Ollama HTTP error: %s", exc.response.status_code)
            raise RuntimeError(f"Ollama request failed: {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.exception("Ollama connection error")
            raise RuntimeError(f"Ollama connection failed: {exc}") from exc
