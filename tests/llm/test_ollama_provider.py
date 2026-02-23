import pytest
import respx
import httpx
from hammer.llm.ollama_provider import OllamaProvider

BASE_URL = "http://localhost:11434"


def test_generate_returns_response_text():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/api/generate").mock(
            return_value=httpx.Response(
                200, json={"response": "def hello():\n    return 'hi'\n"}
            )
        )
        provider = OllamaProvider(base_url=BASE_URL)
        result = provider.generate(
            prompt="fix this", model="qwen3-coder:30b", max_tokens=512, temperature=0.2
        )
    assert "def hello" in result


def test_generate_raises_on_http_error():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/api/generate").mock(return_value=httpx.Response(500))
        provider = OllamaProvider(base_url=BASE_URL)
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            provider.generate("prompt", "model", 512, 0.2)
