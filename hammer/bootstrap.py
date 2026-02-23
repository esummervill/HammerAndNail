"""Repository-level environment bootstrap helpers."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
import httpx

logger = logging.getLogger("hammer.bootstrap")

DEFAULT_LLM_URL = "http://localhost:11434"


def _locate_python() -> str:
    candidates = ("python3.11", "python3", "python")
    for name in candidates:
        path = shutil.which(name)
        if path:
            logger.debug("Using Python interpreter: %s", path)
            return path
    raise RuntimeError("Python 3 interpreter not found in PATH.")


def ensure_repo_env(repo_path: Path) -> Path:
    """Ensure `.venv` exists and local dependencies are installed."""
    venv_path = repo_path / ".venv"
    python_cmd = _locate_python()
    if not venv_path.exists():
        logger.info("Creating virtual environment at %s", venv_path)
        subprocess.run([python_cmd, "-m", "venv", str(venv_path)], check=True)

    pip_exe = venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
    try:
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip", "setuptools", "wheel"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([str(pip_exe), "install", "-e", str(repo_path)], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to install project dependencies inside repository venv.") from exc

    return venv_path


def ensure_ollama_access(model: str, provider: str) -> None:
    """Validate that Ollama is reachable and the requested model is available."""
    if provider.lower() != "ollama":
        return

    llm_url = os.getenv("HAMMER_LLM_URL", DEFAULT_LLM_URL).rstrip("/")
    health_url = f"{llm_url}/health"
    response = httpx.get(health_url, timeout=5)
    if response.status_code == 404:
        logger.warning("Ollama health endpoint %s returned 404; assuming service is reachable.", health_url)
    else:
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Ollama health check failed at {health_url}: {exc}") from exc

    if not shutil.which("ollama"):
        raise RuntimeError("`ollama` CLI not found in PATH; needed to pull models.")

    try:
        ls_result = subprocess.run(["ollama", "ls"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to contact Ollama CLI to list models.") from exc

    if model not in ls_result.stdout:
        logger.info("Model %s not present locally, attempting to pull.", model)
        try:
            subprocess.run(["ollama", "pull", model], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to pull model {model}: {exc}") from exc
