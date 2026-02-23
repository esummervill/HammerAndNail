"""Chat session persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ChatSession:
    """Simple disk-backed chat session used by the interactive guided mode."""

    def __init__(self, repo_path: Path, session_name: str = "session.json") -> None:
        self.repo_path = repo_path
        self.session_path = repo_path / ".hammer" / session_name
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {"messages": []}
        if self.session_path.exists():
            try:
                self.data = json.loads(self.session_path.read_text())
            except Exception:
                self.data = {"messages": []}

    def add_message(self, role: str, content: str) -> None:
        self.data.setdefault("messages", [])
        self.data["messages"].append({"role": role, "content": content})
        self._flush()

    def last_message(self) -> str | None:
        msgs = self.data.get("messages", [])
        return msgs[-1]["content"] if msgs else None

    def _flush(self) -> None:
        self.session_path.write_text(json.dumps(self.data, indent=2))
