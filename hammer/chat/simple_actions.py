"""Simple intent detection for trivial filesystem edits."""
from __future__ import annotations

import re


class DirectoryAction:
    """Represents a request to create a directory tracked via .gitkeep."""

    def __init__(self, path: str) -> None:
        self.path = path.strip().strip("/\\")

    def diff(self) -> str:
        if not self.path:
            raise ValueError("Directory path is required.")
        file_path = f"{self.path}/.gitkeep"
        return "\n".join(
            [
                f"diff --git a/{file_path} b/{file_path}",
                "new file mode 100644",
                "index 0000000..e69de29",
                "--- /dev/null",
                f"+++ b/{file_path}",
                "@@ -0,0 +1 @@",
                "+# placeholder created by Hammer sandbox intent",
                "",
            ]
        )


def detect_directory_action(goal: str) -> DirectoryAction | None:
    """Detect phrases that describe creating a named directory."""
    if not goal:
        return None
    patterns = [
        r"name it (?P<name>[A-Za-z0-9_.\-\/]+)",
        r"create (?:a )?directory(?: called| named)? (?P<name>[A-Za-z0-9_.\-\/]+)",
        r"make (?:a )?directory(?: called| named)? (?P<name>[A-Za-z0-9_.\-\/]+)",
        r"build (?:a )?directory(?: called| named)? (?P<name>[A-Za-z0-9_.\-\/]+)",
        r"create (?:a )?folder(?: called| named)? (?P<name>[A-Za-z0-9_.\-\/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, goal, re.IGNORECASE)
        if match:
            return DirectoryAction(match.group("name"))
    return None
