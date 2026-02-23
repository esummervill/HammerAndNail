import pytest
from unittest.mock import patch
from hammer.tools.registry import ToolRegistry, ToolNotAllowed


def test_registry_has_git_tools():
    registry = ToolRegistry()
    assert registry.has("git_status")
    assert registry.has("git_diff")


def test_registry_blocks_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(ToolNotAllowed):
        registry.call("rm_rf", {})


def test_registry_call_git_status(tmp_path):
    registry = ToolRegistry()
    with patch("hammer.tools.git_tools.run_git_status") as mock_fn:
        mock_fn.return_value = {"status": "clean"}
        result = registry.call("git_status", {"repo": str(tmp_path)})
    assert result["status"] == "clean"


def test_registry_list_tools_returns_all():
    registry = ToolRegistry()
    tools = registry.list_tools()
    names = [t["name"] for t in tools]
    assert "git_status" in names
    assert "pytest" in names
    assert "docker_compose_ps" in names
