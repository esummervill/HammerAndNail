from hammer.chat.simple_actions import detect_directory_action, DirectoryAction


def test_detect_directory_action_simple_named():
    action = detect_directory_action("Create a directory named Sandbox in this repo")
    assert isinstance(action, DirectoryAction)
    assert action.path == "Sandbox"


def test_detect_directory_invalid_returns_none():
    assert detect_directory_action("Please make a change") is None


def test_directory_action_diff_contains_file():
    action = DirectoryAction("nested/dir")
    diff = action.diff()
    assert "nested/dir/.gitkeep" in diff
    assert "+# placeholder created by Hammer sandbox intent" in diff
