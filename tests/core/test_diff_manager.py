import pytest
from hammer.core.diff_manager import extract_diff, DiffValidationError, count_diff_lines

SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
-    pass
+    return "hello"
+
"""

FENCED_DIFF = f"Some text\n```diff\n{SAMPLE_DIFF}```\nMore text"


def test_extract_raw_diff():
    result = extract_diff(SAMPLE_DIFF)
    assert "--- a/foo.py" in result
    assert "+++ b/foo.py" in result
    assert "@@" in result


def test_extract_fenced_diff():
    result = extract_diff(FENCED_DIFF)
    assert "--- a/foo.py" in result
    assert "+++ b/foo.py" in result


def test_extract_empty_raises():
    with pytest.raises(DiffValidationError, match="no unified diff"):
        extract_diff("Just some text with no diff content")


def test_extract_too_large_raises():
    huge_diff = SAMPLE_DIFF + ("+line\n" * 600)
    with pytest.raises(DiffValidationError, match="exceeds maximum"):
        extract_diff(huge_diff, max_lines=500)


def test_count_diff_lines():
    assert count_diff_lines(SAMPLE_DIFF) > 0
