import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from hammer.core.validator import validate_patch, PatchValidationError

SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,3 @@
 def hello():
-    pass
+    return "hello"
"""


def test_validate_patch_success(tmp_path):
    with patch("hammer.core.validator._run_git_apply_check") as mock_check:
        mock_check.return_value = MagicMock(returncode=0, stderr="")
        # Should not raise
        validate_patch(SAMPLE_DIFF, tmp_path)


def test_validate_patch_failure_raises(tmp_path):
    with patch("hammer.core.validator._run_git_apply_check") as mock_check:
        mock_check.return_value = MagicMock(
            returncode=1, stderr="error: patch does not apply"
        )
        with pytest.raises(PatchValidationError, match="patch does not apply"):
            validate_patch(SAMPLE_DIFF, tmp_path)
