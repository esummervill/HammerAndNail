import json
from pathlib import Path
import pytest
from hammer.core.state import RunState, load_state, save_state, new_run_id


def test_new_run_id_is_timestamp_string():
    run_id = new_run_id()
    assert isinstance(run_id, str)
    assert len(run_id) == 15  # YYYYmmdd-HHMMSS


def test_save_and_load_roundtrip(tmp_path):
    state = RunState(
        run_id="20260223-120000",
        repo_path=str(tmp_path),
        directive="fix tests",
        iteration=0,
        branch="EngineerExternal/20260223-120000",
        patches=[],
        test_results=[],
        stable=False,
    )
    save_state(state, tmp_path / ".hammer" / "runs")
    loaded = load_state("20260223-120000", tmp_path / ".hammer" / "runs")
    assert loaded.run_id == state.run_id
    assert loaded.directive == state.directive
    assert loaded.stable is False


def test_load_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_state("nonexistent", tmp_path / ".hammer" / "runs")
