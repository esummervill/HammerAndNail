from click.testing import CliRunner
from hammer.cli import main


def test_cli_run_requires_repo():
    runner = CliRunner()
    result = runner.invoke(main, ["run"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_cli_tools_list():
    runner = CliRunner()
    result = runner.invoke(main, ["tools", "list"])
    assert result.exit_code == 0
    assert "git_status" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
