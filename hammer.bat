@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" (
  set "REPO_ROOT=%SCRIPT_DIR:~0,-1%"
) else (
  set "REPO_ROOT=%SCRIPT_DIR%"
)

set "PYTHON_CMD="
for %%P in (python3.11 python3 python) do (
  where %%P >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=%%P"
    goto :found_python
  )
)

echo Python 3 is required but was not found in PATH. >&2
exit /b 1

:found_python
if not exist "%REPO_ROOT%\\.venv" (
  echo Creating virtual environment at "%REPO_ROOT%\\.venv" using "%PYTHON_CMD%".
  "%PYTHON_CMD%" -m venv "%REPO_ROOT%\\.venv"
)

set "PIP_CMD=%REPO_ROOT%\\.venv\\Scripts\\pip.exe"
"%PIP_CMD%" install -U pip setuptools wheel >nul
"%PIP_CMD%" install -e "%REPO_ROOT%"

set "PATH=%REPO_ROOT%\\.venv\\Scripts;%PATH%"

hammer %*
