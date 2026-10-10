@echo off
REM ---------------------------------------------------------------------------
REM One-click launcher for the Streamlit prototype.
REM This file is intentionally ASCII-only: Windows cmd parses .cmd bytes with the
REM active console codepage, and non-ASCII text here breaks on GBK/UTF-8 mixups.
REM All the human-readable (Chinese) messages live in scripts\run_app.py.
REM
REM Double-click, or run:  run_app.cmd --local-only
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

python -m scripts.run_app %*
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" echo [launcher] exited with code %EXITCODE%
pause
