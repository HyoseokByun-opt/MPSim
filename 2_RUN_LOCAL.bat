@echo off
REM  Runs the simulator for this PC only and opens the browser.
call "%~dp0_common.bat" || (pause & exit /b 1)
echo Starting on http://127.0.0.1:8000  (leave this window open)
start "" http://127.0.0.1:8000
%MPY% "%~dp0app\run_web.py" --host 127.0.0.1 --port 8000
pause
