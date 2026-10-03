@echo off
REM  Serves the simulator to other PCs on the local network.
REM  Leave this window open; closing it stops the server.
REM  Add --token auto to put a shared access token in front of it.
call "%~dp0_common.bat" || (pause & exit /b 1)
echo Starting the server. Leave this window open. Ctrl+C stops it.
echo.
%MPY% "%~dp0app\run_web.py" --port 8000 --concurrency 1 %*
echo.
echo The server has stopped.
pause
