@echo off
REM  Explains why other PCs cannot reach the server. Changes nothing.
call "%~dp0_common.bat" || (pause & exit /b 1)
%MPY% "%~dp0app\run_web.py" --check --port 8000
echo.
echo From another PC test the port, not ping:  curl -m 5 http://SERVER-IP:8000/healthz
pause
