@echo off
REM  Proves the engine on this PC against problems whose answers are known.
REM  About 3-6 minutes. Add --quick for the short set.
call "%~dp0_common.bat" || (pause & exit /b 1)
%MPY% "%~dp0app\selftest.py" %*
pause
