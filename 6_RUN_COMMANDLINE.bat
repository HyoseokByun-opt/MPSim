@echo off
REM  Runs a bundled preset without the browser, e.g.
REM      6_RUN_COMMANDLINE.bat --preset tim_aln
REM      6_RUN_COMMANDLINE.bat --list
call "%~dp0_common.bat" || (pause & exit /b 1)
if "%~1"=="" (
    %MPY% "%~dp0app\run_cli.py" --list
    echo.
    echo Usage: 6_RUN_COMMANDLINE.bat --preset PRESET_ID [--quality fast]
) else (
    %MPY% "%~dp0app\run_cli.py" %*
)
pause
