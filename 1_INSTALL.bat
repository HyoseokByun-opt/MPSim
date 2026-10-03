@echo off
REM ===================================================================
REM  Creates a private conda environment in .\env and checks it.
REM  Nothing outside this folder is changed.
REM  Needs internet (conda-forge) the first time; about 2-3 GB.
REM ===================================================================
cd /d "%~dp0"
echo ==================================================================
echo   MPSim (Material Property Simulation) - install
echo ==================================================================
call "%~dp0_common.bat" >nul 2>nul
if not defined CONDA_EXE_FOUND (
    call "%~dp0_common.bat"
    pause
    exit /b 1
)
if exist "env\python.exe" (
    echo An env\ folder already exists here; reusing it.
    echo To rebuild it, delete the env folder and run this again.
) else (
    echo Creating env\ from environment.yml ...
    "%CONDA_EXE_FOUND%" env create -p "%~dp0env" -f "%~dp0environment.yml"
    if errorlevel 1 (
        echo.
        echo   The environment could not be created.
        echo   Check the internet connection and that conda-forge is reachable.
        pause
        exit /b 1
    )
)
call "%~dp0_common.bat" || (pause & exit /b 1)
echo.
echo Running the self-test (known answers) ...
%MPY% "%~dp0app\selftest.py" --quick
echo.
echo Done. Start the program with 2_RUN_LOCAL.bat or 3_RUN_SERVER.bat
pause
