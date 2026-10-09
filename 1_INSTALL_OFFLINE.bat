@echo off
REM ===================================================================
REM  Offline install - no internet and no conda are needed on this PC.
REM
REM  What this does:
REM    1. unpacks the Python runtime in runtime\ into python\ next to this
REM       file (the official python.org package; nothing is registered in
REM       Windows, nothing outside this folder changes)
REM    2. installs the wheels in wheels\ into it, NASA PuMA included
REM    3. checks every import and runs the short self-test
REM
REM  Deleting the python\ folder undoes everything this script did.
REM  The wheels\ and runtime\ folders are made on a PC with internet by
REM      conda run -p .\env python tools\build_offline.py
REM  Pure ASCII on purpose: cmd.exe reads .bat files in the ANSI code page.
REM ===================================================================
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
echo ==================================================================
echo   MPSim (Material Property Simulation) - offline install (no conda)
echo ==================================================================
echo.

REM Some packages nest folders deeply; Windows limits a path to 260
REM characters unless long paths are switched on. A long folder name here
REM makes the installation fail with "file name too long" (WinError 206).
set "HERE=%~dp0"
if not "%HERE:~70,1%"=="" (
    echo   This folder has a long path:
    echo       %HERE%
    echo   Some packages nest folders deeply and Windows limits a path to 260
    echo   characters. If the installation stops with "file name too long",
    echo   move the folder to a short path such as C:\MPSim and run this again.
    echo.
)

if exist "python\python.exe" (
    echo   A python\ folder already exists here; it will be reused.
    echo   To rebuild it, delete the python folder and run this file again.
    goto :install
)

set "RT="
for %%F in ("runtime\python-3.11*-embed-amd64.zip") do set "RT=%%~fF"
if not defined RT (
    echo   runtime\python-3.11.x-embed-amd64.zip is missing.
    echo   Copy the complete folder made by tools\build_offline.py.
    pause
    exit /b 1
)
if not exist "wheels\pumapy-*.whl" (
    echo   wheels\ has no NASA PuMA wheel. Copy the complete folder made by
    echo   tools\build_offline.py.
    pause
    exit /b 1
)
echo   Unpacking %RT% ...
mkdir python
tar -xf "%RT%" -C python
if not exist "python\python.exe" (
    echo   The runtime could not be unpacked. Check that you may write here.
    pause
    exit /b 1
)
REM The path file of the runtime replaces the usual search path: it must
REM name the standard library, the packages and the program itself.
> "python\python311._pth" (
    echo python311.zip
    echo .
    echo Lib\site-packages
    echo ..\app
    echo import site
)
echo   Installing pip from wheels\ ...
REM a wheel is a zip archive; pip's is pure Python and is simply unpacked
set "PIPWHL="
for %%F in ("wheels\pip-*.whl") do set "PIPWHL=%%~fF"
mkdir "python\Lib\site-packages" 2>nul
tar -xf "%PIPWHL%" -C "python\Lib\site-packages"
"python\python.exe" -m pip --version
if errorlevel 1 (
    echo   pip could not be installed.
    pause
    exit /b 1
)

:install
echo.
echo   Installing the packages from wheels\ - nothing is downloaded ...
"python\python.exe" -m pip install --no-index --find-links "%~dp0wheels" --disable-pip-version-check --no-warn-script-location --constraint "%~dp0constraints-offline.txt" -r "%~dp0requirements-offline.txt"
if errorlevel 1 (
    echo.
    echo   The installation failed; the message above names the package.
    pause
    exit /b 1
)

REM openEMS (the full-wave simulation of the EMI analysis, on by default) is a program of its own:
REM unpacked next to the others, used from tools\openEMS
set "OEMSZIP="
for %%F in ("runtime\openEMS_x64_*.zip") do set "OEMSZIP=%%~fF"
if defined OEMSZIP if not exist "tools\openEMS\openEMS.exe" (
    echo.
    echo   Unpacking openEMS for the EMI full-wave simulation ...
    tar -xf "%OEMSZIP%" -C tools
)

echo.
echo   Checking that every part can be imported ...
"python\python.exe" "%~dp0tools\check_imports.py"
if errorlevel 1 (
    echo.
    echo   Something is still missing - the report above says what.
    pause
    exit /b 1
)

echo.
echo   Running the short self-test (known answers) ...
"python\python.exe" "%~dp0app\selftest.py" --quick
echo.
echo ==================================================================
echo   Done. Start the program with 2_RUN_LOCAL.bat (this PC)
echo   or 3_RUN_SERVER.bat (other PCs on the local network).
echo ==================================================================
pause
exit /b 0
