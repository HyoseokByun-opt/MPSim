@echo off
REM ===================================================================
REM  Finds conda and the environment this bundle runs in.
REM  Sets  MPY = the command that runs Python inside that environment.
REM
REM  Order: a python\ folder next to this file (1_INSTALL_OFFLINE.bat),
REM  an env\ folder next to this file (made by 1_INSTALL.bat),
REM  then a named environment given in MPSIM_ENV, then "mpsim", "rvesim", "puma".
REM  Pure ASCII on purpose: cmd.exe reads .bat files in the ANSI code page.
REM ===================================================================
set "MPSIM_ROOT=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM  The offline installation (1_INSTALL_OFFLINE.bat) comes first: a python
REM  folder next to this file with every package inside, conda not needed.
if exist "%MPSIM_ROOT%python\python.exe" (
    set "MPY="%MPSIM_ROOT%python\python.exe""
    exit /b 0
)

set "CONDA_EXE_FOUND="
where conda >nul 2>nul && set "CONDA_EXE_FOUND=conda"
if not defined CONDA_EXE_FOUND if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE_FOUND=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE_FOUND if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE_FOUND=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE_FOUND if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_EXE_FOUND=%ProgramData%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE_FOUND if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_EXE_FOUND=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE_FOUND (
    echo.
    echo   conda was not found. Install Miniconda first:
    echo       https://docs.conda.io/en/latest/miniconda.html
    echo   NASA PuMA is only available from conda-forge.
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist "%MPSIM_ROOT%env\python.exe" (
    set "MPY="%CONDA_EXE_FOUND%" run -p "%MPSIM_ROOT%env" --no-capture-output python"
    exit /b 0
)
if defined MPSIM_ENV (
    set "MPY="%CONDA_EXE_FOUND%" run -n %MPSIM_ENV% --no-capture-output python"
    exit /b 0
)
for %%E in (mpsim rvesim puma) do (
    "%CONDA_EXE_FOUND%" env list | findstr /R /C:"^%%E " >nul 2>nul && (
        set "MPY="%CONDA_EXE_FOUND%" run -n %%E --no-capture-output python"
        exit /b 0
    )
)
echo.
echo   No environment found. Run 1_INSTALL.bat (internet) or
echo   1_INSTALL_OFFLINE.bat (no internet, no conda) first.
exit /b 1
