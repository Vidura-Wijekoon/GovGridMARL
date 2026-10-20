@echo off
REM GovGridMARL - one-click showcase (Windows)
REM Activates the project environment, replays an IEEE-14 evaluation day with the trained
REM MAPPO policy, rebuilds docs\status\govgridmarl_status.html and opens it in your browser.
REM Options:  Start_Show.bat --validate     (run the 14-check environment validation first)
REM           Start_Show.bat --train        (train a quick MAPPO if no checkpoint exists)
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  where conda >nul 2>nul && call conda activate govgridmarl
)
python --version >nul 2>nul || (echo Python not found. Create the venv first - see README "Installation". & pause & exit /b 1)
python -c "import torch, pettingzoo, hydra, pandapower, matplotlib" >nul 2>nul
if errorlevel 1 (
  echo.
  echo The Python environment is missing part of the stack (torch / pettingzoo / hydra / matplotlib).
  echo Installing from requirements-show.txt needs ~1.5 GB of free disk and a few minutes.
  set /p GO="Install now? [y/N] "
  if /i "%GO%"=="y" (
    python -m pip install --upgrade pip
    python -m pip install -r requirements-show.txt
    python -m pip install -e .
  ) else (
    echo Skipped. Install later with:  pip install -r requirements-show.txt ^&^& pip install -e .
    pause
    exit /b 1
  )
)
python scripts\showcase.py %*
set RC=%ERRORLEVEL%
if not %RC%==0 (echo. & echo Showcase exited with code %RC%. See messages above. & pause)
endlocal
