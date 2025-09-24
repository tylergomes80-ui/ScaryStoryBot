@echo off
title Scary Story Bot - SUPREME RUNNER
color 0A
cls

:menu
echo ============================================================
echo         ███████╗ ██████╗  █████╗ ██████╗ ██╗   ██╗
echo         ██╔════╝██╔═══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
echo         ███████╗██║   ██║███████║██████╔╝ ╚████╔╝ 
echo         ╚════██║██║   ██║██╔══██║██╔═══╝   ╚██╔╝  
echo         ███████║╚██████╔╝██║  ██║██║        ██║   
echo         ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝        ╚═╝   
echo ============================================================
echo               SCARY STORY BOT - STARTUP MENU
echo ============================================================
echo   [1] Run Bot (main.py)
echo   [2] Download Background Videos
echo   [3] Manage Sounds (jumpscares/ambience)
echo   [4] Show Bot Status
echo   [5] Open Output Folder
echo   [U] Push Local Changes to GitHub
echo   [P] Pull Latest Changes from GitHub
echo   [X] Exit
echo ============================================================

set /p choice="Choose an option: "

if "%choice%"=="1" goto runbot
if "%choice%"=="2" goto download
if "%choice%"=="3" goto sounds
if "%choice%"=="4" goto status
if "%choice%"=="5" goto folder
if /I "%choice%"=="U" goto gitpush
if /I "%choice%"=="P" goto gitpull
if /I "%choice%"=="X" exit

goto menu

:runbot
cls
echo [INFO] Starting Bot with heartbeat...
python main.py 2>&1 | powershell -Command ^
  "$input | ForEach-Object { Write-Host ('['+(Get-Date -Format 'HH:mm:ss')+'] ' + $_) }"
pause
goto menu

:download
cls
echo [INFO] Running video downloader...
python download_videos.py
pause
goto menu

:sounds
cls
echo [INFO] Opening sounds folder...
explorer sounds
pause
goto menu

:status
cls
echo [INFO] Bot status check...
tasklist | findstr /I "python"
pause
goto menu

:folder
cls
echo [INFO] Opening output folder...
explorer output
pause
goto menu

:gitpush
cls
echo [GIT] Adding and pushing local changes to GitHub...
git add .
git commit -m "Auto commit from START.bat"
git push -u origin main
pause
goto menu

:gitpull
cls
echo [GIT] Pulling latest changes from GitHub...
git pull origin main
pause
goto menu
