@echo off
setlocal ENABLEDELAYEDEXPANSION
title Reddit Video Bot - GOD MODE
color 0c

:: ----------------------------
:: Check Dependencies
:: ----------------------------
echo Checking dependencies...

where python >nul 2>&1
if errorlevel 1 (
    color 0e
    echo [WARN] Python not found in PATH!
    echo Install Python 3.10+ and add it to PATH.
    echo https://www.python.org/downloads/
    pause
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    color 0e
    echo [WARN] FFmpeg not found in PATH!
    echo Install FFmpeg and add /bin to PATH.
    echo https://ffmpeg.org/download.html
    pause
)

where yt-dlp >nul 2>&1
if errorlevel 1 (
    color 0e
    echo [WARN] yt-dlp not found in PATH!
    echo Install with:  pip install yt-dlp
    pause
)

color 0c

:: ----------------------------
:: Main Menu
:: ----------------------------
:mainmenu
cls
echo ============================================
echo       REDDIT VIDEO BOT - GOD MODE
echo ============================================
echo [1] Run Bot (main.py)
echo [2] Download Background Videos
echo [3] Download Sounds
echo [Q] Quit
echo ============================================
set /p choice="Enter choice: "

if "%choice%"=="1" goto runbot
if "%choice%"=="2" goto dlbg
if "%choice%"=="3" goto dlsounds
if /i "%choice%"=="q" goto end

echo Invalid choice. Try again.
pause
goto mainmenu

:: ----------------------------
:: Run Bot
:: ----------------------------
:runbot
cls
echo [RUN] Starting main.py...
python main.py
if errorlevel 1 (
    color 0e
    echo [ERROR] main.py crashed or Python not found.
)
echo --------------------------------------------
echo Bot finished. Returning to menu...
pause
goto mainmenu

:: ----------------------------
:: Background Downloader
:: ----------------------------
:dlbg
cls
echo [RUN] Starting Background Downloader...
python download_videos.py
if errorlevel 1 (
    color 0e
    echo [ERROR] Background downloader crashed.
)
echo --------------------------------------------
echo Background Downloader finished. Returning...
pause
goto mainmenu

:: ----------------------------
:: Sound Downloader
:: ----------------------------
:dlsounds
cls
echo [RUN] Starting Sound Downloader...
python download_sounds.py
if errorlevel 1 (
    color 0e
    echo [ERROR] Sound downloader crashed.
)
echo --------------------------------------------
echo Sound Downloader finished. Returning...
pause
goto mainmenu

:: ----------------------------
:: Exit
:: ----------------------------
:end
color 07
echo Exiting GOD MODE... bye!
pause
exit
