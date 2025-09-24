@echo off
title Scary Story Bot - AUTO UPDATER
cls
echo ============================================
echo        SCARY STORY BOT - UPDATER
echo ============================================

:: === Config ===
set REPO_URL=https://raw.githubusercontent.com/YOURNAME/ScaryStoryBot/main
set FILES=main.py download_videos.py download_sounds.py START.bat DEPLOY_ALL.bat
set BACKUP_DIR=backups

:: Ensure backup dir exists
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

echo [INFO] Backing up current scripts...
for %%F in (%FILES%) do (
    if exist "%%F" (
        copy /Y "%%F" "%BACKUP_DIR%\%%F_!date:~10,4!-!date:~4,2!-!date:~7,2!_!time:~0,2!-!time:~3,2!.bak" >nul
    )
)

echo [INFO] Downloading latest versions...
for %%F in (%FILES%) do (
    echo - %%F
    powershell -Command "(New-Object Net.WebClient).DownloadFile('%REPO_URL%/%%F','%%F')" || echo [ERROR] Failed: %%F
)

echo.
echo [DONE] Update complete. Restart your bot with START.bat
pause
