@echo off
setlocal enabledelayedexpansion
title Build Scary Story Bot EXE
echo ============================================
echo       Building ScaryStoryBot.exe
echo ============================================

:: Change into the script directory
cd /d "%~dp0"

:: Clean old build/dist folders
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist launcher.spec del launcher.spec

:: Run PyInstaller
pyinstaller --onefile launcher.py

:: Prepare log file
set "LOG_FILE=build_log.txt"

:: Check if exe was built successfully
if exist dist\launcher.exe (
    ren dist\launcher.exe ScaryStoryBot.exe
    copy /Y dist\ScaryStoryBot.exe . >nul

    :: Get size in MB
    for %%I in ("ScaryStoryBot.exe") do set SIZE=%%~zI
    set /a SIZE_MB=%SIZE%/1024/1024

    echo.
    echo Build complete: dist\ScaryStoryBot.exe
    echo Also copied to project root: .\ScaryStoryBot.exe
    echo Size: %SIZE_MB% MB

    :: Log success
    echo [%date% %time%] SUCCESS - ScaryStoryBot.exe (%SIZE_MB% MB) >> "%LOG_FILE%"
) else (
    echo.
    echo [ERROR] Build failed. Check PyInstaller output.

    :: Log failure
    echo [%date% %time%] FAILED - Build error >> "%LOG_FILE%"
)

echo.
echo Log updated: %LOG_FILE%
pause
