@echo off
title Super Background Downloader Pro++
echo ============================================
echo    Super Background Downloader Pro++
echo ============================================

:: Always use Python 3.11 for consistency
"C:\Users\tyler\AppData\Local\Programs\Python\Python311\python.exe" "%~dp0download_videos.py" %*

echo.
pause
