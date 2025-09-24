@echo off
title Sound Downloader Bot Pro++
echo ============================================
echo    Sound Downloader Bot Pro++
echo ============================================

:: Always use Python 3.11 for consistency
"C:\Users\tyler\AppData\Local\Programs\Python\Python311\python.exe" "%~dp0download_sounds.py" %*

echo.
pause
