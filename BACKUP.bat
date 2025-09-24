@echo off
setlocal enabledelayedexpansion

:: === CONFIGURATION ===
set "BACKUP_DIR=backups"
set "MAX_BACKUPS=10"
set "LOG_FILE=%BACKUP_DIR%\backup_log.txt"

:: === PREP ===
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

:: Timestamp (YYYY-MM-DD_HHMMSS)
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (
    set "yyyy=%%c"
    set "mm=%%a"
    set "dd=%%b"
)
for /f "tokens=1-2 delims=:." %%a in ("%time%") do (
    set "hh=%%a"
    set "nn=%%b"
)
if %hh% lss 10 set hh=0%hh%
set "TS=%yyyy%-%mm%-%dd%_%hh%%nn%"

set "ZIPNAME=backup_%TS%.zip"
set "ZIPPATH=%BACKUP_DIR%\%ZIPNAME%"

echo ============================================
echo       Scary Story Bot - Backup Utility
echo ============================================
echo Creating backup: %ZIPNAME%
echo.

:: === BACKUP ===
powershell -NoLogo -NoProfile -Command ^
  "Get-ChildItem -Path . -Recurse -Exclude backups | ForEach-Object { $_.FullName } | Compress-Archive -DestinationPath '%ZIPPATH%' -Force -CompressionLevel Optimal"

if errorlevel 1 (
    echo [ERROR] Backup failed.
    pause
    exit /b 1
)

:: === INFO ===
for %%I in ("%ZIPPATH%") do set SIZE=%%~zI
set /a SIZE_MB=%SIZE%/1024/1024

echo Backup complete.
echo File: %ZIPNAME%
echo Size: %SIZE_MB% MB
echo.

:: === LOGGING ===
echo [%date% %time%] Created %ZIPNAME% (%SIZE_MB% MB) >> "%LOG_FILE%"

:: === ROTATE OLD BACKUPS ===
setlocal disableDelayedExpansion
set count=0
for /f "delims=" %%f in ('dir /b /o-d "%BACKUP_DIR%\backup_*.zip"') do (
    set /a count+=1
    if !count! gtr %MAX_BACKUPS% (
        echo Deleting old backup: %%f
        del "%BACKUP_DIR%\%%f"
    )
)
endlocal

echo Log updated: %LOG_FILE%
echo.
echo [DONE] Backup process finished.
pause
