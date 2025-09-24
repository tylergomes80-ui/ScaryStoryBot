@echo off
title Git Auto Fix + Push
echo ============================================
echo   ScaryStoryBot - Git Auto Fix + Push
echo ============================================

REM Go to project folder
cd /d C:\Users\tyler\RedditVideoBot

REM Step 1: Remove stale lock if exists
if exist .git\index.lock (
    echo [INFO] Removing stale Git lock...
    del /f /q .git\index.lock
)

REM Step 2: Ensure branch is main
for /f "tokens=*" %%i in ('git branch --show-current') do set CURBRANCH=%%i
if not "%CURBRANCH%"=="main" (
    echo [INFO] Renaming branch to main...
    git branch -M main
)

REM Step 3: Add all changes
echo [INFO] Staging all changes...
git add .

REM Step 4: Commit (with timestamp message)
set NOW=%date% %time%
git commit -m "Auto-commit on %NOW%"

REM Step 5: Push to GitHub
echo [INFO] Pushing to GitHub...
git push -u origin main

echo ============================================
echo [DONE] Repo synced with GitHub!
echo ============================================
pause
