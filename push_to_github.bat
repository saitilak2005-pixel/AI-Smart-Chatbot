@echo off
title Push to GitHub
cd /d "%~dp0"
echo =====================================================================
echo   Pushing AI Smart Chatbot to GitHub: saitilak2005-pixel
echo =====================================================================
echo.
echo Connecting to https://github.com/saitilak2005-pixel/AI-Smart-Chatbot...
echo If a browser window pops up, click "Sign in with your browser" to authorize.
echo.
"C:\Users\SRINIVAS\anaconda3\Library\cmd\git.exe" push -u origin main --force
echo.
echo =====================================================================
echo   Complete! Refresh your GitHub page to see all your project files!
echo =====================================================================
pause
