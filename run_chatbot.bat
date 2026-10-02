@echo off
title AI Smart Chatbot Runner
echo ========================================================
echo   Starting AI Smart Chatbot (Streamlit + Cloudflare)
echo ========================================================
echo.

cd /d "%~dp0"

echo 1. Starting Streamlit server...
start /b "" "C:\Users\SRINIVAS\anaconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false

timeout /t 3 /nobreak >nul

echo.
echo 2. Starting Public Cloudflare Tunnel...
echo Look for your public URL below (ending in .trycloudflare.com):
echo --------------------------------------------------------
.\cloudflared.exe tunnel --url http://localhost:8501

pause
