@echo off
title NeuroScan AI - Brain Tumor Classifier
echo ========================================================
echo   Starting Brain Tumor Classification Web Application
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run: uv venv .venv --python 3.12 ^&^& uv pip install -r requirements.txt
    pause
    exit /b 1
)

echo Activating virtual environment and launching Streamlit...
echo The app will open automatically in your default browser.
echo Press Ctrl+C in this terminal window to stop the server.
echo.

.venv\Scripts\python.exe -m streamlit run app.py
pause
