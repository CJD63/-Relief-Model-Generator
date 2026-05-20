@echo off
REM 3D Relief Model Generator Launcher
REM This batch file launches the Streamlit app with optimal settings

cd /d %~dp0

echo ================================================
echo    3D Relief Model Generator
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit is not installed
    echo Install it with: pip install streamlit
    echo.
    pause
    exit /b 1
)

REM Check if port 8501 is already in use
netstat -ano | findstr :8501 >nul 2>&1
if not errorlevel 1 (
    echo WARNING: Port 8501 is already in use
    echo Another Streamlit instance may be running.
    echo Open http://localhost:8501 in your browser.
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit app...
echo Open your browser and navigate to:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

REM Launch Streamlit with appropriate settings
REM --server.headless false: Allow browser to open

python -m streamlit run app.py --server.headless false

pause