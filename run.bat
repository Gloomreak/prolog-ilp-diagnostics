@echo off
rem Starts the app: creates .venv on first run, installs dependencies, launches Streamlit.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python 3.11+ was not found. Install it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"

python -c "import streamlit, sklearn, pandas, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies ...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: dependency installation failed.
        pause
        exit /b 1
    )
)

where swipl >nul 2>nul
if errorlevel 1 (
    if not exist "C:\Program Files\swipl\bin\swipl.exe" (
        echo WARNING: swipl was not found in PATH. Install SWI-Prolog from
        echo          https://www.swi-prolog.org/Download.html and tick "Add swipl to the system PATH",
        echo          or set the path in the app sidebar / in .env ^(SWIPL_PATH^).
    )
)

streamlit run app.py
pause
