#!/usr/bin/env bash
# Linux/macOS launcher (same as run.bat).
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -c "import streamlit, sklearn, pandas, matplotlib" 2>/dev/null || pip install -r requirements.txt
command -v swipl >/dev/null || echo "WARNING: swipl not found in PATH (install swi-prolog)"
exec streamlit run app.py
