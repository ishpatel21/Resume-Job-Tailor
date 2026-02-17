#!/bin/bash
# Resume Job Tailor - Quick Start Script
# Runs the Streamlit app from root directory

cd "$(dirname "$0")"
./.venv/bin/streamlit run src/app_mvp.py
