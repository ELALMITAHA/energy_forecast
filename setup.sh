#!/bin/bash
# setup.sh
# Streamlit Cloud setup script
# Installs only the minimal requirements for the Streamlit app

echo "=== Installing Streamlit app requirements ==="

# Update pip first
python -m pip install --upgrade pip

# Install only the minimal Streamlit requirements
pip install -r streamlit_requirements.txt

echo "=== Streamlit setup completed ==="