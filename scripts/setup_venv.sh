#!/bin/bash

# Setup script for existing virtual environment

# Check if virtual environment is active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "No active virtual environment detected."
    echo "Please activate your virtual environment first and then run this script."
    echo "Example: source your_venv/bin/activate"
    exit 1
fi

# Create directory structure if it doesn't exist
mkdir -p models
mkdir -p scripts

echo "Installing dependencies in the active virtual environment: ${VIRTUAL_ENV}"
pip install --upgrade pip
pip install numpy tensorflow pillow matplotlib

# Create requirements.txt file if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo "Creating requirements.txt file..."
    echo "numpy" > requirements.txt
    echo "tensorflow" >> requirements.txt
    echo "pillow" >> requirements.txt
    echo "matplotlib" >> requirements.txt
fi

echo "Setup complete! You can now run the application with: python digit_recognizer.py"