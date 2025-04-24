#!/bin/bash

# Setup script for new environment

# Create directory structure if it doesn't exist
mkdir -p models
mkdir -p scripts

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 could not be found. Please install Python 3 before proceeding."
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux or macOS
    source venv/bin/activate
fi

echo "Installing dependencies..."
pip install --upgrade pip
pip install numpy tensorflow pillow matplotlib

# Create requirements.txt file
echo "numpy" > requirements.txt
echo "tensorflow" >> requirements.txt
echo "pillow" >> requirements.txt
echo "matplotlib" >> requirements.txt

echo "Setup complete! Activate the virtual environment with:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "    venv\\Scripts\\activate"
else
    echo "    source venv/bin/activate"
fi
echo "Then run the application with: python digit_recognizer.py"