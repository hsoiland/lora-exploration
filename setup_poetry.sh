#!/bin/bash

# Make script exit on first error
set -e

echo "Setting up Poetry environment for loras project..."

# Remove any existing Poetry lock file
if [ -f "poetry.lock" ]; then
    echo "Removing existing poetry.lock file..."
    rm poetry.lock
fi

# Install dependencies without the project itself
echo "Installing dependencies..."
poetry install --no-root

# Install optional GPU dependencies if CUDA is available 
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing xformers..."
    poetry install -E gpu --no-root
    
    # Find CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    echo "Detected CUDA version: $CUDA_VERSION"
    
    if [ "$CUDA_VERSION" -ge "11" ]; then
        # Get more specific CUDA version
        if [ "$CUDA_VERSION" -eq "11" ]; then
            echo "Installing xformers for CUDA 11.x..."
            poetry run pip install xformers --index-url https://download.pytorch.org/whl/cu118
        elif [ "$CUDA_VERSION" -ge "12" ]; then
            echo "Installing xformers for CUDA 12.x..."
            poetry run pip install xformers --index-url https://download.pytorch.org/whl/cu121
        fi
    else
        echo "CUDA version less than 11, xformers may not work properly."
    fi
else
    echo "CUDA not detected, skipping xformers installation."
fi

echo "Poetry setup complete!"
echo "To activate the environment, run: poetry shell"
echo "To run a script with the environment: poetry run python your_script.py" 