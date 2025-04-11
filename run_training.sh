#!/bin/bash

# Make script exit on first error
set -e

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Ensure we have the latest dependencies
if [ ! -f "poetry.lock" ]; then
    echo "Poetry lock file not found. Running setup script first..."
    bash setup_poetry.sh
fi

# Make scripts executable if they aren't already
chmod +x setup_poetry.sh

# Run the training script with Poetry
echo "Starting training with Poetry environment..."

# Command line arguments for test_multi_lora.py
# Uncomment and modify as needed
ARGS=""
# ARGS="--base_model=sdxl-base-1.0 --images_dir=your_image_dir --output_dir=output"

echo "Running: poetry run python test_multi_lora.py $ARGS"
poetry run python test_multi_lora.py $ARGS

echo "Training complete!" 