#!/bin/bash

# Make script exit on first error
set -e

# Make the validation script executable
chmod +x test_lora_weights.py

# Run the validation with Poetry
echo "Running LoRA validation..."
poetry run python test_lora_weights.py 