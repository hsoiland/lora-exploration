#!/bin/bash
set -e

echo "🔧 Setting up LoRA Rust Optimizer..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Build Rust extension
echo "🦀 Building Rust extension..."
poetry run maturin develop

echo "✅ Setup complete! You can now run the example:"
echo "poetry run python examples/test_lora.py" 