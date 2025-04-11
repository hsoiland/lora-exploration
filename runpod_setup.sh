#!/bin/bash
set -e

echo "🔧 Setting up LoRA Rust Optimizer on RunPod..."

# Install system dependencies
apt-get update
apt-get install -y curl build-essential pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Build Rust extension
echo "🦀 Building Rust extension..."
poetry run maturin develop

# Download SDXL base model if not present
if [ ! -d "sdxl-base-1.0" ]; then
    echo "📥 Downloading SDXL base model..."
    poetry run python -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype='float16', use_safetensors=True, variant='fp16').save_pretrained('sdxl-base-1.0')"
    echo "✅ SDXL base model downloaded"
fi

echo "✅ Setup complete! Ready to run training." 