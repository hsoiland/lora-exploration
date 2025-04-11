#!/bin/bash

# Make script exit on first error
set -e

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Make the inference script executable
chmod +x test_lora_inference.py

# Run the inference script with Poetry
echo "Starting LoRA inference test..."

# Generate multiple samples with different prompts
poetry run python test_lora_inference.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --self_lora_path="multi_lora_test/self_attn_lora_test.safetensors" \
  --cross_lora_path="multi_lora_test/cross_attn_lora_test.safetensors" \
  --prompt="A beautiful portrait of a person with glowing eyes, detailed features, professional lighting" \
  --output_dir="lora_outputs" \
  --num_samples=1 \
  --steps=30

# Generate a second sample with a different prompt
poetry run python test_lora_inference.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --self_lora_path="multi_lora_test/self_attn_lora_test.safetensors" \
  --cross_lora_path="multi_lora_test/cross_attn_lora_test.safetensors" \
  --prompt="A dramatic portrait of a person in a futuristic setting, cinematic lighting" \
  --output_dir="lora_outputs" \
  --num_samples=1 \
  --steps=30 \
  --seed=1234

echo "Inference complete! Check the lora_outputs directory for results." 