#!/bin/bash

# Make script exit on first error
set -e

# Make the inference script executable
chmod +x test_diffusers_lora_inference.py

# Run the inference
echo "Starting diffusers-based LoRA inference for SDXL..."

# Run with Poetry if available
if command -v poetry &> /dev/null; then
    echo "Using Poetry for execution..."
    poetry run python test_diffusers_lora_inference.py \
      --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
      --lora_dir="diffusers_lora_output/final_checkpoint" \
      --prompt="a photo of a beautiful gina with blue eyes, intricate, elegant, highly detailed" \
      --negative_prompt="deformed, ugly, disfigured, low quality, blurry" \
      --num_images=4 \
      --lora_scale=0.8 \
      --output_dir="diffusers_lora_outputs"
else
    echo "Poetry not found, using regular Python..."
    python test_diffusers_lora_inference.py \
      --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
      --lora_dir="diffusers_lora_output/final_checkpoint" \
      --prompt="a photo of a beautiful gina with blue eyes, intricate, elegant, highly detailed" \
      --negative_prompt="deformed, ugly, disfigured, low quality, blurry" \
      --num_images=4 \
      --lora_scale=0.8 \
      --output_dir="diffusers_lora_outputs"
fi

echo "Inference complete! Check the diffusers_lora_outputs directory for results." 