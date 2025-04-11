#!/bin/bash

# Make script exit on first error
set -e

# Run with Poetry
echo "Running multi-LoRA inference with text embeddings..."

# Make the script executable
chmod +x test_multi_lora_inference.py

# Run the inference with your specific dataset
poetry run python test_multi_lora_inference.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --self_attn_lora="multi_lora_test/self_attn_lora_test.safetensors" \
  --cross_attn_lora="multi_lora_test/cross_attn_lora_test.safetensors" \
  --output_dir="multi_lora_outputs" \
  --prompt="A beautiful portrait of Gina with detailed features, professional lighting" \
  --self_attn_weight=0.8 \
  --cross_attn_weight=0.8 \
  --image_size=512

echo "Inference complete! Check the multi_lora_outputs directory for results." 