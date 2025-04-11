#!/bin/bash

# Test Multi-LoRA Training and Inference
# This script runs minimal training and inference to verify multi-LoRA functionality
# Optimized for 12GB VRAM

echo "===== TESTING MULTI-LORA SETUP (VRAM-OPTIMIZED) ====="

# Create dummy data directory if it doesn't exist
mkdir -p dummy_data
if [ ! -f dummy_data/test_image_0.jpg ]; then
    echo "Creating dummy test images..."
    # Use Python to create dummy images at smaller 256px size
    python -c "
from PIL import Image
import numpy as np
for i in range(3):
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img.save(f'dummy_data/test_image_{i}.jpg')
print('Created 3 random test images')
"
fi

# Set up output directory
OUTPUT_DIR="multi_lora_test"
mkdir -p $OUTPUT_DIR

# Clear CUDA cache before starting
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Cleared CUDA cache')
"

# Step 1: Run minimal training with two LoRAs
echo -e "\n\n===== STEP 1: TRAINING TWO LORAS (MEMORY-OPTIMIZED) ====="
poetry run python test_multi_lora.py \
  --base_model "sdxl-base-1.0" \
  --images_dir "dummy_data" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 \
  --max_train_steps 5 \
  --image_size 256 \
  --rank 2 \
  --mixed_precision "fp16" \
  --gradient_checkpointing \
  --enable_xformers \
  --prompt "A portrait photo of person with beautiful features"

# Check if training completed and files were created
if [ -f "$OUTPUT_DIR/self_attn_lora_test.safetensors" ] && [ -f "$OUTPUT_DIR/cross_attn_lora_test.safetensors" ]; then
    echo -e "\n✅ Training completed successfully"
else
    echo -e "\n❌ Training failed - LoRA files not found"
    exit 1
fi

# Clear CUDA cache between steps
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Cleared CUDA cache')
"

# Step 2: Test inference with both LoRAs combined
echo -e "\n\n===== STEP 2: TESTING INFERENCE WITH COMBINED LORAS (MEMORY-OPTIMIZED) ====="
poetry run python test_multi_lora_inference.py \
  --base_model "sdxl-base-1.0" \
  --self_attn_lora "$OUTPUT_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora "$OUTPUT_DIR/cross_attn_lora_test.safetensors" \
  --output_dir "$OUTPUT_DIR" \
  --prompt "A portrait photo of a person with detailed features" \
  --image_size 256 \
  --enable_xformers \
  --enable_vae_slicing

# Check if test image was created
if [ -f "$OUTPUT_DIR/multi_lora_test_image.png" ]; then
    echo -e "\n✅ Inference test completed successfully"
    echo "Test images saved to $OUTPUT_DIR/"
else
    echo -e "\n❌ Inference test failed - output image not found"
    exit 1
fi

echo -e "\n===== ALL TESTS COMPLETED SUCCESSFULLY ====="
echo "The multi-LoRA setup works. You can now scale up to full training with both LoRAs."
echo "Memory optimization techniques are enabled for your 12GB VRAM GPU." 