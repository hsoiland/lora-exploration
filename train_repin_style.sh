#!/bin/bash

# Train a multi-LoRA model using the Ilya Repin young women dataset
# Optimized for 12GB VRAM

echo "===== TRAINING ILYA REPIN STYLE LoRA (VRAM-OPTIMIZED) ====="

# Set up output directory
OUTPUT_DIR="repin_style_lora"
mkdir -p $OUTPUT_DIR

# Clear CUDA cache before starting
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Cleared CUDA cache')
"

# Train the LoRA models
echo -e "\n\n===== TRAINING SELF-ATTENTION AND CROSS-ATTENTION LORAs ====="
poetry run python test_multi_lora.py \
  --base_model "sdxl-base-1.0" \
  --images_dir "ilya_repin_young_women" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 5 \
  --max_train_steps 15 \
  --image_size 256 \
  --rank 4 \
  --mixed_precision "fp16" \
  --gradient_checkpointing \
  --enable_xformers \
  --prompt "A portrait of a beautiful young woman in the style of Ilya Repin, fine art, oil painting, classical art"

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

# Test inference with the trained LoRAs
echo -e "\n\n===== TESTING ILYA REPIN STYLE LORA INFERENCE ====="
poetry run python test_multi_lora_inference.py \
  --base_model "sdxl-base-1.0" \
  --self_attn_lora "$OUTPUT_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora "$OUTPUT_DIR/cross_attn_lora_test.safetensors" \
  --output_dir "$OUTPUT_DIR" \
  --prompt "A beautiful portrait of a young woman with intricate details, in the style of Ilya Repin, masterful, oil painting, fine art" \
  --image_size 512 \
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

echo -e "\n===== ILYA REPIN STYLE LORA TRAINING COMPLETE ====="
echo "Your Ilya Repin style LoRA is ready for use."
echo "The LoRA files are located in: $OUTPUT_DIR/"
echo "- Self-attention LoRA: $OUTPUT_DIR/self_attn_lora_test.safetensors"
echo "- Cross-attention LoRA: $OUTPUT_DIR/cross_attn_lora_test.safetensors" 