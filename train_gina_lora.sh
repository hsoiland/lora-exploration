#!/bin/bash

# Train a multi-LoRA model using the Gina face dataset
# Optimized for 12GB VRAM

echo "===== TRAINING GINA FACE LORA (VRAM-OPTIMIZED) ====="

# Set up output directory
OUTPUT_DIR="gina_lora"
mkdir -p $OUTPUT_DIR

# Generate a simple captions file if it doesn't exist
CAPTIONS_FILE="gina_face_cropped/captions.json"
if [ ! -f "$CAPTIONS_FILE" ]; then
    echo "Creating simple captions file..."
    python -c "
import os
import json

image_files = [f for f in os.listdir('gina_face_cropped') if f.endswith('.jpg')]
captions = {}
for img in image_files:
    captions[img] = 'A portrait photograph of Gina, a woman with red hair'

with open('gina_face_cropped/captions.json', 'w') as f:
    json.dump(captions, f, indent=2)
print(f'Created captions for {len(captions)} images')
"
fi

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
  --images_dir "gina_face_cropped" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 5 \
  --max_train_steps 15 \
  --image_size 256 \
  --rank 4 \
  --mixed_precision "fp16" \
  --gradient_checkpointing \
  --enable_xformers \
  --prompt "A photograph of Gina, a woman with red hair and beautiful features"

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
echo -e "\n\n===== TESTING GINA FACE LORA INFERENCE ====="
poetry run python test_multi_lora_inference.py \
  --base_model "sdxl-base-1.0" \
  --self_attn_lora "$OUTPUT_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora "$OUTPUT_DIR/cross_attn_lora_test.safetensors" \
  --output_dir "$OUTPUT_DIR" \
  --prompt "A stunning portrait of Gina, a beautiful woman with red hair in dramatic lighting" \
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

echo -e "\n===== GINA FACE LORA TRAINING COMPLETE ====="
echo "Your Gina face LoRA is ready for use."
echo "The LoRA files are located in: $OUTPUT_DIR/"
echo "- Self-attention LoRA: $OUTPUT_DIR/self_attn_lora_test.safetensors"
echo "- Cross-attention LoRA: $OUTPUT_DIR/cross_attn_lora_test.safetensors"

# Add a trigger word to a text file for reference
echo "Use the trigger words \"Gina\" in your prompts to activate this LoRA" > "$OUTPUT_DIR/trigger_words.txt" 