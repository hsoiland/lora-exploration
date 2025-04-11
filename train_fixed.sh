#!/bin/bash
# Fixed training script for SDXL LoRA with proper text embeddings and dtype fixes

# Set environment variables (uncomment if needed)
# export CUDA_VISIBLE_DEVICES=0

# Training parameters
IMAGES_DIR="gina_face_cropped"  # Directory with training images
OUTPUT_DIR="fixed_lora_output_v3"  # Where to save the model
BASE_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
NUM_EPOCHS=8
BATCH_SIZE=1
MAX_STEPS=500
IMAGE_SIZE=512
LEARNING_RATE=1e-4
SAVE_STEPS=100

# LoRA parameters (conservative to avoid issues)
RANK=4
LORA_ALPHA=16
LORA_DROPOUT=0.05

echo "Starting SDXL LoRA training with fixed text embeddings and dtype handling..."
echo "Images directory: $IMAGES_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the training script
python train_sdxl_lora_simple.py \
  --base_model "$BASE_MODEL" \
  --images_dir "$IMAGES_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $NUM_EPOCHS \
  --train_batch_size $BATCH_SIZE \
  --max_train_steps $MAX_STEPS \
  --image_size $IMAGE_SIZE \
  --learning_rate $LEARNING_RATE \
  --save_steps $SAVE_STEPS \
  --rank $RANK \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --mixed_precision "fp16" \
  --gradient_checkpointing \
  --enable_xformers

echo "Training complete! Model saved to $OUTPUT_DIR" 