#!/bin/bash

# Run the LoRA training with SafeVAE to avoid the fp16/fp32 type issues

# Set output directory
OUTPUT_DIR="gina_safe_lora"
mkdir -p "$OUTPUT_DIR"

# Path to SDXL model
SDXL_PATH="sdxl-base-1.0"
if [ ! -d "$SDXL_PATH" ]; then
  echo "Warning: SDXL model not found at $SDXL_PATH. Make sure the path is correct."
fi

# Path to images
IMAGES_DIR="gina_face_cropped"
if [ ! -d "$IMAGES_DIR" ]; then
  echo "Error: Images directory not found at $IMAGES_DIR"
  exit 1
fi

# Training settings
BATCH_SIZE=4  # Can be higher with safer approach
EPOCHS=50
LORA_RANK=8
STEPS_PER_SAVE=50

echo "==============================================="
echo "    STARTING SAFE LORA TRAINING"
echo "==============================================="
echo "Images directory: $IMAGES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "SDXL model path:  $SDXL_PATH"
echo "Batch size:       $BATCH_SIZE"
echo "Epochs:           $EPOCHS"
echo "LoRA rank:        $LORA_RANK"
echo "==============================================="

# Run the training
python train_safe_vae.py \
  --base_model "$SDXL_PATH" \
  --images_dir "$IMAGES_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --lora_name "gina_safe_lora" \
  --train_batch_size "$BATCH_SIZE" \
  --num_train_epochs "$EPOCHS" \
  --rank "$LORA_RANK" \
  --save_steps "$STEPS_PER_SAVE" \
  --mixed_precision "no" \
  --learning_rate 5e-5

# Make the script executable
# chmod +x run_safe_training.sh 