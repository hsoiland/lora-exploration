#!/bin/bash

# Make script exit on first error
set -e

# Make the training script executable
chmod +x train_gina_multi_lora.py

# Run the training with specific parameters
echo "Starting multi-LoRA training for Gina dataset with text embeddings..."

# Run with Poetry
poetry run python train_gina_multi_lora.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --images_dir="gina_face_cropped" \
  --captions_file="gina_captions.json" \
  --output_dir="gina_multi_lora" \
  --num_train_epochs=5 \
  --train_batch_size=1 \
  --max_train_steps=300 \
  --image_size=512 \
  --learning_rate=5e-5 \
  --rank=8 \
  --save_steps=50 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --enable_xformers

echo "Training complete! Check the gina_multi_lora directory for results." 