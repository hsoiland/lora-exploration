#!/bin/bash

# Make script exit on first error
set -e

# Make the training script executable
chmod +x train_peft_lora.py

# Run the training with specific parameters
echo "Starting diffusers-based LoRA training for SDXL..."

# Run with Poetry if available
if command -v poetry &> /dev/null; then
    echo "Using Poetry for execution..."
    poetry run python train_peft_lora.py \
      --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
      --images_dir="gina_face_cropped" \
      --captions_file="gina_captions.json" \
      --output_dir="diffusers_lora_output" \
      --num_train_epochs=5 \
      --train_batch_size=1 \
      --max_train_steps=300 \
      --image_size=512 \
      --learning_rate=1e-5 \
      --rank=8 \
      --lora_alpha=32 \
      --save_steps=50 \
      --mixed_precision="fp16" \
      --gradient_checkpointing \
      --enable_xformers
else
    echo "Poetry not found, using regular Python..."
    python train_peft_lora.py \
      --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
      --images_dir="gina_face_cropped" \
      --captions_file="gina_captions.json" \
      --output_dir="diffusers_lora_output" \
      --num_train_epochs=5 \
      --train_batch_size=1 \
      --max_train_steps=300 \
      --image_size=512 \
      --learning_rate=1e-5 \
      --rank=8 \
      --lora_alpha=32 \
      --save_steps=50 \
      --mixed_precision="fp16" \
      --gradient_checkpointing \
      --enable_xformers
fi

echo "Training complete! Check the diffusers_lora_output directory for results." 