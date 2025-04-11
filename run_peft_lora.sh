#!/bin/bash

# Make script exit on first error
set -e

# Make the training script executable
chmod +x train_lora_sdxl.py

# Check and install PEFT if it's not installed
if ! python -c "import peft" &> /dev/null; then
    echo "PEFT library not found, installing..."
    pip install peft
    echo "PEFT installed successfully!"
fi

# Run the training with specific parameters
echo "Starting PEFT LoRA training for SDXL..."

# Run with Poetry if available
if command -v poetry &> /dev/null; then
    echo "Using Poetry for execution..."
    # Make sure PEFT is installed in the poetry environment
    poetry add peft
    
    poetry run python train_lora_sdxl.py \
      --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
      --images_dir="gina_face_cropped" \
      --captions_file="gina_captions.json" \
      --output_dir="peft_lora_output" \
      --num_train_epochs=10 \
      --train_batch_size=1 \
      --max_train_steps=500 \
      --image_size=512 \
      --learning_rate=5e-7 \
      --rank=4 \
      --lora_alpha=32 \
      --save_steps=50 \
      --mixed_precision="fp16" \
      --gradient_checkpointing \
      --enable_xformers
else
    echo "Poetry not found, using regular Python..."
    python train_lora_sdxl.py \
      --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
      --images_dir="gina_face_cropped" \
      --captions_file="gina_captions.json" \
      --output_dir="peft_lora_output" \
      --num_train_epochs=10 \
      --train_batch_size=1 \
      --max_train_steps=500 \
      --image_size=512 \
      --learning_rate=5e-7 \
      --rank=4 \
      --lora_alpha=32 \
      --save_steps=50 \
      --mixed_precision="fp16" \
      --gradient_checkpointing \
      --enable_xformers
fi

echo "Training complete! Check the peft_lora_output directory for results."

# Make the test script executable and run an inference test
chmod +x test_peft_lora_inference.py
echo "Running a test inference with the trained model..."

if command -v poetry &> /dev/null; then
    poetry run python test_peft_lora_inference.py \
      --lora_dir="peft_lora_output/final_checkpoint" \
      --num_images=1
else
    python test_peft_lora_inference.py \
      --lora_dir="peft_lora_output/final_checkpoint" \
      --num_images=1
fi

echo "Inference test complete! Check the peft_lora_outputs directory for results." 