#!/bin/bash

# Make script exit on first error
set -e

# Make the training script executable
chmod +x train_sdxl_lora_simple.py

# Check and install PEFT if it's not installed
if ! python -c "import peft" &> /dev/null; then
    echo "PEFT library not found, installing..."
    pip install peft
    echo "PEFT installed successfully!"
fi

# Check if fixed SDXL VAE is downloadable
if ! python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype='auto')" &> /dev/null; then
    echo "Installing diffusers (needed for fixed SDXL VAE)..."
    pip install --upgrade diffusers transformers accelerate
    echo "Dependencies installed successfully!"
fi

# Run the training with specific parameters
echo "Starting simplified SDXL LoRA training..."

python train_sdxl_lora_simple.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --images_dir="gina_face_cropped" \
  --captions_file="gina_captions.json" \
  --output_dir="simple_lora_output" \
  --num_train_epochs=1 \
  --train_batch_size=1 \
  --max_train_steps=10 \
  --image_size=512 \
  --learning_rate=5e-7 \
  --rank=4 \
  --lora_alpha=32 \
  --save_steps=5 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --enable_xformers

echo "Training complete! Check the simple_lora_output directory for results."

# Run a simple test with the trained model
if [ -d "simple_lora_output/final_checkpoint" ]; then
    echo "Running inference test with trained model..."
    python test_peft_lora_inference.py \
      --lora_dir="simple_lora_output/final_checkpoint" \
      --num_images=1
fi 