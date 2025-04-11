#!/bin/bash

# Run the LoRA image generator with the full path to SDXL

# Get the absolute path to the SDXL model
SDXL_PATH="$(pwd)/sdxl-base-1.0"
echo "Looking for SDXL model at: $SDXL_PATH"

# Check if model exists
if [ -d "$SDXL_PATH" ]; then
  echo "Found SDXL model at: $SDXL_PATH"
else
  echo "ERROR: SDXL model not found at: $SDXL_PATH"
  echo "Please make sure you have downloaded the SDXL model or provide the correct path."
  exit 1
fi

# Get the LoRA path
LORA_DIR="$(pwd)/gina_full_lora"
LORA_PATH="${LORA_DIR}/gina_full_lora.safetensors"

# Check if LoRA exists
if [ -f "$LORA_PATH" ]; then
  echo "Found LoRA weights at: $LORA_PATH"
else
  echo "Warning: LoRA weights not found at: $LORA_PATH"
  
  # Try to find any safetensors file in the LoRA directory
  if [ -d "$LORA_DIR" ]; then
    FOUND_LORA=$(find "$LORA_DIR" -name "*.safetensors" | head -1)
    if [ -n "$FOUND_LORA" ]; then
      echo "Found alternative LoRA weights at: $FOUND_LORA"
      LORA_PATH="$FOUND_LORA"
    fi
  fi
fi

# Create output directory if it doesn't exist
mkdir -p generated_images

echo "Running generator with model path: $SDXL_PATH"
echo "Using LoRA path: $LORA_PATH"

# Run the generator script
python generate_with_gina_lora.py \
  --base_model "$SDXL_PATH" \
  --lora_path "$LORA_PATH" \
  --num_images 1 