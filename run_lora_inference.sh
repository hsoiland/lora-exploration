#!/bin/bash

# Make script exit on first error
set -e

# Make the inference script executable
chmod +x test_lora_inference.py

# Check if PEFT is available for the PEFT loading option
PEFT_AVAILABLE=false
if python -c "import peft" &> /dev/null; then
    PEFT_AVAILABLE=true
    echo "PEFT library is available, will test both loading methods."
else
    echo "PEFT library not found, will only test diffusers loading method."
fi

# Define the LoRA directory to test
LORA_DIR="peft_lora_output/final_checkpoint"

# If the final checkpoint doesn't exist, check for the most recent checkpoint
if [ ! -d "$LORA_DIR" ]; then
    echo "Final checkpoint not found, looking for the most recent checkpoint..."
    LATEST_CHECKPOINT=$(ls -d peft_lora_output/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        LORA_DIR=$LATEST_CHECKPOINT
        echo "Using most recent checkpoint: $LORA_DIR"
    else
        echo "No checkpoints found. Please train a model first or specify a valid LoRA directory."
        exit 1
    fi
fi

# Generate some test prompts
PROMPTS=(
    "a beautiful portrait of a woman with blue eyes, intricate, elegant, highly detailed"
    "a woman with blue eyes in a forest, magical atmosphere, cinematic lighting"
    "close-up portrait of a woman, professional photography, award-winning"
    "woman with blue eyes in cyberpunk setting, neon lights, futuristic"
)

# Run inference with diffusers loading method
echo "Running inference using diffusers loading method..."
python test_lora_inference.py \
    --lora_dir="$LORA_DIR" \
    --prompt="${PROMPTS[0]}" \
    --num_images=1 \
    --lora_scale=0.8

# Run inference with PEFT loading method if available
if [ "$PEFT_AVAILABLE" = true ]; then
    echo "Running inference using PEFT loading method..."
    python test_lora_inference.py \
        --lora_dir="$LORA_DIR" \
        --use_peft \
        --prompt="${PROMPTS[1]}" \
        --num_images=1 \
        --lora_scale=0.8
fi

# Generate images with different prompts
echo "Generating variations with different prompts..."
python test_lora_inference.py \
    --lora_dir="$LORA_DIR" \
    --prompt="${PROMPTS[2]}" \
    --num_images=1 \
    --output_dir="lora_inference_outputs/variations"

python test_lora_inference.py \
    --lora_dir="$LORA_DIR" \
    --prompt="${PROMPTS[3]}" \
    --num_images=1 \
    --output_dir="lora_inference_outputs/variations"

echo "Inference complete! Check the lora_inference_outputs directory for results." 