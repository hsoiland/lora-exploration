#!/bin/bash
set -e

# Define paths
LORA_OUTPUT_DIR="/workspace/volume/gina_lora_output"
REPO_DIR="/lora-exploration"
REPO_LORA_DIR="$REPO_DIR/trained_loras"
REPO_SAMPLES_DIR="$REPO_DIR/lora_samples"

# Create directories in the repository if they don't exist
mkdir -p "$REPO_LORA_DIR"
mkdir -p "$REPO_SAMPLES_DIR"

# Copy the LoRA weights
echo "Copying LoRA weights..."
cp "$LORA_OUTPUT_DIR"/gina_faces_lora.safetensors "$REPO_LORA_DIR"/

# Copy the sample images
echo "Copying sample images..."
cp "$LORA_OUTPUT_DIR"/trained_lora_test.png "$REPO_SAMPLES_DIR"/gina_faces_sample.png

# Copy checkpoints if needed (optional)
if [ -d "$LORA_OUTPUT_DIR/checkpoints" ]; then
    mkdir -p "$REPO_LORA_DIR/checkpoints"
    echo "Copying checkpoints..."
    cp "$LORA_OUTPUT_DIR"/checkpoints/* "$REPO_LORA_DIR/checkpoints/"
fi

# Copy any step checkpoints directly in the output directory
find "$LORA_OUTPUT_DIR" -name "*_step_*.safetensors" -exec cp {} "$REPO_LORA_DIR/" \;

echo "✅ Files successfully copied to repository"
echo "LoRA files located at: $REPO_LORA_DIR"
echo "Sample images located at: $REPO_SAMPLES_DIR"

# List the copied files
echo -e "\nLoRA weights:"
ls -lh "$REPO_LORA_DIR"/*.safetensors

echo -e "\nSample images:"
ls -lh "$REPO_SAMPLES_DIR"
