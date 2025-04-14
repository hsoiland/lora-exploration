#!/bin/bash

# Multi-LoRA Training Script for H100
# Trains both the EMO style and Gina portrait LoRAs sequentially

echo "===== MULTI-LORA TRAINING - H100 OPTIMIZED ====="

# Set up directories
EMO_LORA_DIR="emo_style_lora"
GINA_LORA_DIR="gina_portrait_lora"
EMO_INFERENCE_DIR="emo_inference_results"
GINA_INFERENCE_DIR="gina_inference_results"
COMBINED_DIR="combined_inference_results"

mkdir -p $EMO_LORA_DIR
mkdir -p $GINA_LORA_DIR
mkdir -p $EMO_INFERENCE_DIR
mkdir -p $GINA_INFERENCE_DIR
mkdir -p $COMBINED_DIR

# Memory cleanup function
cleanup_memory() {
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Cleared CUDA cache')
"
}

# Run initial memory cleanup
cleanup_memory

# Activate the Python virtual environment
# Use the workspace venv instead of ~/loras/venv
if [ -f "/workspace/venv/bin/activate" ]; then
  source /workspace/venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# ======= EMO STYLE LORA TRAINING =======

# Directory containing EMO style training images
EMO_DATASET_DIR="/workspace/emo_dataset"
# Captions file for EMO dataset (will be automatically detected if exists)
EMO_CAPTIONS_FILE="${EMO_DATASET_DIR}/captions.json"

# Output directory for trained EMO style model
EMO_OUTPUT_DIR="/workspace/emo_lora_model"

# Set model name (for the output file)
EMO_MODEL_NAME="emo_style_v1"

# Set the concept name for metadata
EMO_LORA_CONCEPT="emo_style"

echo "=========================================="
echo "Starting LoRA training for EMO style"
echo "Using dataset: $EMO_DATASET_DIR"
if [ -f "$EMO_CAPTIONS_FILE" ]; then
  echo "Using captions file: $EMO_CAPTIONS_FILE"
else
  echo "No captions file found, will use filenames as captions"
fi
echo "=========================================="

# Training parameters optimized for H100 GPU
python train_sdxl_lora_simple_h100.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --images_dir="${EMO_DATASET_DIR}" \
  --captions_file="${EMO_CAPTIONS_FILE}" \
  --output_dir="${EMO_OUTPUT_DIR}" \
  --model_name="${EMO_MODEL_NAME}" \
  --lora_concept="${EMO_LORA_CONCEPT}" \
  --num_train_epochs=100 \
  --train_batch_size=2 \
  --max_train_steps=1000 \
  --image_size=1024 \
  --learning_rate=1e-4 \
  --save_steps=100 \
  --rank=32 \
  --lora_alpha=32 \
  --lora_dropout=0.05 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --use_pytorch_sdpa

echo "=========================================="
echo "EMO style LoRA training complete!"
echo "Model saved to: $EMO_OUTPUT_DIR/$EMO_MODEL_NAME.safetensors"
echo "==========================================="

# ======= GINA PORTRAIT LORA TRAINING =======

# Directory containing Gina portrait training images
GINA_DATASET_DIR="/workspace/gina_dataset_cropped"
# Specify the BLIP2 captions file for Gina dataset
GINA_CAPTIONS_FILE="/workspace/gina_dataset_cropped/captions.json"

# Output directory for trained Gina portrait model
GINA_OUTPUT_DIR="/workspace/gina_lora_model"

# Set model name (for the output file)
GINA_MODEL_NAME="gina_portrait_v1"

# Set the concept name for metadata
GINA_LORA_CONCEPT="gina_portrait"

echo "=========================================="
echo "Starting LoRA training for Gina portrait"
echo "Using dataset: $GINA_DATASET_DIR"
if [ -f "$GINA_CAPTIONS_FILE" ]; then
  echo "Using captions file: $GINA_CAPTIONS_FILE"
else
  echo "WARNING: BLIP2 captions file not found, will use filenames as captions"
fi
echo "=========================================="

# Training parameters optimized for H100 GPU
python train_sdxl_lora_simple_h100.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --images_dir="/workspace/gina_dataset_cropped" \
  --captions_file="/workspace/gina_dataset_cropped/captions.json" \
  --output_dir="/workspace/gina_lora_model" \
  --model_name="gina_portrait_v1" \
  --num_train_epochs=20 \
  --max_train_steps=2000 \
  --image_size=1024 \
  --learning_rate=1e-4 \
  --save_steps=100 \
  --rank=32 \
  --lora_alpha=32 \
  --lora_dropout=0.05 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --use_pytorch_sdpa

echo "=========================================="
echo "Gina portrait LoRA training complete!"
echo "Model saved to: $GINA_OUTPUT_DIR/$GINA_MODEL_NAME.safetensors"
echo "==========================================="

# ======= INFERENCE TESTING =======

# Create a directory for inference outputs
INFERENCE_DIR="/workspace/inference_outputs"
mkdir -p $INFERENCE_DIR

echo "=========================================="
echo "Creating inference script for testing"
echo "=========================================="

# Generate an inference test script
cat > test_loras.py << 'EOL'
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.utils import make_image_grid
import os

# Create output directory
output_dir = "/workspace/inference_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load VAE and base model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

print("Loading base model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

# Enable efficient attention with SDPA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
pipe.enable_model_cpu_offload()

# Test prompts for each LoRA
prompts = {
    "emo_style": [
        "a scene in emo style, dark colors, dramatic lighting",
        "emo style portrait of a young woman with dark makeup",
    ],
    "gina_portrait": [
        "portrait of gina, detailed face, natural lighting",
        "closeup of gina, professional photography, studio setting",
    ]
}

lora_models = {
    "emo_style": "/workspace/emo_lora_model/emo_style_v1.safetensors",
    "gina_portrait": "/workspace/gina_lora_model/gina_portrait_v1.safetensors"
}

# Generate images with each LoRA
for lora_name, lora_path in lora_models.items():
    print(f"Testing {lora_name} LoRA...")
    
    # Load LoRA weights
    pipe.load_lora_weights(lora_path)
    
    # Generate images with this LoRA
    for i, prompt in enumerate(prompts[lora_name]):
        print(f"Generating image for prompt: {prompt}")
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        
        # Save the image
        output_path = os.path.join(output_dir, f"{lora_name}_{i+1}.png")
        image.save(output_path)
        print(f"Saved to {output_path}")
    
    # Unload LoRA weights before loading the next one
    pipe.unload_lora_weights()

print("All test images generated successfully!")
EOL

echo "=========================================="
echo "Running inference tests"
echo "=========================================="

# Run the inference test script
python test_loras.py

echo "=========================================="
echo "All LoRA training and testing completed successfully!"
echo "Output models:"
echo "- EMO style: $EMO_OUTPUT_DIR/$EMO_MODEL_NAME.safetensors"
echo "- Gina portrait: $GINA_OUTPUT_DIR/$GINA_MODEL_NAME.safetensors"
echo "Test images available in: $INFERENCE_DIR"
echo "==========================================" 