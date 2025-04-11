#!/usr/bin/env python
import os
import torch
from diffusers import StableDiffusionXLPipeline

print("Downloading essential SDXL components for LoRA training...")

# Create output directory
model_dir = "sdxl-base-1.0"
os.makedirs(model_dir, exist_ok=True)

# Download the model - using diffusers ensures all configs come with it
print(f"Downloading to {os.path.abspath(model_dir)}...")

# Use the built-in library to handle downloads properly
try:
    # This will download all required files while skipping already downloaded ones
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Save the entire pipeline to the directory
    pipeline.save_pretrained(model_dir)
    
    print(f"✅ Successfully downloaded SDXL model to {os.path.abspath(model_dir)}")
    print("You can now run the training with your LoRA script!")
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("Try again or download manually from https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0") 