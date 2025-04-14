#!/usr/bin/env python3
"""Test script to check if we can import all necessary modules."""

import os
import sys
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing Rust module
print("\n=== Testing Rust Module ===")
try:
    import src.lora_ops
    print("✅ Rust module imported")
except ImportError as e:
    print(f"❌ Failed to import Rust module: {e}")
    print("ERROR: Rust implementation is required. PyTorch fallbacks are not allowed.")
    print("Please run 'cargo build --release' in the rust_lora directory.")
    sys.exit(1)

try:
    from src.lora_ops import LoraTrainingContext, AdamParams
    print("✅ Rust LORA module classes loaded")
except ImportError as e:
    print(f"❌ Failed to import Rust module classes: {e}")
    print("ERROR: Cannot continue without the Rust implementation.")
    sys.exit(1)

# Try importing diffusers components
print("\n=== Testing Diffusers ===")
try:
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
    print("✅ Diffusers components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import diffusers components: {e}")
    sys.exit(1)

# Try loading the SDXL model
print("\n=== Testing SDXL Loading ===")
try:
    from transformers import CLIPTokenizer
    print("Testing loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained('../sdxl-base-1.0/tokenizer', use_fast=False)
    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    sys.exit(1)

try:
    print("Testing loading UNet...")
    unet = UNet2DConditionModel.from_pretrained('../sdxl-base-1.0/unet')
    print("✅ UNet loaded successfully")
except Exception as e:
    print(f"❌ Error loading UNet: {e}")
    
    # Let's check if we can read the file directly
    try:
        print("\nAttempting to check UNet file...")
        import os
        from safetensors import safe_open
        unet_path = '../sdxl-base-1.0/unet/diffusion_pytorch_model.safetensors'
        print(f"File exists: {os.path.exists(unet_path)}")
        print(f"File size: {os.path.getsize(unet_path)} bytes")
        with safe_open(unet_path, framework="pt") as f:
            metadata = f.metadata()
            print(f"✅ SafeTensors file opened successfully. Metadata: {metadata}")
    except Exception as nested_e:
        print(f"❌ Error examining UNet file: {nested_e}")
    
    sys.exit(1)

print("\n=== Testing Complete ===")
print("✅ All required components are available. Ready to run training with Rust implementation.") 