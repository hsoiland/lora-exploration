#!/usr/bin/env python
import os
from transformers import CLIPTokenizer

print("Downloading CLIP tokenizers for SDXL...")

# Create output directories
base_dir = "sdxl-base-1.0"
os.makedirs(os.path.join(base_dir, "tokenizer"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "tokenizer_2"), exist_ok=True)

# Download both tokenizers
print("Downloading first tokenizer...")
tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    subfolder="tokenizer"
)
tokenizer.save_pretrained(os.path.join(base_dir, "tokenizer"))

print("Downloading second tokenizer...")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    subfolder="tokenizer_2"
)
tokenizer_2.save_pretrained(os.path.join(base_dir, "tokenizer_2"))

print("✅ Successfully downloaded both CLIP tokenizers")
print(f"  - First tokenizer saved to: {os.path.join(base_dir, 'tokenizer')}")
print(f"  - Second tokenizer saved to: {os.path.join(base_dir, 'tokenizer_2')}")
print("\nYou can now run your LoRA training with text conditioning.") 