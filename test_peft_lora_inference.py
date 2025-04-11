#!/usr/bin/env python3
"""
Test script for SDXL LoRA inference
"""

import os
import torch
import argparse
from pathlib import Path
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler
)
import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Test SDXL LoRA inference")
    parser.add_argument("--lora_dir", type=str, default="fixed_lora_output_v3/final_checkpoint")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default="a photo of a beautiful gina with blue eyes, intricate, elegant, highly detailed")
    parser.add_argument("--negative_prompt", type=str, default="deformed, ugly, disfigured, low quality, blurry")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--lora_scale", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="lora_test_outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    print(f"Using random seed: {args.seed}")
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    
    # Create scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.base_model, 
        subfolder="scheduler"
    )
    
    # Load pipeline with mixed precision
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    
    # Enable xformers if available
    if device == "cuda":
        try:
            import xformers
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except ImportError:
            print("xFormers not available, using default attention mechanism")
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {args.lora_dir}")
    
    if os.path.exists(args.lora_dir):
        try:
            pipeline.load_lora_weights(args.lora_dir)
            print("LoRA weights loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            print("⚠️ Continuing with base model only.")
    else:
        print(f"⚠️ LoRA directory {args.lora_dir} not found, continuing with base model only")
    
    # Generate images
    print(f"Generating {args.num_images} images with prompt: {args.prompt}")
    
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}...")
        
        # Set a different seed for each image but keep it deterministic
        image_seed = args.seed + i
        generator = torch.Generator(device=device).manual_seed(image_seed)
        
        # Generate image
        image = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            guidance_scale=args.guidance_scale,
            cross_attention_kwargs={"scale": args.lora_scale}
        ).images[0]
        
        # Save image
        output_path = os.path.join(args.output_dir, f"lora_test_{i+1}_seed_{image_seed}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")
    
    print("✅ Generation complete!")
    print(f"Generated {args.num_images} images in {args.output_dir}")

if __name__ == "__main__":
    main() 