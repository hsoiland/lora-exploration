#!/usr/bin/env python3
"""
Simple script to test LoRA models
"""

import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import tempfile
from safetensors.torch import load_file, save_file
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA models")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base model to use")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to LoRA model")
    parser.add_argument("--token", type=str, default="",
                       help="Trigger token for the LoRA")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="a portrait of a woman",
                       help="Base prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="deformed, ugly, disfigured",
                       help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--lora_scale", type=float, default=0.7,
                       help="Scale of LoRA adapters")
    parser.add_argument("--output_dir", type=str, default="test_images",
                       help="Output directory for generated images")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    ).to(device)
    
    # Use DPM++ scheduler for faster and better results
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Optimize for faster inference
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        
    # Set seed
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Generate without LoRA first
    print("Generating without LoRA...")
    base_prompt = args.prompt
    
    base_image = pipe(
        prompt=base_prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).images[0]
    
    base_path = os.path.join(args.output_dir, "base_image.png")
    base_image.save(base_path)
    print(f"Saved base image to {base_path}")
    
    # Now generate with LoRA
    print(f"Loading LoRA from {args.lora_path}")
    
    # Prepare full prompt with token
    full_prompt = f"{base_prompt} {args.token}".strip()
    
    # Try two different loading methods
    try:
        # Method 1: Try direct loading
        try:
            print("Attempting to load LoRA directly...")
            pipe.load_lora_weights(args.lora_path)
            success = True
        except Exception as e:
            print(f"Direct loading failed: {e}")
            success = False
            
        if not success:
            # Method 2: Try with prefix=None
            print("Attempting to load with prefix=None...")
            pipe.load_lora_weights(args.lora_path, prefix=None)
            
        pipe.fuse_lora(lora_scale=args.lora_scale)
        
        print(f"Generating with LoRA using prompt: {full_prompt}")
        lora_image = pipe(
            prompt=full_prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        
        lora_path = os.path.join(args.output_dir, "lora_image.png")
        lora_image.save(lora_path)
        print(f"Saved LoRA image to {lora_path}")
        
    except Exception as e:
        print(f"Error using LoRA: {e}")
    
    print("Generation complete!")

if __name__ == "__main__":
    main() 