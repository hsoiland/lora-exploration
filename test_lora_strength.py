#!/usr/bin/env python3
"""
Test LoRA at different strength values
"""

import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA at different strengths")
    
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
    parser.add_argument("--num_steps", type=int, default=25,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--min_strength", type=float, default=0.5,
                       help="Minimum LoRA strength")
    parser.add_argument("--max_strength", type=float, default=1.5,
                       help="Maximum LoRA strength")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of strength samples to test")
    parser.add_argument("--output_dir", type=str, default="lora_strength_test",
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
    
    # Prepare strength values to test
    if args.num_samples == 1:
        strengths = [args.min_strength]
    else:
        strengths = [args.min_strength + i * (args.max_strength - args.min_strength) / (args.num_samples - 1) 
                    for i in range(args.num_samples)]
    
    # Prepare full prompt with token
    full_prompt = f"{args.prompt} {args.token}".strip()
    
    # Load the LoRA weights
    try:
        # Always use prefix=None for our custom format
        print(f"Loading LoRA from {args.lora_path} with prefix=None")
        pipe.load_lora_weights(args.lora_path, adapter_name="lora", prefix=None)
        print("LoRA loaded successfully")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return
    
    # Generate base image (without LoRA)
    print("Generating base image without LoRA...")
    base_image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).images[0]
    
    base_path = os.path.join(args.output_dir, "base_no_lora.png")
    base_image.save(base_path)
    print(f"Saved base image to {base_path}")
    
    # Create grid of images for different strengths
    grid_width = 512 * (args.num_samples + 1)  # +1 for base image
    grid_height = 512
    grid_image = Image.new('RGB', (grid_width, grid_height))
    grid_image.paste(base_image.resize((512, 512)), (0, 0))
    
    # Generate images with different strengths
    for i, strength in enumerate(strengths):
        print(f"Generating with strength {strength:.2f}...")
        
        # Fuse the LoRA with specified strength
        pipe.fuse_lora(lora_scale=strength)
        
        # Generate image
        lora_image = pipe(
            prompt=full_prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        
        # Save individual image
        lora_path = os.path.join(args.output_dir, f"strength_{strength:.2f}.png")
        lora_image.save(lora_path)
        
        # Add to grid
        grid_image.paste(lora_image.resize((512, 512)), ((i+1)*512, 0))
        
        # Unfuse for next iteration
        pipe.unfuse_lora()
    
    # Add labels to the grid
    draw = ImageDraw.Draw(grid_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Add text with darker background for readability
    def add_text_with_bg(pos, text):
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        x, y = pos
        
        # Draw background rectangle
        draw.rectangle(
            (x, y, x + text_width + 10, y + text_height + 5),
            fill=(0, 0, 0, 128)
        )
        
        # Draw text
        draw.text((x + 5, y + 2), text, fill=(255, 255, 255), font=font)
    
    # Add labels
    add_text_with_bg((10, grid_height - 30), "Base (No LoRA)")
    
    for i, strength in enumerate(strengths):
        add_text_with_bg(((i+1)*512 + 10, grid_height - 30), f"Strength: {strength:.2f}")
    
    # Add prompt text at top
    add_text_with_bg((10, 10), f"Prompt: {full_prompt}")
    
    # Save grid image
    grid_path = os.path.join(args.output_dir, f"strength_comparison_{args.seed}.png")
    grid_image.save(grid_path)
    print(f"Saved comparison grid to {grid_path}")

if __name__ == "__main__":
    main() 