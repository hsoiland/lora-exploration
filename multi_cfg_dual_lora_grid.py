#!/usr/bin/env python3
"""
Dual LoRA Grid Generator
Generate a 10x10 grid showing combinations of two LoRAs with different alpha values
LoRA 1 alpha values on Y-axis and LoRA 2 alpha values on X-axis, both ranging from 0.5 to 4.0
"""

import os
import argparse
import torch
import gc
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 10x10 grid of Dual LoRA combinations")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base model path or huggingface.co identifier")
    parser.add_argument("--lora1_model", type=str, required=True,
                       help="Path to first LoRA model (.safetensors file) - Y-axis")
    parser.add_argument("--lora2_model", type=str, required=True,
                       help="Path to second LoRA model (.safetensors file) - X-axis")
    parser.add_argument("--lora1_name", type=str, default="LoRA 1",
                       help="Display name for first LoRA (Y-axis)")
    parser.add_argument("--lora2_name", type=str, default="LoRA 2",
                       help="Display name for second LoRA (X-axis)")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, 
                       default="worst quality, low quality, text, watermark, deformed",
                       help="Negative prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="dual_lora_grids",
                       help="Directory to save generated images")
    parser.add_argument("--min_alpha", type=float, default=0.5,
                       help="Minimum LoRA alpha value")
    parser.add_argument("--max_alpha", type=float, default=4.0,
                       help="Maximum LoRA alpha value")
    parser.add_argument("--cfg_scale", type=float, default=10.0,
                       help="CFG scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=18,
                       help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (same for all generations)")
    parser.add_argument("--image_size", type=int, default=1024,
                       help="Size of individual images in the grid")
    
    return parser.parse_args()

def clean_memory():
    """Clean up CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def create_grid(images, lora1_values, lora2_values, args):
    """Create a 10x10 grid of images with LoRA1 on Y-axis and LoRA2 on X-axis"""
    # Get size of individual images
    individual_width, individual_height = images[0][0].size
    
    # Add padding and space for labels
    padding = 10
    header_height = 100
    footer_height = 40
    side_label_width = 60
    top_label_height = 30
    
    # Calculate total grid size
    grid_width = side_label_width + padding + (individual_width + padding) * 10 + padding
    grid_height = header_height + top_label_height + padding + (individual_height + padding) * 10 + padding + footer_height
    
    # Create a dark background
    grid = Image.new('RGB', (grid_width, grid_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)
    
    # Try to load a nice font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("Arial", 24)
        font = ImageFont.truetype("Arial", 14)
        small_font = ImageFont.truetype("Arial", 12)
    except:
        try:
            title_font = ImageFont.truetype("DejaVuSans.ttf", 24)
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
    # Add title
    title = f"Dual LoRA Grid - CFG Scale: {args.cfg_scale}"
    title_w, title_h = draw.textsize(title, font=title_font) if hasattr(draw, 'textsize') else (0, 0)
    draw.text(((grid_width - title_w) // 2, padding), title, fill=(255, 255, 255), font=title_font)
    
    # Add LoRA names subtitle
    lora_text = f"Y-axis: {args.lora1_name} | X-axis: {args.lora2_name}"
    lora_w, lora_h = draw.textsize(lora_text, font=font) if hasattr(draw, 'textsize') else (0, 0)
    draw.text(((grid_width - lora_w) // 2, padding + title_h + 5), 
              lora_text, fill=(200, 255, 200), font=font)
    
    # Add prompt subtitle
    prompt_text = f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}"
    prompt_w, prompt_h = draw.textsize(prompt_text, font=small_font) if hasattr(draw, 'textsize') else (0, 0)
    draw.text(((grid_width - prompt_w) // 2, padding + title_h + lora_h + 10), 
              prompt_text, fill=(200, 200, 200), font=small_font)
    
    # Add X-axis labels (LoRA 2 alpha values)
    for x, alpha2 in enumerate(lora2_values):
        x_pos = side_label_width + padding + (individual_width + padding) * x + individual_width // 2
        draw.text((x_pos - 15, header_height + padding // 2), 
                 f"α2: {alpha2:.1f}", fill=(255, 200, 0), font=small_font)
    
    # Add Y-axis labels (LoRA 1 alpha values)
    for y, alpha1 in enumerate(lora1_values):
        y_pos = header_height + top_label_height + padding + (individual_height + padding) * y + individual_height // 2
        draw.text((padding + 5, y_pos - 5), 
                 f"α1: {alpha1:.1f}", fill=(0, 255, 200), font=small_font)
    
    # Place images in grid
    for y, alpha1 in enumerate(lora1_values):
        for x, alpha2 in enumerate(lora2_values):
            # Calculate position
            x_pos = side_label_width + padding + x * (individual_width + padding)
            y_pos = header_height + top_label_height + padding + y * (individual_height + padding)
            
            # Paste the image
            grid.paste(images[y][x], (x_pos, y_pos))
    
    # Add footer info
    seed_info = f"Seed: {args.seed} | Steps: {args.num_inference_steps}"
    draw.text((padding + side_label_width, grid_height - footer_height + padding), 
              seed_info, fill=(200, 200, 200), font=small_font)
    
    return grid

def load_lora_model(pipe, lora_path, adapter_name):
    """Load LoRA model handling both file and directory paths"""
    try:
        print(f"Loading LoRA model as {adapter_name}: {lora_path}")
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path)
            lora_file = os.path.basename(lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=lora_file)
        else:
            pipe.load_lora_weights(lora_path)
        
        print(f"LoRA model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        return False

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "individual"), exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate 10 alpha values from min_alpha to max_alpha
    lora1_values = np.linspace(args.min_alpha, args.max_alpha, 10)
    lora2_values = np.linspace(args.min_alpha, args.max_alpha, 10)
    
    # Round to one decimal place
    lora1_values = [round(float(alpha), 1) for alpha in lora1_values]
    lora2_values = [round(float(alpha), 1) for alpha in lora2_values]
    
    print(f"LoRA 1 alpha values (Y-axis): {lora1_values}")
    print(f"LoRA 2 alpha values (X-axis): {lora2_values}")
    
    # Set seed
    if args.seed is None:
        args.seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using seed: {args.seed}")
    
    # Initialize generator with seed
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Get current timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean memory before loading model
    clean_memory()
    
    # Initialize pipeline
    print(f"Loading base model: {args.base_model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    
    # Move pipeline to the GPU
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    
    # Load first LoRA model initially
    print(f"Loading first LoRA model ({args.lora1_name}): {args.lora1_model}")
    if not load_lora_model(pipe, args.lora1_model, "lora1"):
        print("Failed to load first LoRA model. Exiting.")
        return
    
    # Store images for the grid (10x10)
    grid_images = [[None for _ in range(10)] for _ in range(10)]
    
    # Generate all combinations
    for y, alpha1 in enumerate(lora1_values):
        for x, alpha2 in enumerate(lora2_values):
            print(f"Generating image for {args.lora1_name}={alpha1:.1f}, {args.lora2_name}={alpha2:.1f}")
            
            # Set the scales for each LoRA
            pipe.fuse_lora(scale=alpha1)
            
            # Load and fuse second LoRA
            if load_lora_model(pipe, args.lora2_model, "lora2"):
                pipe.fuse_lora(scale=alpha2)
            else:
                print("Failed to load second LoRA model for this combination")
                continue
            
            # Generate the image
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                height=args.image_size,
                width=args.image_size,
            ).images[0]
            
            # Save individual image
            img_filename = f"{timestamp}_lora1_{alpha1:.1f}_lora2_{alpha2:.1f}.png"
            img_path = os.path.join(args.output_dir, "individual", img_filename)
            image.save(img_path)
            
            # Add to grid collection
            grid_images[y][x] = image
            
            # Unfuse LoRAs and reload first one for next iteration
            pipe.unfuse_lora()
            if not load_lora_model(pipe, args.lora1_model, "lora1"):
                print("Failed to reload first LoRA model. Exiting.")
                return
    
    # Create and save the grid
    print("Creating grid...")
    grid = create_grid(grid_images, lora1_values, lora2_values, args)
    
    # Save the grid
    grid_filename = f"{timestamp}_dual_lora_grid.png"
    grid_path = os.path.join(args.output_dir, grid_filename)
    grid.save(grid_path)
    print(f"Saved grid to: {grid_path}")
    
    # Save metadata
    meta_filename = f"{timestamp}_dual_lora_grid.txt"
    meta_path = os.path.join(args.output_dir, meta_filename)
    with open(meta_path, "w") as f:
        f.write(f"CFG Scale: {args.cfg_scale}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Negative prompt: {args.negative_prompt}\n")
        f.write(f"LoRA 1 (Y-axis): {args.lora1_model} ({args.lora1_name})\n")
        f.write(f"LoRA 2 (X-axis): {args.lora2_model} ({args.lora2_name})\n")
        f.write(f"Alpha range: {args.min_alpha} to {args.max_alpha}\n")
        f.write(f"Steps: {args.num_inference_steps}\n")
        f.write(f"Base model: {args.base_model}\n")
    
    print("\nGrid generated successfully!")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 