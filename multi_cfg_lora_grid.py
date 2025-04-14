#!/usr/bin/env python3
"""
Multi-CFG LoRA Grid Generator
Generate grids showing LoRA alpha values from 0 to 4 (0.1 increments) 
for multiple CFG scales (3, 5, 7, 9, 11, 13), using the same seed
"""

import os
import sys
import argparse
import torch
import gc
import numpy as np
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LoRA alpha grids for multiple CFG scales")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base model path or huggingface.co identifier")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="Path to LoRA model (.safetensors file)")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, 
                       default="worst quality, low quality, text, watermark, deformed",
                       help="Negative prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="multi_cfg_alpha_grids",
                       help="Directory to save generated images")
    parser.add_argument("--min_alpha", type=float, default=0.0,
                       help="Minimum LoRA alpha value")
    parser.add_argument("--max_alpha", type=float, default=4.0,
                       help="Maximum LoRA alpha value")
    parser.add_argument("--alpha_step", type=float, default=0.1,
                       help="Step size for alpha values")
    parser.add_argument("--cfg_scales", type=str, default="3,5,7,9,11,13",
                       help="Comma-separated list of CFG scales")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                       help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (same for all generations)")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Size of individual images in the grid")
    parser.add_argument("--grid_cols", type=int, default=10,
                       help="Number of columns in each grid")
    
    return parser.parse_args()

def clean_memory():
    """Clean up CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def create_alpha_grid(images, alpha_values, cfg_scale, args, cols=10):
    """Create a grid of images with alpha labels for a specific CFG scale"""
    # Calculate number of rows needed
    num_images = len(images)
    num_rows = math.ceil(num_images / cols)
    
    # Get size of individual images
    individual_width, individual_height = images[0].size
    
    # Add padding and space for labels
    padding = 10
    header_height = 40
    footer_height = 40
    label_width = 60
    
    # Calculate total grid size
    grid_width = (individual_width + padding) * cols + padding + label_width
    grid_height = header_height + (individual_height + padding) * num_rows + padding + footer_height
    
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
    title = f"LoRA Alpha Values Grid - CFG Scale: {cfg_scale}"
    title_w, title_h = draw.textsize(title, font=title_font) if hasattr(draw, 'textsize') else (0, 0)
    draw.text(((grid_width - title_w) // 2, padding), title, fill=(255, 255, 255), font=title_font)
    
    # Add prompt subtitle
    prompt_text = f"Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}"
    prompt_w, prompt_h = draw.textsize(prompt_text, font=small_font) if hasattr(draw, 'textsize') else (0, 0)
    draw.text(((grid_width - prompt_w) // 2, padding + title_h + 5), 
              prompt_text, fill=(200, 200, 200), font=small_font)
    
    # Place images in grid with alpha labels
    for i, (image, alpha) in enumerate(zip(images, alpha_values)):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = label_width + padding + col * (individual_width + padding)
        y = header_height + row * (individual_height + padding)
        
        # Add alpha label
        alpha_label = f"α: {alpha:.1f}"
        draw.text((x + 5, y + 5), alpha_label, fill=(255, 255, 0), font=small_font)
        
        # Paste the image
        grid.paste(image, (x, y))
    
    # Add footer info
    seed_info = f"Seed: {args.seed} | Steps: {args.num_inference_steps}"
    draw.text((padding + label_width, grid_height - footer_height + padding), 
              seed_info, fill=(200, 200, 200), font=small_font)
    
    return grid

def load_lora_model(pipe, lora_path, adapter_name="default"):
    """Load LoRA model handling both file and directory paths"""
    try:
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path)
            lora_file = os.path.basename(lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=lora_file, adapter_name=adapter_name)
        else:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        return True
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        return False

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse CFG scales
    cfg_scales = [float(scale) for scale in args.cfg_scales.split(',')]
    
    # Generate alpha values
    alpha_values = np.arange(args.min_alpha, args.max_alpha + args.alpha_step/2, args.alpha_step)
    alpha_values = [round(float(alpha), 1) for alpha in alpha_values]
    
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
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        print("Model CPU offload not available, using default settings")
    
    # Load LoRA model
    print(f"Loading LoRA model: {args.lora_model}")
    if not load_lora_model(pipe, args.lora_model):
        print("Failed to load LoRA model. Exiting.")
        sys.exit(1)
    
    # Generate grids for each CFG scale
    for cfg_scale in cfg_scales:
        print(f"\n=== Generating grid for CFG Scale: {cfg_scale} ===")
        print(f"Alpha values: {len(alpha_values)} values from {args.min_alpha} to {args.max_alpha}")
        
        # Store images for this CFG scale
        cfg_images = []
        
        # Generate images for each alpha value
        for alpha in alpha_values:
            print(f"Generating image for CFG={cfg_scale}, alpha={alpha:.1f}")
            
            # Set the LoRA adapter weight
            pipe.set_adapters(["default"], adapter_weights=[alpha])
            
            # Generate the image
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=cfg_scale,
                generator=generator,
                height=args.image_size,
                width=args.image_size,
            ).images[0]
            
            # Save individual image
            os.makedirs(os.path.join(args.output_dir, "individual"), exist_ok=True)
            img_filename = f"{timestamp}_cfg{cfg_scale}_alpha{alpha:.1f}.png"
            img_path = os.path.join(args.output_dir, "individual", img_filename)
            image.save(img_path)
            
            # Add to collection for grid
            cfg_images.append(image)
        
        # Create and save grid for this CFG scale
        print(f"Creating grid for CFG scale {cfg_scale}...")
        grid = create_alpha_grid(cfg_images, alpha_values, cfg_scale, args, cols=args.grid_cols)
        
        # Save the grid
        grid_filename = f"{timestamp}_cfg{cfg_scale}_alpha_grid.png"
        grid_path = os.path.join(args.output_dir, grid_filename)
        grid.save(grid_path)
        print(f"Saved grid to: {grid_path}")
        
        # Save metadata
        meta_filename = f"{timestamp}_cfg{cfg_scale}_alpha_grid.txt"
        meta_path = os.path.join(args.output_dir, meta_filename)
        with open(meta_path, "w") as f:
            f.write(f"CFG Scale: {cfg_scale}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Negative prompt: {args.negative_prompt}\n")
            f.write(f"LoRA model: {args.lora_model}\n")
            f.write(f"Alpha range: {args.min_alpha} to {args.max_alpha} (step: {args.alpha_step})\n")
            f.write(f"Steps: {args.num_inference_steps}\n")
            f.write(f"Base model: {args.base_model}\n")
    
    # Unload LoRA weights
    try:
        pipe.unload_lora_weights()
    except:
        print("Note: Could not unload LoRA weights (may be normal in some diffusers versions)")
    
    print("\nAll grids generated successfully!")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 