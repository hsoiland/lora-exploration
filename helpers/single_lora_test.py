#!/usr/bin/env python3
"""
SDXL LoRA Testing Script
Tests a trained LoRA weight file with the SDXL base model, generating a grid
of images with different alpha weights and guidance scales
"""

import os
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
import datetime
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Test a SDXL LoRA model with a grid of alpha values and guidance scales")
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to base model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="Path to the LoRA model safetensors file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for image generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bad quality, worst quality, low quality, low resolution, blurry, text, watermark, logo, signature",
        help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_test_outputs",
        help="Directory to save the generated images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Individual image width (default lower to create grid)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Individual image height (default lower to create grid)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix for LoRA loading. If None, no prefix will be used."
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual images in addition to the grid"
    )
    
    return parser.parse_args()

def create_image_grid(images, rows, cols, labels=None):
    """Create a grid from a list of images with optional labels."""
    w, h = images[0].size
    grid_w, grid_h = cols * w, rows * h
    
    # Create a white background
    grid = Image.new('RGB', (grid_w, grid_h), color='white')
    
    # Add images to the grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        offset = (col * w, row * h)
        grid.paste(img, offset)
    
    # Add labels if provided
    if labels:
        draw = ImageDraw.Draw(grid)
        font = None
        try:
            # Try to use a system font, fallback to default if not available
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
            
        for i, label in enumerate(labels):
            row = i // cols
            col = i % cols
            offset = (col * w + 10, row * h + 10)
            draw.text(offset, label, fill=(255, 0, 0), font=font)
    
    return grid

def main():
    args = parse_args()
    
    print(f"Loading base model: {args.base_model}")
    
    # Initialize pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    # Move the pipeline to the GPU
    pipe = pipe.to("cuda")
    
    # Enable memory optimization
    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        print("Model CPU offload not available, using default settings")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set the random seed if specified
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(f"Using random seed: {seed}")
        args.seed = seed
    
    # Define alpha values and guidance scales for the grid
    alpha_values = np.arange(0, 4.2, 0.2).round(1).tolist()
    guidance_scales = [3, 7, 9, 11]
    
    # Calculate grid dimensions
    n_rows = len(guidance_scales)
    n_cols = len(alpha_values)
    
    print(f"Creating a grid with:")
    print(f"- Alpha values: {alpha_values}")
    print(f"- Guidance scales: {guidance_scales}")
    print(f"- Total images to generate: {n_rows * n_cols}")
    
    # Current timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_images = []
    all_labels = []
    
    # Generate images for each combination
    for guidance_index, guidance_scale in enumerate(guidance_scales):
        for alpha_index, alpha in enumerate(alpha_values):
            print(f"Generating image with guidance scale {guidance_scale} and alpha {alpha}")
            
            # Load the LoRA weights with specified alpha
            print(f"Loading LoRA weights from: {args.lora_model_path} with alpha {alpha}")
            pipe.load_lora_weights(
                os.path.dirname(args.lora_model_path), 
                weight_name=os.path.basename(args.lora_model_path),
                adapter_name="default",
                cross_attention_kwargs={"scale": alpha},
                prefix=args.prefix
            )
            
            # Set the LoRA scale - different versions of diffusers use different methods
            # This line was causing errors, so we're using cross_attention_kwargs in load_lora_weights instead
            # pipe.set_adapters_scale(alpha)
            
            # Generate image
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                generator=generator,
                guidance_scale=guidance_scale,
                width=args.width,
                height=args.height,
            ).images[0]
            
            # Create label for the image
            label = f"α:{alpha} CFG:{guidance_scale}"
            all_images.append(image)
            all_labels.append(label)
            
            if args.save_individual:
                # Save individual image
                filename = f"{timestamp}_alpha{alpha}_cfg{guidance_scale}.png"
                filepath = os.path.join(args.output_dir, filename)
                image.save(filepath)
                
                # Save metadata
                info_filepath = os.path.join(args.output_dir, f"{timestamp}_alpha{alpha}_cfg{guidance_scale}.txt")
                with open(info_filepath, "w") as f:
                    f.write(f"Prompt: {args.prompt}\n")
                    f.write(f"Negative prompt: {args.negative_prompt}\n")
                    f.write(f"Seed: {args.seed}\n")
                    f.write(f"Steps: {args.steps}\n")
                    f.write(f"Guidance scale: {guidance_scale}\n")
                    f.write(f"Alpha: {alpha}\n")
                    f.write(f"LoRA model: {args.lora_model_path}\n")
                    f.write(f"Base model: {args.base_model}\n")
                
                print(f"Saved individual image to {filepath}")
            
            # Unload LoRA weights before loading with a different alpha
            pipe.unload_lora_weights()
    
    # Create the grid
    grid = create_image_grid(all_images, n_rows, n_cols, all_labels)
    
    # Save the grid
    grid_filename = f"{timestamp}_alpha_cfg_grid.png"
    grid_filepath = os.path.join(args.output_dir, grid_filename)
    grid.save(grid_filepath)
    
    # Save grid metadata
    grid_info_filepath = os.path.join(args.output_dir, f"{timestamp}_alpha_cfg_grid.txt")
    with open(grid_info_filepath, "w") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Negative prompt: {args.negative_prompt}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Guidance scales: {guidance_scales}\n")
        f.write(f"Alpha values: {alpha_values}\n")
        f.write(f"LoRA model: {args.lora_model_path}\n")
        f.write(f"Base model: {args.base_model}\n")
    
    print(f"Successfully generated grid image with {len(all_images)} variations in {args.output_dir}")
    print(f"Grid saved to: {grid_filepath}")
    
if __name__ == "__main__":
    main() 