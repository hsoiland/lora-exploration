#!/usr/bin/env python3
"""
Generate a grid of images with different LoRA strength combinations.
Each image uses the same prompt and seed, but with different strength values
for two LoRAs, ranging from 0 to 1.8 in 0.2 increments.
"""

import os
import argparse
import torch
import gc
import numpy as np
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Generate grid of images with different LoRA strength combinations")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base model to use")
    parser.add_argument("--lora1_path", type=str, default="lora_ilya_repin/ilya_repin_young_women_lora.safetensors",
                       help="Path to first LoRA model")
    parser.add_argument("--lora2_path", type=str, default="gina_lora_output/gina_szanyel.safetensors",
                       help="Path to second LoRA model")
    parser.add_argument("--lora1_token", type=str, default="<ilya_repin>",
                       help="Trigger token for first LoRA")
    parser.add_argument("--lora2_token", type=str, default="<georgina_szayel>",
                       help="Trigger token for second LoRA")
    parser.add_argument("--lora1_name", type=str, default="LoRA 1",
                       help="Display name for first LoRA")
    parser.add_argument("--lora2_name", type=str, default="LoRA 2",
                       help="Display name for second LoRA")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, 
                       default="A beautiful portrait of a girl early 20s, with ginger hair <ilya_repin> <georgina_szayel>",
                       help="Prompt to use for all generations")
    parser.add_argument("--negative_prompt", type=str, default="deformed, ugly, disfigured, bad anatomy",
                       help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (same for all images)")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--output_dir", type=str, default="lora_strength_grid",
                       help="Output directory for generated images")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Output image size")
    parser.add_argument("--min_strength", type=float, default=0.0,
                       help="Minimum LoRA strength")
    parser.add_argument("--max_strength", type=float, default=1.8,
                       help="Maximum LoRA strength")
    parser.add_argument("--strength_increment", type=float, default=0.2,
                       help="LoRA strength increment")
    
    return parser.parse_args()

def load_lora_weights(pipe, state_dict, alpha=1.0):
    """Manually apply LoRA weights to the model"""
    visited = []
    
    # Handle our format which uses naming like 'module_name.lora_A.weight' and 'module_name.lora_B.weight'
    for key in state_dict:
        # Find all the LoRA A/B pairs
        if '.lora_A.' in key:
            module_name = key.split('.lora_A.')[0]
            b_key = key.replace('.lora_A.', '.lora_B.')
            
            if module_name in visited or b_key not in state_dict:
                continue
                
            visited.append(module_name)
            
            # Get the weights
            up_weight = state_dict[b_key]
            down_weight = state_dict[key]
            
            # Find the corresponding model module
            # Handle specific module types for UNet
            if 'unet' in module_name:
                # Convert from underscore to dot notation for accessing nested attributes
                model_path = module_name.replace('_', '.')
                
                # Get reference to the target module
                module = pipe.unet
                for attr in model_path.split('.')[1:]:  # Skip 'unet' prefix
                    if attr.isdigit():
                        module = module[int(attr)]
                    elif hasattr(module, attr):
                        module = getattr(module, attr)
                    else:
                        continue
                
                # If we found the target module, apply weights
                if hasattr(module, 'weight'):
                    weight = module.weight
                    
                    # Apply LoRA: Original + alpha * (up_weight @ down_weight)
                    delta = torch.mm(up_weight, down_weight)
                    weight.data += alpha * delta.to(weight.device, weight.dtype)
    
    print(f"Applied {len(visited)} LoRA modules with strength {alpha}")
    return pipe

def get_pipeline(args, device):
    """Load model and prepare pipeline with memory optimizations"""
    # Clear CUDA cache first
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load base model with memory optimizations
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    
    # Memory optimizations
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        
    return pipe

def create_strength_grid(images, strengths, args):
    """Create a grid of images showing different LoRA strength combinations"""
    # Calculate grid dimensions
    num_steps = int((args.max_strength - args.min_strength) / args.strength_increment) + 1
    
    # Calculate image dimensions
    img_width = images[0].width
    img_height = images[0].height
    
    # Add margins and headers
    margin = 5
    header_height = 40
    label_size = 30
    
    # Create grid image
    total_width = label_size + (img_width + margin) * num_steps + margin
    total_height = header_height + label_size + (img_height + margin) * num_steps + margin
    grid_image = Image.new('RGB', (total_width, total_height), color=(20, 20, 20))
    
    # Setup for drawing text
    draw = ImageDraw.Draw(grid_image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
    # Draw title
    title = f"LoRA Strength Grid - {args.lora1_name} vs {args.lora2_name} - Seed: {args.seed}"
    draw.rectangle((0, 0, total_width, header_height), fill=(40, 40, 40))
    draw.text((margin, margin), title, fill=(255, 255, 255), font=font)
    
    # Draw column headers (LoRA 1 strengths)
    for i in range(num_steps):
        strength = args.min_strength + i * args.strength_increment
        x = label_size + i * (img_width + margin) + img_width // 2
        y = header_height + margin // 2
        draw.text((x - 10, y), f"{strength:.1f}", fill=(255, 255, 255), font=small_font)
    
    # Draw row headers (LoRA 2 strengths)
    for i in range(num_steps):
        strength = args.min_strength + i * args.strength_increment
        x = margin
        y = header_height + label_size + i * (img_height + margin) + img_height // 2
        draw.text((x, y - 5), f"{strength:.1f}", fill=(255, 255, 255), font=small_font)
    
    # Draw axis labels
    draw.text((total_width // 2, header_height - 20), f"{args.lora1_name} Strength →", 
              fill=(255, 255, 255), font=font)
    draw.text((margin, total_height // 2), f"{args.lora2_name} Strength →", 
              fill=(255, 255, 255), font=font, anchor="mm")
    
    # Place images in grid
    for i, (image, (lora1_strength, lora2_strength)) in enumerate(zip(images, strengths)):
        # Calculate grid position
        col = int((lora1_strength - args.min_strength) / args.strength_increment)
        row = int((lora2_strength - args.min_strength) / args.strength_increment)
        
        # Calculate pixel position
        x = label_size + margin + col * (img_width + margin)
        y = header_height + label_size + margin + row * (img_height + margin)
        
        # Paste the image
        grid_image.paste(image, (x, y))
    
    return grid_image

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use the same prompt for all configurations
    full_prompt = args.prompt
    print(f"\nUsing prompt for all configurations: {full_prompt}")
    
    # Set up generator with seed
    seed = args.seed
    print(f"\nGenerating with seed: {seed}")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Calculate strength values
    num_steps = int((args.max_strength - args.min_strength) / args.strength_increment) + 1
    strengths = np.linspace(args.min_strength, args.max_strength, num_steps)
    
    # Store all generated images and their strength combinations
    all_images = []
    all_strengths = []
    
    # Load LoRA state dicts once
    print("Loading LoRA weights...")
    lora1_state_dict = load_file(args.lora1_path)
    lora2_state_dict = load_file(args.lora2_path)
    
    total_combinations = len(strengths) * len(strengths)
    current_combination = 1
    
    for lora1_strength in strengths:
        for lora2_strength in strengths:
            print(f"\n[{current_combination}/{total_combinations}] Generating with {args.lora1_name}: {lora1_strength:.1f}, {args.lora2_name}: {lora2_strength:.1f}")
            
            # Load fresh pipeline for each combination to avoid weight accumulation
            pipe = get_pipeline(args, device)
            
            try:
                # Apply LoRA weights if strength > 0
                if lora1_strength > 0:
                    pipe = load_lora_weights(pipe, lora1_state_dict, alpha=lora1_strength)
                
                if lora2_strength > 0:
                    pipe = load_lora_weights(pipe, lora2_state_dict, alpha=lora2_strength)
                
                # Generate image
                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    height=args.image_size,
                    width=args.image_size
                ).images[0]
                
                # Save individual image
                img_path = os.path.join(args.output_dir, f"lora1_{lora1_strength:.1f}_lora2_{lora2_strength:.1f}_seed_{seed}.png")
                image.save(img_path)
                
                all_images.append(image)
                all_strengths.append((lora1_strength, lora2_strength))
                
            except Exception as e:
                print(f"Error generating image: {e}")
                # Create a blank image with error text
                error_img = Image.new('RGB', (args.image_size, args.image_size), color=(30, 0, 0))
                draw = ImageDraw.Draw(error_img)
                draw.text((args.image_size//2, args.image_size//2), "ERROR", fill=(255, 255, 255), anchor="mm")
                all_images.append(error_img)
                all_strengths.append((lora1_strength, lora2_strength))
            
            # Free memory
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            
            current_combination += 1
    
    # Create and save the strength grid
    if len(all_images) > 0:
        try:
            print("\nCreating strength grid image...")
            grid = create_strength_grid(all_images, all_strengths, args)
            grid_path = os.path.join(args.output_dir, f"lora_strength_grid_seed_{seed}.png")
            grid.save(grid_path)
            print(f"Saved strength grid to {grid_path}")
        except Exception as e:
            print(f"Error creating strength grid: {e}")

if __name__ == "__main__":
    main() 