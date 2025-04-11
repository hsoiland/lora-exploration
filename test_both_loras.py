#!/usr/bin/env python3
"""
Test both LoRAs at different strength combinations
"""

import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Test both LoRAs at different strengths")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base model to use")
    parser.add_argument("--lora1_path", type=str, default="lora_ilya_repin/ilya_repin_young_women_lora.safetensors",
                       help="Path to first LoRA model")
    parser.add_argument("--lora2_path", type=str, default="lora_gina/gina_face_cropped_lora.safetensors",
                       help="Path to second LoRA model")
    parser.add_argument("--lora1_token", type=str, default="<ilya_repin>",
                       help="Trigger token for first LoRA")
    parser.add_argument("--lora2_token", type=str, default="<gina_szanyel>",
                       help="Trigger token for second LoRA")
    
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
    parser.add_argument("--lora1_strength", type=float, default=0.7,
                       help="Strength of first LoRA")
    parser.add_argument("--lora2_strength", type=float, default=1.2,
                       help="Strength of second LoRA")
    parser.add_argument("--output_dir", type=str, default="lora_tests",
                       help="Output directory for generated images")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Output image size")
    
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
    
    print(f"Applied {len(visited)} LoRA modules")
    return pipe

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create test prompts
    prompts = {
        "base": args.prompt,
        "lora1": f"{args.prompt} {args.lora1_token}",
        "lora2": f"{args.prompt} {args.lora2_token}",
        "both": f"{args.prompt} {args.lora1_token} {args.lora2_token}"
    }
    
    # Create grid of smaller images for memory efficiency
    grid_width = args.image_size * 2
    grid_height = args.image_size * 2
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # Function to load model and clear memory
    def get_pipeline():
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
    
    # 1. Generate base image (no LoRA)
    print("\n[1/4] Generating base image without LoRA...")
    pipe = get_pipeline()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    base_image = pipe(
        prompt=prompts["base"],
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        height=args.image_size,
        width=args.image_size
    ).images[0]
    
    grid_image.paste(base_image, (0, 0))
    
    # Save base image
    base_path = os.path.join(args.output_dir, f"base_seed_{args.seed}.png")
    base_image.save(base_path)
    print(f"Saved base image to {base_path}")
    
    # Free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. Generate with first LoRA only
    print("\n[2/4] Generating with LoRA 1 only...")
    pipe = get_pipeline()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    try:
        print(f"Loading first LoRA from {args.lora1_path}")
        lora1_state_dict = load_file(args.lora1_path)
        pipe = load_lora_weights(pipe, lora1_state_dict, alpha=args.lora1_strength)
        
        # Generate with LoRA 1
        print(f"Generating with LoRA 1 using prompt: {prompts['lora1']}")
        lora1_image = pipe(
            prompt=prompts["lora1"],
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size
        ).images[0]
        
        # Save and add to grid
        lora1_path = os.path.join(args.output_dir, f"lora1_seed_{args.seed}_strength_{args.lora1_strength}.png")
        lora1_image.save(lora1_path)
        print(f"Saved LoRA 1 image to {lora1_path}")
        grid_image.paste(lora1_image, (args.image_size, 0))
        
    except Exception as e:
        print(f"Error with LoRA 1: {e}")
    
    # Free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # 3. Generate with second LoRA only
    print("\n[3/4] Generating with LoRA 2 only...")
    pipe = get_pipeline()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    try:
        print(f"Loading second LoRA from {args.lora2_path}")
        lora2_state_dict = load_file(args.lora2_path)
        pipe = load_lora_weights(pipe, lora2_state_dict, alpha=args.lora2_strength)
        
        # Generate with LoRA 2
        print(f"Generating with LoRA 2 using prompt: {prompts['lora2']}")
        lora2_image = pipe(
            prompt=prompts["lora2"],
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size
        ).images[0]
        
        # Save and add to grid
        lora2_path = os.path.join(args.output_dir, f"lora2_seed_{args.seed}_strength_{args.lora2_strength}.png")
        lora2_image.save(lora2_path)
        print(f"Saved LoRA 2 image to {lora2_path}")
        grid_image.paste(lora2_image, (0, args.image_size))
        
    except Exception as e:
        print(f"Error with LoRA 2: {e}")
    
    # Free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # 4. Generate with both LoRAs
    print("\n[4/4] Generating with both LoRAs...")
    pipe = get_pipeline()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    try:
        # Load and apply both LoRAs
        print(f"Loading and applying both LoRAs")
        lora1_state_dict = load_file(args.lora1_path)
        pipe = load_lora_weights(pipe, lora1_state_dict, alpha=args.lora1_strength)
        
        lora2_state_dict = load_file(args.lora2_path)
        pipe = load_lora_weights(pipe, lora2_state_dict, alpha=args.lora2_strength)
        
        # Generate with both LoRAs
        print(f"Generating with both LoRAs using prompt: {prompts['both']}")
        both_image = pipe(
            prompt=prompts["both"],
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size
        ).images[0]
        
        # Save and add to grid
        both_path = os.path.join(args.output_dir, f"both_loras_seed_{args.seed}.png")
        both_image.save(both_path)
        print(f"Saved combined LoRAs image to {both_path}")
        grid_image.paste(both_image, (args.image_size, args.image_size))
        
    except Exception as e:
        print(f"Error with combined LoRAs: {e}")
    
    # Add labels to the grid
    draw = ImageDraw.Draw(grid_image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
    
    # Add text with darker background for readability
    def add_text_with_bg(pos, text):
        x, y = pos
        # Get text size
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        
        # Draw background rectangle
        draw.rectangle(
            (x, y, x + text_width + 10, y + text_height + 5),
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((x + 5, y + 2), text, fill=(255, 255, 255), font=font)
    
    # Add labels for each quadrant
    add_text_with_bg((10, 10), f"Base")
    add_text_with_bg((args.image_size + 10, 10), f"{args.lora1_token} ({args.lora1_strength})")
    add_text_with_bg((10, args.image_size + 10), f"{args.lora2_token} ({args.lora2_strength})")
    add_text_with_bg((args.image_size + 10, args.image_size + 10), f"Both LoRAs")
    
    # Add generation settings at bottom
    settings = f"Seed: {args.seed} | Prompt: {args.prompt}"
    add_text_with_bg((10, grid_height - 30), settings)
    
    # Save grid image
    grid_path = os.path.join(args.output_dir, f"comparison_grid_seed_{args.seed}.png")
    grid_image.save(grid_path)
    print(f"\nSaved comparison grid to {grid_path}")

if __name__ == "__main__":
    main() 