#!/usr/bin/env python3
"""
Compare image generation with and without LoRA
Generates images with identical parameters with and without LoRA applied
"""

import os
import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from src.rust_lora_self_attn import apply_lora_to_self_attention, find_self_attention_modules

def parse_args():
    parser = argparse.ArgumentParser(description="Compare image generation with and without LoRA")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                      help="Base model path or HF repo ID")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to your trained LoRA weights file")
    parser.add_argument("--prompt", type=str, default="A portrait of a woman, photorealistic, detailed, studio lighting",
                       help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="low quality, bad anatomy, worse quality, watermark",
                       help="Negative prompt")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=0.8, help="LoRA strength (0-1)")
    parser.add_argument("--output_dir", type=str, default="lora_comparison", help="Output directory")
    parser.add_argument("--num_comparisons", type=int, default=1, help="Number of different comparisons to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading base model: {args.base_model}")
    
    # Load scheduler
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.base_model, 
        subfolder="scheduler"
    )
    
    # Load pipeline with all required components
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            scheduler=scheduler,
            use_safetensors=True
        )
        pipe = pipe.to("cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to download from Hugging Face...")
        
        # If local load fails, try to download from Hugging Face
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            scheduler=scheduler,
            use_safetensors=True
        )
        pipe = pipe.to("cuda")
    
    # Find target modules for LoRA
    print("Finding target modules for LoRA...")
    target_modules = find_self_attention_modules(pipe.unet)
    
    # Optimize for slightly faster inference
    pipe.enable_vae_slicing()
    
    # Generate multiple comparison pairs if requested
    for i in range(args.num_comparisons):
        # Use specified seed or different seeds for each comparison
        if args.num_comparisons > 1:
            current_seed = args.seed + i
        else:
            current_seed = args.seed
        
        print(f"Comparison {i+1}/{args.num_comparisons} (Seed: {current_seed})")
        
        # Set the seed
        generator = torch.Generator(device="cuda").manual_seed(current_seed)
        torch.manual_seed(current_seed)
        
        # FIRST: Generate image WITHOUT LoRA
        print(f"Generating image WITHOUT LoRA...")
        without_lora = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        
        # Save the image
        without_lora_path = os.path.join(args.output_dir, f"without_lora_seed{current_seed}.png")
        without_lora.save(without_lora_path)
        print(f"Saved image to: {without_lora_path}")
        
        # SECOND: Now apply LoRA
        print(f"Applying LoRA weights from: {args.lora_path} (alpha={args.alpha})")
        
        # Load and apply LoRA weights
        try:
            original_weights = apply_lora_to_self_attention(
                pipe.unet,
                args.lora_path,
                alpha=args.alpha,
                target_modules=target_modules
            )
            
            # Generate with LoRA
            print(f"Generating image WITH LoRA...")
            # Create a new generator with the same seed for exact comparison
            generator = torch.Generator(device="cuda").manual_seed(current_seed)
            
            with_lora = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            
            # Save the image with LoRA
            with_lora_path = os.path.join(args.output_dir, f"with_lora_seed{current_seed}.png")
            with_lora.save(with_lora_path)
            print(f"Saved image to: {with_lora_path}")
            
            # Save a side-by-side comparison (optional)
            try:
                from PIL import Image
                # Create a new image with both results side by side
                total_width = without_lora.width * 2
                max_height = max(without_lora.height, with_lora.height)
                
                comparison = Image.new('RGB', (total_width, max_height))
                comparison.paste(without_lora, (0, 0))
                comparison.paste(with_lora, (without_lora.width, 0))
                
                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comparison)
                
                # Try to load a font, use default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except:
                    font = ImageFont.load_default()
                
                # Add text
                draw.text((10, 10), "Without LoRA", fill=(255, 255, 255), font=font)
                draw.text((without_lora.width + 10, 10), f"With LoRA (α={args.alpha})", fill=(255, 255, 255), font=font)
                
                # Save comparison
                comparison_path = os.path.join(args.output_dir, f"comparison_seed{current_seed}.png")
                comparison.save(comparison_path)
                print(f"Saved comparison to: {comparison_path}")
            except Exception as e:
                print(f"Couldn't create side-by-side comparison: {e}")
            
            # Restore original weights
            print("Restoring original model weights...")
            from src.rust_lora_self_attn import restore_weights
            restore_weights(pipe.unet, original_weights)
            
        except Exception as e:
            print(f"Error applying LoRA: {e}")
    
    print("✅ Comparison complete!")
    
    # Print a summary of files generated
    print("\nFiles generated:")
    for i in range(args.num_comparisons):
        current_seed = args.seed + i if args.num_comparisons > 1 else args.seed
        print(f"  - Seed {current_seed}:")
        print(f"    - {os.path.join(args.output_dir, f'without_lora_seed{current_seed}.png')}")
        print(f"    - {os.path.join(args.output_dir, f'with_lora_seed{current_seed}.png')}")
        print(f"    - {os.path.join(args.output_dir, f'comparison_seed{current_seed}.png')}")

if __name__ == "__main__":
    main() 