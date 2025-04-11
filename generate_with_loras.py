#!/usr/bin/env python3
"""
Generate images using different combinations of LoRA models
"""

import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with LoRAs")
    
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
    parser.add_argument("--num_batches", type=int, default=3,
                       help="Number of batches to generate with different seeds")
    parser.add_argument("--start_seed", type=int, default=42,
                       help="Starting seed value")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--lora_scale", type=float, default=0.7,
                       help="Scale of LoRA adapters")
    parser.add_argument("--output_dir", type=str, default="generated_images",
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
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except (ImportError, AttributeError):
            print("⚠️ xFormers not available, using default attention")
    
    # Prepare prompts for different combinations
    prompts = {
        "no_lora": args.prompt,
        "lora1_only": f"{args.prompt} {args.lora1_token}",
        "lora2_only": f"{args.prompt} {args.lora2_token}",
        "both_loras": f"{args.prompt} {args.lora1_token} {args.lora2_token}"
    }
    
    # For each batch (different seed)
    for batch_idx in range(args.num_batches):
        seed = args.start_seed + batch_idx
        current_seed = torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"\n{'='*40}\nGenerating batch {batch_idx+1}/{args.num_batches} with seed {seed}\n{'='*40}")
        
        # Create a grid of images (2x2)
        result_width = 1024 * 2
        result_height = 1024 * 2
        result_image = Image.new('RGB', (result_width, result_height))
        
        # Stage 1: Generate with no LoRA
        print("Generating without any LoRA...")
        image_no_lora = pipe(
            prompt=prompts["no_lora"],
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        
        # Place in top-left of grid
        result_image.paste(image_no_lora, (0, 0))
        
        # Stage 2: Load and generate with first LoRA only
        print(f"Loading first LoRA: {args.lora1_path}")
        try:
            pipe.unload_lora_weights()  # Make sure we start fresh
            # Use prefix=None to match our format
            pipe.load_lora_weights(args.lora1_path, adapter_name="lora1", prefix=None)
            pipe.fuse_lora(lora_scale=args.lora_scale)
            
            print(f"Generating with first LoRA only...")
            image_lora1 = pipe(
                prompt=prompts["lora1_only"],
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            
            # Place in top-right of grid
            result_image.paste(image_lora1, (1024, 0))
            
            # Unfuse before loading the next LoRA
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"Error with first LoRA: {e}")
            # Use base image as fallback
            result_image.paste(image_no_lora, (1024, 0))
        
        # Stage 3: Load and generate with second LoRA only
        print(f"Loading second LoRA: {args.lora2_path}")
        try:
            pipe.unload_lora_weights()  # Make sure we start fresh
            # Use prefix=None to match our format
            pipe.load_lora_weights(args.lora2_path, adapter_name="lora2", prefix=None)
            pipe.fuse_lora(lora_scale=args.lora_scale)
            
            print(f"Generating with second LoRA only...")
            image_lora2 = pipe(
                prompt=prompts["lora2_only"],
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            
            # Place in bottom-left of grid
            result_image.paste(image_lora2, (0, 1024))
            
            # Unfuse before loading both LoRAs
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"Error with second LoRA: {e}")
            # Use base image as fallback
            result_image.paste(image_no_lora, (0, 1024))
        
        # Stage 4: Load and generate with both LoRAs
        print(f"Loading both LoRAs...")
        try:
            pipe.unload_lora_weights()  # Make sure we start fresh
            # Load both LoRAs with prefix=None
            pipe.load_lora_weights(args.lora1_path, adapter_name="lora1", prefix=None)
            pipe.load_lora_weights(args.lora2_path, adapter_name="lora2", prefix=None)
            
            # Fuse both LoRAs
            pipe.fuse_lora(lora_scale=args.lora_scale, adapter_name="lora1")
            pipe.fuse_lora(lora_scale=args.lora_scale, adapter_name="lora2")
            
            print(f"Generating with both LoRAs...")
            image_both_loras = pipe(
                prompt=prompts["both_loras"],
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            
            # Place in bottom-right of grid
            result_image.paste(image_both_loras, (1024, 1024))
            
            # Unfuse all LoRAs
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"Error with combined LoRAs: {e}")
            # Use base image as fallback
            result_image.paste(image_no_lora, (1024, 1024))
        
        # Add labels to the grid
        draw = ImageDraw.Draw(result_image)
        # Try to use a font if available, otherwise use default
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            try:
                # Try DejaVu Sans as alternative
                font = ImageFont.truetype("DejaVuSans.ttf", 36)
            except IOError:
                font = ImageFont.load_default()
        
        # Add dark background for text readability
        def add_text_with_bg(draw, xy, text, fill_text=(255, 255, 255), fill_bg=(0, 0, 0, 180), font=font):
            x, y = xy
            # Draw semi-transparent background
            text_bbox = draw.textbbox((x, y), text, font=font)
            bg_rect = (
                text_bbox[0] - 5,  # left
                text_bbox[1] - 5,  # top
                text_bbox[2] + 5,  # right
                text_bbox[3] + 5,  # bottom
            )
            overlay = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(bg_rect, fill=fill_bg)
            result_image.paste(overlay, (0, 0), overlay)
            
            # Draw text
            draw.text((x, y), text, fill=fill_text, font=font)
        
        # Add labels for each quadrant
        add_text_with_bg(draw, (10, 10), f"Base: {args.prompt}")
        add_text_with_bg(draw, (1034, 10), f"LoRA 1: {args.lora1_token}\n{args.prompt}")
        add_text_with_bg(draw, (10, 1034), f"LoRA 2: {args.lora2_token}\n{args.prompt}")
        add_text_with_bg(draw, (1034, 1034), f"Combined: {args.lora1_token} {args.lora2_token}\n{args.prompt}")
        
        # Add generation settings at the bottom
        settings_text = (
            f"Seed: {seed} | Steps: {args.num_steps} | CFG: {args.guidance_scale} | "
            f"LoRA Scale: {args.lora_scale} | Neg Prompt: {args.negative_prompt}"
        )
        add_text_with_bg(draw, (10, result_height - 50), settings_text)
        
        # Save the result
        output_path = os.path.join(args.output_dir, f"comparison_seed_{seed}.png")
        result_image.save(output_path)
        print(f"Saved comparison image to {output_path}")

if __name__ == "__main__":
    main() 