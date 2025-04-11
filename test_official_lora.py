#!/usr/bin/env python3
"""
Inference script for testing diffusers' official LoRA weights
"""

import os
import torch
import argparse
import json
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Test official diffusers LoRA weights")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base model to load")
    parser.add_argument("--lora_dir", type=str, default="official_lora_output/final_checkpoint",
                        help="Directory with LoRA weights")
    
    # Inference parameters
    parser.add_argument("--prompt", type=str, 
                        default="a photo of a beautiful gina with blue eyes, intricate, elegant, highly detailed",
                        help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, 
                        default="deformed, ugly, disfigured, low quality, blurry",
                        help="Negative prompt for image generation")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512],
                        help="Generated image size (width, height)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--lora_scale", type=float, default=0.8,
                        help="Scale for LoRA weights")
    parser.add_argument("--output_dir", type=str, default="official_lora_outputs",
                        help="Directory to save generated images")
    
    # Sampling parameters
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of denoising steps")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        # Use a random seed and report it for reproducibility
        args.seed = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print(f"Loading base model: {args.base_model}")
    # Load SDXL base model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    
    # Use DPM++ scheduler for better quality
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Enable memory optimizations
    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
        print("✅ xFormers memory efficient attention enabled")
    except (ImportError, AttributeError):
        print("⚠️ xFormers not available, using default attention")
    
    # Load LoRA weights
    if not os.path.exists(args.lora_dir):
        raise ValueError(f"LoRA directory {args.lora_dir} does not exist!")
    
    print(f"Loading LoRA weights from: {args.lora_dir}")
    pipeline.load_lora_weights(args.lora_dir)
    
    # Read adapter config if available
    adapter_config_path = os.path.join(args.lora_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
            print(f"LoRA config: rank={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
    
    print("Model loaded successfully!")
    
    # Generate images
    print(f"Generating {args.num_images} images with prompt: {args.prompt}")
    
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}...")
        
        # Generate the image
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.image_size[0],
                height=args.image_size[1],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                cross_attention_kwargs={"scale": args.lora_scale},
            ).images[0]
        
        # Save the image
        output_path = os.path.join(args.output_dir, f"official_lora_sample_{i+1}_seed_{args.seed}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")
    
    print("✅ Generation complete!")
    print(f"Generated {args.num_images} images in {args.output_dir}")

if __name__ == "__main__":
    main() 