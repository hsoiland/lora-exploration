#!/usr/bin/env python3
"""
Simple image generation with LoRA using manual weight application
"""

import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
import re
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with LoRA")
    
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
    parser.add_argument("--lora_scale", type=float, default=1.0,
                       help="Scale of LoRA adapters")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                       help="Output directory for generated images")
    
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
                        print(f"Could not find module: {model_path}")
                        continue
                
                # If we found the target module, apply weights
                if hasattr(module, 'weight'):
                    weight = module.weight
                    
                    # Apply LoRA: Original + alpha * (up_weight @ down_weight)
                    delta = torch.mm(up_weight, down_weight)
                    weight.data += alpha * delta.to(weight.device, weight.dtype)
                    print(f"Applied LoRA to {module_name}")
    
    print(f"Applied {len(visited)} LoRA modules")
    return pipe

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
    
    # Prepare full prompt with token
    full_prompt = f"{args.prompt} {args.token}".strip()
    
    # First, generate base image without LoRA
    base_image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).images[0]
    
    # Save base image
    base_path = os.path.join(args.output_dir, f"base_seed_{args.seed}.png")
    base_image.save(base_path)
    print(f"Saved base image to {base_path}")
    
    # Load and apply LoRA weights manually
    try:
        print(f"Loading LoRA weights from {args.lora_path}")
        lora_state_dict = load_file(args.lora_path)
        pipe = load_lora_weights(pipe, lora_state_dict, alpha=args.lora_scale)
        
        # Generate with LoRA
        print(f"Generating with LoRA using prompt: {full_prompt}")
        lora_image = pipe(
            prompt=full_prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        
        # Save LoRA image
        lora_path = os.path.join(args.output_dir, f"lora_seed_{args.seed}_strength_{args.lora_scale}.png")
        lora_image.save(lora_path)
        print(f"Saved LoRA image to {lora_path}")
        
    except Exception as e:
        print(f"Error applying LoRA: {e}")

if __name__ == "__main__":
    main() 