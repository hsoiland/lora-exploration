#!/usr/bin/env python3
"""
Inference script for testing diffusers-based LoRA weights
"""

import os
import torch
import argparse
import json
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoRAAttnProcessor
from safetensors.torch import load_file
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Test diffusers LoRA weights with inference")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base model to load")
    parser.add_argument("--lora_dir", type=str, default="diffusers_lora_output/final_checkpoint",
                        help="Directory with diffusers LoRA weights")
    
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
    parser.add_argument("--output_dir", type=str, default="diffusers_lora_outputs",
                        help="Directory to save generated images")
    
    # Sampling parameters
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of denoising steps")
    
    return parser.parse_args()

def load_lora_weights(pipeline, weights_path, rank=4, alpha=1.0):
    """Load LoRA weights from safetensors file into the pipeline"""
    # Load weights
    state_dict = load_file(weights_path)
    
    # Get the device
    device = pipeline.device
    
    # Load the JSON config if exists
    config_path = os.path.join(os.path.dirname(weights_path), "training_args.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            rank = config.get("rank", rank)
            alpha = config.get("alpha", alpha)
    
    print(f"Loading LoRA weights with rank={rank}, alpha={alpha}")
    
    # Set up attention processors
    lora_attn_procs = {}
    lora_keys = [k for k in state_dict.keys() if "lora" in k]
    for key in lora_keys:
        # Parse the attention processor name
        attn_processor_name = key.split(".")[0]
        if attn_processor_name not in lora_attn_procs:
            # Get cross attention dimensions from UNet
            for name, module in pipeline.unet.named_modules():
                if name == attn_processor_name:
                    # Determine if this is cross-attention or self-attention
                    if name.endswith("attn1"):
                        # Self-attention
                        cross_attention_dim = None
                    elif name.endswith("attn2") or name.startswith("mid_block"):
                        # Cross-attention
                        cross_attention_dim = pipeline.unet.config.cross_attention_dim
                    else:
                        # Skip
                        continue
                        
                    lora_attn_procs[attn_processor_name] = LoRAAttnProcessor(
                        rank=rank,
                        lora_alpha=alpha,
                        use_to_k=True,
                        use_to_v=True,
                        use_to_q=True,
                        use_to_out=True,
                        cross_attention_dim=cross_attention_dim,
                    )
                    break
    
    # Load weights into attention processors
    for key in lora_keys:
        attn_processor_name, weight_name = key.split(".", 1)
        if attn_processor_name in lora_attn_procs:
            # Get the processor
            processor = lora_attn_procs[attn_processor_name]
            
            # Determine which weight to load into
            if "to_q_lora" in weight_name:
                if "up" in weight_name:
                    processor.to_q_lora.up.weight.data = state_dict[key].to(device)
                elif "down" in weight_name:
                    processor.to_q_lora.down.weight.data = state_dict[key].to(device)
                    
            elif "to_k_lora" in weight_name:
                if "up" in weight_name:
                    processor.to_k_lora.up.weight.data = state_dict[key].to(device)
                elif "down" in weight_name:
                    processor.to_k_lora.down.weight.data = state_dict[key].to(device)
                    
            elif "to_v_lora" in weight_name:
                if "up" in weight_name:
                    processor.to_v_lora.up.weight.data = state_dict[key].to(device)
                elif "down" in weight_name:
                    processor.to_v_lora.down.weight.data = state_dict[key].to(device)
                    
            elif "to_out_lora" in weight_name:
                if "up" in weight_name:
                    processor.to_out_lora.up.weight.data = state_dict[key].to(device)
                elif "down" in weight_name:
                    processor.to_out_lora.down.weight.data = state_dict[key].to(device)
    
    # Set the attention processors
    pipeline.unet.set_attn_processor(lora_attn_procs)
    
    return pipeline, lora_attn_procs

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
    
    # Use Euler scheduler for better quality
    pipeline.scheduler = EulerDiscreteScheduler.from_pretrained(
        args.base_model, 
        subfolder="scheduler",
        torch_dtype=torch.float16
    )
    
    # Enable memory optimizations
    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
        print("✅ xFormers memory efficient attention enabled")
    except (ImportError, AttributeError):
        print("⚠️ xFormers not available, using default attention")
    
    # Load LoRA weights
    lora_path = os.path.join(args.lora_dir, "lora_weights.safetensors")
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA weights file {lora_path} does not exist!")
    
    print(f"Loading LoRA weights from: {lora_path}")
    pipeline, _ = load_lora_weights(pipeline, lora_path)
    
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
        output_path = os.path.join(args.output_dir, f"diffusers_lora_sample_{i+1}_seed_{args.seed}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")
    
    print("✅ Generation complete!")
    print(f"Generated {args.num_images} images in {args.output_dir}")

if __name__ == "__main__":
    main() 