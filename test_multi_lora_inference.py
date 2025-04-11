#!/usr/bin/env python3
"""
Test inference with multiple LoRAs applied simultaneously
Optimized for 12GB VRAM GPUs
"""

import os
import torch
import argparse
import gc
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file

from diffusers import StableDiffusionXLPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Test multiple LoRAs in inference")
    parser.add_argument("--base_model", type=str, default="sdxl-base-1.0")
    parser.add_argument("--self_attn_lora", type=str, default="multi_lora_test/self_attn_lora_test.safetensors")
    parser.add_argument("--cross_attn_lora", type=str, default="multi_lora_test/cross_attn_lora_test.safetensors")
    parser.add_argument("--output_dir", type=str, default="multi_lora_test")
    parser.add_argument("--prompt", type=str, default="A photo of a smiling person with red hair")
    parser.add_argument("--self_attn_weight", type=float, default=0.7)
    parser.add_argument("--cross_attn_weight", type=float, default=0.7)
    parser.add_argument("--image_size", type=int, default=256,
                     help="Output image size (smaller = faster)")
    parser.add_argument("--enable_xformers", action="store_true", default=True,
                     help="Enable xformers memory efficient attention")
    parser.add_argument("--enable_vae_slicing", action="store_true", default=True,
                     help="Enable VAE slicing to reduce memory usage")
    return parser.parse_args()

def apply_lora_to_model(pipeline, lora_path, target_modules, alpha=0.7):
    """Apply a LoRA to selected modules of the model"""
    print(f"Applying LoRA from {lora_path}")
    
    if not os.path.exists(lora_path):
        print(f"⚠️ LoRA file not found: {lora_path}")
        return False
    
    # Load LoRA weights
    try:
        lora_weights = load_file(lora_path)
        print(f"Loaded {len(lora_weights)} weights from {lora_path}")
    except Exception as e:
        print(f"⚠️ Failed to load LoRA: {e}")
        return False

    # Find all matching modules in the UNet
    all_modules = {}
    for name, module in pipeline.unet.named_modules():
        if hasattr(module, "weight") and name in target_modules:
            all_modules[name] = module
    
    # Apply LoRA weights
    applied_count = 0
    for name, weight in lora_weights.items():
        # Parse module name from parameter name (e.g. "down.lora.down.weight" -> "down")
        module_name = name.split(".lora")[0]
        
        if "lora_up.weight" in name and module_name in all_modules:
            up_weight = weight
            down_weight = lora_weights.get(name.replace("lora_up", "lora_down"))
            
            if down_weight is not None:
                module = all_modules[module_name]
                # Store original weight
                if not hasattr(module, "original_weight"):
                    module.original_weight = module.weight.data.clone()
                
                # Calculate LoRA contribution
                # W = W_0 + alpha * (up × down)
                lora_contribution = torch.matmul(up_weight, down_weight)
                
                # Apply LoRA with scaling factor
                scaled_contribution = alpha * lora_contribution
                module.weight.data = module.original_weight + scaled_contribution
                applied_count += 1
    
    print(f"Applied {applied_count} LoRA modules")
    return applied_count > 0

def get_target_modules(model, pattern):
    """Get list of modules matching the pattern"""
    target_modules = []
    for name, module in model.named_modules():
        if pattern in name and hasattr(module, 'weight'):
            target_modules.append(name)
    return target_modules

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("🧪 TESTING MULTI-LORA INFERENCE 🧪".center(50))
    print("=" * 50 + "\n")
    
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load model
    print(f"Loading base model from {args.base_model}...")
    try:
        # Load with minimal components for memory efficiency
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # Apply memory optimizations
        if args.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xFormers memory efficient attention enabled")
            except (ImportError, AttributeError):
                print("⚠️ xFormers not available, using default attention")
        
        if args.enable_vae_slicing:
            pipeline.enable_vae_slicing()
            print("✅ VAE slicing enabled")
            
        # Other memory optimizations
        pipeline.enable_attention_slicing()
        print("✅ Attention slicing enabled")
        
        # Offload to CPU when not in use
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            print("✅ Model CPU offload enabled")
        else:
            # Manually move to device
            pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Using dummy pipeline for testing...")
        from diffusers import UNet2DConditionModel, AutoencoderKL
        # Create a minimal pipeline for testing
        pipeline = StableDiffusionXLPipeline(
            unet=UNet2DConditionModel.from_pretrained("diffusers/unet-dummy"),
            vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse"),
            scheduler=None,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            feature_extractor=None,
        )
    
    # Get target modules
    self_attn_modules = get_target_modules(pipeline.unet, "attn1")
    cross_attn_modules = get_target_modules(pipeline.unet, "attn2")
    
    print(f"Self-attention modules: {len(self_attn_modules)}")
    print(f"Cross-attention modules: {len(cross_attn_modules)}")
    
    # Apply first LoRA (self-attention)
    applied_self = apply_lora_to_model(
        pipeline, 
        args.self_attn_lora, 
        self_attn_modules, 
        alpha=args.self_attn_weight
    )
    
    # Apply second LoRA (cross-attention)
    applied_cross = apply_lora_to_model(
        pipeline, 
        args.cross_attn_lora, 
        cross_attn_modules, 
        alpha=args.cross_attn_weight
    )
    
    if not applied_self and not applied_cross:
        print("⚠️ No LoRA weights were successfully applied")
        return
    
    # Generate an image with lower memory usage
    print(f"Generating test image with prompt: '{args.prompt}'")
    try:
        # Use smaller image size and fewer steps for memory efficiency
        image = pipeline(
            prompt=args.prompt,
            num_inference_steps=20,  # Reduced steps
            guidance_scale=7.5,
            height=args.image_size,
            width=args.image_size,
        ).images[0]
        
        # Save the image
        output_path = os.path.join(args.output_dir, "multi_lora_test_image.png")
        image.save(output_path)
        print(f"✅ Image saved to {output_path}")
        
        # Free up memory before generating comparison images
        torch.cuda.empty_cache()
        gc.collect()
        
        # Generate comparison images with each LoRA individually
        # Only if the output size is small enough for memory
        if args.image_size <= 512:
            test_variants = {}
            
            if applied_self:
                # Reset model weights
                print("Generating image with self-attention LoRA only...")
                for name, module in pipeline.unet.named_modules():
                    if hasattr(module, "original_weight"):
                        module.weight.data = module.original_weight.clone()
                
                # Apply only self-attn LoRA
                apply_lora_to_model(pipeline, args.self_attn_lora, self_attn_modules, alpha=args.self_attn_weight)
                torch.cuda.empty_cache()  # Clear cache before generation
                test_variants["self_only"] = pipeline(
                    prompt=args.prompt,
                    num_inference_steps=20,  # Reduced steps
                    guidance_scale=7.5,
                    height=args.image_size,
                    width=args.image_size,
                ).images[0]
                
                # Clear cache after generation
                torch.cuda.empty_cache()
                gc.collect()
            
            if applied_cross:
                # Reset model weights
                print("Generating image with cross-attention LoRA only...")
                for name, module in pipeline.unet.named_modules():
                    if hasattr(module, "original_weight"):
                        module.weight.data = module.original_weight.clone()
                
                # Apply only cross-attn LoRA
                apply_lora_to_model(pipeline, args.cross_attn_lora, cross_attn_modules, alpha=args.cross_attn_weight)
                torch.cuda.empty_cache()  # Clear cache before generation
                test_variants["cross_only"] = pipeline(
                    prompt=args.prompt,
                    num_inference_steps=20,  # Reduced steps
                    guidance_scale=7.5,
                    height=args.image_size,
                    width=args.image_size,
                ).images[0]
                
                # Clear cache after generation
                torch.cuda.empty_cache()
                gc.collect()
            
            # Save comparison images
            for variant_name, img in test_variants.items():
                variant_path = os.path.join(args.output_dir, f"multi_lora_test_{variant_name}.png")
                img.save(variant_path)
                print(f"✅ {variant_name.replace('_', ' ').title()} image saved to {variant_path}")
        else:
            print("Skipping comparison images due to memory constraints at this resolution")
        
    except Exception as e:
        print(f"⚠️ Error generating image: {e}")
        print("This could be due to incompletely trained LoRAs or GPU memory issues.")
    
    # Clear cache one final time
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n✅ Multi-LoRA inference test complete!")

if __name__ == "__main__":
    main() 