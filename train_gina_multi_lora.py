#!/usr/bin/env python3
"""
Multi-LoRA training script with text embeddings
Optimized for training on the Gina dataset
"""

import os
import torch
import argparse
import gc
import numpy as np
import json
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import save_file

from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

# Import from the main training script
from train_full_lora import (
    FaceDataset, 
    apply_lora_layer, 
    initialize_lora_weights
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-LoRA training with text embeddings")
    
    # Basic parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--images_dir", type=str, default="gina_face_cropped")
    parser.add_argument("--captions_file", type=str, default="gina_captions.json")
    parser.add_argument("--output_dir", type=str, default="gina_multi_lora")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500, 
                       help="Max training steps to limit training time")
    parser.add_argument("--image_size", type=int, default=512, 
                       help="Image size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoints every X steps")
    
    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, 
                        help="LoRA rank")
    
    # Memory optimization parameters
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], 
                        default="fp16", help="Mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--enable_xformers", action="store_true", default=True,
                        help="Enable memory efficient attention with xformers if available")
    
    return parser.parse_args()

def get_self_attention_modules(unet):
    """Get self-attention modules for the first LoRA"""
    target_modules = []
    
    for name, module in unet.named_modules():
        if "attn1" in name and hasattr(module, 'weight') and any(x in name for x in ["to_q", "to_k", "to_v"]):
            target_modules.append(name)
            
    return target_modules

def get_cross_attention_modules(unet):
    """Get cross-attention modules for the second LoRA"""
    target_modules = []
    
    for name, module in unet.named_modules():
        if "attn2" in name and hasattr(module, 'weight') and any(x in name for x in ["to_q", "to_k", "to_v"]):
            target_modules.append(name)
            
    return target_modules

def get_dtype_from_args(args):
    """Get the torch data type based on the mixed precision setting"""
    if args.mixed_precision == "fp16":
        return torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float32

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🌟 MULTI-LORA TRAINING WITH TEXT EMBEDDINGS 🌟".center(60))
    print("=" * 60 + "\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model components
    print(f"Loading UNet from {args.base_model}...")
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model,
        subfolder="unet",
        torch_dtype=torch.float32
    ).to(device)
    
    # Set up VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.base_model,
        subfolder="vae", 
        torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    
    # Load text encoders and tokenizers
    print("Loading text encoders and tokenizers...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer_2")
        
        # Load text encoders with output_hidden_states=True to get pooled output
        text_encoder = CLIPTextModel.from_pretrained(
            args.base_model, 
            subfolder="text_encoder", 
            torch_dtype=torch.float32,
            output_hidden_states=True
        ).to(device)
        
        text_encoder_2 = CLIPTextModel.from_pretrained(
            args.base_model, 
            subfolder="text_encoder_2", 
            torch_dtype=torch.float32,
            output_hidden_states=True
        ).to(device)
        
        # Set to eval mode and freeze
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        
        use_text_conditioning = True
        print("✅ Text encoders loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading text encoders: {e}")
        print("Falling back to zero conditioning...")
        tokenizer = None
        tokenizer_2 = None
        text_encoder = None
        text_encoder_2 = None
        use_text_conditioning = False
    
    # Get target modules for the two different LoRAs
    self_attn_modules = get_self_attention_modules(unet)
    cross_attn_modules = get_cross_attention_modules(unet)
    
    print(f"Found {len(self_attn_modules)} self-attention modules")
    print(f"Found {len(cross_attn_modules)} cross-attention modules")
    
    # Enable memory optimizations
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")
    
    # Enable xformers memory efficient attention if available
    if args.enable_xformers:
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except (ImportError, AttributeError):
            print("⚠️ xFormers not available, using default attention")
    
    # Initialize LoRA weights with FP32 (more stable for training)
    self_attn_lora = initialize_lora_weights(
        unet, self_attn_modules, args.rank, device=device, dtype=torch.float32
    )
    
    cross_attn_lora = initialize_lora_weights(
        unet, cross_attn_modules, args.rank, device=device, dtype=torch.float32
    )
    
    # Setup dataset
    print(f"Setting up dataset from {args.images_dir}...")
    
    # Check if captions file exists
    if os.path.exists(args.captions_file):
        print(f"Using captions from {args.captions_file}")
    else:
        print(f"Captions file {args.captions_file} not found, using default captions")
    
    # Create dataset
    dataset = FaceDataset(
        images_dir=args.images_dir,
        captions_file=args.captions_file if os.path.exists(args.captions_file) else None,
        image_size=args.image_size,
        use_cache=True,
        tokenizer=tokenizer if use_text_conditioning else None,
        tokenizer_2=tokenizer_2 if use_text_conditioning else None
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid issues with tokenizers
    )
    
    # Create minimal optimizer for both LoRAs
    all_params = list(self_attn_lora.values()) + list(cross_attn_lora.values())
    for param in all_params:
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(all_params, lr=args.learning_rate)
    
    # Set up noise scheduler
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    
    # Set up mixed precision training
    amp_enabled = args.mixed_precision != "no" and device == "cuda"
    try:
        # Try the newer API first (PyTorch >= 2.0)
        scaler = torch.amp.GradScaler(device_type='cuda') if amp_enabled and args.mixed_precision == "fp16" else None
    except TypeError:
        # Fall back to the older API (PyTorch < 2.0)
        scaler = torch.cuda.amp.GradScaler() if amp_enabled and args.mixed_precision == "fp16" else None
    
    # Training loop
    print(f"Starting training for {args.num_train_epochs} epochs, max {args.max_train_steps} steps...")
    unet.requires_grad_(False)  # Freeze UNet
    
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(range(min(args.max_train_steps - global_step, len(dataloader))), 
                          desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(dataloader):
            if global_step >= args.max_train_steps:
                break
                
            # Get images
            img_tensor = batch["image"].to(device)
            
            # Encode to latent space
            with torch.no_grad():
                # Get latent representation
                try:
                    latent = vae.encode(img_tensor).latent_dist.sample() * 0.18215
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    print("Using random latent as fallback...")
                    latent = torch.randn(img_tensor.shape[0], 4, args.image_size // 8, args.image_size // 8, device=device)
                
                # Add noise
                noise = torch.randn_like(latent)
                timesteps = torch.randint(0, 1000, (latent.shape[0],), device=device)
                noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)
                
                # Process text conditioning
                if use_text_conditioning and "input_ids" in batch and "input_ids_2" in batch:
                    try:
                        # Get the text embeddings for conditioning
                        with torch.no_grad():
                            # Process with first text encoder (CLIP ViT-L/14)
                            text_encoder_output_1 = text_encoder(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                output_hidden_states=True,
                                return_dict=True
                            )
                            # The embedding comes from the penultimate layer
                            text_embeddings_1 = text_encoder_output_1.hidden_states[-2]
                            
                            # Process with second text encoder (CLIP ViT-G/14)
                            text_encoder_output_2 = text_encoder_2(
                                input_ids=batch["input_ids_2"].to(device),
                                attention_mask=batch["attention_mask_2"].to(device),
                                output_hidden_states=True, 
                                return_dict=True
                            )
                            # The embedding comes from the penultimate layer
                            text_embeddings_2 = text_encoder_output_2.hidden_states[-2]
                            
                            # Get the pooled output for text_embeds
                            pooled_output_1 = text_encoder_output_1.pooler_output
                            pooled_output_2 = text_encoder_output_2.pooler_output
                            
                            # Ensure pooled outputs have the right shape for the model
                            text_embeds = torch.cat([pooled_output_1, pooled_output_2], dim=-1)
                            
                            # Concatenate embeddings from both text encoders
                            encoder_hidden_states = torch.concat([text_embeddings_1, text_embeddings_2], dim=-1)
                            
                            # Create time_ids for SDXL specific conditioning
                            # Format: [h, w, crop_top, crop_left, crop_h, crop_w]
                            time_ids = torch.tensor(
                                [args.image_size, args.image_size, 0, 0, args.image_size, args.image_size],
                                device=device
                            ).unsqueeze(0).repeat(img_tensor.shape[0], 1)  # Add batch dimension
                            
                            # Prepare the added_cond_kwargs
                            added_cond_kwargs = {
                                "text_embeds": text_embeds,
                                "time_ids": time_ids
                            }
                            
                    except Exception as e:
                        print(f"Error processing text: {e}")
                        # Create dummy encoder hidden states and cond_kwargs
                        encoder_hidden_states = torch.zeros((img_tensor.shape[0], 77, 2048), device=device)
                        added_cond_kwargs = {
                            "text_embeds": torch.zeros((img_tensor.shape[0], 1280), device=device),
                            "time_ids": torch.zeros((img_tensor.shape[0], 6), device=device)
                        }
                else:
                    # Create dummy encoder hidden states with correct shape for SDXL
                    encoder_hidden_states = torch.zeros((img_tensor.shape[0], 77, 2048), device=device)
                    
                    # Create dummy added condition kwargs
                    added_cond_kwargs = {
                        "text_embeds": torch.zeros((img_tensor.shape[0], 1280), device=device),
                        "time_ids": torch.zeros((img_tensor.shape[0], 6), device=device)
                    }
            
            # Apply both LoRAs
            restore_self_attn = apply_lora_layer(unet, self_attn_modules, self_attn_lora)
            restore_cross_attn = apply_lora_layer(unet, cross_attn_modules, cross_attn_lora)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                # Create a noise prediction target
                target = noise
                
                # Try the newer API first (PyTorch >= 2.0)
                try:
                    autocast_context = torch.amp.autocast(device_type='cuda', enabled=amp_enabled)
                except TypeError:
                    # Fall back to the older API (PyTorch < 2.0)
                    autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled)
                
                # Use the appropriate autocast context
                with autocast_context:
                    # Forward pass through UNet
                    noise_pred = unet(
                        noisy_latent,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, target)
                
                # Backward pass with scaling if enabled
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Log step info
                progress_bar.set_postfix(loss=loss.item())
                
            except Exception as e:
                print(f"Error in training step: {e}")
                # Continue to next sample
                restore_self_attn()
                restore_cross_attn()
                progress_bar.update(1)
                global_step += 1
                continue
            
            # Remove LoRA adaptations
            restore_self_attn()
            restore_cross_attn()
            
            progress_bar.update(1)
            global_step += 1
            
            # Save checkpoints
            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                # Save self-attention LoRA
                self_lora_path = os.path.join(args.output_dir, f"self_attn_lora_step_{global_step}.safetensors")
                self_weights = {k: v.detach().cpu() for k, v in self_attn_lora.items()}
                save_file(self_weights, self_lora_path)
                
                # Save cross-attention LoRA
                cross_lora_path = os.path.join(args.output_dir, f"cross_attn_lora_step_{global_step}.safetensors")
                cross_weights = {k: v.detach().cpu() for k, v in cross_attn_lora.items()}
                save_file(cross_weights, cross_lora_path)
                
                print(f"✅ Checkpoint saved at step {global_step}")
            
            # Clean up CUDA cache periodically
            if step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    # Final checkpoint
    # Save self-attention LoRA
    self_lora_path = os.path.join(args.output_dir, f"self_attn_lora_final.safetensors")
    self_weights = {k: v.detach().cpu() for k, v in self_attn_lora.items()}
    save_file(self_weights, self_lora_path)
    
    # Save cross-attention LoRA
    cross_lora_path = os.path.join(args.output_dir, f"cross_attn_lora_final.safetensors")
    cross_weights = {k: v.detach().cpu() for k, v in cross_attn_lora.items()}
    save_file(cross_weights, cross_lora_path)
    
    # Save parameter counts and module names for reference
    with open(os.path.join(args.output_dir, "lora_info.json"), 'w') as f:
        info = {
            "self_attention": {
                "module_count": len(self_attn_modules),
                "parameter_count": sum(p.numel() for p in self_attn_lora.values()),
                "modules": self_attn_modules
            },
            "cross_attention": {
                "module_count": len(cross_attn_modules),
                "parameter_count": sum(p.numel() for p in cross_attn_lora.values()),
                "modules": cross_attn_modules
            }
        }
        json.dump(info, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Self-attention LoRA saved to: {self_lora_path}")
    print(f"Cross-attention LoRA saved to: {cross_lora_path}")
    print("Module information saved to lora_info.json")

if __name__ == "__main__":
    main() 