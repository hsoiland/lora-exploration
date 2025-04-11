#!/usr/bin/env python3
"""
Multi-LoRA training script for SDXL
Uses diffusers' built-in LoRA support instead of PEFT
Optimized for proper tensor shapes and memory efficiency
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
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

# Import FaceDataset from train_full_lora.py
from train_full_lora import FaceDataset

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training for SDXL")
    
    # Basic parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--images_dir", type=str, default="gina_face_cropped")
    parser.add_argument("--captions_file", type=str, default="gina_captions.json")
    parser.add_argument("--output_dir", type=str, default="diffusers_lora_output")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    
    # LoRA parameters
    parser.add_argument("--rank", type=int, default=16, 
                        help="LoRA rank (higher = more capacity, more VRAM)")
    parser.add_argument("--lora_alpha", type=float, default=32, 
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    
    # Memory optimization parameters
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], 
                        default="fp16", help="Mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--enable_xformers", action="store_true", default=True,
                        help="Enable memory efficient attention with xformers if available")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 DIFFUSERS LORA TRAINING FOR SDXL 🚀".center(60))
    print("=" * 60 + "\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load UNet model
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
    
    # Enable memory optimizations for UNet
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
    
    # Create and apply LoRA attention processors
    print(f"Applying LoRA (rank={args.rank}, alpha={args.lora_alpha}) to UNet...")
    
    # Initialize the LoRA attention processors
    lora_attn_procs = {}
    
    # Get a list of all attention processors
    for name in unet.attn_processors.keys():
        cross_attention_dim = None
        if name.startswith("mid_block"):
            # Set cross-attention dimension for mid-block processors
            cross_attention_dim = unet.config.cross_attention_dim
        elif name.endswith("attn1.processor"):
            # Self-attention processor
            cross_attention_dim = None
        elif name.endswith("attn2.processor"):
            # Cross-attention processor
            cross_attention_dim = unet.config.cross_attention_dim
        else:
            # Skip anything that doesn't fit the pattern
            continue
            
        # Create LoRA processor using the correct constructor
        lora_attn_procs[name] = LoRAAttnProcessor2_0(
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            use_to_k=True,     # Apply LoRA to key projection
            use_to_v=True,     # Apply LoRA to value projection
            use_to_q=True,     # Apply LoRA to query projection
            use_to_out=True,   # Apply LoRA to output projection
            lora_dropout=args.lora_dropout,
            cross_attention_dim=cross_attention_dim,
        )
    
    # Set the attention processors to be the LoRA ones
    unet.set_attn_processor(lora_attn_procs)
    
    # Get the LoRA parameters to optimize
    lora_layers = AttnProcsLayers(unet.attn_processors)
    
    # Print number of trainable parameters
    trainable_params = 0
    all_params = 0
    for param in unet.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of total)")
    
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
    
    # Set up optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
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
    unet.train()  # Set UNet to training mode
    
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
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
                
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
                progress_bar.update(1)
                global_step += 1
                continue
            
            progress_bar.update(1)
            global_step += 1
            
            # Save checkpoints
            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save LoRA weights
                lora_state_dict = {}
                for name, attn_processor in unet.attn_processors.items():
                    if isinstance(attn_processor, LoRAAttnProcessor2_0):
                        # Save the weights
                        if hasattr(attn_processor, "to_q_lora"):
                            lora_state_dict[f"{name}.to_q_lora.down.weight"] = attn_processor.to_q_lora.down.weight.data
                            lora_state_dict[f"{name}.to_q_lora.up.weight"] = attn_processor.to_q_lora.up.weight.data
                            
                        if hasattr(attn_processor, "to_k_lora"):
                            lora_state_dict[f"{name}.to_k_lora.down.weight"] = attn_processor.to_k_lora.down.weight.data
                            lora_state_dict[f"{name}.to_k_lora.up.weight"] = attn_processor.to_k_lora.up.weight.data
                            
                        if hasattr(attn_processor, "to_v_lora"):
                            lora_state_dict[f"{name}.to_v_lora.down.weight"] = attn_processor.to_v_lora.down.weight.data
                            lora_state_dict[f"{name}.to_v_lora.up.weight"] = attn_processor.to_v_lora.up.weight.data
                            
                        if hasattr(attn_processor, "to_out_lora"):
                            lora_state_dict[f"{name}.to_out_lora.down.weight"] = attn_processor.to_out_lora.down.weight.data
                            lora_state_dict[f"{name}.to_out_lora.up.weight"] = attn_processor.to_out_lora.up.weight.data
                
                # Save in safetensors format
                save_file(lora_state_dict, os.path.join(checkpoint_dir, "lora_weights.safetensors"))
                
                # Save training args
                with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
                    training_args = {
                        "rank": args.rank,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                    }
                    json.dump(training_args, f, indent=2)
                
                print(f"✅ Checkpoint saved at step {global_step}")
            
            # Clean up CUDA cache periodically
            if step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    # Final checkpoint
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    # Save LoRA weights
    lora_state_dict = {}
    for name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, LoRAAttnProcessor2_0):
            # Save the weights
            if hasattr(attn_processor, "to_q_lora"):
                lora_state_dict[f"{name}.to_q_lora.down.weight"] = attn_processor.to_q_lora.down.weight.data
                lora_state_dict[f"{name}.to_q_lora.up.weight"] = attn_processor.to_q_lora.up.weight.data
                
            if hasattr(attn_processor, "to_k_lora"):
                lora_state_dict[f"{name}.to_k_lora.down.weight"] = attn_processor.to_k_lora.down.weight.data
                lora_state_dict[f"{name}.to_k_lora.up.weight"] = attn_processor.to_k_lora.up.weight.data
                
            if hasattr(attn_processor, "to_v_lora"):
                lora_state_dict[f"{name}.to_v_lora.down.weight"] = attn_processor.to_v_lora.down.weight.data
                lora_state_dict[f"{name}.to_v_lora.up.weight"] = attn_processor.to_v_lora.up.weight.data
                
            if hasattr(attn_processor, "to_out_lora"):
                lora_state_dict[f"{name}.to_out_lora.down.weight"] = attn_processor.to_out_lora.down.weight.data
                lora_state_dict[f"{name}.to_out_lora.up.weight"] = attn_processor.to_out_lora.up.weight.data
    
    # Save in safetensors format
    save_file(lora_state_dict, os.path.join(final_checkpoint_dir, "lora_weights.safetensors"))
    
    # Save model information
    with open(os.path.join(args.output_dir, "lora_info.json"), 'w') as f:
        info = {
            "base_model": args.base_model,
            "rank": args.rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "image_size": args.image_size,
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "training_steps": global_step
        }
        json.dump(info, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {final_checkpoint_dir}")
    print("Model information saved to lora_info.json")

if __name__ == "__main__":
    main() 