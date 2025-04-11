#!/usr/bin/env python3
"""
Simplified SDXL LoRA training script using diffusers' built-in LoRA support
"""

import os
import torch
import argparse
import json
import math
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import save_file
import traceback
import numpy as np

from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Import PEFT for LoRA configuration
from peft import LoraConfig, get_peft_model

# Create a simple dataset class
class SimpleImageDataset(Dataset):
    def __init__(
        self,
        images_dir,
        image_size=512,
        captions_file=None,
        tokenizer=None,
        tokenizer_2=None,
        caption_ext=".txt",
        use_caption_files=False,
        use_cache=True,
        default_caption="a photo of a person"
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.default_caption = default_caption
        self.caption_ext = caption_ext
        self.use_caption_files = use_caption_files
        self.use_cache = use_cache
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.captions = {}
        self.cached_samples = {}
        
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, "r") as f:
                self.captions = json.load(f)
        
        # Ensure images directory exists
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            print(f"Creating images directory: {self.images_dir}")
            self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_paths = sorted([
            p for p in self.images_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ])
        
        print(f"Found {len(self.image_paths)} images in {images_dir}")
        
        # Create a dummy image if no real images found
        if len(self.image_paths) == 0:
            print("No images found! Creating a dummy image for testing.")
            dummy_image_path = self.images_dir / "dummy_image.png"
            # Create a small random image and save it
            dummy_img = Image.fromarray(
                (np.random.rand(image_size, image_size, 3) * 255).astype(np.uint8)
            )
            dummy_img.save(dummy_image_path)
            self.image_paths = [dummy_image_path]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_caption(self, image_path):
        # Try to get caption from JSON file
        if image_path.name in self.captions:
            return self.captions[image_path.name]
        
        # Try to get caption from sidecar TXT file
        if self.use_caption_files:
            caption_path = image_path.with_suffix(self.caption_ext)
            if caption_path.exists():
                with open(caption_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        
        # Use default caption as fallback
        return self.default_caption
    
    def tokenize(self, caption):
        if self.tokenizer is None or self.tokenizer_2 is None:
            return None, None, None, None
        
        # Tokenize with first tokenizer (CLIP ViT-L/14)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize with second tokenizer (CLIP ViT-G/14)
        tokens_2 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return (
            tokens.input_ids.squeeze(0),
            tokens.attention_mask.squeeze(0),
            tokens_2.input_ids.squeeze(0),
            tokens_2.attention_mask.squeeze(0)
        )
    
    def __getitem__(self, idx):
        if idx in self.cached_samples and self.use_cache:
            return self.cached_samples[idx]
        
        image_path = self.image_paths[idx]
        
        # Load and transform image
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image as fallback
            img_tensor = torch.randn(3, self.image_size, self.image_size)
        
        # Get caption and tokenize
        caption = self.get_caption(image_path)
        
        if self.tokenizer and self.tokenizer_2:
            input_ids, attention_mask, input_ids_2, attention_mask_2 = self.tokenize(caption)
            
            sample = {
                "image": img_tensor,
                "caption": caption,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_ids_2": input_ids_2,
                "attention_mask_2": attention_mask_2,
            }
        else:
            sample = {
                "image": img_tensor,
                "caption": caption,
            }
        
        if self.use_cache:
            self.cached_samples[idx] = sample
        
        return sample

def save_as_safetensors(unet, save_path, metadata=None):
    """Save the LoRA weights as safetensors format."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract LoRA weights from the PEFT model
    lora_state_dict = {}
    
    # Collect all LoRA weights
    for name, module in unet.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # Handle layer modules that have lora_A and lora_B attributes
            a_tensor = module.lora_A.default.weight if hasattr(module.lora_A, "default") else module.lora_A.weight
            b_tensor = module.lora_B.default.weight if hasattr(module.lora_B, "default") else module.lora_B.weight
            
            # Get module name for safetensors format
            module_name = name.replace(".", "_")
            
            # Save LoRA A and B matrices
            lora_state_dict[f"{module_name}.lora_A.weight"] = a_tensor
            lora_state_dict[f"{module_name}.lora_B.weight"] = b_tensor
    
    print(f"Saving {len(lora_state_dict)} LoRA tensors to {save_path}")
    
    # Add metadata
    if metadata is None:
        metadata = {}
    
    # Save using safetensors
    save_file(lora_state_dict, save_path, metadata)
    print(f"Successfully saved safetensors file to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training for SDXL using diffusers")
    
    # Basic parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--images_dir", type=str, default="gina_face_cropped", 
                        help="Directory containing training images. Use comma-separated values for multiple datasets.")
    parser.add_argument("--captions_file", type=str, default=None,
                        help="Path to captions file. Can be comma-separated for multiple datasets.")
    parser.add_argument("--output_dir", type=str, default="official_lora_output",
                        help="Output directory. Can be comma-separated for multiple datasets.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for the final model file. Can be comma-separated for multiple datasets.")
    parser.add_argument("--lora_concept", type=str, default=None,
                        help="Concept name to be used in metadata. Can be comma-separated for multiple datasets.")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    
    # LoRA parameters
    parser.add_argument("--rank", type=int, default=4, 
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
    parser.add_argument("--enable_xformers", action="store_true", default=False,
                        help="Enable memory efficient attention with xformers if available")
    
    return parser.parse_args()

def train_lora(
    base_model,
    images_dir,
    captions_file,
    output_dir,
    model_name,
    lora_concept,
    num_train_epochs,
    train_batch_size,
    max_train_steps,
    image_size,
    learning_rate,
    save_steps,
    rank,
    lora_alpha,
    lora_dropout,
    mixed_precision,
    gradient_checkpointing,
    enable_xformers
):
    """
    Train a LoRA model on a single dataset
    """
    print("\n" + "=" * 60)
    print(f"🚀 TRAINING LORA FOR: {images_dir} 🚀".center(60))
    print("=" * 60 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load UNet model
    print(f"Loading UNet from {base_model}...")
    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32
    ).to(device)
    
    # Set up VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix" if mixed_precision == "fp16" else base_model,
        subfolder="vae" if mixed_precision != "fp16" else None,
        torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32
    ).to(device)
    vae.requires_grad_(False)
    
    # Load text encoders and tokenizers
    print("Loading text encoders and tokenizers...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
        
        # Load text encoders
        text_encoder = CLIPTextModel.from_pretrained(
            base_model, 
            subfolder="text_encoder", 
            torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
        ).to(device)
        
        text_encoder_2 = CLIPTextModel.from_pretrained(
            base_model, 
            subfolder="text_encoder_2", 
            torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
        ).to(device)
        
        # Set to eval mode and freeze
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        
        # For better error handling and no GPU OOM
        default_caption = "a photo of a person"
        try:
            print("Testing tokenization and text encoders with the basic caption...")
            # Test tokenization on CPU first
            test_tokens = tokenizer(
                default_caption,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            test_tokens_2 = tokenizer_2(
                default_caption,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            use_text_conditioning = True
            print("✅ Text tokenization successful!")
        except Exception as e:
            print(f"⚠️ Text tokenization failed: {e}")
            print(traceback.format_exc())
            print("Falling back to zero conditioning...")
            use_text_conditioning = False
            
    except Exception as e:
        print(f"⚠️ Error loading text encoders: {e}")
        print("Falling back to zero conditioning...")
        tokenizer = None
        tokenizer_2 = None
        text_encoder = None
        text_encoder_2 = None
        use_text_conditioning = False
    
    # Enable memory optimizations for UNet
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")
    
    # Enable xformers memory efficient attention if available
    if enable_xformers:
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except (ImportError, AttributeError):
            print("⚠️ xFormers not available, using default attention")
    
    # Set up LoRA configuration
    print(f"Setting up LoRA with rank={rank}, alpha={lora_alpha}...")
    
    # Define the target modules for SDXL UNet - more focused and specific list
    target_modules = [
        "to_q",        # Query projection in attention
        "to_k",        # Key projection in attention
        "to_v",        # Value projection in attention
        "to_out.0"     # Output projection in attention
    ]
    
    # Create a proper PEFT LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # Apply LoRA config to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
    
    # Setup dataset
    print(f"Setting up dataset from {images_dir}...")
    
    # Create dataset
    dataset = SimpleImageDataset(
        images_dir=images_dir,
        image_size=image_size,
        captions_file=captions_file if captions_file and os.path.exists(captions_file) else None,
        tokenizer=tokenizer if use_text_conditioning else None,
        tokenizer_2=tokenizer_2 if use_text_conditioning else None,
        default_caption="a photo of a person"
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid issues with tokenizers
    )
    
    # Set up optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Set up noise scheduler
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")
    
    # Calculate number of steps
    num_update_steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = max_train_steps or (num_train_epochs * num_update_steps_per_epoch)
    
    # Set up mixed precision training
    amp_enabled = mixed_precision != "no" and device == "cuda"
    try:
        # Try the newer API first (PyTorch >= 2.0)
        scaler = torch.amp.GradScaler(device_type='cuda') if amp_enabled and mixed_precision == "fp16" else None
    except TypeError:
        # Fall back to the older API (PyTorch < 2.0)
        scaler = torch.cuda.amp.GradScaler() if amp_enabled and mixed_precision == "fp16" else None
    
    # Training loop
    print(f"Starting training for {num_train_epochs} epochs, max {max_train_steps} steps...")
    unet.train()  # Set UNet to training mode
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps))
    
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break
                
            # Get images
            img_tensor = batch["image"].to(device)
            
            # Encode to latent space
            with torch.no_grad():
                # Get latent representation
                try:
                    # Convert image tensor to float16 if using mixed precision
                    if mixed_precision == "fp16":
                        img_tensor = img_tensor.to(dtype=torch.float16)
                        
                    latent = vae.encode(img_tensor).latent_dist.sample() * 0.18215
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    print("Using random latent as fallback...")
                    latent = torch.randn(img_tensor.shape[0], 4, image_size // 8, image_size // 8, device=device)
                    # Ensure random latent has the correct dtype
                    if mixed_precision == "fp16":
                        latent = latent.to(dtype=torch.float16)
                
                # Add noise
                noise = torch.randn_like(latent)
                timesteps = torch.randint(0, 1000, (latent.shape[0],), device=device)
                noisy_latent = noise_scheduler.add_noise(latent, noise, timesteps)
                
                # Ensure noisy_latent has the correct dtype
                if mixed_precision == "fp16":
                    noisy_latent = noisy_latent.to(dtype=torch.float16)
                
                # Process text conditioning
                if use_text_conditioning and "input_ids" in batch and "input_ids_2" in batch:
                    try:
                        # Explicitly ensure batch elements are on the correct device
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        input_ids_2 = batch["input_ids_2"].to(device)
                        attention_mask_2 = batch["attention_mask_2"].to(device)
                        
                        # Process inputs safely with safe execution
                        try:
                            # Get the text embeddings for conditioning
                            with torch.no_grad():
                                # Process with first text encoder (CLIP ViT-L/14)
                                encoder_output_1 = text_encoder(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                )
                                encoder_hidden_states_1 = encoder_output_1.last_hidden_state
                                
                                # Process with second text encoder (CLIP ViT-G/14)
                                encoder_output_2 = text_encoder_2(
                                    input_ids=input_ids_2,
                                    attention_mask=attention_mask_2,
                                )
                                encoder_hidden_states_2 = encoder_output_2.last_hidden_state
                                
                                # Get the batch size
                                batch_size = img_tensor.shape[0]
                                
                                # Concatenate the outputs from both text encoders along the hidden dimension
                                encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                                
                                # Get pooled output for SDXL conditioning (second encoder only)
                                text_embeds = encoder_output_2.pooler_output
                                
                                # Create time_ids for SDXL specific conditioning
                                # Format: [h, w, crop_top, crop_left, crop_h, crop_w]
                                # All images are resized to image_size x image_size with no cropping
                                time_ids = torch.tensor(
                                    [[image_size, image_size, 0, 0, image_size, image_size]],
                                    device=device
                                ).repeat(batch_size, 1)  # Repeat to match batch size
                                
                                # Prepare the added_cond_kwargs
                                added_cond_kwargs = {
                                    "text_embeds": text_embeds,
                                    "time_ids": time_ids
                                }
                        except Exception as e:
                            print(f"Error in text encoder forward pass: {e}")
                            print(traceback.format_exc())
                            raise  # Re-raise to be caught by the outer try/except
                                
                    except Exception as e:
                        print(f"Error processing text: {e}")
                        print(traceback.format_exc())
                        # Create dummy encoder hidden states and cond_kwargs
                        batch_size = img_tensor.shape[0]
                        encoder_hidden_states = torch.zeros((batch_size, 77, 2048), device=device)
                        added_cond_kwargs = {
                            "text_embeds": torch.zeros((batch_size, 1280), device=device),
                            "time_ids": torch.zeros((batch_size, 6), device=device)
                        }
                else:
                    # Create dummy encoder hidden states with correct shape for SDXL
                    batch_size = img_tensor.shape[0]
                    encoder_hidden_states = torch.zeros((batch_size, 77, 2048), device=device)
                    
                    # Create dummy added condition kwargs
                    added_cond_kwargs = {
                        "text_embeds": torch.zeros((batch_size, 1280), device=device),
                        "time_ids": torch.zeros((batch_size, 6), device=device)
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
                
                # Ensure all tensors have the same dtype
                # Convert all tensors to float16 if using mixed precision
                if amp_enabled and mixed_precision == "fp16":
                    noisy_latent = noisy_latent.to(dtype=torch.float16)
                    encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)
                    added_cond_kwargs["text_embeds"] = added_cond_kwargs["text_embeds"].to(dtype=torch.float16)
                    added_cond_kwargs["time_ids"] = added_cond_kwargs["time_ids"].to(dtype=torch.float16)
                
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
                traceback_str = traceback.format_exc()
                print(f"Traceback: {traceback_str}")
                # Continue to next sample
                progress_bar.update(1)
                global_step += 1
                continue
            
            progress_bar.update(1)
            global_step += 1
            
            # Save checkpoints
            if global_step % save_steps == 0 or global_step == max_train_steps:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save the PEFT model
                unet.save_pretrained(checkpoint_dir)
                
                # Save adapter config as JSON for compatibility with other tools
                with open(os.path.join(checkpoint_dir, "adapter_config.json"), "w") as f:
                    adapter_config_dict = {
                        "r": rank,
                        "lora_alpha": lora_alpha,
                        "target_modules": target_modules,
                        "lora_dropout": lora_dropout,
                        "bias": "none"
                    }
                    json.dump(adapter_config_dict, f, indent=4)
                
                # Also save as safetensors at checkpoints
                if model_name:
                    safetensors_path = os.path.join(checkpoint_dir, f"{model_name}.safetensors")
                    metadata = {
                        "model_type": "lora",
                        "base_model": base_model,
                        "rank": str(rank),
                        "alpha": str(lora_alpha)
                    }
                    if lora_concept:
                        metadata["concept"] = lora_concept
                    
                    save_as_safetensors(unet, safetensors_path, metadata)
                
                print(f"✅ Checkpoint saved at step {global_step}")
            
            # Clean up CUDA cache periodically
            if step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if global_step >= max_train_steps:
                break
        
    # Final checkpoint
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    # Save the PEFT model
    unet.save_pretrained(final_checkpoint_dir)
    
    # Save adapter config as JSON for compatibility with other tools
    with open(os.path.join(final_checkpoint_dir, "adapter_config.json"), "w") as f:
        adapter_config_dict = {
            "r": rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "lora_dropout": lora_dropout,
            "bias": "none"
        }
        json.dump(adapter_config_dict, f, indent=4)
    
    # Save model information
    with open(os.path.join(output_dir, "lora_info.json"), 'w') as f:
        info = {
            "base_model": base_model,
            "rank": rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "image_size": image_size,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "training_steps": global_step
        }
        if lora_concept:
            info["concept"] = lora_concept
        json.dump(info, f, indent=2)
    
    # Save as safetensors
    if model_name:
        safetensors_path = os.path.join(output_dir, f"{model_name}.safetensors")
        metadata = {
            "model_type": "lora",
            "base_model": base_model,
            "rank": str(rank),
            "alpha": str(lora_alpha)
        }
        if lora_concept:
            metadata["concept"] = lora_concept
        
        save_as_safetensors(unet, safetensors_path, metadata)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {final_checkpoint_dir}")
    print(f"Safetensors model saved to: {safetensors_path}")
    print("Model information saved to lora_info.json")
    
    return final_checkpoint_dir

def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🚀 SIMPLIFIED SDXL LORA TRAINING 🚀".center(60))
    print("=" * 60 + "\n")
    
    # Check for multiple datasets
    images_dirs = args.images_dir.split(",")
    captions_files = args.captions_file.split(",") if args.captions_file else [None] * len(images_dirs)
    output_dirs = args.output_dir.split(",") if "," in args.output_dir else [f"{args.output_dir}_{i}" for i in range(len(images_dirs))]
    model_names = args.model_name.split(",") if args.model_name else [f"lora_{Path(img_dir).name}" for img_dir in images_dirs]
    lora_concepts = args.lora_concept.split(",") if args.lora_concept else [None] * len(images_dirs)
    
    # Ensure all lists have the same length
    if len(captions_files) < len(images_dirs):
        captions_files.extend([None] * (len(images_dirs) - len(captions_files)))
    if len(output_dirs) < len(images_dirs):
        output_dirs.extend([f"{args.output_dir}_{i}" for i in range(len(output_dirs), len(images_dirs))])
    if len(model_names) < len(images_dirs):
        model_names.extend([f"lora_{Path(img_dir).name}" for img_dir in images_dirs[len(model_names):]])
    if len(lora_concepts) < len(images_dirs):
        lora_concepts.extend([None] * (len(images_dirs) - len(lora_concepts)))
    
    # Train each dataset separately
    for i, (images_dir, captions_file, output_dir, model_name, lora_concept) in enumerate(
        zip(images_dirs, captions_files, output_dirs, model_names, lora_concepts)
    ):
        print(f"\n\n{'#'*80}\nTraining dataset {i+1}/{len(images_dirs)}: {images_dir}\n{'#'*80}\n")
        
        # Strip whitespace
        images_dir = images_dir.strip()
        if captions_file:
            captions_file = captions_file.strip()
        output_dir = output_dir.strip()
        model_name = model_name.strip()
        if lora_concept:
            lora_concept = lora_concept.strip()
        
        train_lora(
            base_model=args.base_model,
            images_dir=images_dir,
            captions_file=captions_file if captions_file else os.path.join(images_dir, "captions.json"),
            output_dir=output_dir,
            model_name=model_name,
            lora_concept=lora_concept,
            num_train_epochs=args.num_train_epochs,
            train_batch_size=args.train_batch_size,
            max_train_steps=args.max_train_steps,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            save_steps=args.save_steps,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            mixed_precision=args.mixed_precision,
            gradient_checkpointing=args.gradient_checkpointing,
            enable_xformers=args.enable_xformers
        )

if __name__ == "__main__":
    main() 