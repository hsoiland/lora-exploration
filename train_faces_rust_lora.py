import os
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from safetensors.torch import save_file
from src.rust_lora_self_attn import find_self_attention_modules, apply_lora_to_self_attention, apply_lora_rust

class FaceDataset(Dataset):
    """Dataset for training on face images"""
    
    def __init__(self, image_dir, size=512, tokenizer=None):
        """Initialize dataset from directory of images"""
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Image transformations pipeline
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])
        
        # Store tokenizer for optional text conditioning
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Optional caption is just the filename without extension
        caption = image_path.stem
        
        return {
            "image": image,
            "caption": caption
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA using face images with Rust backend")
    parser.add_argument("--base_model", type=str, default="sdxl-base-1.0", 
                      help="Path to base model")
    parser.add_argument("--images_dir", type=str, default="gina_face_cropped", 
                      help="Path to face images")
    parser.add_argument("--output_dir", type=str, default="gina_rust_lora_output", 
                      help="Directory to save LoRA weights and samples")
    parser.add_argument("--lora_name", type=str, default="gina_faces_rust_lora", 
                      help="Name for the trained LoRA")
    parser.add_argument("--rank", type=int, default=8, 
                      help="Rank of LoRA matrices")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                      help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1, 
                      help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                      help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=100, 
                      help="Save checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--image_size", type=int, default=512,
                      help="Size of training images")
    parser.add_argument("--use_rust", type=bool, default=True,
                      help="Use Rust backend for LoRA application")
    return parser.parse_args()

# Initialize LoRA weights for self-attention modules
def train_lora_for_self_attention(unet, rank=4, target_modules=None, use_rust=True):
    """Initialize LoRA weights for self-attention modules"""
    lora_weights = {}
    
    print(f"Initializing LoRA weights for {len(target_modules)} target modules with rank {rank}")
    
    for module_name in target_modules:
        module = None
        for name, mod in unet.named_modules():
            if name == module_name:
                module = mod
                break
                
        if module is not None and hasattr(module, 'weight'):
            # Initialize LoRA matrices
            in_features = module.weight.shape[1]
            out_features = module.weight.shape[0]
            
            print(f"Creating LoRA for {module_name}: shape {out_features}x{in_features}, rank {rank}")
            
            # Initialize with small random values
            lora_down = torch.randn(rank, in_features) * 0.02
            lora_up = torch.randn(out_features, rank) * 0.02
            
            # Store weights
            lora_weights[f"{module_name}.lora_down.weight"] = lora_down
            lora_weights[f"{module_name}.lora_up.weight"] = lora_up
    
    return lora_weights

def main():
    try:
        args = parse_args()
        
        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load base model
        print(f"Loading base model from {args.base_model}...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,  # Use float32 for training
            use_safetensors=True
        )
        
        # Set up UNet for training
        unet = pipeline.unet
        vae = pipeline.vae.to(dtype=torch.float32)  # Keep VAE in float32 for encoding stability
        
        # Get noise scheduler for training
        noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        unet.to(device)
        vae.to(device)
        
        # Find self-attention modules
        self_attn_modules = find_self_attention_modules(unet)
        print(f"Found {len(self_attn_modules)} self-attention modules in UNet")
        
        # Create initial LoRA weights for self-attention modules
        lora_weights = train_lora_for_self_attention(
            unet=unet,
            rank=args.rank,
            target_modules=self_attn_modules,
            use_rust=args.use_rust
        )
        
        # Create dataset
        dataset = FaceDataset(
            image_dir=args.images_dir,
            size=args.image_size,
            tokenizer=pipeline.tokenizer
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Set up optimizer
        # Extract trainable parameters from lora_weights
        trainable_params = []
        lora_params = {}
        
        for key, weight in lora_weights.items():
            param = torch.nn.Parameter(weight.clone().to(device), requires_grad=True)
            trainable_params.append(param)
            lora_params[key] = param
        
        print(f"Setting up optimizer with {len(trainable_params)} trainable parameters")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        
        # Training loop
        unet.train()
        vae.eval()
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
        total_steps = num_update_steps_per_epoch * args.num_train_epochs
        
        print(f"Starting training for {args.num_train_epochs} epochs ({total_steps} steps)...")
        progress_bar = tqdm(range(total_steps))
        global_step = 0
        
        for epoch in range(args.num_train_epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Get images from batch
                images = batch["image"].to(device)
                
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # Add noise to latents
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Create conditioning (using empty conditioning for simplicity)
                batch_size = images.shape[0]
                encoder_hidden_states = torch.zeros(
                    (batch_size, 77, 2048),  # SDXL text encoder dim
                    device=device
                )
                
                # Prepare additional conditioning kwargs needed by SDXL
                added_cond_kwargs = {
                    "text_embeds": torch.zeros((batch_size, 1280), device=device),
                    "time_ids": torch.zeros((batch_size, 6), device=device)
                }
                
                # Apply LoRA weights to unet
                with torch.no_grad():
                    # Temporarily inject our LoRA weights
                    for module_name in self_attn_modules:
                        module = None
                        for name, mod in unet.named_modules():
                            if name == module_name:
                                module = mod
                                break
                        
                        if module is not None and hasattr(module, 'weight'):
                            # Get corresponding LoRA weights
                            lora_down_key = f"{module_name}.lora_down.weight"
                            lora_up_key = f"{module_name}.lora_up.weight"
                            
                            if lora_down_key in lora_params and lora_up_key in lora_params:
                                lora_down = lora_params[lora_down_key]
                                lora_up = lora_params[lora_up_key]
                                
                                # Apply LoRA with Rust if enabled
                                if args.use_rust:
                                    try:
                                        updated_weight = apply_lora_rust(
                                            module.weight,
                                            lora_down,
                                            lora_up,
                                            1.0  # Use alpha=1.0 during training
                                        )
                                        module.weight.data = updated_weight
                                    except Exception as e:
                                        # Fall back to PyTorch if Rust fails
                                        print(f"⚠️ Rust failed for {module_name}: {e}")
                                        lora_delta = torch.matmul(lora_up, lora_down)
                                        module.weight.data = module.weight.data + lora_delta
                                else:
                                    # Use PyTorch implementation
                                    lora_delta = torch.matmul(lora_up, lora_down)
                                    module.weight.data = module.weight.data + lora_delta
                
                # Forward pass through unet
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                
                # Backward pass and optimize
                loss.backward()
                
                # Restore original weights
                with torch.no_grad():
                    for module_name in self_attn_modules:
                        module = None
                        for name, mod in unet.named_modules():
                            if name == module_name:
                                module = mod
                                break
                        
                        if module is not None and hasattr(module, 'weight'):
                            # Get corresponding LoRA weights
                            lora_down_key = f"{module_name}.lora_down.weight"
                            lora_up_key = f"{module_name}.lora_up.weight"
                            
                            if lora_down_key in lora_params and lora_up_key in lora_params:
                                lora_down = lora_params[lora_down_key]
                                lora_up = lora_params[lora_up_key]
                                
                                # Remove LoRA effect (restore original weights)
                                if args.use_rust:
                                    try:
                                        # Invert the LoRA effect by using negative alpha
                                        updated_weight = apply_lora_rust(
                                            module.weight,
                                            lora_down,
                                            lora_up,
                                            -1.0  # Negative alpha to remove effect
                                        )
                                        module.weight.data = updated_weight
                                    except Exception as e:
                                        # Fall back to PyTorch
                                        lora_delta = torch.matmul(lora_up, lora_down)
                                        module.weight.data = module.weight.data - lora_delta
                                else:
                                    # Use PyTorch
                                    lora_delta = torch.matmul(lora_up, lora_down)
                                    module.weight.data = module.weight.data - lora_delta
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}/{args.num_train_epochs} - Loss: {loss.item():.4f}")
                
                global_step += 1
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"{args.lora_name}_step_{global_step}.safetensors")
                    
                    # Save current LoRA weights
                    save_weights = {}
                    for key, param in lora_params.items():
                        save_weights[key] = param.detach().cpu()
                    
                    save_file(save_weights, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final LoRA weights
        final_lora_path = os.path.join(args.output_dir, f"{args.lora_name}.safetensors")
        
        # Prepare weights for saving
        final_weights = {}
        print(f"Preparing {len(lora_params)} LoRA parameters for saving...")
        
        for key, param in lora_params.items():
            # Process tensor using the recommended method from our unit test
            tensor = param.detach().cpu().to(torch.float32)
            print(f"Processing parameter {key}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            final_weights[key] = tensor
        
        # Save weights
        print(f"Saving LoRA weights to {final_lora_path}...")
        try:
            # Print keys for debugging
            print(f"Keys being saved: {list(final_weights.keys())}")
            save_file(final_weights, final_lora_path)
            print(f"✅ Saved final LoRA weights to {final_lora_path}")
        except Exception as e:
            print(f"❌ Error saving weights: {e}")
            # Try alternate saving method
            print("Attempting alternate saving method with PyTorch native format...")
            torch.save({key: tensor for key, tensor in final_weights.items()}, 
                      final_lora_path.replace('.safetensors', '.pt'))
            print(f"✅ Saved weights in PyTorch format instead")
            final_lora_path = final_lora_path.replace('.safetensors', '.pt')
        
        # Generate a test image with the trained LoRA
        print("Generating test image with trained LoRA...")
        
        # Create a new pipeline for inference
        inference_pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        
        # Apply trained LoRA using our Rust implementation
        apply_lora_to_self_attention(
            inference_pipeline.unet,
            final_lora_path,
            alpha=0.8,  # Use lower alpha for initial testing
            use_batch=True  # Use batch processing for inference
        )
        
        # Generate test image
        test_prompt = "A portrait of a woman with red hair, photorealistic, studio lighting, highly detailed"
        test_image = inference_pipeline(
            prompt=test_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
        
        # Save test image
        test_image_path = os.path.join(args.output_dir, "trained_lora_test.png")
        test_image.save(test_image_path)
        print(f"Saved test image to {test_image_path}")
        
        print("✅ Training complete!")
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main() 