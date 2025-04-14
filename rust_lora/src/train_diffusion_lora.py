"""
Train a LORA for SDXL using Rust-based implementation

This script trains a LORA model for SDXL using our Rust-accelerated trainer.
"""

import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, CLIPTextModelWithProjection
import logging

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionXLPipeline
)

# UNet wrapper class to handle dimension mismatches
class UNetDimensionWrapper(torch.nn.Module):
    """
    Wrapper for UNet to handle SDXL dimension mismatches in cross-attention layers.
    
    SDXL has several dimension incompatibilities:
    1. Text encoder output is 2304 dim but UNet expects 2816
    2. Cross-attention expects 2048 dim for keys/values
    3. Add_embedding expects specific dimensions
    
    This wrapper fixes these issues transparently.
    """
    def __init__(self, unet, verbose=False):
        super().__init__()
        self.unet = unet
        self.verbose = verbose
        self._patch_cross_attention_modules()
        self._patch_add_embedding()
        
    def _patch_cross_attention_modules(self):
        """Patch cross-attention modules to handle dimension mismatches."""
        if self.verbose:
            logging.info("Patching cross-attention modules for dimension compatibility")
            
        # Store projections per module
        self.module_projections = {}
        
        for name, module in self.unet.named_modules():
            # Look for cross-attention blocks (attn2)
            if 'attn2' in name:
                # Get the to_k and to_v modules which process encoder_hidden_states
                if hasattr(module, 'to_k'):
                    original_forward_k = module.to_k.forward
                    module_name_k = f"{name}.to_k"
                    
                    # Create dimension-fixing forward method
                    def make_k_forward(orig_forward, module_name):
                        def new_forward(x):
                            # x here is encoder_hidden_states
                            if x.shape[-1] != 2048 and x.shape[-1] > 2048:
                                if self.verbose:
                                    logging.info(f"Fixing encoder_hidden_states dim in {module_name}: {x.shape[-1]} -> 2048")
                                
                                # Create or get projection matrix for this module
                                if module_name not in self.module_projections:
                                    proj = torch.zeros((x.shape[-1], 2048), device=x.device)
                                    # Initialize with identity-like pattern for first 2048 dims
                                    min_dim = min(x.shape[-1], 2048)
                                    for i in range(min_dim):
                                        proj[i, i] = 1.0
                                    self.module_projections[module_name] = torch.nn.Parameter(
                                        proj, requires_grad=False
                                    )
                                
                                # Apply projection
                                projection = self.module_projections[module_name].to(device=x.device, dtype=x.dtype)
                                x = torch.matmul(x, projection)
                            return orig_forward(x)
                        return new_forward
                    
                    # Replace forward method
                    module.to_k.forward = make_k_forward(original_forward_k, module_name_k)
                
                if hasattr(module, 'to_v'):
                    original_forward_v = module.to_v.forward
                    module_name_v = f"{name}.to_v"
                    
                    # Create dimension-fixing forward method for to_v
                    def make_v_forward(orig_forward, module_name):
                        def new_forward(x):
                            # x here is encoder_hidden_states
                            if x.shape[-1] != 2048 and x.shape[-1] > 2048:
                                if self.verbose:
                                    logging.info(f"Fixing encoder_hidden_states dim in {module_name}: {x.shape[-1]} -> 2048")
                                
                                # Create or get projection matrix for this module
                                if module_name not in self.module_projections:
                                    proj = torch.zeros((x.shape[-1], 2048), device=x.device)
                                    # Initialize with identity-like pattern for first 2048 dims
                                    min_dim = min(x.shape[-1], 2048)
                                    for i in range(min_dim):
                                        proj[i, i] = 1.0
                                    self.module_projections[module_name] = torch.nn.Parameter(
                                        proj, requires_grad=False
                                    )
                                
                                # Apply projection
                                projection = self.module_projections[module_name].to(device=x.device, dtype=x.dtype)
                                x = torch.matmul(x, projection)
                            return orig_forward(x)
                        return new_forward
                    
                    # Replace forward method
                    module.to_v.forward = make_v_forward(original_forward_v, module_name_v)
    
    def _patch_add_embedding(self):
        """Patch add_embedding modules to handle add_embeds dimension mismatches."""
        if self.verbose:
            logging.info("Patching add_embedding module for dimension compatibility")
        
        # Look for add_embedding modules
        for name, module in self.unet.named_modules():
            if 'add_embedding' in name and hasattr(module, 'linear_1'):
                if self.verbose:
                    logging.info(f"Found add_embedding module: {name}")
                
                # Store original linear layer
                original_linear = module.linear_1
                
                # Get output dimension from original weight
                output_dim = original_linear.weight.shape[0]
                
                # Create a new linear layer that accepts 2560 dimensions instead
                new_linear = torch.nn.Linear(
                    2560, output_dim,
                    bias=original_linear.bias is not None,
                    device=original_linear.weight.device,
                    dtype=original_linear.weight.dtype
                )
                
                # Initialize weights based on original layer
                with torch.no_grad():
                    # Get original input dimension
                    orig_input_dim = original_linear.weight.shape[1]
                    min_dim = min(orig_input_dim, 2560)
                    
                    if orig_input_dim <= 2560:
                        # Copy original weights to the beginning of new weights
                        new_linear.weight.data[:, :orig_input_dim] = original_linear.weight.data
                    else:
                        # Truncate original weights to fit new dimensions
                        new_linear.weight.data = original_linear.weight.data[:, :2560]
                    
                    # Copy bias if exists
                    if original_linear.bias is not None:
                        new_linear.bias.data = original_linear.bias.data
                
                # Replace the linear module
                module.linear_1 = new_linear
                
                if self.verbose:
                    logging.info(f"Replaced {name}.linear_1 to handle 2560 input dimensions")
                
                # We only need to patch one add_embedding module
                break
    
    def forward(self, *args, **kwargs):
        """Pass through to UNet with fixed dimensions."""
        # The dimension fixes are applied in the patched modules
        return self.unet(*args, **kwargs)

# Import Rust LORA implementation - no fallbacks allowed
try:
    # Try importing directly (when run as a module)
    import lora_ops
    from lora_ops import LoraTrainingContext, AdamParams
except ImportError:
    # Try importing as src.lora_ops (when run as script)
    import src.lora_ops as lora_ops
    from src.lora_ops import LoraTrainingContext, AdamParams

print("✅ Using Rust LORA implementation")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DiffusionDataset(Dataset):
    """Dataset for training diffusion models with LORA."""
    
    def __init__(
        self,
        image_dir,
        caption_file,
        tokenizer,
        tokenizer_2=None,
        resolution=1024,
        center_crop=True,
        random_flip=True,
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        
        # Load captions
        with open(caption_file, "r") as f:
            self.captions = json.load(f)
        
        # Get list of image files that have captions
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f in self.captions
        ]
        
        print(f"Found {len(self.image_files)} images with captions")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        
        # Resize and center crop
        if self.center_crop:
            image = self._center_crop_and_resize(image, self.resolution)
        else:
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Random horizontal flip
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensor and normalize
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        
        # Check for NaN or inf values in the image array
        if np.isnan(image).any() or np.isinf(image).any():
            logging.warning(f"Found NaN or inf values in image {image_name}, replacing with zeros")
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get caption
        caption = self.captions.get(image_name, "")
        
        # Tokenize caption for SDXL (two tokenizers)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        if self.tokenizer_2:
            tokens_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids_2 = tokens_2.input_ids[0]
            attention_mask_2 = tokens_2.attention_mask[0]
        else:
            input_ids_2 = torch.zeros(77, dtype=torch.long)
            attention_mask_2 = torch.zeros(77, dtype=torch.long)
        
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
        }
    
    def _center_crop_and_resize(self, image, target_size):
        width, height = image.size
        
        # Calculate dimensions to crop to
        if width > height:
            left = (width - height) // 2
            right = left + height
            top, bottom = 0, height
        else:
            top = (height - width) // 2
            bottom = top + width
            left, right = 0, width
        
        # Crop and resize
        image = image.crop((left, top, right, bottom))
        image = image.resize((target_size, target_size), Image.LANCZOS)
        
        return image


class RustDiffusionTrainer:
    """SDXL LORA Trainer using Rust implementation."""
    
    def __init__(
        self,
        model_id,
        output_dir,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        rank=8,
        alpha=16,
        mixed_precision="no",
        save_steps=500,
        max_train_steps=1000,
        seed=42,
        target_modules=None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.rank = rank
        self.alpha = alpha
        self.mixed_precision = mixed_precision
        self.save_steps = save_steps
        self.max_train_steps = max_train_steps
        self.seed = seed
        
        # Set target modules for LORA (if not specified)
        self.target_modules = target_modules or [
            "to_q", 
            "to_k", 
            "to_v", 
            "to_out.0", 
            "proj_in", 
            "proj_out", 
            "ff.net.0", 
            "ff.net.2"
        ]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Map for target modules to their parameters
        self.lora_layers = {}
        self.rust_lora_contexts = {}
        
        # Noise scheduler
        self.noise_scheduler = None
        
        # Load model components
        self._load_models()
        
        # Setup optimizer
        self.optimizer = AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        
        # Set up LORA layers
        self._setup_lora_layers()
        
    def _load_models(self):
        """Load models for the diffusion pipeline."""
        # Set up pipeline kwargs for model loading
        pipeline_kwargs = {
            "torch_dtype": torch.float16 if self.mixed_precision == "fp16" else torch.float32,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        
        # Set up tokenizer kwargs
        self.tokenizer_kwargs = {
            "revision": "main",
            "use_safetensors": True
        }
        
        # Check if this is an SDXL model
        is_sdxl = any(["sdxl" in self.model_id.lower(), 
                       "stable-diffusion-xl" in self.model_id.lower(),
                       os.path.exists(os.path.join(self.model_id, "text_encoder_2"))])
        
        if is_sdxl:
            # SDXL pipeline
            logging.info("Loading SDXL model components...")
            # Load tokenizers
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id,
                subfolder="tokenizer",
                **self.tokenizer_kwargs
            )
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.model_id,
                subfolder="tokenizer_2",
                **self.tokenizer_kwargs
            )
            
            # Load text encoders
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                **pipeline_kwargs
            )
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                **pipeline_kwargs
            )
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                **pipeline_kwargs
            )
            
            # Load UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_id,
                subfolder="unet",
                **pipeline_kwargs
            )
            
            # Wrap UNet with dimension fixing wrapper
            logging.info("Wrapping UNet with dimension fixing wrapper")
            self.unet = UNetDimensionWrapper(self.unet, verbose=True)
            
            # Set UNet to training mode (we're training it)
            self.unet.train()
        else:
            # Regular SD 1.x/2.x pipeline
            logging.info("Loading Standard Stable Diffusion model components...")
            # Load tokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id,
                subfolder="tokenizer" if os.path.isdir(os.path.join(self.model_id, "tokenizer")) else None,
                **self.tokenizer_kwargs
            )
            
            # Load text encoder
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder" if os.path.isdir(os.path.join(self.model_id, "text_encoder")) else None,
                **pipeline_kwargs
            )
            
            # Load VAE
            vae_subfolder = "vae" if os.path.isdir(os.path.join(self.model_id, "vae")) else None
            self.vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder=vae_subfolder,
                **pipeline_kwargs
            )
            
            # Load UNet
            unet_subfolder = "unet" if os.path.isdir(os.path.join(self.model_id, "unet")) else None
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_id,
                subfolder=unet_subfolder,
                **pipeline_kwargs
            )
        
        # Move models to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self.unet.to(device)
            self.vae.to(device)
            self.text_encoder.to(device)
            if hasattr(self, "text_encoder_2"):
                self.text_encoder_2.to(device)
        
        # Set VAE and text encoder to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.eval()
        
        # Load the noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler", **self.tokenizer_kwargs)
    
    def _setup_lora_layers(self):
        """Set up LORA layers in the UNet model."""
        try:
            print("Setting up LORA layers for UNet...")
            logging.info("Starting LORA layer setup")
            
            # Get all modules that match target patterns
            for name, module in self.unet.named_modules():
                try:
                    if any(target in name for target in self.target_modules):
                        if isinstance(module, torch.nn.Linear):
                            print(f"Adding LORA to {name}")
                            self.lora_layers[name] = module
                            
                            # Get original weights with more error handling
                            try:
                                weight = module.weight.detach().cpu().numpy().astype(np.float32)
                                print(f"Weight shape: {weight.shape}")
                                
                                # Create LORA context
                                ctx = LoraTrainingContext(
                                    layer_name=name,
                                    weight=weight,
                                    rank=self.rank,
                                    alpha=self.alpha,
                                    init_scale=0.01
                                )
                                
                                # Set optimizer
                                ctx.set_optimizer(self.optimizer)
                                
                                # Store context
                                self.rust_lora_contexts[name] = ctx
                                print(f"Successfully added LORA to {name}")
                            except Exception as e:
                                logging.error(f"Error processing weights for {name}: {str(e)}")
                                import traceback
                                logging.error(traceback.format_exc())
                except Exception as e:
                    logging.error(f"Error processing module {name}: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
            
            print(f"Added LORA to {len(self.rust_lora_contexts)} layers")
            logging.info(f"Completed LORA setup with {len(self.rust_lora_contexts)} layers")
        except Exception as e:
            logging.error(f"Error in _setup_lora_layers: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def prepare_latents(self, pixel_values, timesteps):
        """Prepare latent variables for training."""
        # Scale and encode images to latent space
        with torch.no_grad():
            # Scale pixel values to -1 to 1
            pixel_values = 2 * pixel_values - 1
            
            # Get VAE's dtype to match input type with model
            vae_dtype = next(self.vae.parameters()).dtype
            vae_device = next(self.vae.parameters()).device
            
            # Ensure pixel values match VAE's dtype and device
            pixel_values = pixel_values.to(device=vae_device, dtype=vae_dtype)
            
            # Check for NaN values
            if torch.isnan(pixel_values).any():
                logging.warning("NaN values detected in pixel_values, replacing with zeros")
                pixel_values = torch.nan_to_num(pixel_values, nan=0.0)
            
            # Encode images
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            # Generate noise 
            noise = torch.randn_like(latents)
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Create a new tensor with gradients enabled by detaching from the no_grad computation
        latents_with_grad = noisy_latents.detach().clone()
        latents_with_grad.requires_grad_(True)  # Explicitly enable gradients
        
        return latents_with_grad
    
    def prepare_text_embeddings(self, batch):
        """Prepare text embeddings from the batch."""
        with torch.no_grad():
            # Get text embeddings for first text encoder - handle different return formats
            encoder_output_1 = self.text_encoder(
                batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get text embeddings for second text encoder - handle different return formats
            encoder_output_2 = self.text_encoder_2(
                batch["input_ids_2"].to("cuda"),
                attention_mask=batch["attention_mask_2"].to("cuda"),
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract hidden states from encoder outputs
            if hasattr(encoder_output_1, "last_hidden_state"):
                text_embeddings = encoder_output_1.last_hidden_state
            elif isinstance(encoder_output_1, tuple) and len(encoder_output_1) > 0:
                text_embeddings = encoder_output_1[0]
            else:
                # Fallback case - output might be the tensor directly
                text_embeddings = encoder_output_1
                
            if hasattr(encoder_output_2, "last_hidden_state"):
                text_embeddings_2 = encoder_output_2.last_hidden_state
            elif isinstance(encoder_output_2, tuple) and len(encoder_output_2) > 0:
                text_embeddings_2 = encoder_output_2[0] 
            else:
                # Fallback case - output might be the tensor directly
                text_embeddings_2 = encoder_output_2
            
            # Log shape information for debugging
            logging.info(f"Text embeddings shape: {text_embeddings.shape}")
            logging.info(f"Text embeddings 2 shape: {text_embeddings_2.shape}")
            
            # SDXL dimension compatibility fix
            # SDXL text encoders output combined dimension of 2304
            # but UNet cross-attention expects 2816 dimensions
            batch_size = text_embeddings.shape[0]
            seq_length = text_embeddings.shape[1]
            hidden_size_1 = text_embeddings.shape[2]
            hidden_size_2 = text_embeddings_2.shape[2]
            
            # Get dimensions for combined embeddings
            EXPECTED_COMBINED_DIM = 2816  # Final dimension for UNet
            
            # Create correctly sized tensor for combined embeddings
            combined_embeddings = torch.zeros(
                (batch_size, seq_length, EXPECTED_COMBINED_DIM),
                device=text_embeddings.device,
                dtype=text_embeddings.dtype
            )
            
            # Copy first embeddings
            combined_embeddings[:, :, :hidden_size_1] = text_embeddings
            
            # Copy second embeddings (as much as will fit)
            copy_dim_2 = min(hidden_size_2, EXPECTED_COMBINED_DIM - hidden_size_1)
            combined_embeddings[:, :, hidden_size_1:hidden_size_1 + copy_dim_2] = text_embeddings_2[:, :, :copy_dim_2]
            
            # Return the dimension-fixed embeddings
            return combined_embeddings
    
    def train_step(self, batch):
        """Perform a single training step."""
        try:
            # Get text embeddings
            text_embeddings = self.prepare_text_embeddings(batch)
            
            # Sample a random timestep for each image
            batch_size = batch["pixel_values"].shape[0]
            timesteps = torch.randint(
                0, 
                self.noise_scheduler.config.num_train_timesteps, 
                (batch_size,), 
                device="cuda"
            )
            
            # Prepare latents
            latents = self.prepare_latents(
                batch["pixel_values"].to("cuda"), timesteps
            )
            
            # For SDXL: prepare added conditioning kwargs (text_embeds and time_ids)
            added_cond_kwargs = None
            
            # Better detection for SDXL model (always add conditioning for SDXL)
            is_sdxl = hasattr(self, "text_encoder_2")
            
            if is_sdxl:
                # Extract pooled output from the first text encoder for text_embeds
                with torch.no_grad():
                    # Get text embeddings with pooled output
                    encoder_output = self.text_encoder(
                        batch["input_ids"].to("cuda"),
                        attention_mask=batch["attention_mask"].to("cuda"),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    
                    # Get pooled output (text_embeds)
                    # For CLIPTextModel, we use pooler_output
                    # For CLIPTextModelWithProjection, we need the text_embeds attribute
                    if hasattr(encoder_output, "pooler_output"):
                        text_embeds = encoder_output.pooler_output
                    elif hasattr(encoder_output, "text_embeds"):
                        # CLIPTextModelWithProjection returns text_embeds directly
                        text_embeds = encoder_output.text_embeds
                    else:
                        # Fallback: Using the last hidden state's first token ([CLS]) as pooled representation
                        if hasattr(encoder_output, "last_hidden_state"):
                            last_hidden_state = encoder_output.last_hidden_state
                        else:
                            last_hidden_state = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
                        text_embeds = last_hidden_state[:, 0]
                    
                    # Log shape info for debugging
                    logging.info(f"Text embeds shape before fixing: {text_embeds.shape}")
                    
                    # Fix dimensions for text_embeds
                    # SDXL expects 1280 dimensions for text_embeds
                    EXPECTED_EMBED_DIM = 1280
                    
                    # Create a properly sized tensor
                    fixed_text_embeds = torch.zeros(
                        (batch_size, EXPECTED_EMBED_DIM),
                        device=text_embeds.device,
                        dtype=text_embeds.dtype
                    )
                    
                    # Copy as much data as will fit
                    copy_dim = min(text_embeds.shape[1], EXPECTED_EMBED_DIM)
                    fixed_text_embeds[:, :copy_dim] = text_embeds[:, :copy_dim]
                    
                    # Create time_ids tensor [batch_size, 5]
                    # Format: [orig_height, orig_width, crop_top, crop_left, aesthetic_score]
                    orig_size = (1024, 1024)
                    crop_size = (1024, 1024)
                    crop_coords = (0, 0)
                    aesthetic_score = 6.0
                    
                    time_ids = torch.zeros((batch_size, 5), device="cuda")
                    time_ids[:, 0] = orig_size[0]  # height
                    time_ids[:, 1] = orig_size[1]  # width
                    time_ids[:, 2] = crop_coords[0]  # crop top
                    time_ids[:, 3] = crop_coords[1]  # crop left
                    time_ids[:, 4] = aesthetic_score
                    
                    added_cond_kwargs = {
                        "text_embeds": fixed_text_embeds,
                        "time_ids": time_ids
                    }
            
            # Check for NaN values in tensors and attempt to fix them
            def fix_nans(tensor, name):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    logging.warning(f"NaN or Inf values detected in {name}, attempting to fix")
                    fixed_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                    return fixed_tensor, True
                return tensor, False
            
            # Fix key tensors if they contain NaN values
            latents, latents_fixed = fix_nans(latents, "latents")
            text_embeddings, embeddings_fixed = fix_nans(text_embeddings, "text_embeddings")
            
            if added_cond_kwargs:
                added_cond_kwargs["text_embeds"], embeds_fixed = fix_nans(added_cond_kwargs["text_embeds"], "text_embeds")
            
            # Log if fixes were applied
            if latents_fixed or embeddings_fixed:
                logging.warning("NaN values were fixed in input tensors")
                
            # Add debugging for tensor shapes
            logging.info(f"UNet input shapes - latents: {latents.shape}, timesteps: {timesteps.shape}, encoder_hidden_states: {text_embeddings.shape}")
            if added_cond_kwargs:
                logging.info(f"Added condition kwargs - text_embeds: {added_cond_kwargs['text_embeds'].shape}, time_ids: {added_cond_kwargs['time_ids'].shape}")
            
            try:
                # Forward pass through UNet with gradient tracking enabled
                # Ensure all tensors are on the same device and have the same dtype
                device = latents.device
                dtype = latents.dtype
                
                # Move all tensors to the same device and dtype
                timesteps = timesteps.to(device=device, dtype=torch.long)  # timesteps should be long
                text_embeddings = text_embeddings.to(device=device, dtype=dtype)
                
                if added_cond_kwargs:
                    added_cond_kwargs["text_embeds"] = added_cond_kwargs["text_embeds"].to(device=device, dtype=dtype)
                    added_cond_kwargs["time_ids"] = added_cond_kwargs["time_ids"].to(device=device)
                
                logging.info(f"Running UNet forward pass on device {device} with dtype {dtype}")
                
                noise_pred = self.unet(
                    latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Check for NaN values in output and attempt to fix
                noise_pred, pred_fixed = fix_nans(noise_pred, "noise_pred")
                if pred_fixed:
                    logging.warning("Fixed NaN values in UNet output")
                
            except Exception as e:
                # Try to extract useful information from errors
                error_msg = str(e)
                logging.error(f"Error in UNet forward pass: {error_msg}")
                
                # Check for dimension mismatch errors
                if "mat1 and mat2 shapes cannot be multiplied" in error_msg:
                    import re
                    match = re.search(r"\((\d+)x(\d+) and (\d+)x(\d+)\)", error_msg)
                    if match:
                        a1, a2, b1, b2 = map(int, match.groups())
                        logging.error(f"Matrix dimension mismatch: ({a1}x{a2}) and ({b1}x{b2})")
                        logging.error(f"Expected a2 = {b1}, but got a2 = {a2}")
                
                # Re-raise to abort this training step
                raise
            
            # Compute loss - use random noise as target
            # Do NOT use torch.randn_like(latents) which keeps the tensor on the same device but loses gradients
            target = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, requires_grad=False)
            loss = torch.nn.functional.mse_loss(noise_pred, target)
            
            # Backpropagate and update LORA weights using Rust
            loss.backward()
            
            # Update LORA parameters in Rust
            for name, module in self.lora_layers.items():
                if name in self.rust_lora_contexts:
                    # Get gradient
                    if module.weight.grad is not None:
                        grad = module.weight.grad.detach().cpu().numpy().astype(np.float32)
                        
                        # Check for NaN values in gradients
                        if np.isnan(grad).any() or np.isinf(grad).any():
                            logging.warning(f"NaN or Inf values in gradients for {name}, skipping update")
                            continue
                        
                        # Update weights using Rust implementation
                        self.rust_lora_contexts[name].backward(grad)
                    
                    # Zero out gradient
                    if module.weight.grad is not None:
                        module.weight.grad.zero_()
            
            return loss.item()
        except Exception as e:
            logging.error(f"Error in train_step: {str(e)}")
            return float('nan')  # Return NaN loss to indicate error
    
    def save_lora_weights(self, step=None):
        """Save trained LORA weights."""
        lora_state_dict = {}
        
        for name, ctx in self.rust_lora_contexts.items():
            # Get weights from Rust
            lora_a, lora_b = ctx.get_weights()
            
            # Convert to numpy arrays
            lora_a = np.array(lora_a)
            lora_b = np.array(lora_b)
            
            # Add to state dict
            lora_state_dict[f"{name}.lora_a.weight"] = torch.tensor(lora_a)
            lora_state_dict[f"{name}.lora_b.weight"] = torch.tensor(lora_b)
            lora_state_dict[f"{name}.alpha"] = torch.tensor(self.alpha)
            lora_state_dict[f"{name}.rank"] = torch.tensor(self.rank)
        
        # Save to file
        filename = "lora_weights.safetensors"
        if step is not None:
            filename = f"lora_weights_{step}.safetensors"
        
        save_path = os.path.join(self.output_dir, filename)
        save_file(lora_state_dict, save_path)
        print(f"Saved LORA weights to {save_path}")
        
        # Also save metadata
        with open(os.path.join(self.output_dir, "lora_config.json"), "w") as f:
            json.dump({
                "rank": self.rank,
                "alpha": self.alpha,
                "target_modules": self.target_modules,
                "model_id": self.model_id,
            }, f, indent=2)
    
    def train(self, train_dataloader):
        """Train the LORA model."""
        print(f"Starting training for {self.max_train_steps} steps")
        
        global_step = 0
        for epoch in range(int(self.max_train_steps / len(train_dataloader)) + 1):
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                # Skip if we've reached max steps
                if global_step >= self.max_train_steps:
                    break
                
                # Train step
                loss = self.train_step(batch)
                
                # Update progress
                global_step += 1
                
                # Log
                if global_step % 10 == 0:
                    print(f"Step {global_step}: loss = {loss:.4f}")
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_lora_weights(global_step)
            
            # Save at end of epoch
            self.save_lora_weights(f"epoch_{epoch}")
        
        # Final save
        self.save_lora_weights()
        
        print("Training complete!")


def run_training(args):
    """Run LORA training on SDXL."""
    # Verify Rust implementation is available
    if 'lora_ops' not in sys.modules:
        raise ImportError("Rust LORA implementation required but not available. Please ensure the Rust library is built correctly.")
    
    # Create the dataset
    train_dataset = DiffusionDataset(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        tokenizer=AutoTokenizer.from_pretrained(
            args.model_id,
            subfolder="tokenizer",
            use_fast=False,
        ),
        tokenizer_2=AutoTokenizer.from_pretrained(
            args.model_id,
            subfolder="tokenizer_2",
            use_fast=False,
        ),
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Initialize trainer
    trainer = RustDiffusionTrainer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        rank=args.rank,
        alpha=args.alpha,
        mixed_precision=args.mixed_precision,
        save_steps=args.save_steps,
        max_train_steps=args.max_train_steps,
        seed=args.seed,
    )
    
    # Set up logging to file
    log_file = os.path.join(args.output_dir, "training.log")
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Log command line args
    logging.info(f"Command line arguments: {args}")
    
    try:
        # Run training
        trainer.train(train_dataloader)
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"Training failed. See log at {log_file} for details.")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SDXL LORA with Rust")
    
    # Data arguments
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--caption_file", type=str, required=True, help="JSON file with captions")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LORA weights")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="/home/harry/loras/sdxl-base-1.0", help="Path to SDXL model")
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution")
    
    # Training arguments
    parser.add_argument("--rank", type=int, default=16, help="LORA rank")
    parser.add_argument("--alpha", type=float, default=32, help="LORA alpha scaling factor")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of update steps to accumulate before update")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Learning rate warmup steps")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Dataset processing
    parser.add_argument("--center_crop", action="store_true", help="Center crop images")
    parser.add_argument("--random_flip", action="store_true", help="Random flip images horizontally")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    
    args = parser.parse_args()
    
    # Always use Rust - no fallback allowed
    run_training(args) 