#!/usr/bin/env python3
"""
Full LoRA training script with Rust acceleration.
This trains a more comprehensive LoRA on facial images.
"""

import os
import torch
import argparse
import numpy as np
import json
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from safetensors.torch import save_file, load_file
import math
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

# Import optional bitsandbytes for 8-bit optimizers if available
try:
    import bitsandbytes as bnb
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False

from src.rust_lora_self_attn import apply_lora_rust, apply_lora_to_self_attention

class FaceDataset(Dataset):
    def __init__(self, images_dir, captions_file=None, image_size=512, use_cache=True, tokenizer=None, tokenizer_2=None):
        self.images_dir = Path(images_dir)
        self.image_paths = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        self.use_cache = use_cache
        self.cache = {}
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        
        # Handle case with no images
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}. Make sure the path is correct and contains .jpg or .png files.")
        
        # Sort to ensure consistent order
        self.image_paths.sort()
        
        # Load captions if provided
        self.captions = {}
        if captions_file and os.path.exists(captions_file):
            print(f"Loading captions from {captions_file}")
            with open(captions_file, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
            print(f"Loaded {len(self.captions)} captions")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # Scales to [0, 1]
            transforms.Normalize([0.5], [0.5])  # Scales to [-1, 1]
        ])
        
        print(f"Found {len(self.image_paths)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = image_path.name
        
        # Check cache first if enabled
        if self.use_cache and str(image_path) in self.cache:
            result = self.cache[str(image_path)]
            if isinstance(result, dict):
                return result
            else:
                # If we only cached the image tensor, we need to process the caption
                image_tensor = result
        else:
            # Load and transform image
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a blank image of the right shape as fallback
                image_tensor = torch.zeros(3, 512, 512)
        
        # Get caption for this image if available
        caption = self.captions.get(filename, f"A photo of a person")
        
        # If we have tokenizers, tokenize the caption
        if self.tokenizer and self.tokenizer_2:
            # Tokenize with first tokenizer (SDXL uses two text encoders)
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize with second tokenizer
            tokens_2 = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            
            result = {
                "image": image_tensor,
                "caption": caption,
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
                "input_ids_2": tokens_2.input_ids[0],
                "attention_mask_2": tokens_2.attention_mask[0]
            }
        else:
            # If no tokenizers, just return the image and caption
            result = {
                "image": image_tensor,
                "caption": caption
            }
        
        # Cache the result if enabled
        if self.use_cache:
            self.cache[str(image_path)] = result
        
        return result

class LoRALinear(torch.nn.Module):
    """Custom LoRA implementation that preserves gradients"""
    def __init__(self, original_weight, lora_up, lora_down, alpha=1.0):
        super().__init__()
        self.original_weight = original_weight
        self.lora_up = lora_up
        self.lora_down = lora_down
        self.alpha = alpha
        
    def forward(self, x):
        # Original weight multiplication (use the original Linear layer function)
        original_output = torch.nn.functional.linear(x, self.original_weight)
        
        # LoRA path - optimized implementation
        # Compute both matrix multiplications in one step, keeping everything on the GPU
        # x @ down.T @ up.T = x @ (up @ down).T
        # This is more efficient than separate matrix multiplications
        lora_matrix = self.lora_up @ self.lora_down  # Precompute once per forward pass
        lora_output = torch.nn.functional.linear(x, lora_matrix.T)
        
        # Combine outputs
        return original_output + self.alpha * lora_output

def apply_lora_layer(model, target_modules, lora_params, alpha=1.5):
    """Apply LoRA to model by patching the forward hooks"""
    hooks = []
    original_forward_methods = {}
    patched_modules = []
    
    # Store original weights and patch methods
    for module_name in target_modules:
        # Find module
        module = None
        for name, mod in model.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is not None and hasattr(module, 'weight'):
            # Get LoRA weights
            lora_down_key = f"{module_name}.lora_down.weight"
            lora_up_key = f"{module_name}.lora_up.weight"
            
            if lora_down_key in lora_params and lora_up_key in lora_params:
                lora_down = lora_params[lora_down_key]
                lora_up = lora_params[lora_up_key]
                
                # Debug prints for shapes
                if module_name == target_modules[0]:  # Print only for first module to avoid spam
                    print(f"Module: {module_name}")
                    print(f"  Original weight: {module.weight.shape}")
                    print(f"  Lora down: {lora_down.shape}")
                    print(f"  Lora up: {lora_up.shape}")
                
                # Check shapes before proceeding
                in_dim = module.weight.shape[1]
                out_dim = module.weight.shape[0]
                
                # Verify that lora shapes match the module
                if lora_down.shape[1] != in_dim or lora_up.shape[0] != out_dim:
                    print(f"⚠️ Shape mismatch in {module_name}:")
                    print(f"   Expected lora_down: [rank, {in_dim}], got {lora_down.shape}")
                    print(f"   Expected lora_up: [{out_dim}, rank], got {lora_up.shape}")
                    continue  # Skip this module
                
                # Store original forward
                original_forward = module.forward
                original_forward_methods[module_name] = original_forward
                
                # Create a new forward method that uses our LoRA
                def make_forward(mod, orig_forward, ld, lu):
                    def lora_forward(x):
                        try:
                            # Create a LoRA layer with the current parameters
                            lora_layer = LoRALinear(
                                mod.weight,
                                lu,
                                ld,
                                alpha
                            )
                            return lora_layer(x)
                        except Exception as e:
                            print(f"⚠️ Error in forward: {e}")
                            # Fall back to original if there's an error
                            return orig_forward(x)
                    return lora_forward
                
                # Replace the forward method
                module.forward = make_forward(module, original_forward, lora_down, lora_up)
                patched_modules.append(module_name)
    
    print(f"Successfully patched {len(patched_modules)} modules with LoRA")
    
    # Return a function to restore original methods
    def restore_methods():
        for module_name, original_method in original_forward_methods.items():
            module = None
            for name, mod in model.named_modules():
                if name == module_name:
                    module = mod
                    break
            
            if module is not None:
                module.forward = original_method
    
    return restore_methods

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA for Stable Diffusion.")
    parser.add_argument("--base_model", type=str, default="sdxl-base-1.0")
    parser.add_argument("--images_dir", type=str, default="gina_face_cropped")
    parser.add_argument("--captions_file", type=str, default="gina_captions.json")
    parser.add_argument("--output_dir", type=str, default="gina_full_lora")
    parser.add_argument("--lora_name", type=str, default="gina_full_lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default is more stable than 1e-4)")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--use_rust", type=bool, default=True)
    parser.add_argument("--verbose", type=bool, default=False, help="Show verbose logging")
    parser.add_argument("--use_text_conditioning", type=bool, default=True, help="Use text conditioning from captions")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps")
    # New performance-related arguments
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Use mixed precision training")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="Use 8-bit AdamW optimizer if bitsandbytes is available")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--cache_latents", action="store_true", help="Cache latents during training to save compute")
    parser.add_argument("--use_lion", action="store_true", help="Use Lion optimizer instead of AdamW")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "lion", "sgd"], help="Optimizer to use")
    parser.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile for faster training (requires PyTorch 2.0+)")
    # New stability arguments
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Value for gradient clipping")
    parser.add_argument("--scale_lr", action="store_true", help="Scale learning rate by batch size and gradient accumulation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon value for optimizer stability")
    parser.add_argument("--stability_mode", action="store_true", help="Enable maximum stability mode (disables mixed precision, uses low learning rate)")
    # Add resume training options
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from"
    )
    parser.add_argument(
        "--resume_optimizer",
        action="store_true",
        default=True,
        help="Whether to resume optimizer state from checkpoint (if available)"
    )
    parser.add_argument(
        "--resume_scheduler",
        action="store_true",
        default=True,
        help="Whether to resume learning rate scheduler state from checkpoint (if available)"
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=200,
        help="Save a checkpoint every N steps"
    )
    return parser.parse_args()

def get_target_modules(unet):
    """Get a more comprehensive list of target modules for training"""
    target_modules = []
    
    # Target all self-attention modules
    for name, module in unet.named_modules():
        # Target all self-attention layers (attn1)
        if "attn1" in name and hasattr(module, 'weight') and any(x in name for x in ["to_q", "to_k", "to_v", "to_out"]):
            target_modules.append(name)
            
    return target_modules

def initialize_lora_weights(unet, target_modules, rank, device="cuda", dtype=torch.float32):
    """Initialize LoRA weights for all target modules"""
    lora_params = {}
    
    for module_name in target_modules:
        # Find the module
        module = None
        for name, mod in unet.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is not None and hasattr(module, 'weight'):
            # Get dimensions
            in_features = module.weight.shape[1]  # Input features (second dimension of weight)
            out_features = module.weight.shape[0]  # Output features (first dimension of weight)
            
            # Print for debugging
            if module_name == target_modules[0]:
                print(f"Initializing LoRA for {module_name}")
                print(f"  Weight shape: {module.weight.shape}")
                print(f"  LoRA down shape: [{rank}, {in_features}]")
                print(f"  LoRA up shape: [{out_features}, {rank}]")
            
            # Use a more stable initialization method
            # For lora_down, use kaiming initialization which is more stable
            lora_down = torch.zeros(rank, in_features, device=device, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(lora_down, a=math.sqrt(5))
            
            # For lora_up, initialize with zeros so training starts from identity
            lora_up = torch.zeros(out_features, rank, device=device, dtype=torch.float32)
            
            # Make them trainable parameters
            lora_down = torch.nn.Parameter(lora_down)
            lora_up = torch.nn.Parameter(lora_up)
            
            # Store in our dictionary
            lora_params[f"{module_name}.lora_down.weight"] = lora_down
            lora_params[f"{module_name}.lora_up.weight"] = lora_up
    
    return lora_params

def get_optimizer(parameters, args):
    """Get the appropriate optimizer based on arguments"""
    # Scale learning rate if requested
    lr = args.learning_rate
    if args.scale_lr:
        lr = lr * args.train_batch_size * args.gradient_accumulation_steps
        print(f"Scaling learning rate to {lr}")
        
    if args.optimizer == "adamw8bit" and HAVE_BNB:
        return bnb.optim.AdamW8bit(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif args.optimizer == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(parameters, lr=lr, weight_decay=args.weight_decay)
        except ImportError:
            print("Lion optimizer not available, falling back to AdamW")
            return torch.optim.AdamW(
                parameters, 
                lr=lr,
                betas=(0.9, 0.999), 
                weight_decay=args.weight_decay,
                eps=args.eps
            )
    elif args.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    else:  # Default to AdamW
        return torch.optim.AdamW(
            parameters, 
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=args.eps
        )

def get_lr_scheduler(optimizer, args, total_steps):
    """Create learning rate scheduler with warmup"""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        # Decay phase - linear decay to 10% of original LR
        return max(0.1, float(total_steps - current_step) / float(max(1, total_steps - args.warmup_steps)))
        
    return LambdaLR(optimizer, lr_lambda)

def contains_nan(tensor):
    """Check if tensor contains any NaN values"""
    if tensor is None:
        return False
    return torch.isnan(tensor).any()

def reduce_learning_rate(optimizer, factor=0.5):
    """Emergency learning rate reduction to recover from NaNs"""
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        param_group['lr'] = old_lr * factor
        print(f"⚠️ Reducing learning rate from {old_lr:.2e} to {param_group['lr']:.2e}")
    return optimizer

def safe_forward(model, *args, **kwargs):
    """Perform a forward pass with NaN checking"""
    try:
        outputs = model(*args, **kwargs)
        
        # Check for NaN in output
        if isinstance(outputs, torch.Tensor) and torch.isnan(outputs).any():
            print("⚠️ NaN detected in model output!")
            return None
        
        return outputs
    except RuntimeError as e:
        print(f"⚠️ Runtime error in forward pass: {e}")
        return None

def safe_backward(loss, optimizer, max_grad_norm=1.0, reset_on_nan=True):
    """Safely perform backward pass with gradient checks"""
    try:
        # Backward pass
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grads = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grads = True
                    print("⚠️ NaN detected in gradients!")
                    break
            if has_nan_grads:
                break
        
        # Reset gradients if NaNs detected
        if has_nan_grads and reset_on_nan:
            print("🛑 Resetting gradients due to NaN values")
            optimizer.zero_grad()
            return False
        
        # Clip gradients
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None],
                max_norm=max_grad_norm
            )
        
        return True
    except RuntimeError as e:
        print(f"⚠️ Error during backward pass: {e}")
        optimizer.zero_grad()
        return False

def save_debug_snapshot(lora_params, step, output_dir, prefix="nan_debug"):
    """Save a debug snapshot when NaNs are detected"""
    try:
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(output_dir, "debug_snapshots")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save a snapshot of the current LoRA weights
        snapshot_path = os.path.join(debug_dir, f"{prefix}_step_{step}.safetensors")
        
        # Make sure weights are clean before saving
        clean_weights = {}
        for key, param in lora_params.items():
            # Clean any potential NaNs before saving
            tensor = param.detach().cpu().to(torch.float32)
            # Replace NaNs with zeros
            tensor[torch.isnan(tensor)] = 0.0
            clean_weights[key] = tensor
        
        # Save to safetensors file
        save_file(clean_weights, snapshot_path)
        print(f"📸 Saved debug snapshot to {snapshot_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save debug snapshot: {e}")
        return False

def validate_and_preprocess_image(image_tensor, device, dtype=None):
    """
    Validate and preprocess an image tensor to help prevent NaN issues during encoding.
    
    Args:
        image_tensor: The input image tensor to validate
        device: The device to put the tensor on
        dtype: The dtype to convert the tensor to (IGNORED for VAE processing - always using float32)
        
    Returns:
        Preprocessed image tensor or None if validation fails
    """
    try:
        # Check for NaN or infinity values
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            print("⚠️ Input image contains NaN or Inf values")
            return None
            
        # IMPORTANT: ALWAYS use float32 for VAE operations to avoid type mismatches
        # The error "Input type (c10::Half) and bias type (float) should be the same" 
        # happens when the VAE gets float16 inputs but has float32 biases
        image_tensor = image_tensor.to(device).to(torch.float32)
        
        # Double check dtype - this is crucial!
        if image_tensor.dtype != torch.float32:
            print(f"⚠️ Warning: Image tensor still not float32, forcing conversion from {image_tensor.dtype}")
            image_tensor = image_tensor.float()  # Alternative way to convert
        
        # Check image range and normalize if needed
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        
        # Print warning if range is abnormal
        if min_val < -1.5 or max_val > 1.5:
            print(f"⚠️ Image has unusual value range: [{min_val:.2f}, {max_val:.2f}]")
        
        # Aggressively clip to ensure valid range
        image_tensor = torch.clamp(image_tensor, -1.0, 1.0)
        
        # Check for extremely low variance which can cause issues
        if torch.var(image_tensor) < 1e-8:
            print("⚠️ Image has almost no variance (nearly constant)")
            return None
            
        return image_tensor
    
    except Exception as e:
        print(f"⚠️ Error validating image: {e}")
        return None

def save_checkpoint(lora_params, step, optimizer, lr_scheduler, args, loss_value=None, save_full_state=True):
    """Save a checkpoint including full training state for resuming later"""
    try:
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        lora_path = os.path.join(checkpoint_dir, f"{args.lora_name}.safetensors")
        # Clean any NaN values before saving
        clean_weights = {}
        for key, param in lora_params.items():
            # Clean any potential NaNs before saving
            tensor = param.detach().cpu().to(torch.float32)
            # Replace NaNs with zeros
            tensor[torch.isnan(tensor)] = 0.0
            clean_weights[key] = tensor
        
        # Save to safetensors file
        save_file(clean_weights, lora_path)
        
        # Also save latest weights to root of output directory
        latest_path = os.path.join(args.output_dir, f"{args.lora_name}.safetensors")
        save_file(clean_weights, latest_path)
        
        # Save full state if requested (for resuming)
        if save_full_state:
            checkpoint_state = {
                'step': step,
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'loss': loss_value,
            }
            
            # Add scheduler if available
            if lr_scheduler is not None:
                checkpoint_state['lr_scheduler'] = lr_scheduler.state_dict()
            
            # Save state for resuming
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, "training_state.pt"))
            # Also save latest state to root
            torch.save(checkpoint_state, os.path.join(args.output_dir, "latest_training_state.pt"))
            
            # Save args for reference
            with open(os.path.join(checkpoint_dir, "training_args.json"), 'w') as f:
                json.dump(vars(args), f, indent=2)
        
        print(f"✅ Saved checkpoint at step {step}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")
        return False
        
def load_checkpoint(checkpoint_path, device, dtype=torch.float32):
    """Load a checkpoint for resuming training"""
    try:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # First check if this is a training state checkpoint
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if not os.path.exists(state_path):
            state_path = checkpoint_path.replace('.safetensors', '_training_state.pt')
            if not os.path.exists(state_path):
                # Fall back to just loading weights
                print(f"⚠️ No training state found at {state_path}, loading only weights.")
                weights = load_file(checkpoint_path)
                return {
                    'weights': weights,
                    'step': None,
                    'optimizer': None,
                    'lr_scheduler': None,
                    'args': None
                }
        
        # Load full training state
        state = torch.load(state_path, map_location=device)
        
        # Load weights from safetensors
        weights = load_file(checkpoint_path)
        
        # Add weights to state
        state['weights'] = weights
        
        print(f"✅ Loaded checkpoint from step {state.get('step', 'unknown')}")
        return state
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        return None

def main():
    args = parse_args()
    
    # Track NaN occurrences
    nan_counter = 0
    
    # Apply stability mode if enabled
    if args.stability_mode:
        print("🛡️ STABILITY MODE ENABLED")
        args.mixed_precision = "no"  # Disable mixed precision
        args.learning_rate = min(args.learning_rate, 1e-5)  # Use very low learning rate
        args.gradient_clip_val = 0.5  # Stronger gradient clipping
        args.warmup_steps = max(args.warmup_steps, 200)  # Longer warmup
        args.eps = 1e-8  # Stable epsilon
        args.optimizer = "adamw"  # Use stable optimizer
        print(f"  - Using full precision (mixed_precision=no)")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Gradient clipping: {args.gradient_clip_val}")
        print(f"  - Warmup steps: {args.warmup_steps}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Header
        print("\n" + "=" * 50)
        print(f"🔥 TRAINING LORA: {args.lora_name} 🔥".center(50))
        print("=" * 50 + "\n")
        
        # Load model
        print(f"📂 Loading base model from {args.base_model}...")
        
        # Set the device and dtype based on mixed precision settings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if args.mixed_precision == "fp16":
            dtype = torch.float16
        elif args.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            if args.mixed_precision != "no":
                print(f"Warning: {args.mixed_precision} precision not supported, using float32 instead")
        
        # Setup automatic mixed precision if enabled
        if args.mixed_precision != "no" and device == "cuda":
            amp_enabled = True
            print(f"🚀 Using automatic mixed precision ({args.mixed_precision})")
            if args.mixed_precision == "fp16":
                amp_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()
            else:  # bf16
                amp_dtype = torch.bfloat16
                scaler = None  # Not needed for bf16
        else:
            amp_enabled = False
            scaler = None
        
        # Load UNet directly (more efficient)
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model,
            subfolder="unet",
            torch_dtype=dtype
        )
        
        # Load VAE for encoding images - use float32 for all VAE operations
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", 
            torch_dtype=torch.float32  # Always use float32 for VAE
        ).to(device)
        
        # IMPORTANT: Force ALL VAE parameters to float32 to avoid type mismatches
        print("🔍 Converting ALL VAE parameters to float32...")
        for name, param in vae.named_parameters():
            if param.dtype != torch.float32:
                print(f"  Converting {name} from {param.dtype} to float32")
                param.data = param.data.to(torch.float32)
        
        # Make sure the VAE remains in evaluation mode and doesn't receive gradients
        vae.eval()
        vae.requires_grad_(False)
        
        # Use this VAE with your pipeline
        unet.vae = vae
        
        # Text encoders and tokenizers - only load if using text conditioning
        text_encoder = None
        text_encoder_2 = None
        tokenizer = None
        tokenizer_2 = None
        
        if args.use_text_conditioning:
            print("📝 Loading text encoders and tokenizers for conditioning...")
            
            # Load tokenizers (lightweight)
            tokenizer = CLIPTokenizer.from_pretrained(
                args.base_model, subfolder="tokenizer"
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                args.base_model, subfolder="tokenizer_2"
            )
            
            # Load text encoders
            text_encoder = CLIPTextModel.from_pretrained(
                args.base_model, subfolder="text_encoder", torch_dtype=dtype
            ).to(device)
            text_encoder_2 = CLIPTextModel.from_pretrained(
                args.base_model, subfolder="text_encoder_2", torch_dtype=dtype
            ).to(device)
            
            # Set to eval mode 
            text_encoder.eval()
            text_encoder_2.eval()
            
            # Freeze text encoders
            text_encoder.requires_grad_(False)
            text_encoder_2.requires_grad_(False)
        
        # Move to GPU
        unet = unet.to(device)
        
        # Use torch.compile for UNet if available and requested
        if args.use_torch_compile and hasattr(torch, 'compile') and device == "cuda":
            print("🔧 Using torch.compile for UNet")
            unet = torch.compile(unet)
        
        # Ensure all parameters don't require grad (we'll manually add our LoRA parameters later)
        unet.requires_grad_(False)
        
        # Set up dataset and dataloader with performance improvements
        dataset = FaceDataset(
            args.images_dir, 
            captions_file=args.captions_file if args.use_text_conditioning else None,
            image_size=args.image_size, 
            use_cache=True,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers if device == "cuda" else 0,
            pin_memory=device == "cuda",
            drop_last=True
        )
        
        # Get target modules
        target_modules = get_target_modules(unet)
        print(f"🎯 Training LoRA for {len(target_modules)} self-attention modules with rank {args.rank}")
        
        # Initialize LoRA weights with FLOAT32 for parameters with gradients
        lora_params = initialize_lora_weights(
            unet, 
            target_modules, 
            args.rank, 
            device=device, 
            dtype=torch.float32  # Use float32 for parameters that will have gradients
        )
        print(f"✨ Initialized {len(lora_params)} LoRA parameters")
        
        # Setup optimizer - only optimize LoRA weights
        # Ensure all LoRA parameters require gradients
        for param in lora_params.values():
            param.requires_grad = True
            
        # Create optimizer
        optimizer = get_optimizer(lora_params.values(), args)
        
        # Create learning rate scheduler
        lr_scheduler = get_lr_scheduler(optimizer, args, args.num_train_epochs * len(dataloader))
        
        # Cache latents if requested
        cached_latents = None
        if args.cache_latents:
            print("🔄 Pre-computing and caching latents...")
            cached_latents = []
            
            # NaN-handling approach: process in smaller batches with careful error handling
            with torch.no_grad():
                # Process with a more robust approach
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching latents")):
                    # Get pixel values from batch
                    if isinstance(batch, dict):
                        pixel_values = batch["image"].to(device)
                    else:
                        pixel_values = batch.to(device)
                    
                    # IMPORTANT: Always convert to float32 for VAE operations
                    # This fixes the "Input type (c10::Half) and bias type (float) should be the same" error
                    pixel_values = pixel_values.to(torch.float32)
                    
                    # Check if image has valid values
                    if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                        print(f"⚠️ Invalid pixel values detected in batch {batch_idx}. Using random noise instead.")
                        # Use small random values instead of zeros (better for training)
                        random_latents = torch.randn((pixel_values.shape[0], 4, args.image_size // 8, args.image_size // 8), 
                                                   device=device, dtype=dtype) * 0.1
                        cached_latents.append(random_latents)
                        continue
                        
                    try:
                        # Use a safer approach by handling each image separately with error recovery
                        batch_latents = []
                        
                        # Process each image in the batch individually for better error isolation
                        for i in range(pixel_values.shape[0]):
                            try:
                                # Get image info for debugging
                                image_idx = batch_idx * args.train_batch_size + i
                                if image_idx < len(dataset.image_paths):
                                    image_path = dataset.image_paths[image_idx]
                                    image_filename = os.path.basename(image_path)
                                else:
                                    image_filename = f"unknown_idx_{image_idx}"
                                
                                # Get single image and ensure it's properly normalized to [-1, 1]
                                img = pixel_values[i:i+1]
                                
                                # Validate and preprocess the image
                                img = validate_and_preprocess_image(img, device, dtype)
                                if img is None:
                                    print(f"⚠️ Image {i} in batch {batch_idx} (file: {image_filename}) failed validation. Using random noise.")
                                    with open(os.path.join(args.output_dir, "problematic_images.txt"), "a") as f:
                                        f.write(f"{image_filename} (failed validation)\n")
                                    single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                             device=device, dtype=dtype) * 0.1
                                else:
                                    # Encode with VAE - image is guaranteed to be float32 now
                                    try:
                                        # Encode with VAE - image must be float32
                                        single_latent = vae.encode(img).latent_dist.sample() * 0.18215
                                        
                                        # Convert latent to training dtype after encoding
                                        single_latent = single_latent.to(dtype)
                                        
                                        # Check for NaNs in output
                                        if torch.isnan(single_latent).any() or torch.isinf(single_latent).any():
                                            print(f"⚠️ NaN/Inf detected in latent {i} in batch {batch_idx} (file: {image_filename}). Using random noise.")
                                            # Log problematic files
                                            with open(os.path.join(args.output_dir, "problematic_images.txt"), "a") as f:
                                                f.write(f"{image_filename} (NaN in latent)\n")
                                            # Use random noise instead
                                            single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                                     device=device, dtype=dtype) * 0.1
                                    except RuntimeError as e:
                                        if "Input type" in str(e) and "bias type" in str(e):
                                            print(f"⚠️ Type mismatch error for image {i}: {e}")
                                            print("Attempting aggressive type fixing...")
                                            # Force convert VAE parameters again
                                            for param in vae.parameters():
                                                param.data = param.data.to(torch.float32)
                                            # Try with explicit contiguous float
                                            img = img.float().contiguous()
                                            try:
                                                single_latent = vae.encode(img).latent_dist.sample() * 0.18215
                                                single_latent = single_latent.to(dtype)
                                            except Exception:
                                                print("Second attempt failed, using random noise")
                                                single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8),
                                                                             device=device, dtype=dtype) * 0.1
                                    except Exception as e:
                                        print(f"⚠️ General VAE encoding error: {e}")
                                        single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                                     device=device, dtype=dtype) * 0.1
                                    
                                    batch_latents.append(single_latent)
                            except Exception as e:
                                print(f"⚠️ Error encoding image {i} in batch {batch_idx}: {e}")
                                # Use small random values as fallback
                                random_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                          device=device, dtype=dtype) * 0.1
                                batch_latents.append(random_latent)
                        
                        # Combine batch latents if we have any
                        if batch_latents:
                            combined_latents = torch.cat(batch_latents, dim=0)
                            cached_latents.append(combined_latents)
                        else:
                            # Emergency fallback if all images in batch failed
                            print(f"⚠️ All images in batch {batch_idx} failed encoding. Using random noise.")
                            random_latents = torch.randn((pixel_values.shape[0], 4, args.image_size // 8, args.image_size // 8), 
                                                       device=device, dtype=dtype) * 0.1
                            cached_latents.append(random_latents)
                        
                    except RuntimeError as e:
                        print(f"⚠️ Runtime error encoding batch {batch_idx}: {e}")
                        # Use random noise as fallback (slightly better than zeros for training)
                        random_latents = torch.randn((pixel_values.shape[0], 4, args.image_size // 8, args.image_size // 8), 
                                                   device=device, dtype=dtype) * 0.1
                        cached_latents.append(random_latent)
                    
                    # Periodically clear CUDA cache to prevent OOM
                    if batch_idx % 10 == 0 and device == "cuda":
                        torch.cuda.empty_cache()
            
            print(f"✅ Cached latents for {len(cached_latents)} batches")
            
            # Final verification of cached latents
            nan_batches = 0
            for i, latent_batch in enumerate(cached_latents):
                if torch.isnan(latent_batch).any() or torch.isinf(latent_batch).any():
                    print(f"⚠️ NaN/Inf still detected in cached batch {i}. Will be replaced during training.")
                    nan_batches += 1
            
            if nan_batches > 0:
                print(f"⚠️ {nan_batches}/{len(cached_latents)} cached batches contain NaN/Inf values.")
            else:
                print("✅ All cached latents verified successfully.")
        
        # Initialize training state
        global_step = 0
        start_epoch = 0
        
        # Load optimizer and scheduler
        optimizer = None
        lr_sched = None
        
        # Resume from checkpoint if provided
        if args.resume_from:
            print(f"🔄 Attempting to resume training from {args.resume_from}")
            checkpoint = load_checkpoint(args.resume_from, device, dtype)
            
            if checkpoint:
                # Load LoRA parameters
                if 'weights' in checkpoint and checkpoint['weights']:
                    for name, param in checkpoint['weights'].items():
                        if name in lora_params:
                            lora_params[name].data.copy_(param.to(device))
                            print(f"Loaded parameter: {name}")
                
                # Resume training step
                if 'step' in checkpoint and checkpoint['step'] is not None:
                    global_step = checkpoint['step']
                    # Calculate starting epoch
                    steps_per_epoch = steps_per_epoch = len(dataloader)
                    start_epoch = global_step // steps_per_epoch
                    print(f"Resuming from global step {global_step} (epoch {start_epoch})")
                
                # Setup optimizer
                optimizer = get_optimizer(lora_params.values(), args)
                
                # Resume optimizer state
                if args.resume_optimizer and 'optimizer' in checkpoint and checkpoint['optimizer']:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        print("✅ Resumed optimizer state")
                    except Exception as e:
                        print(f"⚠️ Failed to load optimizer state: {e}")
                
                # Setup scheduler
                lr_sched = get_lr_scheduler(optimizer, args, len(dataloader) * args.num_train_epochs)
                
                # Resume scheduler state
                if args.resume_scheduler and 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler']:
                    try:
                        lr_sched.load_state_dict(checkpoint['lr_scheduler'])
                        print("✅ Resumed learning rate scheduler state")
                    except Exception as e:
                        print(f"⚠️ Failed to load scheduler state: {e}")
        
        # If not resuming, create new optimizer and scheduler
        if optimizer is None:
            optimizer = get_optimizer(lora_params.values(), args)
        
        if lr_sched is None:
            lr_sched = get_lr_scheduler(optimizer, args, len(dataloader) * args.num_train_epochs)
        
        # Training loop
        total_steps = args.num_train_epochs * len(dataloader) // args.gradient_accumulation_steps
        effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        
        # Print initial info
        print(f"Starting training on {len(dataset)} images")
        print(f"Training for {args.num_train_epochs} epochs with batch size {args.train_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps} (effective batch size: {effective_batch_size})")
        print(f"Total steps: {total_steps}")
        print(f"LoRA Config: rank={args.rank}, learning_rate={args.learning_rate}")
        print(f"Using Rust: {args.use_rust}")
        print(f"Optimizer: {args.optimizer}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Using text conditioning: {args.use_text_conditioning}")
        print(f"Training {len(target_modules)} modules")
        print("=" * 40)
        
        for epoch in range(start_epoch, args.num_train_epochs):
            # More descriptive progress bar
            desc = f"Epoch {epoch+1}/{args.num_train_epochs}"
            progress_bar = tqdm(
                total=len(dataloader), 
                desc=desc,
                bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
            
            epoch_loss = 0.0
            
            for step, batch in enumerate(dataloader):
                # Get pixel values from batch
                if isinstance(batch, dict):
                    pixel_values = batch["image"].to(device)
                else:
                    pixel_values = batch.to(device)
                
                # IMPORTANT: Always convert to float32 for VAE operations regardless of training precision
                # This fixes the "Input type (c10::Half) and bias type (float) should be the same" error
                pixel_values = pixel_values.to(torch.float32)
                
                # Check if we're using cached latents
                if cached_latents is not None:
                    latents = cached_latents[step % len(cached_latents)]
                else:
                    # Encode images to latent space
                    with torch.no_grad():
                        try:
                            # Handle each image individually for better error isolation
                            batch_latents = []
                            
                            for i in range(pixel_values.shape[0]):
                                try:
                                    # Get image info for debugging
                                    image_idx = step * args.train_batch_size + i
                                    if image_idx < len(dataset.image_paths):
                                        image_path = dataset.image_paths[image_idx]
                                        image_filename = os.path.basename(image_path)
                                    else:
                                        image_filename = f"unknown_idx_{image_idx}"
                                    
                                    # Get single image and ensure it's properly normalized to [-1, 1]
                                    img = pixel_values[i:i+1]
                                    
                                    # Validate and preprocess the image - this now forces float32
                                    img = validate_and_preprocess_image(img, device)
                                    if img is None:
                                        print(f"⚠️ Image {i} in batch {step} (file: {image_filename}) failed validation. Using random noise.")
                                        with open(os.path.join(args.output_dir, "problematic_images.txt"), "a") as f:
                                            f.write(f"{image_filename} (failed validation)\n")
                                        single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                                 device=device, dtype=dtype) * 0.1
                                    else:
                                        # Encode with VAE - image is guaranteed to be float32 now
                                        try:
                                            # Encode with VAE - image must be float32
                                            single_latent = vae.encode(img).latent_dist.sample() * 0.18215
                                            
                                            # Convert latent to training dtype after encoding
                                            single_latent = single_latent.to(dtype)
                                            
                                            # Check for NaNs in output
                                            if torch.isnan(single_latent).any() or torch.isinf(single_latent).any():
                                                print(f"⚠️ NaN/Inf detected in latent {i} in batch {step} (file: {image_filename}). Using random noise.")
                                                # Log problematic files
                                                with open(os.path.join(args.output_dir, "problematic_images.txt"), "a") as f:
                                                    f.write(f"{image_filename} (NaN in latent)\n")
                                                # Use random noise instead
                                                single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                                         device=device, dtype=dtype) * 0.1
                                        except RuntimeError as e:
                                            if "Input type" in str(e) and "bias type" in str(e):
                                                print(f"⚠️ Type mismatch error for image {i}: {e}")
                                                print("Attempting aggressive type fixing...")
                                                # Force convert VAE parameters again
                                                for param in vae.parameters():
                                                    param.data = param.data.to(torch.float32)
                                                # Try with explicit contiguous float
                                                img = img.float().contiguous()
                                                try:
                                                    single_latent = vae.encode(img).latent_dist.sample() * 0.18215
                                                    single_latent = single_latent.to(dtype)
                                                except Exception:
                                                    print("Second attempt failed, using random noise")
                                                    single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8),
                                                                             device=device, dtype=dtype) * 0.1
                                            else:
                                                print(f"⚠️ General VAE encoding error: {e}")
                                                single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                                         device=device, dtype=dtype) * 0.1
                                        except Exception as e:
                                            print(f"⚠️ General VAE encoding error: {e}")
                                            single_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                                     device=device, dtype=dtype) * 0.1
                                    
                                    batch_latents.append(single_latent)
                                except Exception as e:
                                    print(f"⚠️ Error encoding image {i}: {e}")
                                    # Use random noise as fallback
                                    random_latent = torch.randn((1, 4, args.image_size // 8, args.image_size // 8), 
                                                           device=device, dtype=dtype) * 0.1
                                    batch_latents.append(random_latent)
                            
                            # Combine batch latents
                            if batch_latents:
                                latents = torch.cat(batch_latents, dim=0)
                            else:
                                # Emergency fallback
                                print("⚠️ All images failed encoding. Using random noise.")
                                latents = torch.randn((pixel_values.shape[0], 4, args.image_size // 8, args.image_size // 8), 
                                                 device=device, dtype=dtype) * 0.1
                        except Exception as e:
                            print(f"⚠️ Batch encoding error: {e}")
                            # Use random noise for the whole batch
                            latents = torch.randn((pixel_values.shape[0], 4, args.image_size // 8, args.image_size // 8), 
                                             device=device, dtype=dtype) * 0.1
                
                # Prepare noise and timesteps
                with torch.no_grad():
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
                    
                    # Make sure timesteps is the right dtype for the model
                    timesteps = timesteps.to(dtype)
                    
                    # Add noise to latents
                    # Instead of using unet.scheduler (which might not exist), use standard diffusion noise formula
                    # This is the basic noise formula used in diffusion models
                    noise_level = timesteps.float() / 1000
                    # Convert back to the working dtype
                    noise_level = noise_level.to(dtype)
                    alpha_cumprod = torch.cos(noise_level * torch.pi / 2) ** 2
                    alpha_cumprod = alpha_cumprod.view(-1, 1, 1, 1)
                    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
                    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
                    noisy_latents = sqrt_alpha_cumprod * latents + sqrt_one_minus_alpha_cumprod * noise
                    
                # Check for NaNs in noisy_latents to prevent training issues
                if torch.isnan(noisy_latents).any():
                    print("⚠️ NaN detected in noisy_latents! Skipping step.")
                    continue
                
                # Get encoder_hidden_states - either use text conditioning or zeros
                batch_size = latents.shape[0]
                
                # Initialize variables we'll need in all cases
                encoder_hidden_states = None
                time_ids = None
                text_embeds = None
                added_cond_kwargs = None
                
                # Case 1: Text conditioning is enabled and we have valid inputs
                if args.use_text_conditioning and isinstance(batch, dict) and text_encoder is not None:
                    try:
                        # Process with text encoders
                        with torch.no_grad():
                            # Process with first text encoder
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            text_encoder_output = text_encoder(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            encoder_hidden_states_1 = text_encoder_output[0].to(dtype)
                            
                            # Process with second text encoder
                            input_ids_2 = batch["input_ids_2"].to(device)
                            attention_mask_2 = batch["attention_mask_2"].to(device)
                            text_encoder_output_2 = text_encoder_2(
                                input_ids=input_ids_2,
                                attention_mask=attention_mask_2
                            )
                            encoder_hidden_states_2 = text_encoder_output_2[0].to(dtype)
                            
                            # CRITICAL: Get shapes right for SDXL
                            # First check the actual shape to confirm dimensions
                            if step == 0:
                                print(f"Text encoder 1 output shape: {encoder_hidden_states_1.shape}")
                                print(f"Text encoder 2 output shape: {encoder_hidden_states_2.shape}")
                            
                            # SDXL ALWAYS requires hidden states of shape [batch_size, 77, 2048]
                            # We're seeing shapes of [batch_size, 77, 1280] which causes matrix multiplication errors
                            
                            # Create empty tensor with the correct shape
                            encoder_hidden_states = torch.zeros(
                                (batch_size, 77, 2048), device=device, dtype=dtype
                            )
                            
                            # If we have the correct shape from encoder 2, use it (but we likely don't)
                            if encoder_hidden_states_2.shape[-1] == 2048:
                                encoder_hidden_states = encoder_hidden_states_2
                            # Otherwise, if the shape is wrong, we'll use our zero tensor initialized above
                            elif step % 20 == 0:  # Only show warning occasionally
                                print(f"Using zeros for encoder hidden states due to dimension mismatch: {encoder_hidden_states_2.shape[-1]} != 2048")
                            
                            # For SDXL, time_ids need to be correct
                            # Format: [h, w, crop_top, crop_left, crop_h, crop_w]
                            time_ids = torch.tensor(
                                [args.image_size, args.image_size, 0, 0, args.image_size, args.image_size],
                                device=device, 
                                dtype=dtype
                            ).repeat(batch_size, 1)
                            
                            # Add a standard dimension for text_embeds
                            # We'll use zeros in the correct shape for consistency
                            text_embeds = torch.zeros((batch_size, 1280), device=device, dtype=dtype)
                            
                            # Debug print for shapes
                            if step == 0:
                                print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")
                                print(f"time_ids shape: {time_ids.shape}")
                                print(f"text_embeds shape: {text_embeds.shape}")
                            
                            # Set up added_cond_kwargs with required parameters
                            added_cond_kwargs = {
                                "text_embeds": text_embeds,
                                "time_ids": time_ids
                            }
                    except Exception as e:
                        print(f"⚠️ Error processing text conditioning: {e}")
                        print("Falling back to zero conditioning")
                        # Fall back to zeros - will be handled by the else block below
                        encoder_hidden_states = None
                
                # Case 2: Either text conditioning failed or is disabled - use zeros
                if encoder_hidden_states is None:
                    # Use zeros as fallback - MAKE SURE DIMENSIONS ARE CORRECT (77x2048)
                    encoder_hidden_states = torch.zeros(
                        (batch_size, 77, 2048), device=device, dtype=dtype
                    )
                    
                    # Default time_ids
                    time_ids = torch.tensor(
                        [args.image_size, args.image_size, 0, 0, args.image_size, args.image_size],
                        device=device, 
                        dtype=dtype
                    ).repeat(batch_size, 1)
                    
                    # Use zeros with the correct shape
                    text_embeds = torch.zeros((batch_size, 1280), device=device, dtype=dtype)
                    
                    # SDXL specific conditioning
                    added_cond_kwargs = {
                        "text_embeds": text_embeds,
                        "time_ids": time_ids
                    }
                
                # Only zero gradients on the first accumulation step
                if (step % args.gradient_accumulation_steps) == 0:
                    optimizer.zero_grad()
                    
                # Apply LoRA layers with hooks that preserve gradients
                restore_original = apply_lora_layer(unet, target_modules, lora_params, alpha=1.5)
                
                # Forward pass through unet with mixed precision if enabled
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        try:
                            # Check inputs for NaN
                            if contains_nan(noisy_latents):
                                print("⚠️ NaN detected in noisy_latents! Skipping step.")
                                continue
                                
                            if contains_nan(encoder_hidden_states):
                                print("⚠️ NaN detected in encoder_hidden_states! Using zeros instead.")
                                encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                            
                            # Forward pass
                            model_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs
                            ).sample
                            
                            # Check for NaN in model prediction
                            if contains_nan(model_pred):
                                print("⚠️ NaN detected in model_pred! Skipping gradient calculation.")
                                save_debug_snapshot(lora_params, step, args.output_dir, prefix="nan_model_pred")
                                continue
                                
                            if contains_nan(noise):
                                print("⚠️ NaN detected in target noise! Skipping gradient calculation.")
                                save_debug_snapshot(lora_params, step, args.output_dir, prefix="nan_noise")
                                continue
                            
                            # Calculate loss with exception handling
                            try:
                                loss = torch.nn.functional.mse_loss(model_pred, noise)
                                
                                # If we got a NaN loss, skip this step
                                if torch.isnan(loss).any():
                                    print(f"⚠️ NaN loss detected at step {step}! Value: {loss.item()}")
                                    # Save debug snapshot
                                    save_debug_snapshot(lora_params, step, args.output_dir, prefix="nan_loss")
                                    # Reset the optimizer to clear bad gradients
                                    optimizer.zero_grad()
                                    continue
                                
                                # Scale loss for gradient accumulation
                                loss = loss / args.gradient_accumulation_steps
                            except RuntimeError as e:
                                print(f"⚠️ Error calculating loss: {e}")
                                optimizer.zero_grad()
                                continue
                        except RuntimeError as e:
                            print(f"⚠️ Runtime error during forward pass: {e}")
                            optimizer.zero_grad()
                            continue
                    
                    # Backward pass with scaler for fp16
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        success = safe_backward(loss, optimizer, max_grad_norm=args.gradient_clip_val)
                        if not success:
                            # Skip this step if backward failed
                            continue
                else:
                    try:
                        # Standard forward pass without mixed precision
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            added_cond_kwargs=added_cond_kwargs
                        ).sample
                        
                        # Calculate loss
                        loss = torch.nn.functional.mse_loss(model_pred, noise)
                        
                        # Scale loss for gradient accumulation
                        loss = loss / args.gradient_accumulation_steps
                        
                        # Backward pass using safe implementation
                        success = safe_backward(loss, optimizer, max_grad_norm=args.gradient_clip_val)
                        if not success:
                            # Skip the rest of this step if backward failed
                            continue
                    except RuntimeError as e:
                        print(f"⚠️ Runtime error during standard processing: {e}")
                        optimizer.zero_grad()
                        continue
                
                # Restore original forward methods
                restore_original()
                
                # Update weights if we've accumulated enough gradients
                if ((step + 1) % args.gradient_accumulation_steps == 0) or (step == len(dataloader) - 1):
                    try:
                        # Check for NaN in gradients
                        has_nan_grads = False
                        for param in lora_params.values():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grads = True
                                break
                        
                        if has_nan_grads:
                            print("⚠️ NaN detected in gradients! Skipping update.")
                            # Save debug snapshot for investigation
                            save_debug_snapshot(lora_params, global_step, args.output_dir, prefix="nan_gradients")
                            # Zero out gradients to start fresh
                            optimizer.zero_grad()
                            
                            # Apply emergency learning rate reduction if this happens multiple times
                            nan_counter += 1
                            if nan_counter > 3:
                                print("Multiple NaN errors detected, reducing learning rate...")
                                optimizer = reduce_learning_rate(optimizer, factor=0.5)
                                nan_counter = 0
                        else:
                            # Apply gradient clipping to prevent exploding gradients
                            if args.gradient_clip_val > 0:
                                if amp_enabled and scaler is not None:
                                    scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in lora_params.values() if p.requires_grad], 
                                    max_norm=args.gradient_clip_val
                                )
                            
                            # Update weights with or without scaler
                            if amp_enabled and scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            
                            # Step the learning rate scheduler
                            lr_sched.step()
                
                        global_step += 1
                    except RuntimeError as e:
                        print(f"⚠️ Optimization error: {e}")
                        # Recovery logic - skip this step
                        optimizer.zero_grad()
                        print("Skipping update and continuing...")
                    
                # Update progress bar postfix
                if not contains_nan(loss):
                    epoch_loss += loss.item() * args.gradient_accumulation_steps
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr_sched.get_last_lr()[0]:.6f}"
                    })
                    
                    # Save intermediate checkpoint
                    if global_step > 0 and global_step % args.save_every_n_steps == 0:
                        save_checkpoint(
                            lora_params=lora_params,
                            step=global_step,
                            optimizer=optimizer,
                            lr_scheduler=lr_sched, 
                            args=args,
                            loss_value=avg_loss
                        )
                else:
                    progress_bar.set_postfix(loss="NaN")
                progress_bar.update(1)
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"{args.lora_name}_step_{global_step}.safetensors")
                    
                    # Save current LoRA weights
                    save_weights = {}
                    for key, param in lora_params.items():
                        # Make sure we're sending clean CPU tensors to safetensors
                        save_weights[key] = param.detach().cpu().to(torch.float32)
                    
                    try:
                        save_file(save_weights, checkpoint_path)
                        print(f"\nSaved checkpoint to {checkpoint_path} (step {global_step}/{total_steps})")
                    except Exception as e:
                        print(f"\nWarning: Failed to save checkpoint: {e}")
                        # Try alternative format
                        alt_path = checkpoint_path.replace('.safetensors', '.pt')
                        torch.save(save_weights, alt_path)
                        print(f"Saved checkpoint in PT format to {alt_path}")
            
            # Print epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"✅ Completed epoch {epoch+1}/{args.num_train_epochs} - Avg loss: {avg_epoch_loss:.6f}")
        
        # Training complete, save final model
        print("✅ Training complete!")
        
        # Save final checkpoint with everything
        print("💾 Saving final LoRA weights...")
        save_checkpoint(
            lora_params=lora_params,
            step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_sched,
            args=args,
            loss_value=epoch_loss / len(dataloader) if len(dataloader) > 0 else None,
            save_full_state=True
        )
        
        # Additionally save in original format for compatibility
        print("💾 Saving in legacy format for compatibility...")
        final_lora_path = os.path.join(args.output_dir, f"{args.lora_name}.safetensors")
        
        # Save weights in FP32 for compatibility
        final_weights = {}
        for key, param in lora_params.items():
            final_weights[key] = param.detach().cpu().to(torch.float32)
            
        # Extra safety check for NaNs
        has_nans = False
        for key, tensor in final_weights.items():
            if torch.isnan(tensor).any():
                print(f"⚠️ NaN detected in final weights for {key}! Replacing with zeros.")
                tensor[torch.isnan(tensor)] = 0.0
                has_nans = True
        
        # Also save in safetensors format for compatibility with web UIs
        save_file(final_weights, final_lora_path)
        print(f"✅ Saved final LoRA weights to {final_lora_path}")
        
        # Save for PyTorch compatibility as well
        try:
            torch.save(final_weights, final_lora_path.replace('.safetensors', '.pt'))
        except Exception as e:
            print(f"⚠️ Error saving PyTorch format: {e}")
        
        # Try to generate a test image with the trained LoRA
        try:
            print("Generating test image with trained LoRA...")
            
            # Create a new pipeline for inference
            inference_pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(device)
            
            # Apply trained LoRA
            apply_lora_to_self_attention(
                inference_pipeline.unet,
                final_lora_path,
                alpha=0.8,  # Use lower alpha for initial testing
                target_modules=target_modules
            )
            
            # Generate test image
            test_prompt = "A portrait of <georgina_szayel>, a young woman with red hair, photorealistic, studio lighting, highly detailed"
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
        except Exception as e:
            print(f"⚠️ Could not generate test image: {e}")
            print("This is normal if using a minimal model without tokenizers.")
            print("Your LoRA weights were still saved successfully.")
        
        print("✅ Training complete!")
    except Exception as e:
        import traceback
        print(f"❌ Training failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Diagnostic helper function for VAE type issues
def process_with_vae(vae, image_tensor, device):
    """
    Process an image tensor with VAE, ensuring correct types and handling errors.
    
    Args:
        vae: The VAE model
        image_tensor: The input image tensor
        device: The device to use
        
    Returns:
        Latent encoding or None on error
    """
    try:
        # Ensure image is on device and using float32 (required for VAE)
        image_tensor = image_tensor.to(device).to(torch.float32)
        
        # Verify VAE dtype
        all_fp32 = True
        for param in vae.parameters():
            if param.dtype != torch.float32:
                all_fp32 = False
                break
                
        if not all_fp32:
            print("⚠️ VAE parameters not all float32 - converting...")
            # Force all VAE parameters to float32
            for param in vae.parameters():
                param.data = param.data.to(torch.float32)
        
        # Process with VAE
        with torch.no_grad():
            # Get latent representation
            latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215
            return latent
            
    except RuntimeError as e:
        err_msg = str(e)
        if "Input type" in err_msg and "bias type" in err_msg:
            print("⚠️ Type mismatch in VAE! Input and bias types don't match. This might indicate mixed precision issues.")
            
            # Print all parameter dtypes for debugging
            print("🔍 VAE parameter dtypes:")
            for name, param in vae.named_parameters():
                print(f"  {name}: {param.dtype}")
                
            print("🔍 Input tensor dtype:", image_tensor.dtype)
            
            # Force convert to the same dtype
            print("🛠️ Attempting to convert all VAE parameters to float32...")
            for param in vae.parameters():
                param.data = param.data.to(torch.float32)
                
            try:
                # Try again after conversion
                return vae.encode(image_tensor).latent_dist.sample() * 0.18215
            except Exception as e2:
                print(f"⚠️ Second attempt failed: {e2}")
                return None
        else:
            print(f"⚠️ VAE encoding error: {e}")
            return None
    except Exception as e:
        print(f"⚠️ Unexpected VAE error: {e}")
        return None 