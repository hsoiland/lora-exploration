"""
LoRA (Low-Rank Adaptation) implementation for SDXL using Rust backend.
This version specifically targets self-attention layers for simplified dimensions.
"""

import torch
import lora_ops  # Our Rust backend module
from safetensors.torch import load_file, save_file
from typing import List, Dict, Tuple, Optional, Union


def apply_lora_rust(
    tensor: torch.Tensor, 
    lora_down: torch.Tensor, 
    lora_up: torch.Tensor, 
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Apply LoRA transformation to a weight tensor using our Rust backend.
    
    Args:
        tensor: Original weight tensor
        lora_down: LoRA down projection (A matrix)
        lora_up: LoRA up projection (B matrix)
        alpha: Scaling factor for LoRA effect
        
    Returns:
        Updated weight tensor
    """
    # Handle device and type conversions
    device = tensor.device
    original_dtype = tensor.dtype
    
    # Convert to float32 and CPU for Rust
    tensor_cpu = tensor.detach().cpu().to(torch.float32)
    lora_down_cpu = lora_down.detach().cpu().to(torch.float32)
    lora_up_cpu = lora_up.detach().cpu().to(torch.float32)
    
    # Call our Rust function
    try:
        updated_tensor = lora_ops.apply_lora(
            tensor_cpu.numpy(), 
            lora_down_cpu.numpy(), 
            lora_up_cpu.numpy(), 
            alpha
        )
        
        # Convert back to PyTorch tensor
        updated_tensor = torch.from_numpy(updated_tensor).to(device).to(original_dtype)
        return updated_tensor
    except ValueError as e:
        # If Rust fails, fall back to PyTorch (though this shouldn't happen with self-attention only)
        print(f"⚠️ Rust implementation failed: {e}")
        print("Falling back to PyTorch implementation")
        
        # For self-attention layers, the standard LoRA formula works: W' = W + alpha * (BA)
        lora_delta = torch.matmul(lora_up.to(device), lora_down.to(device))
        updated_tensor = tensor + alpha * lora_delta.to(tensor.dtype)
        return updated_tensor


def is_self_attention_layer(name: str) -> bool:
    """
    Check if a layer is a self-attention layer.
    
    Args:
        name: Layer name
        
    Returns:
        True if it's a self-attention layer, False otherwise
    """
    # Self-attention layers in SDXL have "attn1" in their name
    return "attn1" in name and any(x in name for x in ["to_q", "to_k", "to_v"])


def find_self_attention_modules(model) -> List[str]:
    """
    Find all self-attention modules in an SDXL model.
    
    Args:
        model: SDXL UNet model
        
    Returns:
        List of module names for self-attention layers
    """
    target_modules = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if is_self_attention_layer(name):
                target_modules.append(name)
    
    return target_modules


def apply_lora_to_self_attention(
    model, 
    lora_path: str, 
    alpha: float = 1.0, 
    target_modules: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply LoRA to self-attention layers in an SDXL model.
    
    Args:
        model: SDXL UNet model
        lora_path: Path to LoRA weights file (.safetensors)
        alpha: Scaling factor for LoRA effect
        target_modules: Optional list of specific target modules
        
    Returns:
        Dictionary of original weights for restoration
    """
    # Load LoRA weights
    lora_state = load_file(lora_path)
    
    # Store original weights for restoration
    original_weights = {}
    
    # Find target modules if not provided
    if target_modules is None:
        target_modules = find_self_attention_modules(model)
        filtered_modules = []
        for name in target_modules:
            down_key = f"{name}.lora_down.weight"
            up_key = f"{name}.lora_up.weight"
            if down_key in lora_state and up_key in lora_state:
                filtered_modules.append(name)
        target_modules = filtered_modules
    
    print(f"Applying LoRA to {len(target_modules)} self-attention modules")
    
    # Apply LoRA to each target module
    for module_name in target_modules:
        # Get LoRA weights
        down_key = f"{module_name}.lora_down.weight"
        up_key = f"{module_name}.lora_up.weight"
        
        if down_key not in lora_state or up_key not in lora_state:
            print(f"⚠️ Missing LoRA weights for {module_name}, skipping")
            continue
        
        lora_down = lora_state[down_key]
        lora_up = lora_state[up_key]
        
        # Find target module
        module = None
        for name, mod in model.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is None or not hasattr(module, 'weight'):
            print(f"⚠️ Module {module_name} not found or has no weight")
            continue
        
        # Store original weight
        print(f"🎯 Applying LoRA to self-attention: {module_name}")
        original_weights[module_name] = module.weight.data.clone()
        
        # Apply LoRA
        with torch.no_grad():
            updated_weight = apply_lora_rust(
                module.weight,
                lora_down,
                lora_up,
                alpha
            )
            module.weight.data.copy_(updated_weight)
    
    print(f"✅ Successfully applied LoRA to {len(original_weights)} self-attention modules")
    return original_weights


def restore_weights(model, original_weights: Dict[str, torch.Tensor]) -> None:
    """
    Restore original weights of modules.
    
    Args:
        model: Model to restore
        original_weights: Dictionary of original weights from apply_lora_to_self_attention
    """
    for module_name, weight in original_weights.items():
        for name, module in model.named_modules():
            if name == module_name and hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.copy_(weight)
    
    print(f"✅ Restored {len(original_weights)} modules to original weights")


def train_lora_for_self_attention(
    unet,
    output_path: str,
    rank: int = 4,
    target_modules: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Initialize empty LoRA weights for self-attention layers.
    This can be used as a starting point for training.
    
    Args:
        unet: UNet model to create LoRA for
        output_path: Path to save LoRA weights
        rank: Rank of LoRA matrices
        target_modules: Optional list of specific target modules
        
    Returns:
        Dictionary of LoRA weights
    """
    # Find target modules if not provided
    if target_modules is None:
        target_modules = find_self_attention_modules(unet)
    
    # Create LoRA weights
    lora_weights = {}
    
    for module_name in target_modules:
        # Find module
        module = None
        for name, mod in unet.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is None or not hasattr(module, 'weight'):
            print(f"⚠️ Module {module_name} not found or has no weight")
            continue
        
        # Get weight shape
        weight = module.weight
        weight_shape = weight.shape
        
        # Initialize LoRA weights
        # LoRA down: [rank, dim_out], LoRA up: [dim_in, rank]
        lora_down = torch.zeros((rank, weight_shape[0]), dtype=weight.dtype, device='cpu')
        lora_up = torch.zeros((weight_shape[1], rank), dtype=weight.dtype, device='cpu')
        
        # Initialize with small random values
        torch.nn.init.kaiming_uniform_(lora_down, a=5**0.5)
        torch.nn.init.zeros_(lora_up)
        
        # Add to dictionary
        lora_weights[f"{module_name}.lora_down.weight"] = lora_down
        lora_weights[f"{module_name}.lora_up.weight"] = lora_up
    
    # Save weights
    save_file(lora_weights, output_path)
    print(f"✅ Created LoRA weights for {len(target_modules)} self-attention modules, saved to {output_path}")
    
    return lora_weights 