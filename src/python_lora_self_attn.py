import torch
import torch.nn as nn
from typing import List, Dict, Any

def find_self_attention_modules(model: nn.Module) -> List[str]:
    """Find all self-attention modules in the model"""
    self_attn_modules = []
    
    for name, module in model.named_modules():
        # Look for attention modules in SDXL
        if any(attn_name in name.lower() for attn_name in ['attn', 'attention']):
            if hasattr(module, 'to_q') or hasattr(module, 'q_proj'):
                self_attn_modules.append(name)
    
    return self_attn_modules

def train_lora_for_self_attention(
    unet: nn.Module,
    output_path: str,
    rank: int = 4,
    target_modules: List[str] = None
) -> Dict[str, torch.Tensor]:
    """Initialize LoRA weights for self-attention modules"""
    lora_weights = {}
    
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
            
            # Initialize with small random values
            lora_down = torch.randn(rank, in_features) * 0.02
            lora_up = torch.randn(out_features, rank) * 0.02
            
            # Store weights
            lora_weights[f"{module_name}.lora_down.weight"] = lora_down
            lora_weights[f"{module_name}.lora_up.weight"] = lora_up
    
    return lora_weights

def apply_lora_to_self_attention(
    unet: nn.Module,
    lora_path: str,
    alpha: float = 1.0
) -> None:
    """Apply LoRA weights to self-attention modules"""
    # Load LoRA weights
    if lora_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        lora_weights = load_file(lora_path)
    else:
        lora_weights = torch.load(lora_path)
    
    # Apply weights to modules
    for name, module in unet.named_modules():
        if hasattr(module, 'weight'):
            # Check if we have LoRA weights for this module
            lora_down_key = f"{name}.lora_down.weight"
            lora_up_key = f"{name}.lora_up.weight"
            
            if lora_down_key in lora_weights and lora_up_key in lora_weights:
                lora_down = lora_weights[lora_down_key]
                lora_up = lora_weights[lora_up_key]
                
                # Apply LoRA: W = W + alpha * (up @ down)
                delta = torch.matmul(lora_up, lora_down)
                module.weight.data += alpha * delta 