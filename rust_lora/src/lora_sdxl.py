import torch
import numpy as np
from safetensors.torch import load_file, save_file

def apply_lora_sdxl(tensor: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Applies a LoRA transformation to a weight tensor, designed specifically for SDXL.
    Handles both self-attention and cross-attention dimensions properly.
    
    Args:
        tensor: The original weight (e.g. attention or MLP weight)
        lora_A: LoRA down matrix (rank reduction)
        lora_B: LoRA up matrix (rank expansion)
        alpha: Scaling factor
    
    Returns:
        Updated weight tensor with LoRA applied
    """
    # Get device and dtype from the original tensor
    device = tensor.device
    original_dtype = tensor.dtype
    
    # Move all to the same device and convert to float32 for better precision
    tensor = tensor.to(device).to(torch.float32)
    lora_A = lora_A.to(device).to(torch.float32) 
    lora_B = lora_B.to(device).to(torch.float32)
    
    # Print shapes for debugging
    weight_shape = tensor.shape
    lora_A_shape = lora_A.shape
    lora_B_shape = lora_B.shape
    
    print(f"Weight shape: {weight_shape}")
    print(f"LoRA A shape: {lora_A_shape}")
    print(f"LoRA B shape: {lora_B_shape}")
    
    # Based on analysis of SDXL architecture from the paper, we need special handling
    # For SDXL LoRA's typical format:
    # - A is [rank, dim_out] - the "down" projection 
    # - B is [dim_in, rank] - the "up" projection
    # The original weight is [dim_out, dim_in]
    
    # Calculate LoRA update based on shape analysis
    with torch.no_grad():
        # Handle common case for self-attention layers in SDXL
        # For these layers, weight is [1280, 1280] or [640, 640]
        # Check for proper order by shape
        if lora_A.shape[1] == tensor.shape[0] and lora_B.shape[0] == tensor.shape[1]:
            # Standard BA format: W' = W + alpha * BA
            delta_weight = torch.matmul(lora_A.transpose(0, 1), lora_B.transpose(0, 1))
            tensor = tensor + alpha * delta_weight
            print("  ✓ Applied LoRA with standard multiplication (A.T @ B.T)")
            return tensor.to(original_dtype)
            
        # Specific cross-attention case with [1280, 2048] weight shape
        elif tensor.shape[1] == 2048 and lora_A.shape[1] == 1280 and lora_B.shape[0] == 2048:
            # Cross-attention case: different input (text) dim and attention dim
            delta_weight = torch.matmul(lora_A.transpose(0, 1), lora_B.transpose(0, 1))
            tensor = tensor + alpha * delta_weight
            print("  ✓ Applied LoRA for cross-attention (A.T @ B.T)")
            return tensor.to(original_dtype)
        
        # Handle case where the matrix needs to be further transposed
        else:
            # Reshape to match weight dimension - this is a direct way to 
            # ensure we get the right dimensions regardless of the exact layout
            # Convert lora_A to have the first dimension match tensor's first dimension
            # Convert lora_B to have the second dimension match tensor's second dimension
            
            # For matrices where rank is the second dim in A and first in B
            if lora_A.shape[0] == tensor.shape[0] and lora_B.shape[1] == tensor.shape[1]:
                delta_weight = torch.matmul(lora_A, lora_B)
                tensor = tensor + alpha * delta_weight
                print("  ✓ Applied LoRA with direct multiplication (A @ B)")
                return tensor.to(original_dtype)
            
            # Specific case for certain layers with dimensions reversed
            elif lora_A.shape[1] == tensor.shape[1] and lora_B.shape[0] == tensor.shape[0]:
                delta_weight = torch.matmul(lora_B.transpose(0, 1), lora_A.transpose(0, 1))
                tensor = tensor + alpha * delta_weight
                print("  ✓ Applied LoRA with reversed multiplication (B.T @ A.T)")
                return tensor.to(original_dtype)
    
    # Special Case: Try specifically with hand-calculated dimensions for SDXL
    # Based on what we know from the paper's architecture
    try:
        # Reshape the matrices to match the expected dimensions
        # This is specifically tailored for our known [8, dim] and [dim, 8] matrices
        rank = lora_A.shape[0]  # Usually 8 in our implementation
        
        # For typical attention layers: reshape to produce [dim_out, dim_in]
        # Create a tensor of zeroes with the same shape as the weight tensor
        final_AB = torch.zeros_like(tensor)
        
        # Apply a matrix outer product-like operation
        for i in range(rank):
            a_row = lora_A[i].unsqueeze(1)  # [dim_out, 1]
            b_col = lora_B[:, i].unsqueeze(0)  # [1, dim_in]
            final_AB += torch.matmul(a_row, b_col)
        
        tensor = tensor + alpha * final_AB
        print("  ✓ Applied LoRA using outer product approach")
        return tensor.to(original_dtype)
    except Exception as e:
        print(f"  ✗ Failed to apply LoRA special case: {e}")
    
    # If we couldn't apply LoRA with any of the above methods
    print(f"  ✗ Failed to apply LoRA - couldn't match dimensions")
    return tensor.to(original_dtype)

def inject_sdxl_lora(model, lora_path: str, alpha=1.0):
    """
    Injects LoRA weights into an SDXL model.
    Handles both self-attention and cross-attention layers properly.
    
    Args:
        model: Your PyTorch model (e.g., UNet from SDXL)
        lora_path: Path to .safetensors file with LoRA weights
        alpha: LoRA scaling factor
    
    Returns:
        Dictionary of original weights for later restoration
    """
    # Load LoRA weights
    lora_state = load_file(lora_path)
    
    # Track original weights to restore later if needed
    original_weights = {}
    
    # Extract target modules from LoRA file
    target_modules = []
    for key in lora_state.keys():
        if ".lora_down.weight" in key:
            module_name = key.split(".lora_down.weight")[0]
            target_modules.append(module_name)
    
    print(f"Found {len(target_modules)} target modules in LoRA file")
    
    # Apply LoRA to each target module
    for module_name in target_modules:
        down_key = f"{module_name}.lora_down.weight"
        up_key = f"{module_name}.lora_up.weight"
        
        if down_key not in lora_state or up_key not in lora_state:
            print(f"⚠️ Skipping {module_name}: Missing LoRA weights")
            continue
        
        lora_down = lora_state[down_key]
        lora_up = lora_state[up_key]
        
        # Find the target module
        target_module = None
        for name, module in model.named_modules():
            if name == module_name and hasattr(module, 'weight'):
                target_module = module
                break
        
        if target_module is None:
            print(f"❌ Module {module_name} not found in model")
            continue
        
        # Save original weight
        print(f"🎯 Applying LoRA to: {module_name}")
        original_weights[module_name] = target_module.weight.data.clone()
        
        # Apply LoRA
        with torch.no_grad():
            updated_weight = apply_lora_sdxl(
                target_module.weight,
                lora_down,
                lora_up,
                alpha
            )
            target_module.weight.data.copy_(updated_weight)
    
    print(f"✅ Applied LoRA to {len(original_weights)} modules")
    return original_weights

def restore_original_weights(model, original_weights):
    """
    Restores original weights of a model after LoRA application.
    
    Args:
        model: The model with LoRA weights
        original_weights: Dictionary of original weights
    """
    for module_name, original_weight in original_weights.items():
        for name, module in model.named_modules():
            if name == module_name and hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.copy_(original_weight)
                break
    
    print(f"✅ Restored original weights for {len(original_weights)} modules") 