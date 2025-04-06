import torch
import lora_ops  # This is your Rust module
from safetensors.torch import load_file, save_file
import numpy as np

def apply_lora_to_tensor(tensor: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Applies a LoRA transformation to a single weight tensor using your Rust-backed module.
    Handles dimension mismatches that can occur in cross-attention layers.
    
    tensor: The original weight (e.g. attention or MLP weight)
    lora_A, lora_B: LoRA weights (rank-decomposed)
    alpha: Scaling factor
    """
    # Handle float16 tensors by converting to float32 for Rust interop
    original_dtype = tensor.dtype
    
    # Print shapes for debugging
    weight_shape = tensor.shape
    lora_A_shape = lora_A.shape
    lora_B_shape = lora_B.shape
    
    print(f"Weight shape: {weight_shape}")
    print(f"LoRA A shape: {lora_A_shape}")
    print(f"LoRA B shape: {lora_B_shape}")
    
    # Convert to float32 NumPy for Rust interop
    x_np = tensor.detach().cpu().to(torch.float32).numpy()
    A_np = lora_A.detach().cpu().to(torch.float32).numpy()
    B_np = lora_B.detach().cpu().to(torch.float32).numpy()
    
    # The expected shapes for the Rust function are:
    # x: [output_dim, input_dim]
    # A: [rank, output_dim]
    # B: [input_dim, rank]
    
    # Calculate expected LoRA contribution
    try:
        # Try the standard approach first
        out_np = lora_ops.apply_lora(x_np, A_np, B_np, alpha)
    except ValueError as e:
        print(f"⚠️ Dimension mismatch detected: {e}")
        print("Falling back to PyTorch implementation for this layer")
        
        # Fall back to PyTorch implementation
        # For cross-attention layers, the dimensions need to be handled differently
        # W' = W + alpha * BA
        with torch.no_grad():
            x_tensor = torch.tensor(x_np)
            A_tensor = torch.tensor(A_np)
            B_tensor = torch.tensor(B_np)
            
            # Try to determine the correct multiplication order
            try:
                # Method 1: BA
                lora_delta = B_tensor.T @ A_tensor
                if lora_delta.shape == x_tensor.shape:
                    out_np = (x_tensor + alpha * lora_delta).numpy()
                else:
                    # Method 2: AB
                    lora_delta = A_tensor.T @ B_tensor
                    if lora_delta.shape == x_tensor.shape:
                        out_np = (x_tensor + alpha * lora_delta).numpy()
                    else:
                        # Method 3: Try with transposed weights
                        lora_delta = B_tensor @ A_tensor.T
                        if lora_delta.shape == x_tensor.shape:
                            out_np = (x_tensor + alpha * lora_delta).numpy()
                        else:
                            # Method 4: Last attempt
                            lora_delta = A_tensor @ B_tensor.T
                            if lora_delta.shape == x_tensor.shape:
                                out_np = (x_tensor + alpha * lora_delta).numpy()
                            else:
                                raise ValueError(f"Cannot find compatible multiplication order. Shapes: tensor {x_tensor.shape}, A {A_tensor.shape}, B {B_tensor.shape}")
            except Exception as e2:
                raise ValueError(f"PyTorch fallback also failed: {e2}") from e

    # Convert back to tensor and original dtype
    return torch.from_numpy(out_np).to(tensor.device).to(original_dtype)

def inject_lora_into_model(model, lora_path: str, target_layer_names: list[str], alpha=1.0):
    """
    Injects LoRA weights into specified layers of the model.
    model: Your PyTorch model (e.g., UNet or CLIP)
    lora_path: Path to .safetensors or .pt with LoRA keys like `layer.lora_down.weight`
    target_layer_names: List of base layer names (e.g., ['attn1.to_q', 'mlp.fc1'])
    alpha: LoRA scaling
    """
    lora_state = load_file(lora_path)
    
    # Collect target modules from the LoRA file
    if not target_layer_names:
        print("No target modules specified, extracting from LoRA file...")
        target_layer_names = []
        for key in lora_state.keys():
            if ".lora_down.weight" in key:
                module_name = key.split(".lora_down.weight")[0]
                target_layer_names.append(module_name)
        print(f"Found {len(target_layer_names)} target modules in LoRA file")

    for base_name in target_layer_names:
        down_key = f"{base_name}.lora_down.weight"
        up_key   = f"{base_name}.lora_up.weight"

        if down_key not in lora_state or up_key not in lora_state:
            print(f"⚠️ Skipping {base_name}: Missing LoRA weights")
            continue

        lora_down = lora_state[down_key]
        lora_up = lora_state[up_key]

        target_layer = dict(model.named_modules()).get(base_name)
        if target_layer is None:
            print(f"❌ Layer {base_name} not found in model.")
            continue

        with torch.no_grad():
            print(f"🎯 Applying LoRA to: {base_name}")
            updated = apply_lora_to_tensor(target_layer.weight, lora_down, lora_up, alpha)
            target_layer.weight.copy_(updated)

    print("✅ LoRA injection complete.") 