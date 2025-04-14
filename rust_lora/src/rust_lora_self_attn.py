"""
LoRA (Low-Rank Adaptation) implementation for SDXL using Rust backend.
This version specifically targets self-attention layers for simplified dimensions.
"""

import torch
import lora_ops  # Our Rust backend module
from safetensors.torch import load_file, save_file
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import os
import json
import time
from pathlib import Path

# LoRA configuration class
class LoRAConfig:
    """Configuration class for LoRA parameters"""
    
    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout_rate: float = 0.0,
        target_modules: Optional[List[str]] = None,
        use_bias: bool = False,
        scaling_factor: float = 1.0
    ):
        """
        Initialize LoRA configuration.
        
        Args:
            rank: Rank of LoRA matrices
            alpha: Scaling factor for LoRA effect
            dropout_rate: Dropout probability for training
            target_modules: List of target module names
            use_bias: Whether to use bias
            scaling_factor: Additional scaling for gradients
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.target_modules = target_modules or []
        self.use_bias = use_bias
        self.scaling_factor = scaling_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout_rate": self.dropout_rate,
            "target_modules": self.target_modules,
            "use_bias": self.use_bias,
            "scaling_factor": self.scaling_factor
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """Create config from dictionary"""
        return cls(
            rank=config_dict.get("rank", 4),
            alpha=config_dict.get("alpha", 1.0),
            dropout_rate=config_dict.get("dropout_rate", 0.0),
            target_modules=config_dict.get("target_modules", []),
            use_bias=config_dict.get("use_bias", False),
            scaling_factor=config_dict.get("scaling_factor", 1.0)
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'LoRAConfig':
        """Load config from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save(self, config_path: str) -> None:
        """Save config to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


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
    device = tensor.device
    original_dtype = tensor.dtype
    
    # Always use the Rust backend by moving data to CPU if needed
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
        raise RuntimeError(f"Rust LORA implementation failed: {e}. Cannot continue without Rust.")


def apply_lora_batch_rust(
    tensor: torch.Tensor,
    lora_pairs: List[Tuple[torch.Tensor, torch.Tensor, float]]
) -> torch.Tensor:
    """
    Apply multiple LoRA transformations to a weight tensor using Rust backend.
    
    Args:
        tensor: Original weight tensor
        lora_pairs: List of (lora_down, lora_up, alpha) tuples
        
    Returns:
        Updated weight tensor
    """
    device = tensor.device
    original_dtype = tensor.dtype
    
    # Always use the Rust backend by moving data to CPU if needed
    tensor_cpu = tensor.detach().cpu().to(torch.float32)
    
    # Convert each pair
    rust_pairs = []
    for lora_down, lora_up, alpha in lora_pairs:
        lora_down_cpu = lora_down.detach().cpu().to(torch.float32).numpy()
        lora_up_cpu = lora_up.detach().cpu().to(torch.float32).numpy()
        rust_pairs.append((lora_down_cpu, lora_up_cpu, float(alpha)))
    
    try:
        # Call our batch function
        updated_tensor = lora_ops.apply_lora_batch(
            tensor_cpu.numpy(),
            rust_pairs
        )
        
        # Convert back to PyTorch tensor
        return torch.from_numpy(updated_tensor).to(device).to(original_dtype)
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(f"Rust batch LORA implementation failed: {e}. Cannot continue without Rust.")


def is_self_attention_layer(name: str) -> bool:
    """
    Check if a layer is a self-attention layer.
    
    Args:
        name: Layer name
        
    Returns:
        True if it's a self-attention layer, False otherwise
    """
    # Self-attention layers in SDXL have "attn1" in their name
    return "attn1" in name and any(x in name for x in ["to_q", "to_k", "to_v", "to_out"])


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


def validate_lora_weights(lora_state: Dict[str, torch.Tensor], target_modules: List[str]) -> Dict[str, bool]:
    """
    Validate LoRA weights against target modules.
    
    Args:
        lora_state: Dictionary of LoRA weights
        target_modules: List of target module names
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    for module_name in target_modules:
        down_key = f"{module_name}.lora_down.weight"
        up_key = f"{module_name}.lora_up.weight"
        
        # Check if both keys exist
        both_exist = down_key in lora_state and up_key in lora_state
        validation_results[module_name] = both_exist
        
        if both_exist:
            lora_down = lora_state[down_key]
            lora_up = lora_state[up_key]
            
            # Additional checks for shapes
            if lora_down.dim() != 2 or lora_up.dim() != 2:
                validation_results[module_name] = False
                print(f"⚠️ Invalid LoRA tensor dimensions for {module_name}")
    
    return validation_results


def apply_lora_to_self_attention(
    model, 
    lora_path: str, 
    alpha: Optional[float] = None, 
    target_modules: Optional[List[str]] = None,
    use_batch: bool = True,
    config: Optional[LoRAConfig] = None
) -> Dict[str, torch.Tensor]:
    """
    Apply LoRA to self-attention layers in an SDXL model.
    
    Args:
        model: SDXL UNet model
        lora_path: Path to LoRA weights file (.safetensors)
        alpha: Scaling factor for LoRA effect (overrides config if provided)
        target_modules: Optional list of specific target modules (overrides config if provided)
        use_batch: Whether to use batch processing
        config: Optional LoRA configuration (loaded from metadata if available)
        
    Returns:
        Dictionary of original weights for restoration
    """
    # Load LoRA weights
    lora_state = load_file(lora_path)
    
    # Try to load config from metadata
    if config is None:
        config_path = lora_path.replace('.safetensors', '_config.json')
        if os.path.exists(config_path):
            config = LoRAConfig.from_file(config_path)
            print(f"Loaded LoRA config from {config_path}")
        else:
            # Create default config
            config = LoRAConfig()
    
    # Override config with function parameters if provided
    effective_alpha = alpha if alpha is not None else config.alpha
    effective_target_modules = target_modules if target_modules is not None else config.target_modules
    
    if not effective_target_modules:
        effective_target_modules = find_self_attention_modules(model)
    
    # Store original weights for restoration
    original_weights = {}
    
    # Validate weights
    validation_results = validate_lora_weights(lora_state, effective_target_modules)
    valid_modules = [name for name, valid in validation_results.items() if valid]
    
    print(f"Applying LoRA (rank={config.rank}, alpha={effective_alpha}) to {len(valid_modules)} out of {len(effective_target_modules)} self-attention modules")
    
    if use_batch:
        # Group modules by shape for batch processing
        shape_groups = {}
        
        for module_name in valid_modules:
            # Find target module
            module = None
            for name, mod in model.named_modules():
                if name == module_name:
                    module = mod
                    break
            
            if module is None or not hasattr(module, 'weight'):
                continue
            
            # Store by shape
            weight_shape = tuple(module.weight.shape)
            if weight_shape not in shape_groups:
                shape_groups[weight_shape] = []
            
            shape_groups[weight_shape].append(module_name)
        
        # Process each shape group
        for shape, modules in shape_groups.items():
            if not modules:
                continue
            
            print(f"Batch processing {len(modules)} modules with shape {shape}")
            
            # Extract and store original weights
            weight_modules = []
            lora_pairs = []
            
            for module_name in modules:
                # Find module
                module = None
                for name, mod in model.named_modules():
                    if name == module_name:
                        module = mod
                        break
                
                if module is None or not hasattr(module, 'weight'):
                    continue
                
                # Get LoRA weights
                down_key = f"{module_name}.lora_down.weight"
                up_key = f"{module_name}.lora_up.weight"
                lora_down = lora_state[down_key]
                lora_up = lora_state[up_key]
                
                # Store original weight
                original_weights[module_name] = module.weight.data.clone()
                
                # Add to batch
                weight_modules.append(module)
                lora_pairs.append((lora_down, lora_up, effective_alpha))
            
            # Apply batch update to all modules in group
            if weight_modules:
                for i, module in enumerate(weight_modules):
                    updated_weight = apply_lora_rust(
                        module.weight, 
                        lora_pairs[i][0], 
                        lora_pairs[i][1], 
                        lora_pairs[i][2]
                    )
                    module.weight.data.copy_(updated_weight)
    else:
        # Apply LoRA to each target module individually
        for module_name in valid_modules:
            # Get LoRA weights
            down_key = f"{module_name}.lora_down.weight"
            up_key = f"{module_name}.lora_up.weight"
            
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
                    effective_alpha
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


class LoRACheckpointer:
    """Class to handle LoRA checkpointing during training"""
    
    def __init__(
        self, 
        output_dir: str, 
        base_name: str = "lora_checkpoint", 
        save_every: int = 1000,
        keep_last_n: int = 5,
        save_optimizer: bool = True
    ):
        """
        Initialize LoRA checkpointer.
        
        Args:
            output_dir: Directory to save checkpoints
            base_name: Base name for checkpoint files
            save_every: Save checkpoint every N steps
            keep_last_n: Number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.last_save_step = -1
        self.checkpoints = []
    
    def should_save(self, global_step: int) -> bool:
        """Check if checkpoint should be saved at current step"""
        return global_step % self.save_every == 0 and global_step > self.last_save_step
    
    def save_checkpoint(
        self, 
        lora_weights: Dict[str, torch.Tensor], 
        config: LoRAConfig,
        global_step: int, 
        optimizer_state: Optional[Dict] = None,
        train_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save LoRA checkpoint.
        
        Args:
            lora_weights: Dictionary of LoRA weights
            config: LoRA configuration
            global_step: Current training step
            optimizer_state: Optional optimizer state
            train_info: Optional training information
            
        Returns:
            Path to saved checkpoint
        """
        if not self.should_save(global_step):
            return ""
        
        # Prepare checkpoint name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.base_name}_{global_step}_{timestamp}"
        
        # Save weights
        weights_path = self.output_dir / f"{checkpoint_name}.safetensors"
        save_file(lora_weights, str(weights_path))
        
        # Save config
        config_path = self.output_dir / f"{checkpoint_name}_config.json"
        config.save(str(config_path))
        
        # Save optimizer state if requested
        if optimizer_state is not None and self.save_optimizer:
            optimizer_path = self.output_dir / f"{checkpoint_name}_optimizer.pt"
            torch.save(optimizer_state, str(optimizer_path))
        
        # Save training info if provided
        if train_info is not None:
            train_info_path = self.output_dir / f"{checkpoint_name}_train_info.json"
            with open(train_info_path, "w") as f:
                json.dump(train_info, f, indent=2)
        
        # Track checkpoint
        self.checkpoints.append(checkpoint_name)
        self.last_save_step = global_step
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(weights_path)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the recent N"""
        if len(self.checkpoints) > self.keep_last_n:
            checkpoints_to_remove = self.checkpoints[:-self.keep_last_n]
            for checkpoint_name in checkpoints_to_remove:
                # Remove weights
                weights_path = self.output_dir / f"{checkpoint_name}.safetensors"
                if weights_path.exists():
                    weights_path.unlink()
                
                # Remove config
                config_path = self.output_dir / f"{checkpoint_name}_config.json"
                if config_path.exists():
                    config_path.unlink()
                
                # Remove optimizer state
                optimizer_path = self.output_dir / f"{checkpoint_name}_optimizer.pt"
                if optimizer_path.exists():
                    optimizer_path.unlink()
                
                # Remove training info
                train_info_path = self.output_dir / f"{checkpoint_name}_train_info.json"
                if train_info_path.exists():
                    train_info_path.unlink()
            
            # Update checkpoints list
            self.checkpoints = self.checkpoints[-self.keep_last_n:]
    
    def load_latest_checkpoint(self) -> Tuple[Optional[str], Optional[LoRAConfig], Optional[Dict]]:
        """
        Load latest checkpoint.
        
        Returns:
            Tuple of (weights_path, config, optimizer_state)
        """
        # Find all checkpoints
        checkpoint_files = list(self.output_dir.glob(f"{self.base_name}_*.safetensors"))
        if not checkpoint_files:
            return None, None, None
        
        # Sort by modification time (latest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        # Extract base name without extension
        base_name = latest_checkpoint.stem
        
        # Load config
        config_path = self.output_dir / f"{base_name}_config.json"
        config = LoRAConfig.from_file(str(config_path)) if config_path.exists() else None
        
        # Load optimizer state
        optimizer_path = self.output_dir / f"{base_name}_optimizer.pt"
        optimizer_state = torch.load(str(optimizer_path)) if optimizer_path.exists() and self.save_optimizer else None
        
        return str(latest_checkpoint), config, optimizer_state


def train_lora_for_self_attention(
    unet,
    output_path: str,
    config: Optional[LoRAConfig] = None,
    target_modules: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Initialize empty LoRA weights for self-attention layers.
    This can be used as a starting point for training.
    
    Args:
        unet: UNet model to create LoRA for
        output_path: Path to save LoRA weights
        config: Optional LoRA configuration
        target_modules: Optional list of specific target modules (overrides config)
        
    Returns:
        Dictionary of LoRA weights
    """
    # Set default config if none provided
    if config is None:
        config = LoRAConfig()
    
    # Override target modules if provided
    effective_target_modules = target_modules if target_modules is not None else config.target_modules
    
    # Find target modules if list is empty
    if not effective_target_modules:
        effective_target_modules = find_self_attention_modules(unet)
    
    # Create LoRA weights
    lora_weights = {}
    
    for module_name in effective_target_modules:
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
        
        # Initialize LoRA weights with configured rank
        lora_down = torch.zeros((config.rank, weight_shape[0]), dtype=weight.dtype, device='cpu')
        lora_up = torch.zeros((weight_shape[1], config.rank), dtype=weight.dtype, device='cpu')
        
        # Initialize with small random values using scaling_factor from config
        torch.nn.init.kaiming_uniform_(lora_down, a=5**0.5 * config.scaling_factor)
        torch.nn.init.zeros_(lora_up)
        
        # Add to dictionary
        lora_weights[f"{module_name}.lora_down.weight"] = lora_down
        lora_weights[f"{module_name}.lora_up.weight"] = lora_up
        
        # Add bias if configured
        if config.use_bias:
            lora_bias = torch.zeros(weight_shape[0], dtype=weight.dtype, device='cpu')
            lora_weights[f"{module_name}.lora_bias"] = lora_bias
    
    # Save weights
    save_file(lora_weights, output_path)
    
    # Save config alongside weights
    config_path = output_path.replace('.safetensors', '_config.json')
    config.save(config_path)
    
    print(f"✅ Created LoRA weights (rank={config.rank}) for {len(effective_target_modules)} self-attention modules")
    print(f"✅ Saved to {output_path} with config at {config_path}")
    
    return lora_weights 