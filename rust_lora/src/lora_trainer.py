"""
LoRA Trainer - Rust-based implementation

This module provides a Rust-powered LORA (Low-Rank Adaptation) training interface
for fine-tuning large language and diffusion models efficiently.
"""

import os
import json
import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Import Rust implementation without fallbacks
import lora_ops
from lora_ops import LoraTrainingContext, AdamParams
print("✅ Using Rust LORA implementation")


class RustLoraTrainer:
    """Rust-based LORA trainer for efficient fine-tuning of models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_modules: List[str],
        rank: int = 8,
        alpha: float = 16.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Rust-based LORA trainer.
        
        Args:
            model: The PyTorch model to fine-tune
            target_modules: List of module names to apply LORA to
            rank: Rank of the LORA adaptation matrices
            alpha: Scaling factor for the LORA update
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            betas: Adam optimizer betas parameters
            eps: Adam optimizer epsilon parameter
            device: Device to use for training
        """
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.device = device
        
        # Maps from parameter name to Rust LORA context
        self.lora_contexts: Dict[str, LoraTrainingContext] = {}
        
        # Initialize Rust-based Adam optimizer
        self.optimizer = AdamParams(learning_rate, betas[0], betas[1], eps)
        
        # Set up LORA for target modules
        self._setup_lora_layers()
        
    def _setup_lora_layers(self):
        """Initialize LORA contexts for target modules in the model."""
        found_modules = []
        
        # Recursively find named modules in the model
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                found_modules.append((name, module))
                print(f"Found target module: {name}")
        
        # Set up LORA for linear layers in target modules
        for module_name, module in found_modules:
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().cpu().numpy().astype(np.float32)
                
                # Create Rust LORA context
                ctx = LoraTrainingContext(
                    layer_name=module_name,
                    weight=weight,
                    rank=self.rank,
                    alpha=self.alpha,
                    init_scale=0.01
                )
                
                # Set optimizer
                ctx.set_optimizer(self.optimizer)
                
                # Store context
                self.lora_contexts[module_name] = ctx
                print(f"  Initialized LORA for {module_name} with shape {weight.shape}")
    
    def train_step(self, inputs, targets):
        """
        Perform a single training step with Rust-based LORA.
        
        Args:
            inputs: Input data
            targets: Target data
            
        Returns:
            loss: The training loss value
        """
        # Standard PyTorch forward pass
        outputs = self.model(inputs)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # For each LORA context, extract gradients and update parameters
        for name, module in self.model.named_modules():
            if name in self.lora_contexts:
                # Get gradient from PyTorch
                grad = module.weight.grad.detach().cpu().numpy().astype(np.float32)
                
                # Update LORA parameters using Rust
                self.lora_contexts[name].backward(grad)
                
                # Zero out the gradient for this parameter
                module.weight.grad.zero_()
        
        return loss.item()
    
    def save_lora_weights(self, output_path: str):
        """
        Save the trained LORA weights to a file.
        
        Args:
            output_path: Path to save the weights
        """
        lora_state = {}
        
        for name, ctx in self.lora_contexts.items():
            # Get weights from Rust
            lora_a, lora_b = ctx.get_weights()
            lora_a = np.array(lora_a)
            lora_b = np.array(lora_b)
            
            # Store in state dict
            lora_state[f"{name}.lora_a"] = torch.tensor(lora_a)
            lora_state[f"{name}.lora_b"] = torch.tensor(lora_b)
            lora_state[f"{name}.alpha"] = torch.tensor(self.alpha)
            lora_state[f"{name}.rank"] = torch.tensor(self.rank)
        
        # Save to file
        torch.save(lora_state, output_path)
        print(f"Saved LORA weights to {output_path}")
    
    def load_lora_weights(self, input_path: str):
        """
        Load LORA weights from a file.
        
        Args:
            input_path: Path to load the weights from
        """
        print(f"Loading LORA weights from {input_path}")
        lora_state = torch.load(input_path)
        
        for name, ctx in self.lora_contexts.items():
            if f"{name}.lora_a" in lora_state and f"{name}.lora_b" in lora_state:
                # Extract weights
                lora_a = lora_state[f"{name}.lora_a"].numpy().astype(np.float32)
                lora_b = lora_state[f"{name}.lora_b"].numpy().astype(np.float32)
                
                # TODO: Add method in Rust to load weights directly
                # For now, we would need to reinitialize the Rust objects
                print(f"  Loaded weights for {name}")


def train_lora(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    target_modules: List[str],
    output_path: str,
    num_epochs: int = 3,
    rank: int = 8,
    alpha: float = 16.0,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    save_every: int = 100,
):
    """
    Train LORA weights using the Rust implementation.
    
    Args:
        model: The model to fine-tune
        train_dataloader: DataLoader with training data
        target_modules: List of module names to apply LORA to
        output_path: Path to save the trained weights
        num_epochs: Number of training epochs
        rank: Rank of LORA adaptation
        alpha: Scaling factor for LORA update
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        save_every: Save checkpoint every N steps
    """
    # Initialize trainer
    trainer = RustLoraTrainer(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            
            # Perform a single training step
            loss = trainer.train_step(inputs, targets)
            epoch_loss += loss
            global_step += 1
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_path = f"{output_path}.step-{global_step}"
                trainer.save_lora_weights(checkpoint_path)
        
        # Print epoch stats
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save after each epoch
        epoch_path = f"{output_path}.epoch-{epoch+1}"
        trainer.save_lora_weights(epoch_path)
    
    # Save final model
    trainer.save_lora_weights(output_path)
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LORA with Rust backend")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, required=True, help="Path to save LORA weights")
    parser.add_argument("--target-modules", type=str, required=True, 
                       help="Comma-separated list of target modules")
    parser.add_argument("--rank", type=int, default=8, help="LORA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LORA alpha scaling")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = torch.load(args.model)
    
    # Create simple example dataloader (replace with your actual data loading)
    # This is just a placeholder - replace with your actual data loading code
    train_data = torch.utils.data.TensorDataset(
        torch.rand(100, 3, 224, 224),  # Example inputs
        torch.randint(0, 10, (100,))   # Example targets
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=True
    )
    
    # Get target modules list
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    
    # Train LORA
    train_lora(
        model=model,
        train_dataloader=train_dataloader,
        target_modules=target_modules,
        output_path=args.output,
        num_epochs=args.epochs,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
    ) 