# Rust LORA Implementation

A fast LORA (Low-Rank Adaptation) implementation with Rust backend for performance. Supports both inference and training.

## Setup

1. Build the Rust extension:
```bash
cargo build --release
```

2. Use the package for inference:
```bash
python -m src --model <path> --lora <path> --layers <layer-names> --alpha <value> --output <path>
```

## Training with Rust

The implementation now supports efficient Rust-based training:

```bash
python -m src.lora_trainer --model <model-path> --data <data-path> --output <output-path> --target-modules <modules> --rank 8 --alpha 16.0 --epochs 3 --lr 1e-4
```

### Training Parameters

- `--model`: Path to the PyTorch model file
- `--data`: Path to training data directory
- `--output`: Path to save the trained LORA weights
- `--target-modules`: Comma-separated list of module names to apply LORA to
- `--rank`: Rank of LORA decomposition matrices (default: 8)
- `--alpha`: Scaling factor for LORA updates (default: 16.0)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 1e-4)

## Key Benefits of Rust-based LORA

- **Performance**: Training operations are accelerated by native Rust implementation
- **Memory Efficiency**: Uses less memory than standard PyTorch implementations
- **Precision Control**: Full control over numeric precision and computation

## Usage as a Library

```python
# For Inference
from rust_lora.src import inject_lora_into_model, apply_lora_to_qkv

# Load your model
model = ...

# Apply LORA
inject_lora_into_model(model, lora_path, layer_names, alpha)

# For Training
from rust_lora.src.lora_trainer import train_lora, RustLoraTrainer

# Train a model with LORA
train_lora(
    model=model,
    train_dataloader=dataloader,
    target_modules=["attention", "mlp"],
    output_path="trained_lora.pt",
    num_epochs=3,
    rank=8,
    alpha=16.0
) 