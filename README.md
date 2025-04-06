# LoRA Rust Optimizer

This package provides an efficient implementation of Low-Rank Adaptation (LoRA) for PyTorch models using a Rust backend. It allows you to apply LoRA transformations to model weights with better performance than pure Python implementations.

## Setup

This project uses Poetry for Python dependency management and Maturin for Rust integration:

```bash
# Install dependencies and build the Rust module
poetry install
poetry run maturin develop
```

## Usage

### As a Python Module

```python
import torch
from loras.lora import inject_lora_into_model

# Load your model
model = torch.load("path/to/model.pth")

# Define which layers to apply LoRA to
target_layers = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj"
]

# Apply LoRA weights to your model
inject_lora_into_model(
    model=model,
    lora_path="path/to/lora.safetensors",
    target_layer_names=target_layers,
    alpha=1.0  # scaling factor
)

# Save the merged model
torch.save(model, "path/to/merged_model.pth")
```

### Command Line

```bash
# Apply LoRA weights to a model
poetry run python -m loras \
    --model path/to/model.pth \
    --lora path/to/lora.safetensors \
    --layers "model.layers.0.self_attn.q_proj,model.layers.0.self_attn.k_proj" \
    --alpha 1.0 \
    --output path/to/merged_model.pth
```

## How it Works

LoRA applies the transformation: `x + alpha * (B @ A @ x)` where:
- `x` is the original weight tensor
- `A` and `B` are low-rank decomposition matrices
- `alpha` is a scaling factor

Our implementation offloads the matrix multiplication operations to Rust for better performance, particularly useful for large models. # lora-exploration
