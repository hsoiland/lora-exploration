# SDXL LoRA Training with Rust Acceleration

Custom LoRA training implementation for Stable Diffusion XL with Rust-accelerated operations.

## Features

- Train LoRAs for SDXL that target self-attention layers
- Rust-accelerated operations for better performance
- Built-in face training optimization
- CPU/GPU hybrid processing for maximum efficiency
- Full and lightweight training modes
- Text conditioning support
- Automatic mixed precision support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/loras.git
cd loras

# Install Python dependencies
pip install -r requirements.txt

# Build Rust extension
cd src/lora_ops && maturin develop && cd ../..
```

### Download SDXL Base Model

```bash
# Option 1: Use Python to download all components
python -c "import torch; from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', local_dir='sdxl-base-1.0', local_dir_use_symlinks=False)"

# Option 2: For minimal model (UNet + VAE only)
python download_minimal_sdxl.py
```

### Training a Face LoRA

1. **Prepare images**: Place 10-30 face images in a directory (e.g., `your_face_cropped/`)
2. **Train LoRA**:

```bash
# Standard training
python train_full_lora.py \
  --output_dir your_lora \
  --images_dir your_face_cropped \
  --lora_name your_lora \
  --num_train_epochs 300 \
  --rank 8 \
  --learning_rate 5e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision no

# For stronger impression
python train_full_lora.py \
  --output_dir your_strong_lora \
  --images_dir your_face_cropped \
  --lora_name your_strong_lora \
  --num_train_epochs 500 \
  --rank 16 \
  --learning_rate 8e-5 \
  --stability_mode
```

### Testing Your LoRA

```bash
# Generate test images
python test_lora.py \
  --lora_path your_lora/your_lora.safetensors \
  --alpha 1.0 \
  --prompt "A portrait of <your_character>, photorealistic, detailed" \
  --num_images 3

# Compare with/without LoRA
python compare_lora.py \
  --lora_path your_lora/your_lora.safetensors \
  --alpha 1.2 \
  --prompt "Portrait of <your_character> in a snowy setting" \
  --seed 42
```

## Resuming Training from a Checkpoint

If your training is interrupted, you can now easily resume it from where you left off:

```bash
python train_full_lora.py \
  --output_dir=your_output_dir \
  --resume_from=your_output_dir/checkpoint-1000/your_lora_name.safetensors \
  --train_data_dir=your_data_dir \
  [other original arguments]
```

This will:
1. Load the LoRA weights from the checkpoint
2. Restore the optimizer state (learning rates, momentum)
3. Restore the scheduler state 
4. Continue training from the exact step where it was interrupted

You can also resume from the latest checkpoint by using:

```bash
python train_full_lora.py \
  --output_dir=your_output_dir \
  --resume_from=your_output_dir/your_lora_name.safetensors \
  --train_data_dir=your_data_dir \
  [other original arguments]
```

The automatic checkpointing system saves checkpoints every 200 steps by default, but you can customize this with `--save_every_n_steps=N`.

If your training is interrupted on RunPod or another cloud provider and you need to start from scratch, make sure to maintain the same output directory structure. Checkpoints are saved both in the output_dir root and in `/checkpoint-STEP` subdirectories.

## How It Works

This implementation focuses on training LoRA weights for self-attention layers in SDXL's UNet, which are most responsible for subject identity and features. For improved performance:

1. All CPU operations use Rust-accelerated matrix multiplication
2. All GPU operations use PyTorch's native implementation to avoid unnecessary transfers
3. Training optimizations include mixed precision, gradient accumulation, and more

## Advanced Options

### Memory Optimization

- Lower memory usage: `--train_batch_size 1 --mixed_precision fp16`
- Higher quality: `--train_batch_size 1 --gradient_accumulation_steps 4 --mixed_precision no`
- Maximum stability: `--stability_mode`

### Training Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--rank` | LoRA capacity | 4 (small), 8 (medium), 16 (large) |
| `--learning_rate` | How fast to learn | 1e-4 (aggressive), 5e-5 (balanced), 1e-5 (stable) |
| `--num_train_epochs` | Training iterations | 100 (fast), 300 (good), 500+ (best) |
| `--alpha` | LoRA strength for testing | 0.8 (subtle), 1.2 (balanced), 1.8+ (strong) |

### Custom Target Modules

By default, training targets all self-attention layers. To customize:

```python
# Edit get_target_modules in train_full_lora.py
def get_target_modules(unet):
    # Add your custom target module selection logic here
    return custom_modules
```

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size, use mixed precision, or lower the rank
- **NaN Loss Values**: Enable stability mode, lower learning rate, or disable mixed precision
- **Poor Results**: Increase training epochs, increase rank, or check your training images
- **Build Errors**: Ensure Rust and maturin are installed properly

## License

MIT
