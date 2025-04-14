# SDXL LoRA Training with Rust Acceleration

Custom LoRA training implementation for Stable Diffusion XL with various acceleration methods and utilities.

## Components

### Core Training Scripts

- **train_sdxl_lora_simple_h100.py**: Advanced training script optimized for H100 GPUs with:
  - Multiple precision modes (fp16, bf16)
  - PyTorch's native scaled dot product attention
  - Gradient accumulation and checkpointing
  - Extensive command-line options

- **rust_lora**: High-performance Rust implementation for LoRA operations
  - Rust-accelerated matrix operations
  - Native integration with PyTorch for training and inference

### Utilities and Visualization

- **multi_cfg_dual_lora_grid.py**: Generate comparison grids showing combinations of two LoRAs
  - Creates 10x10 grids with different alpha values
  - Visualizes impact of different strength settings

- **lora-explainer-2**: Interactive web application for visualizing LoRA training
  - Built with React, TypeScript and Vite
  - Animated visualizations of LoRA training process
  - Educational tool for understanding LoRA adaptation

### Training LoRAs

1. **Prepare images**: Place 10-30 face images in a directory (e.g., `your_face_cropped/`)
2. **Choose a training script**:

```bash
# H100-optimized training 
python train_sdxl_lora_simple_h100.py \
  --output_dir your_h100_lora \
  --images_dir your_dataset \
  --model_name your_lora_name \
  --num_train_epochs 300 \
  --rank 16 \
  --learning_rate 8e-5 \
  --mixed_precision bf16 \
  --use_pytorch_sdpa \
  --train_batch_size 4
```

### Testing and Visualization

```bash

# Create LoRA combination grids
python multi_cfg_dual_lora_grid.py \
  --lora1_model path/to/first_lora.safetensors \
  --lora2_model path/to/second_lora.safetensors \
  --lora1_name "Character Style" \
  --lora2_name "Background Style" \
  --prompt "A detailed portrait in a scenic environment" \
  --cfg_scale 10.0

# Run the LoRA explainer web app
cd lora-explainer-2 && npm install && npm run dev
```

## How It Works

This implementation focuses on training LoRA weights for self-attention layers in SDXL's UNet, which are most responsible for subject identity and features:

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

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size, use mixed precision, or lower the rank
- **NaN Loss Values**: Enable stability mode, lower learning rate, or disable mixed precision
- **Poor Results**: Increase training epochs, increase rank, or check your training images
- **Build Errors**: Ensure Rust and maturin are installed properly

## License

MIT
