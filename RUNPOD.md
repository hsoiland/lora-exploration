# Running LoRA Rust Optimizer on RunPod

This guide explains how to set up and run the LoRA Rust Optimizer repository on RunPod.

## Setup Instructions

1. Create a new RunPod instance with PyTorch
   - Recommended: Select a GPU template with PyTorch pre-installed
   - Minimum 16GB VRAM recommended for SDXL

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/loras.git
   cd loras
   ```

3. Run the setup script:
   ```bash
   chmod +x runpod_setup.sh
   ./runpod_setup.sh
   ```
   
   The setup script will:
   - Install required system dependencies
   - Install Rust and Poetry
   - Install Python dependencies
   - Build the Rust extension
   - Automatically download the SDXL base model

## Training a Face LoRA

1. Prepare your dataset:
   - Place face images in a directory (e.g., `face_images/`)
   - Images should be consistent in style/subject
   - You can use the included `face_crop.py` script to crop faces from photos:
     ```bash
     poetry run python face_crop.py --input your_photos/ --output face_images/
     ```

2. Run the training script:
   ```bash
   poetry run python train_faces_self_attn_lora.py \
       --base_model sdxl-base-1.0 \
       --images_dir face_images \
       --output_dir trained_lora \
       --lora_name my_face_lora \
       --num_train_epochs 10
   ```

   Additional options:
   - `--rank`: LoRA rank (default: 4)
   - `--learning_rate`: Training learning rate (default: 1e-4)
   - `--train_batch_size`: Batch size (default: 1)
   - `--image_size`: Training image size (default: 512)

3. The trained LoRA will be saved to `trained_lora/my_face_lora.safetensors`

## Using Your Trained LoRA

You can use your trained LoRA with:

1. This repository's generation script
2. Tools like ComfyUI, A1111, or other SDXL frontends

For ComfyUI, load it as a standard SDXL LoRA with your preferred weight.

## Troubleshooting

- If you encounter CUDA out of memory errors, reduce batch size or image size
- If the Rust extension fails to build, ensure you have the required system dependencies
- For other issues, check the error logs and create an issue in the repository

## Resource Management

- Training on SDXL can be memory intensive
- Monitor GPU memory usage with `nvidia-smi`
- Consider using smaller batch sizes for larger models
- The SDXL model requires approximately 13GB of disk space 