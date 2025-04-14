#!/bin/bash
# Run Rust-based LORA training on the emo dataset

set -e  # Exit on error
set -x  # Print commands as they execute

echo "=== Building Rust Library ==="
# Make sure the Rust library is built and copied to the right location
cd "$(dirname "$0")"
cargo build --release
cp target/release/liblora_ops.so src/lora_ops.so
echo "✅ Rust library built successfully"

echo "=== Setting Up Environment ==="
# Create output directory
mkdir -p ../emo_rust_lora

# Print CUDA info
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Test importing the modules - this will exit if Rust modules can't be loaded
echo "=== Testing Imports ==="
python test_imports.py

# Run with very minimal settings first as a test
echo "=== Starting Rust-Only LORA Training ==="
PYTHONPATH=. python src/train_diffusion_lora.py \
  --image_dir ../emo_dataset \
  --caption_file ../emo_dataset/captions.json \
  --output_dir ../emo_rust_lora \
  --model_id ../sdxl-base-1.0 \
  --rank 4 \
  --alpha 8 \
  --train_batch_size 1 \
  --max_train_steps 5 \
  --learning_rate 1e-4 \
  --center_crop \
  --random_flip \
  --save_steps 3

echo "=== Training Complete ==="
echo "LORA weights saved to ../emo_rust_lora/" 