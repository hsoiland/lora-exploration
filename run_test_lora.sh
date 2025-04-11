#!/bin/bash

# Set error handling
set -e

echo "========================================================"
echo "       🚀 SDXL LoRA RunPod Test Script"
echo "========================================================"

# Step 1: Download the correct SDXL model files
echo "📦 Step 1: Downloading SDXL model files with configs..."
python download_minimal_sdxl.py

# Step 2: Fix the indentation issues in the training script
echo "🔧 Step 2: Fixing indentation in training script..."
python fix_indentation.py

# Step 3: Run a small test training
echo "🏃 Step 3: Running a quick test training (5 epochs)..."
echo ""
python train_full_lora_fixed.py \
  --output_dir=test_lora \
  --images_dir=gina_face_cropped \
  --lora_name=test_lora \
  --num_train_epochs=5 \
  --rank=4 \
  --use_rust=True \
  --learning_rate=1e-4 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --save_every_n_steps=10 \
  --cache_latents=True

echo ""
echo "✅ Test complete! If no errors appeared, your setup is working."
echo "You can now run a full training with your desired parameters."
echo ""
echo "Example full training command:"
echo "python train_full_lora_fixed.py --output_dir=full_training --images_dir=gina_face_cropped --lora_name=gina_lora --num_train_epochs=300 --rank=16 --use_rust=True --learning_rate=1e-4 --train_batch_size=4 --gradient_accumulation_steps=1 --save_every_n_steps=200 --cache_latents=True"
echo "========================================================" 