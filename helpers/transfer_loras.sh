#!/bin/bash

# Default SSH connection values
SSH_HOST="root@103.196.86.29"
SSH_PORT="14347"
SSH_KEY="~/.ssh/my_custom_key"

# Parse command line arguments
if [ $# -ge 1 ]; then
    # Check if the first argument includes spaces (likely a quoted string with options)
    if [[ "$1" == *" "* ]]; then
        # Split the quoted string into an array
        read -ra SSH_ARGS <<< "$1"
        
        # First part is the hostname
        SSH_HOST="${SSH_ARGS[0]}"
        
        # Process the rest of the arguments
        for ((i=1; i<${#SSH_ARGS[@]}; i++)); do
            case "${SSH_ARGS[i]}" in
                -p)
                    SSH_PORT="${SSH_ARGS[i+1]}"
                    ((i++))
                    ;;
                -i)
                    SSH_KEY="${SSH_ARGS[i+1]}"
                    ((i++))
                    ;;
            esac
        done
    else
        # Just a simple hostname
        SSH_HOST="$1"
        shift
        
        # Check for additional arguments
        while [ $# -gt 0 ]; do
            case "$1" in
                -p)
                    SSH_PORT="$2"
                    shift 2
                    ;;
                -i)
                    SSH_KEY="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown option: $1"
                    shift
                    ;;
            esac
        done
    fi
fi

# Remote workspace directory
REMOTE_DIR="/workspace"

echo "===== TRANSFERRING FILES TO REMOTE SERVER ====="
echo "Using SSH connection: $SSH_HOST (port $SSH_PORT, key $SSH_KEY)"
echo "Target directory: $REMOTE_DIR"

# Create remote directories
ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "mkdir -p $REMOTE_DIR/ilya_repin_style $REMOTE_DIR/src/lora_ops/src $REMOTE_DIR/repin_lora $REMOTE_DIR/gina_dataset_cropped $REMOTE_DIR/emo_dataset"

# First, create a tar archive excluding large safetensors files
cd ~/loras
tar --exclude="*.safetensors" --exclude="sdxl-base-1.0" --exclude="venv" -czf /tmp/loras_transfer.tar.gz .

# Transfer the archive
scp -P "$SSH_PORT" -i "$SSH_KEY" /tmp/loras_transfer.tar.gz "$SSH_HOST:/tmp/"

# Extract on the remote server with --no-same-owner to fix permission issues
ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "cd $REMOTE_DIR && tar --no-same-owner -xzf /tmp/loras_transfer.tar.gz"

# Create separate tar files for the new datasets to ensure they're transferred properly
echo "Creating archive for gina_dataset_cropped..."
cd ~/loras
tar -czf /tmp/gina_dataset.tar.gz gina_dataset_cropped

echo "Creating archive for emo_dataset..."
tar -czf /tmp/emo_dataset.tar.gz emo_dataset

# Transfer the dataset archives
echo "Transferring gina_dataset_cropped..."
scp -P "$SSH_PORT" -i "$SSH_KEY" /tmp/gina_dataset.tar.gz "$SSH_HOST:/tmp/"

echo "Transferring emo_dataset..."
scp -P "$SSH_PORT" -i "$SSH_KEY" /tmp/emo_dataset.tar.gz "$SSH_HOST:/tmp/"

# Extract the datasets on the remote server with --no-same-owner
echo "Extracting datasets on remote server..."
ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "cd $REMOTE_DIR && tar --no-same-owner -xzf /tmp/gina_dataset.tar.gz && tar --no-same-owner -xzf /tmp/emo_dataset.tar.gz"

# Check if images are present
IMAGE_COUNT_ILYA=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "ls -1 $REMOTE_DIR/ilya_repin_style/*.jpg 2>/dev/null | wc -l")
if [ "$IMAGE_COUNT_ILYA" -eq 0 ]; then
    echo "⚠️ No images found in the remote ilya_repin_style directory! You need to transfer your Ilya Repin images."
    echo "Use this command to transfer images:"
    echo "scp -P $SSH_PORT -i $SSH_KEY ~/loras/ilya_repin_style/*.jpg $SSH_HOST:$REMOTE_DIR/ilya_repin_style/"
fi

IMAGE_COUNT_GINA=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "ls -1 $REMOTE_DIR/gina_dataset_cropped/*.jpg 2>/dev/null | wc -l")
if [ "$IMAGE_COUNT_GINA" -eq 0 ]; then
    echo "⚠️ No images found in the remote gina_dataset_cropped directory!"
fi

IMAGE_COUNT_EMO=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "ls -1 $REMOTE_DIR/emo_dataset/*.jpg 2>/dev/null | wc -l")
if [ "$IMAGE_COUNT_EMO" -eq 0 ]; then
    echo "⚠️ No images found in the remote emo_dataset directory!"
fi

# Clean up
rm /tmp/loras_transfer.tar.gz /tmp/gina_dataset.tar.gz /tmp/emo_dataset.tar.gz
ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "rm /tmp/loras_transfer.tar.gz /tmp/gina_dataset.tar.gz /tmp/emo_dataset.tar.gz"

echo "==============================================================="
echo "Transfer complete! All files except safetensors and SDXL base model transferred."
echo ""
echo "To run on server:"
echo "1. ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST"
echo "2. cd $REMOTE_DIR"
echo ""
echo "3. Install dependencies:"
echo "   apt-get update && apt-get install -y python3-pip python3-venv python3-dev build-essential"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install --upgrade pip"
echo "   pip install numpy==1.24.3"
echo "   # Install PyTorch 2.2.1 which is compatible with stable xformers"
echo "   pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121"
echo "   pip install diffusers transformers safetensors pillow tqdm accelerate huggingface_hub"
echo "   # Install compatible xformers for memory efficient attention"
echo "   pip install xformers==0.0.23"
echo ""
echo "   # Install Rust and maturin for the Rust LoRA module:"
echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
echo "   source \$HOME/.cargo/env"
echo "   pip install maturin"
echo "   cd $REMOTE_DIR/src/lora_ops && maturin develop && cd $REMOTE_DIR"
echo ""
echo "4. Download minimal SDXL (required for training):"
echo "   python download_minimal_sdxl.py"
echo "   python download_tokenizers.py"
echo ""
echo "5. Process Ilya Repin style images and start training:"
echo "   python process_repin.py"
echo "   python fix_vae.py"
echo ""
echo "6. Training options:"
echo ""
echo "   a) Train Ilya Repin style LoRA:"
echo "      python train_full_lora.py \\"
echo "        --output_dir=repin_lora \\"
echo "        --images_dir=processed_repin \\"
echo "        --captions_file=repin_captions.json \\"
echo "        --lora_name=repin_style \\"
echo "        --num_train_epochs=50 \\"
echo "        --rank=32 \\"
echo "        --learning_rate=1e-4 \\"
echo "        --train_batch_size=8 \\"
echo "        --gradient_accumulation_steps=1 \\"
echo "        --use_text_conditioning=True \\"
echo "        --cache_latents \\"
echo "        --mixed_precision=bf16 \\"
echo "        --enable_xformers_memory_efficient_attention \\"
echo "        --use_rust=True \\"
echo "        --save_every_n_steps=25"
echo ""
echo "   b) Train Gina LoRA:"
echo "      python train_full_lora.py \\"
echo "        --output_dir=gina_lora \\"
echo "        --images_dir=gina_dataset_cropped \\"
echo "        --captions_file=gina_dataset_cropped/captions_blip2.json \\"
echo "        --lora_name=gina_portrait \\"
echo "        --num_train_epochs=50 \\"
echo "        --rank=32 \\"
echo "        --learning_rate=1e-4 \\"
echo "        --train_batch_size=8 \\"
echo "        --gradient_accumulation_steps=1 \\"
echo "        --use_text_conditioning=True \\"
echo "        --cache_latents \\"
echo "        --mixed_precision=bf16 \\"
echo "        --enable_xformers_memory_efficient_attention \\"
echo "        --use_rust=True \\"
echo "        --save_every_n_steps=25"
echo ""
echo "   c) Train Emo Style LoRA:"
echo "      python train_full_lora.py \\"
echo "        --output_dir=emo_lora \\"
echo "        --images_dir=emo_dataset \\"
echo "        --captions_file=emo_dataset/captions.json \\"
echo "        --lora_name=emo_style \\"
echo "        --num_train_epochs=50 \\"
echo "        --rank=32 \\"
echo "        --learning_rate=1e-4 \\"
echo "        --train_batch_size=8 \\"
echo "        --gradient_accumulation_steps=1 \\"
echo "        --use_text_conditioning=True \\"
echo "        --cache_latents \\"
echo "        --mixed_precision=bf16 \\"
echo "        --enable_xformers_memory_efficient_attention \\"
echo "        --use_rust=True \\"
echo "        --save_every_n_steps=25"
echo ""
echo "7. To check training progress from another terminal:"
echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST \"tail -f $REMOTE_DIR/repin_lora/train.log\""
echo "   # Or for other datasets:"
echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST \"tail -f $REMOTE_DIR/gina_lora/train.log\""
echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST \"tail -f $REMOTE_DIR/emo_lora/train.log\""
echo ""
echo "8. To download your trained LoRA files back to your local machine:"
echo "   scp -P $SSH_PORT -i $SSH_KEY $SSH_HOST:$REMOTE_DIR/repin_lora/repin_style.safetensors ~/Downloads/"
echo "   scp -P $SSH_PORT -i $SSH_KEY $SSH_HOST:$REMOTE_DIR/gina_lora/gina_portrait.safetensors ~/Downloads/"
echo "   scp -P $SSH_PORT -i $SSH_KEY $SSH_HOST:$REMOTE_DIR/emo_lora/emo_style.safetensors ~/Downloads/"
echo ""
echo "9. To test your LoRA with comparisons:"
echo "   python compare_lora.py \\"
echo "     --lora_path=repin_lora/repin_style.safetensors \\"
echo "     --prompt=\"Portrait of a person in the style of classical Russian painting, detailed\" \\"
echo "     --output_dir=repin_comparison \\"
echo "     --num_comparisons=1 \\"
echo "     --alpha=0.8"
echo ""
echo "10. To download the comparison images:"
echo "    scp -P $SSH_PORT -i $SSH_KEY \"$SSH_HOST:$REMOTE_DIR/repin_comparison/*.png\" ~/Downloads/"
echo ""
echo "11. To run training with nohup (continues even if connection drops):"
echo "    cd $REMOTE_DIR && source venv/bin/activate && nohup python train_full_lora.py \\"
echo "      --output_dir=repin_lora \\"
echo "      --images_dir=processed_repin \\"
echo "      --captions_file=repin_captions.json \\"
echo "      --lora_name=repin_style \\"
echo "      --num_train_epochs=50 \\"
echo "      --rank=32 \\"
echo "      --learning_rate=1e-4 \\"
echo "      --train_batch_size=8 \\"
echo "      --gradient_accumulation_steps=1 \\"
echo "      --use_text_conditioning=True \\"
echo "      --cache_latents \\"
echo "      --mixed_precision=bf16 \\"
echo "      --use_rust=True \\"
echo "      --save_every_n_steps=25 \\"
echo "      > repin_lora/training.log 2>&1 &"
echo ""
echo "    # Check progress with:"
echo "    tail -f repin_lora/training.log"
echo ""
echo "    # If you need to stop the training:"
echo "    ps aux | grep train_full_lora.py"
echo "    kill [PROCESS_ID]"
echo ""
echo "===============================================================" 