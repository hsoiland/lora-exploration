#!/bin/bash
# Script to download the generated dual LoRA grid images from the server

# Server details
REMOTE_USER="root"
REMOTE_HOST="63.141.33.78"
REMOTE_PORT="22163"
SSH_KEY="~/.ssh/my_custom_key"
REMOTE_PATH="/workspace/dual_lora_outputs/"
LOCAL_PATH="./dual_lora_results/"

# Create local directory
mkdir -p "$LOCAL_PATH"

echo "Downloading dual LoRA grid images from server..."
echo "From: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "To: $LOCAL_PATH"

# Download using SCP (just the grid files, not the individual images)
scp -i "$SSH_KEY" -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/*dual_lora_alpha_grid.png" "$LOCAL_PATH"

# Download metadata files as well
scp -i "$SSH_KEY" -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/*dual_lora_alpha_grid.txt" "$LOCAL_PATH"

echo "Download completed. Check $LOCAL_PATH for the grid images." 