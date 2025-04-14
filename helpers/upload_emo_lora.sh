#!/bin/bash
# Script to upload the emo_style_lora model to another server
# This uses SCP with the appropriate SSH key

# Server details based on SSH command
REMOTE_USER="root"
REMOTE_HOST="63.141.33.78"
REMOTE_PORT="22163"
REMOTE_PATH="/workspace/emo_style_lora/"
SSH_KEY="~/.ssh/my_custom_key"

# Source path
LOCAL_MODEL="./emo_style_lora/emo_style_lora.safetensors"

# Check if model exists locally
if [ ! -f "$LOCAL_MODEL" ]; then
    echo "Error: Local model not found at $LOCAL_MODEL"
    echo "Make sure you have downloaded the model using pull_emo_lora.sh first."
    exit 1
fi

# Get model size for verification
MODEL_SIZE=$(du -h "$LOCAL_MODEL" | cut -f1)
echo "Uploading emo style LoRA model ($MODEL_SIZE)..."
echo "From: $LOCAL_MODEL"
echo "To: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "Using SSH key: $SSH_KEY"
echo "Port: $REMOTE_PORT"

# Confirm upload
read -p "Continue with upload? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Create the remote directory if it doesn't exist
echo "Creating remote directory if needed..."
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

# Upload using SCP
echo "Uploading model using SCP..."
scp -i "$SSH_KEY" -P "$REMOTE_PORT" "$LOCAL_MODEL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# Check if the upload was successful
if [ $? -eq 0 ]; then
    echo "Success! LoRA model uploaded to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    
    # Verify the file exists on the remote server
    echo "Verifying file on remote server..."
    REMOTE_SIZE=$(ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "du -h $REMOTE_PATH/$(basename $LOCAL_MODEL) | cut -f1")
    
    echo "Local file size: $MODEL_SIZE"
    echo "Remote file size: $REMOTE_SIZE"
    
    if [ -n "$REMOTE_SIZE" ]; then
        echo "Verification successful. File exists on remote server."
    else
        echo "Warning: Could not verify file size on remote server."
    fi
else
    echo "Error: Upload failed."
    exit 1
fi

echo "Upload process completed." 