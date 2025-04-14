#!/bin/bash

# Script to download the gina_style_lora.safetensors model from the server

# Remote connection details
REMOTE_SERVER="root@63.141.33.78"
REMOTE_PORT="22163"
SSH_KEY="~/.ssh/my_custom_key"
REMOTE_PATH="/workspace/lora-trained/gina_style_lora.safetensors"

# Local destination
LOCAL_DIR="./gina_style_lora"
LOCAL_PATH="$LOCAL_DIR/gina_style_lora.safetensors"

# Create directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Display information
echo "Attempting to pull the gina style LoRA model..."
echo "From: $REMOTE_SERVER:$REMOTE_PATH"
echo "To: $LOCAL_PATH"
echo "Using SSH key: $SSH_KEY"
echo "Port: $REMOTE_PORT"

# Use SCP to copy the file
echo "Using SCP to copy the file..."
scp -i "$SSH_KEY" -P "$REMOTE_PORT" "$REMOTE_SERVER:$REMOTE_PATH" "$LOCAL_PATH"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Success! LoRA model copied to $LOCAL_PATH"
    
    # Display file size
    FILE_SIZE=$(du -h "$LOCAL_PATH" | cut -f1)
    echo "File size: $FILE_SIZE"
    
    echo ""
    echo "To upload this model to your other server, use:"
    echo "scp -i [YOUR_KEY] $LOCAL_PATH [USERNAME]@[OTHER_SERVER]:[DESTINATION_PATH]"
else
    echo "Error: Failed to download the LoRA model."
    exit 1
fi 