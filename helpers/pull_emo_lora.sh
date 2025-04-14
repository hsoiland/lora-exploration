#!/bin/bash
# Script to pull the emo style LoRA model from the remote server

# Remote connection details
REMOTE_HOST="root@205.196.17.18"
REMOTE_PORT="9022"
SSH_KEY="~/.ssh/my_custom_key"
REMOTE_PATH="/workspace/emo_style_lora/emo_style_lora.safetensors"

# Local paths
LOCAL_DIR="./emo_style_lora"
LOCAL_PATH="$LOCAL_DIR/emo_style_lora.safetensors"

# Create local directory
mkdir -p $LOCAL_DIR

# Display information
echo "Attempting to pull the emo style LoRA model..."
echo "From: $REMOTE_HOST:$REMOTE_PATH"
echo "To: $LOCAL_PATH"
echo "Using SSH key: $SSH_KEY"
echo "Port: $REMOTE_PORT"

# Use SCP with the provided SSH details
if command -v scp &> /dev/null; then
  echo "Using SCP to copy the file..."
  scp -P $REMOTE_PORT -i $SSH_KEY "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
  
  if [ $? -ne 0 ]; then
    echo "SCP failed. Please check your connection details and that the file exists."
    exit 1
  fi
else
  echo "SCP command not found. Please install SSH client."
  exit 1
fi

# Check if the file was successfully copied
if [ -f "$LOCAL_PATH" ]; then
  echo "Success! LoRA model copied to $LOCAL_PATH"
  
  # Get file size
  FILE_SIZE=$(ls -lh "$LOCAL_PATH" | awk '{print $5}')
  echo "File size: $FILE_SIZE"
  
  # Prepare for upload to other server
  echo ""
  echo "To upload this model to your other server, use:"
  echo "scp -i [YOUR_KEY] $LOCAL_PATH [USERNAME]@[OTHER_SERVER]:[DESTINATION_PATH]"
  echo ""
  echo "Or you can run the test script locally: ./test_emo_lora.sh"
else
  echo "Failed to copy the model file."
  echo "Please ensure the source path is correct and accessible."
  exit 1
fi 