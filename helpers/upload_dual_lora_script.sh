#!/bin/bash
# Script to update the dual LoRA run script on the server with the second LoRA

# Server details
REMOTE_USER="root"
REMOTE_HOST="63.141.33.78"
REMOTE_PORT="22163"
SSH_KEY="~/.ssh/my_custom_key"

# Update the run script on the remote server with both LoRAs
echo "Updating run script on server with both LoRAs..."
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "cat > /workspace/scripts/run_dual_lora.sh << 'EOF'
#!/bin/bash
# Script to run dual LoRA grid generation with both emo_style and gina_style LoRAs

# Create output directory
mkdir -p /workspace/dual_lora_outputs

# Activate virtual environment
source /workspace/venv/bin/activate

# Run the script
cd /workspace/scripts
python fixed_dual_lora_grid.py \\
  --lora1_model /workspace/emo_style_lora/emo_style_lora.safetensors \\
  --lora2_model /workspace/lora-trained/gina_style_lora.safetensors \\
  --lora1_name \"Emo Style\" \\
  --lora2_name \"Gina Style\" \\
  --prompt \"<gina_szanyel>, <XAJI0Y6D>, a woman with black hair, side bangs, pale skin, high quality photo of Georgina, heavy eyeliner, black choker, band t-shirt, mirror selfie, Canon 35mm\" \\
  --output_dir /workspace/dual_lora_outputs \\
  --min_alpha 0.0 \\
  --max_alpha 1.0 \\
  --alpha_step 0.2 \\
  --cfg_scales 3.0,7.0 \\
  --num_inference_steps 30 \\
  --image_size 1024

echo \"Dual LoRA grid generation completed!\"
echo \"Results saved to: /workspace/dual_lora_outputs\"
EOF"

# Make the remote run script executable
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "chmod +x /workspace/scripts/run_dual_lora.sh"

echo ""
echo "Updated run script on server at: /workspace/scripts/run_dual_lora.sh"
echo "This script will now use both LoRAs:"
echo "1. Emo Style: /workspace/emo_style_lora/emo_style_lora.safetensors"
echo "2. Gina Style: /workspace/lora-trained/gina_style_lora.safetensors"
echo ""
echo "To run it, connect via SSH and execute: /workspace/scripts/run_dual_lora.sh" 