#!/bin/bash

# Multi-dataset LoRA Training Script
# This script trains two separate LoRA models and then combines them for testing
# Optimized for 12GB VRAM

echo "===== MULTI-DATASET LORA TRAINING ====="

# Set up output directories
REPIN_DIR="repin_style_lora"
GINA_DIR="gina_lora"
COMBINED_DIR="combined_lora"

mkdir -p $REPIN_DIR
mkdir -p $GINA_DIR
mkdir -p $COMBINED_DIR

# Generate captions for Gina dataset if needed
GINA_CAPTIONS="gina_face_cropped/captions.json"
if [ ! -f "$GINA_CAPTIONS" ]; then
    echo "Creating captions for Gina dataset..."
    python -c "
import os
import json

image_files = [f for f in os.listdir('gina_face_cropped') if f.endswith('.jpg')]
captions = {}
for img in image_files:
    captions[img] = 'A portrait photograph of Gina, a woman with red hair'

with open('gina_face_cropped/captions.json', 'w') as f:
    json.dump(captions, f, indent=2)
print(f'Created captions for {len(captions)} images')
"
fi

# Memory cleanup function
cleanup_memory() {
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('Cleared CUDA cache')
"
}

# Run initial memory cleanup
cleanup_memory

# STEP 1: Train Repin Style LoRA
echo -e "\n\n===== STEP 1: TRAINING REPIN STYLE LORA ====="
poetry run python test_multi_lora.py \
  --base_model "sdxl-base-1.0" \
  --images_dir "ilya_repin_young_women" \
  --output_dir "$REPIN_DIR" \
  --num_train_epochs 5 \
  --max_train_steps 15 \
  --image_size 256 \
  --rank 4 \
  --mixed_precision "fp16" \
  --gradient_checkpointing \
  --enable_xformers \
  --prompt "A portrait of a beautiful young woman in the style of Ilya Repin, fine art, oil painting, classical art"

if [ ! -f "$REPIN_DIR/self_attn_lora_test.safetensors" ] || [ ! -f "$REPIN_DIR/cross_attn_lora_test.safetensors" ]; then
    echo -e "\n❌ Repin style LoRA training failed"
    exit 1
fi

echo -e "✅ Repin style LoRA training complete"
cleanup_memory

# STEP 2: Train Gina Face LoRA
echo -e "\n\n===== STEP 2: TRAINING GINA FACE LORA ====="
poetry run python test_multi_lora.py \
  --base_model "sdxl-base-1.0" \
  --images_dir "gina_face_cropped" \
  --output_dir "$GINA_DIR" \
  --num_train_epochs 5 \
  --max_train_steps 15 \
  --image_size 256 \
  --rank 4 \
  --mixed_precision "fp16" \
  --gradient_checkpointing \
  --enable_xformers \
  --prompt "A photograph of Gina, a woman with red hair and beautiful features"

if [ ! -f "$GINA_DIR/self_attn_lora_test.safetensors" ] || [ ! -f "$GINA_DIR/cross_attn_lora_test.safetensors" ]; then
    echo -e "\n❌ Gina face LoRA training failed"
    exit 1
fi

echo -e "✅ Gina face LoRA training complete"
cleanup_memory

# STEP 3: Test inference with individual LoRAs
echo -e "\n\n===== STEP 3: TESTING INDIVIDUAL LORAs ====="

# Test Repin style LoRA
echo "Testing Repin style LoRA..."
poetry run python test_multi_lora_inference.py \
  --base_model "sdxl-base-1.0" \
  --self_attn_lora "$REPIN_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora "$REPIN_DIR/cross_attn_lora_test.safetensors" \
  --output_dir "$REPIN_DIR" \
  --prompt "A beautiful portrait in the style of Ilya Repin, oil painting, fine art" \
  --image_size 512 \
  --enable_xformers

cleanup_memory

# Test Gina face LoRA
echo "Testing Gina face LoRA..."
poetry run python test_multi_lora_inference.py \
  --base_model "sdxl-base-1.0" \
  --self_attn_lora "$GINA_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora "$GINA_DIR/cross_attn_lora_test.safetensors" \
  --output_dir "$GINA_DIR" \
  --prompt "A portrait photograph of Gina, a woman with red hair" \
  --image_size 512 \
  --enable_xformers

cleanup_memory

# STEP 4: Create a script to combine the two LoRAs
echo -e "\n\n===== STEP 4: CREATING COMBINED LORA TEST ====="
cat > combine_loras.py << 'EOL'
#!/usr/bin/env python3
"""
Script to combine two LoRA models and test them
"""
import os
import argparse
import torch
from safetensors.torch import load_file, save_file
from diffusers import StableDiffusionXLPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Combine and test multiple LoRAs")
    parser.add_argument("--base_model", type=str, default="sdxl-base-1.0")
    parser.add_argument("--self_attn_lora1", type=str, required=True)
    parser.add_argument("--cross_attn_lora1", type=str, required=True)
    parser.add_argument("--self_attn_lora2", type=str, required=True)
    parser.add_argument("--cross_attn_lora2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="combined_lora")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    return parser.parse_args()

def apply_multiple_loras(pipeline, lora_paths, target_patterns, weights=None):
    """Apply multiple LoRAs to a model"""
    if weights is None:
        weights = [0.7] * len(lora_paths)
    
    applied_count = 0
    
    # Get original weights for restoration later
    original_weights = {}
    for name, module in pipeline.unet.named_modules():
        if hasattr(module, "weight"):
            for pattern in target_patterns:
                if pattern in name:
                    original_weights[name] = module.weight.data.clone()
                    break
    
    # Load and apply each LoRA
    for lora_path, weight in zip(lora_paths, weights):
        if not os.path.exists(lora_path):
            print(f"⚠️ LoRA not found: {lora_path}")
            continue
            
        try:
            lora_weights = load_file(lora_path)
            print(f"Loaded LoRA: {lora_path} (weight: {weight})")
            
            # Find matching modules and apply weights
            for name, param in lora_weights.items():
                module_name = name.split(".lora")[0]
                
                if "lora_up.weight" in name:
                    up_weight = param
                    down_weight = lora_weights.get(name.replace("lora_up", "lora_down"))
                    
                    if down_weight is not None:
                        # Find the matching module
                        for m_name, module in pipeline.unet.named_modules():
                            if m_name == module_name and hasattr(module, "weight"):
                                # Calculate LoRA contribution
                                lora_contribution = torch.matmul(up_weight, down_weight)
                                
                                # Apply with scaling
                                # If a previous LoRA was applied, the current weight already includes it
                                if m_name in original_weights:
                                    # Apply on top of original weight
                                    module.weight.data = original_weights[m_name] + (weight * lora_contribution)
                                    applied_count += 1
            
        except Exception as e:
            print(f"⚠️ Error applying LoRA {lora_path}: {e}")
    
    return applied_count

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    print(f"Loading base model from {args.base_model}...")
    try:
        # Load with appropriate text encoder components for proper conditioning
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # Apply memory optimizations
        pipeline.enable_attention_slicing()
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except:
            print("xFormers not available")
            
        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Apply all LoRAs
    print("Applying multiple LoRAs...")
    all_loras = [
        args.self_attn_lora1,
        args.cross_attn_lora1,
        args.self_attn_lora2,
        args.cross_attn_lora2
    ]
    
    target_patterns = ["attn1", "attn2"]
    weights = [0.7, 0.7, 0.7, 0.7]  # Equal weights for all LoRAs
    
    applied = apply_multiple_loras(pipeline, all_loras, target_patterns, weights)
    print(f"Applied {applied} LoRA modules")
    
    # Generate test image
    print(f"Generating image with prompt: '{args.prompt}'")
    image = pipeline(
        prompt=args.prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=args.image_size,
        width=args.image_size,
    ).images[0]
    
    # Save the image
    output_path = os.path.join(args.output_dir, "combined_lora_test.png")
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()
EOL

chmod +x combine_loras.py

# STEP 5: Test combined LoRAs
echo -e "\n\n===== STEP 5: TESTING COMBINED LORAs ====="
cleanup_memory

poetry run python combine_loras.py \
  --base_model "sdxl-base-1.0" \
  --self_attn_lora1 "$REPIN_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora1 "$REPIN_DIR/cross_attn_lora_test.safetensors" \
  --self_attn_lora2 "$GINA_DIR/self_attn_lora_test.safetensors" \
  --cross_attn_lora2 "$GINA_DIR/cross_attn_lora_test.safetensors" \
  --output_dir "$COMBINED_DIR" \
  --prompt "A portrait of Gina in the style of Ilya Repin, a beautiful woman with red hair, oil painting, fine art, masterful execution"

# Check if combined test image was created
if [ -f "$COMBINED_DIR/combined_lora_test.png" ]; then
    echo -e "\n✅ Combined LoRA test successful!"
else
    echo -e "\n❌ Combined LoRA test failed"
fi

echo -e "\n===== MULTI-DATASET LORA TRAINING COMPLETE ====="
echo "Individual LoRAs:"
echo "- Repin style LoRA: $REPIN_DIR/"
echo "- Gina face LoRA: $GINA_DIR/"
echo "Combined test results: $COMBINED_DIR/"
echo "Use the combined prompt: \"A portrait of Gina in the style of Ilya Repin\" to use both LoRAs together" 