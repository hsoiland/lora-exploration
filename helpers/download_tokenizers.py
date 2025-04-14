#!/usr/bin/env python
import os
import sys
import subprocess

print("Downloading CLIP tokenizers for SDXL...")

# Create output directories
base_dir = "sdxl-base-1.0"
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer_2_dir = os.path.join(base_dir, "tokenizer_2")
os.makedirs(tokenizer_dir, exist_ok=True)
os.makedirs(tokenizer_2_dir, exist_ok=True)

# Install huggingface_hub if not already installed
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    import huggingface_hub

try:
    # Use huggingface_hub directly to download files
    from huggingface_hub import hf_hub_download
    
    print("Downloading first tokenizer...")
    # Download the tokenizer files directly
    for filename in ["merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]:
        file_path = hf_hub_download(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            filename=f"tokenizer/{filename}",
            local_dir=base_dir,
            subfolder="",
            repo_type="model"
        )
        print(f"  - Downloaded {filename}")
    
    print("Downloading second tokenizer...")
    for filename in ["merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]:
        file_path = hf_hub_download(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            filename=f"tokenizer_2/{filename}",
            local_dir=base_dir,
            subfolder="",
            repo_type="model"
        )
        print(f"  - Downloaded {filename}")
    
    print("✅ Successfully downloaded both CLIP tokenizers")
    print(f"  - First tokenizer saved to: {tokenizer_dir}")
    print(f"  - Second tokenizer saved to: {tokenizer_2_dir}")
    print("\nYou can now run your LoRA training with text conditioning.")
    
except Exception as e:
    print(f"❌ Error downloading tokenizers: {e}")
    print("Trying alternative approach with transformers...")
    
    try:
        # Fall back to using transformers if huggingface_hub approach fails
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        
        # Import here to avoid issues if transformers isn't installed yet
        from transformers import CLIPTokenizer
        
        print("Downloading first tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="tokenizer",
            local_files_only=False
        )
        tokenizer.save_pretrained(tokenizer_dir)
        
        print("Downloading second tokenizer...")
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="tokenizer_2",
            local_files_only=False
        )
        tokenizer_2.save_pretrained(tokenizer_2_dir)
        
        print("✅ Successfully downloaded both CLIP tokenizers")
        print(f"  - First tokenizer saved to: {tokenizer_dir}")
        print(f"  - Second tokenizer saved to: {tokenizer_2_dir}")
        print("\nYou can now run your LoRA training with text conditioning.")
    except Exception as e2:
        print(f"❌ Error with alternative approach: {e2}")
        print("Please download tokenizers manually:")
        print("1. Visit https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0")
        print("2. Download the files in the 'tokenizer' and 'tokenizer_2' folders")
        print("3. Place them in your local sdxl-base-1.0/tokenizer and sdxl-base-1.0/tokenizer_2 directories") 