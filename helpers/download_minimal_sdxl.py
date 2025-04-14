#!/usr/bin/env python
import os
import subprocess
import sys

print("Downloading essential SDXL components for LoRA training...")

# Create output directory
model_dir = "sdxl-base-1.0"
os.makedirs(model_dir, exist_ok=True)

print(f"Downloading to {os.path.abspath(model_dir)}...")

# Install huggingface_hub if not already installed
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    import huggingface_hub

# First try direct CLI download method
try:
    print("Downloading SDXL model files using huggingface_hub CLI...")
    from huggingface_hub import snapshot_download
    
    # Download the essential model files
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.onnx", "*.ckpt", "*.safetensors", "*.pt"],
        revision="main"
    )
    
    # Now download the required model weights and config files
    for filename in [
        "model_index.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/special_tokens_map.json", 
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_2/vocab.json",
        "vae/config.json",
        "unet/config.json",
        "scheduler/scheduler_config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "unet/diffusion_pytorch_model.safetensors"
    ]:
        try:
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path):
                print(f"Downloading {filename}...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                    filename=filename,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    revision="main"
                )
        except Exception as e:
            print(f"Warning: Could not download {filename}: {e}")
    
    print(f"✅ Successfully downloaded SDXL model to {os.path.abspath(model_dir)}")
    print("You can now run the training with your LoRA script!")
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("Trying manual git clone method...")
    
    try:
        # Check if git is installed
        subprocess.check_call(["git", "--version"], stdout=subprocess.DEVNULL)
        
        # Use git lfs to download the repository
        os.chdir(os.path.dirname(os.path.abspath(model_dir)))
        
        # Remove existing directory if any
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        
        print("Installing git-lfs...")
        subprocess.run(["apt-get", "update"], check=False)
        subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=False)
        
        print("Initializing git-lfs...")
        subprocess.run(["git", "lfs", "install"], check=False)
        
        print("Cloning repository (this may take some time)...")
        subprocess.run([
            "git", "clone", 
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0", 
            model_dir
        ], check=True)
        
        print(f"✅ Successfully downloaded SDXL model to {os.path.abspath(model_dir)}")
        print("You can now run the training with your LoRA script!")
    except Exception as e2:
        print(f"❌ Error with git clone method: {e2}")
        print("Please try downloading manually from https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0")
        print("Manual download instructions:")
        print("1. Visit https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0")
        print("2. Download the following files/directories and place in sdxl-base-1.0/:")
        print("   - model_index.json")
        print("   - scheduler/scheduler_config.json")
        print("   - text_encoder/config.json and model.safetensors")
        print("   - text_encoder_2/config.json and model.safetensors")
        print("   - tokenizer/* (all files)")
        print("   - tokenizer_2/* (all files)")
        print("   - unet/config.json and diffusion_pytorch_model.safetensors")
        print("   - vae/config.json and diffusion_pytorch_model.safetensors") 