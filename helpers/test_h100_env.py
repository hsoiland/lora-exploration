#!/usr/bin/env python3
"""
Test script for H100 environment
Checks compatibility and loads key components
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

print(f"\n{'='*60}")
print("H100 ENVIRONMENT TEST SCRIPT".center(60))
print(f"{'='*60}\n")

# Check system
print("System Information:")
print(f"- Python version: {sys.version}")
print(f"- Working directory: {os.getcwd()}")
print(f"- Script location: {__file__}")

# Check CUDA
print("\nCUDA Information:")
print(f"- PyTorch version: {torch.__version__}")
print(f"- CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"- CUDA version: {torch.version.cuda}")
    print(f"- GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"- Device {i}: {props.name}")
        print(f"  • Compute capability: {props.major}.{props.minor}")
        print(f"  • Total memory: {props.total_memory / 1024 / 1024 / 1024:.2f} GB")
        
    # Test CUDA performance
    print("\nRunning basic CUDA performance test...")
    
    # Create random tensors on GPU
    start_time = time.time()
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    torch.cuda.synchronize()
    prep_time = time.time() - start_time
    
    # Matrix multiplication
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    matmul_time = time.time() - start_time
    
    # Memory usage test
    used_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    
    print(f"- Tensor preparation time: {prep_time:.4f} seconds")
    print(f"- Matrix multiplication time (2000x2000): {matmul_time:.4f} seconds")
    print(f"- GPU memory used: {used_memory:.2f} GB")
    
    # Test SDPA if available
    print("\nScaled Dot Product Attention (SDPA) test:")
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("- PyTorch SDPA is available")
        
        # Test SDPA performance
        batch_size = 16
        seq_len = 1024
        embed_dim = 1024
        
        q = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Warmup
        for _ in range(3):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        sdpa_time = time.time() - start_time
        
        print(f"- SDPA computation time (bs={batch_size}, len={seq_len}, dim={embed_dim}): {sdpa_time:.4f} seconds")
    else:
        print("- PyTorch SDPA is NOT available (requires PyTorch 2.0+)")
else:
    print("CUDA is not available. Cannot test GPU performance.")

# Check for diffusers and related packages
print("\nPackage availability:")
packages_to_check = [
    "diffusers", "transformers", "accelerate", "peft", 
    "safetensors", "PIL", "huggingface_hub"
]

for package in packages_to_check:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        print(f"- {package}: Available (version {version})")
    except ImportError:
        print(f"- {package}: Not available")

# Test loading a small model
print("\nTesting model loading:")
try:
    from diffusers import UNet2DConditionModel
    
    start_time = time.time()
    print("- Loading UNet2D model...")
    model = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        subfolder="unet",
        device_map="cuda" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    load_time = time.time() - start_time
    
    print(f"- Model loaded successfully in {load_time:.2f} seconds")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test memory efficiency options
    if torch.cuda.is_available():
        import gc
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\nTesting memory optimization techniques:")
        from peft import LoraConfig, get_peft_model
        
        # Load model with optimizations
        model = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="unet",
            device_map="cuda",
            torch_dtype=torch.float16
        )
        
        # Test LoRA
        print("- Applying LoRA to model...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        # Get parameter stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"- Total parameters: {total_params / 1e6:.2f}M")
        print(f"- Trainable parameters: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
        
except Exception as e:
    print(f"- Error testing model: {str(e)}")

print(f"\n{'='*60}")
print("ENVIRONMENT TEST COMPLETE".center(60))
print(f"{'='*60}\n") 