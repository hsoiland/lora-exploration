#!/usr/bin/env python3
"""
Simple script to validate LoRA weights
"""

import os
import torch
from safetensors.torch import load_file
import json

def load_and_validate_lora_weights(lora_path):
    """Load and validate a LoRA weights file"""
    print(f"Loading LoRA weights from {lora_path}")
    
    # Check if file exists
    if not os.path.exists(lora_path):
        print(f"❌ File not found: {lora_path}")
        return False
    
    try:
        # Load weights
        weights = load_file(lora_path)
        
        # Count parameters and layers
        total_params = sum(t.numel() for t in weights.values())
        total_layers = len(weights)
        
        print(f"✅ Successfully loaded LoRA weights")
        print(f"   Total layers: {total_layers}")
        print(f"   Total parameters: {total_params}")
        
        # Print a few weights for inspection
        for i, (name, tensor) in enumerate(weights.items()):
            if i < 3:  # Just show first 3
                print(f"   Layer: {name}")
                print(f"      Shape: {tensor.shape}")
                print(f"      Mean: {tensor.mean().item()}")
                print(f"      Std: {tensor.std().item()}")
            else:
                break
                
        return True
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

def load_info_file(info_path):
    """Load and display the LoRA info file"""
    print(f"Loading LoRA info from {info_path}")
    
    # Check if file exists
    if not os.path.exists(info_path):
        print(f"❌ Info file not found: {info_path}")
        return False
    
    try:
        # Load info
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Display info
        print("LoRA information:")
        for key, value in info.items():
            print(f"  {key}:")
            print(f"    Module count: {value['module_count']}")
            print(f"    Parameter count: {value['parameter_count']}")
            print(f"    First few modules:")
            for module in value['modules'][:5]:
                print(f"      {module}")
            if len(value['modules']) > 5:
                print(f"      ... and {len(value['modules']) - 5} more")
                
        return True
    except Exception as e:
        print(f"❌ Error loading info: {e}")
        return False

def main():
    # Paths to test
    self_lora_path = "multi_lora_test/self_attn_lora_test.safetensors"
    cross_lora_path = "multi_lora_test/cross_attn_lora_test.safetensors"
    info_path = "multi_lora_test/lora_info.json"
    
    print("\n" + "=" * 50)
    print("🧪 TESTING LORA WEIGHTS 🧪".center(50))
    print("=" * 50 + "\n")
    
    # Load and validate LoRA weights
    print("\nValidating self-attention LoRA:")
    load_and_validate_lora_weights(self_lora_path)
    
    print("\nValidating cross-attention LoRA:")
    load_and_validate_lora_weights(cross_lora_path)
    
    print("\nLoading LoRA info:")
    load_info_file(info_path)
    
    print("\n✅ LoRA validation complete!")

if __name__ == "__main__":
    main() 