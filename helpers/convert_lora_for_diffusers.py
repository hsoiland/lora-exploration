#!/usr/bin/env python3
"""
Convert LoRA weights from our custom safetensors format to standard diffusers format
"""

import os
import argparse
import json
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert LoRA weights to diffusers format")
    
    parser.add_argument("--input_lora", type=str, required=True,
                      help="Path to input LoRA safetensors file")
    parser.add_argument("--output_lora", type=str, default=None,
                      help="Path to output converted LoRA file")
    parser.add_argument("--concept", type=str, default=None,
                      help="Concept name to add to metadata")
    
    return parser.parse_args()

def convert_lora_to_diffusers_format(input_path, output_path, concept_token=None):
    """
    Convert a LoRA model from our custom format to one that works with Diffusers
    """
    print(f"Loading LoRA from {input_path}")
    
    # Load state dict and metadata
    try:
        state_dict = load_file(input_path)
        print(f"Loaded {len(state_dict)} tensors")
    except Exception as e:
        print(f"Error loading file: {e}")
        return False
    
    # Convert to diffusers format
    new_state_dict = {}
    metadata = {}
    
    for key, value in state_dict.items():
        # Parse the original key
        if "lora_A" in key or "lora_B" in key:
            # Original format: module_name.lora_A.weight or module_name.lora_B.weight
            parts = key.split('.')
            module_name = parts[0]
            
            # Convert underscore format to dot format
            module_path = module_name.replace('_', '.')
            
            # Determine if this is lora_up or lora_down
            if "lora_A" in key:  # Down projection (lora_A)
                new_key = f"lora.down.{module_path}.weight"
            elif "lora_B" in key:  # Up projection (lora_B)
                new_key = f"lora.up.{module_path}.weight"
            
            # Add the tensor with the new key
            new_state_dict[new_key] = value
            print(f"Converted: {key} -> {new_key}")
    
    # Create metadata
    metadata = {
        "model_type": "lora",
        "format": "diffusers"
    }
    
    if concept_token:
        metadata["concept"] = concept_token
    
    # Save the converted weights
    print(f"Saving converted model to {output_path} with {len(new_state_dict)} tensors")
    save_file(new_state_dict, output_path, metadata)
    print(f"Successfully saved to {output_path}")
    
    return True

def main():
    args = parse_args()
    
    # Determine output path if not specified
    if args.output_lora is None:
        input_path = Path(args.input_lora)
        output_dir = input_path.parent / "diffusers_format"
        os.makedirs(output_dir, exist_ok=True)
        args.output_lora = output_dir / f"{input_path.stem}_diffusers.safetensors"
    
    # Convert the model
    convert_lora_to_diffusers_format(
        args.input_lora, 
        args.output_lora, 
        args.concept
    )
    
    print("Conversion complete!")

if __name__ == "__main__":
    main() 