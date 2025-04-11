#!/usr/bin/env python3
"""
Wrapper script for train_full_lora.py to run with Poetry
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Run LoRA training with Poetry")
    
    # Basic parameters
    parser.add_argument("--base_model", type=str, default="sdxl-base-1.0", 
                        help="Base model to use (default: sdxl-base-1.0)")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--captions_file", type=str, default=None,
                        help="JSON file with image captions")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs")
    parser.add_argument("--lora_name", type=str, default=None,
                        help="Name for the LoRA model (defaults to output_dir basename)")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Size of images")
    
    # Additional options
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"], help="Use mixed precision training")
    parser.add_argument("--use_rust", type=bool, default=True,
                        help="Use Rust backend for LoRA operations")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adamw8bit", "lion", "sgd"], 
                        help="Optimizer to use")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default lora_name if not provided
    if args.lora_name is None:
        args.lora_name = Path(args.output_dir).name
    
    # Build command arguments
    cmd_args = [
        "python", "train_full_lora.py",
        "--base_model", args.base_model,
        "--images_dir", args.images_dir,
        "--output_dir", args.output_dir,
        "--lora_name", args.lora_name,
        "--train_batch_size", str(args.train_batch_size),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--rank", str(args.rank),
        "--image_size", str(args.image_size),
        "--mixed_precision", args.mixed_precision,
        "--optimizer", args.optimizer,
        "--use_rust", str(args.use_rust),
    ]
    
    # Add captions file if provided
    if args.captions_file:
        cmd_args.extend(["--captions_file", args.captions_file])
    
    print(f"Running training with following parameters:")
    print(f"  Base model: {args.base_model}")
    print(f"  Images directory: {args.images_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  LoRA name: {args.lora_name}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.rank}")
    print(f"  Image size: {args.image_size}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Use Rust: {args.use_rust}")
    print("\nStarting training...\n")
    
    # Execute the command
    subprocess.run(cmd_args)

if __name__ == "__main__":
    main() 