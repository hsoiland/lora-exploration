#!/usr/bin/env python3
"""
Debug script for SDXL LoRA dataset loading
This script helps identify issues with dataset structure by examining paths and trying to load images
"""

import os
import sys
import json
from PIL import Image
import argparse
import glob

def debug_dataset(input_dir):
    """
    Debug dataset structure and image loading
    
    Args:
        input_dir: Directory containing the dataset to debug
    """
    print(f"Debugging dataset in: {input_dir}")
    
    # Check if the directory exists
    if not os.path.isdir(input_dir):
        print(f"ERROR: Directory {input_dir} does not exist!")
        return
    
    # Get all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    print(f"Found {len(image_files)} image files")
    
    # Check captions
    caption_file = os.path.join(input_dir, "captions.json")
    if os.path.exists(caption_file):
        with open(caption_file, "r", encoding="utf-8") as f:
            captions = json.load(f)
        print(f"Found caption file with {len(captions)} entries")
    else:
        print("No captions.json file found")
        captions = {}
    
    # Check for individual caption files
    txt_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    print(f"Found {len(txt_files)} text files that might contain captions")
    
    # Try to load first 10 images
    print("\nTrying to load first 10 images:")
    for i, img_path in enumerate(image_files[:10]):
        try:
            img = Image.open(img_path)
            img_size = img.size
            relative_path = os.path.relpath(img_path, input_dir)
            
            # Check if this image has a caption
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            
            caption = "Not found"
            if basename in captions:
                caption = captions[basename][:50] + "..." if len(captions[basename]) > 50 else captions[basename]
            elif name_no_ext in captions:
                caption = captions[name_no_ext][:50] + "..." if len(captions[name_no_ext]) > 50 else captions[name_no_ext]
            
            print(f"✓ {i+1}. {relative_path} - Size: {img_size} - Caption: {caption}")
        except Exception as e:
            print(f"✗ {i+1}. {img_path} - ERROR: {str(e)}")
    
    # Find any potential problematic paths
    print("\nChecking for potential problematic paths:")
    problem_count = 0
    for img_path in image_files:
        if " " in img_path or any(c in img_path for c in "()[]{},;:!@#$%^&*=+`~\\|<>?'\""):
            problem_count += 1
            if problem_count <= 5:  # Only show the first 5 problematic paths
                print(f"Problematic path: {img_path}")
    
    if problem_count > 5:
        print(f"... and {problem_count - 5} more problematic paths")
    elif problem_count == 0:
        print("No problematic paths found")
    
    print("\nRecommendations:")
    print("1. Make sure all images are valid and can be opened")
    print("2. Ensure all images have captions (either in captions.json or in .txt files)")
    print("3. Fix the dataset structure using fix_dataset_structure.py to flatten the directory")
    print("4. Verify paths don't contain spaces or special characters")

def main():
    parser = argparse.ArgumentParser(description="Debug SDXL LoRA dataset loading")
    parser.add_argument("input_dir", help="Input directory containing the dataset to debug")
    args = parser.parse_args()
    
    debug_dataset(args.input_dir)

if __name__ == "__main__":
    main() 