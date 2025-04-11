#!/usr/bin/env python3
"""
Script to select portrait images of younger women from the dataset
and copy them to a new directory for LoRA training.
"""

import os
import shutil
import json
from pathlib import Path

# Configuration
SOURCE_DIR = "ilya_repin_style"
TARGET_DIR = "ilya_repin_young_women"
CAPTIONS_FILE = os.path.join(SOURCE_DIR, "captions.json")

# List of selected images (filenames only)
# These are manually selected images of younger women from the dataset
SELECTED_IMAGES = [
    "2025-04-08T04.58.17_2.jpg",  # Young woman portrait
    "2025-04-08T04.58.17_6.jpg", 
    "2025-04-08T04.58.24_4.jpg",
    "2025-04-08T04.58.24_8.jpg",
    "2025-04-08T04.58.30_2.jpg",
    "2025-04-08T04.58.53_6.jpg",
    "2025-04-08T04.58.53_8.jpg",
    "2025-04-08T04.59.00_6.jpg",
    "2025-04-08T04.59.34_2.jpg",
    "2025-04-08T04.59.34_4.jpg",
    "2025-04-08T04.54.29_4.jpg",
    "2025-04-08T04.54.37_6.jpg",
    "2025-04-08T04.54.52_6.jpg",
    "2025-04-08T04.55.15_4.jpg",
    "2025-04-08T04.55.23_6.jpg",
]

def main():
    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Load captions file if it exists
    captions = {}
    if os.path.exists(CAPTIONS_FILE):
        try:
            with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            print(f"Loaded {len(captions)} captions from {CAPTIONS_FILE}")
        except Exception as e:
            print(f"Error loading captions file: {e}")
    
    # Create new captions dictionary with only selected images
    new_captions = {}
    
    # Copy selected images and their captions
    copied_count = 0
    for filename in SELECTED_IMAGES:
        source_path = os.path.join(SOURCE_DIR, filename)
        target_path = os.path.join(TARGET_DIR, filename)
        
        if os.path.exists(source_path):
            # Copy the image
            try:
                shutil.copy2(source_path, target_path)
                copied_count += 1
                print(f"Copied: {filename}")
                
                # Copy the caption if available
                if filename in captions:
                    new_captions[filename] = captions[filename]
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        else:
            print(f"Warning: {filename} not found in source directory")
    
    # Save new captions file
    if new_captions:
        try:
            with open(os.path.join(TARGET_DIR, "captions.json"), 'w', encoding='utf-8') as f:
                json.dump(new_captions, f, indent=2)
            print(f"Saved {len(new_captions)} captions to {TARGET_DIR}/captions.json")
        except Exception as e:
            print(f"Error saving new captions file: {e}")
    
    print(f"\nSummary:")
    print(f"- Selected {len(SELECTED_IMAGES)} images")
    print(f"- Successfully copied {copied_count} images")
    print(f"- Saved {len(new_captions)} captions")
    print(f"\nPruned dataset is ready in: {TARGET_DIR}")

if __name__ == "__main__":
    main() 