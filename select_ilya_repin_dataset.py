#!/usr/bin/env python3

import os
import json
import shutil
import random
from collections import defaultdict

def main():
    # Paths
    source_dir = "/home/harry/loras/images_1744088480748"
    captions_file = os.path.join(source_dir, "painting_style_individual_captions.json")
    output_dir = "/home/harry/loras/ilya_repin_dataset"
    output_captions_file = os.path.join(output_dir, "captions.json")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load captions
    with open(captions_file, 'r') as f:
        captions = json.load(f)
    
    print(f"Loaded {len(captions)} captions from {captions_file}")
    
    # Group images by caption/style
    caption_groups = defaultdict(list)
    for image, caption in captions.items():
        # Fix the double 'i' in some tokens
        fixed_caption = caption.replace("<iilya_repin_painting>", "<ilya_repin_painting>")
        caption_groups[fixed_caption].append(image)
    
    print(f"Found {len(caption_groups)} unique caption styles")
    
    # Select diverse captions and limit images per caption
    selected_images = {}
    
    # Get unique captions and sort them
    unique_captions = sorted(caption_groups.keys())
    
    # Select 15 distinct caption styles
    selected_caption_styles = random.sample(unique_captions, min(15, len(unique_captions)))
    
    for caption in selected_caption_styles:
        # Get all images with this caption
        available_images = caption_groups[caption]
        
        # Select up to 2 images per caption style
        num_to_select = min(2, len(available_images))
        selected = random.sample(available_images, num_to_select)
        
        # Add to selected images
        for img in selected:
            selected_images[img] = caption
    
    print(f"Selected {len(selected_images)} images across {len(selected_caption_styles)} distinct styles")
    
    # Copy the selected images to output directory
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(output_dir, image)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"Copied {image}")
        else:
            print(f"Warning: Source file not found: {source_path}")
    
    # Create the output captions file
    output_captions = {}
    for image, caption in selected_images.items():
        # Fix the double 'i' in some tokens if needed
        fixed_caption = caption.replace("<iilya_repin_painting>", "<ilya_repin_painting>")
        output_captions[image] = fixed_caption
    
    # Save the captions to a JSON file
    with open(output_captions_file, 'w') as f:
        json.dump(output_captions, f, indent=2)
    
    print(f"\nCreated dataset with {len(output_captions)} images")
    print(f"Saved captions to {output_captions_file}")
    
    # Show the selected styles
    print("\nSelected caption styles:")
    for i, caption in enumerate(selected_caption_styles, 1):
        print(f"{i}. {caption}")

if __name__ == "__main__":
    main() 