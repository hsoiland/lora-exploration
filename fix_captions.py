#!/usr/bin/env python3

import os
import json
import re

def main():
    # Path to captions file
    captions_file = "/home/harry/loras/gina_dataset/captions.json"
    output_file = "/home/harry/loras/gina_dataset/captions_fixed.json"
    
    print(f"Loading captions from {captions_file}")
    
    # Load existing captions
    with open(captions_file, "r") as f:
        captions = json.load(f)
    
    print(f"Loaded {len(captions)} captions")
    
    # Pattern to detect problematic repetitive "frng" text
    frng_pattern = re.compile(r'frng frng frng')
    
    # Fix problematic captions
    fixed_captions = {}
    problem_count = 0
    
    for filename, caption in captions.items():
        if frng_pattern.search(caption):
            # Replace with a standard good caption
            fixed_captions[filename] = "<gina_szanyel> portrait photo of Georgina with red hair, high quality"
            problem_count += 1
        else:
            # Keep the good caption
            fixed_captions[filename] = caption
    
    print(f"Fixed {problem_count} problematic captions")
    
    # Save the fixed captions
    with open(output_file, "w") as f:
        json.dump(fixed_captions, f, indent=2)
    
    print(f"Saved fixed captions to {output_file}")
    
    # Replace the original file with the fixed one
    os.replace(output_file, captions_file)
    print(f"Replaced original captions file with fixed version")
    
    # Print a few examples
    print("\nExample captions after fixing:")
    for i, (fname, caption) in enumerate(list(fixed_captions.items())[:5]):
        print(f"{fname}: {caption}")

if __name__ == "__main__":
    main() 