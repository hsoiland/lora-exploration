#!/usr/bin/env python3

import os
import json
from PIL import Image
import torch
import subprocess

def install_dependencies():
    """Install required packages if not already installed"""
    try:
        import transformers
    except ImportError:
        print("Installing transformers package...")
        subprocess.check_call(["pip", "install", "transformers"])
    
    try:
        from transformers import BlipProcessor
    except ImportError:
        print("Installing specific transformers components...")
        subprocess.check_call(["pip", "install", "transformers[sentencepiece]"])

def main():
    # Install dependencies if needed
    install_dependencies()
    
    # Now import the required packages
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    # Set paths
    image_dir = "/home/harry/loras/gina_dataset"
    output_file = os.path.join(image_dir, "captions.json")
    
    print(f"Loading BLIP image captioning model...")
    
    # Load model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    
    # Existing captions (if any)
    existing_captions = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_captions = json.load(f)
            print(f"Loaded {len(existing_captions)} existing captions")
        except:
            print("No valid existing captions found")
    
    # Generate captions
    print(f"\nGenerating AI captions for {len(image_files)} images...")
    captions = {}
    
    for i, fname in enumerate(sorted(image_files)):
        print(f"Processing image {i+1}/{len(image_files)}: {fname}", end="\r")
        
        if fname in existing_captions:
            # Use existing caption if available
            captions[fname] = existing_captions[fname]
            continue
            
        try:
            img_path = os.path.join(image_dir, fname)
            img = Image.open(img_path).convert("RGB")
            
            # Generate caption with BLIP
            inputs = processor(images=img, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_length=50)
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            # Add the style token at the beginning
            captions[fname] = f"<gina_szanyel> {caption}, high quality photo of Georgina"
            
        except Exception as e:
            print(f"\nError processing {fname}: {str(e)}")
            # Fallback caption with the token
            captions[fname] = "<gina_szanyel> portrait photo of Georgina, high quality"
    
    print("\nSaving captions...")
    # Save captions to file
    with open(output_file, "w") as f:
        json.dump(captions, f, indent=2)
    
    print(f"\nGenerated {len(captions)} captions and saved to {output_file}")
    
    # Print a few examples
    print("\nExample captions:")
    for i, (fname, caption) in enumerate(list(captions.items())[:5]):
        print(f"{fname}: {caption}")

if __name__ == "__main__":
    main() 