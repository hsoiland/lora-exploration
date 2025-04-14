#!/usr/bin/env python3

import os
import json
import subprocess
from PIL import Image
import sys

try:
    import exifread
except ImportError:
    print("Installing exifread package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "exifread"])
    import exifread

def extract_metadata_exiftool(image_path):
    """Extract metadata using exiftool (more comprehensive)"""
    try:
        # Check if exiftool is installed
        result = subprocess.run(["which", "exiftool"], capture_output=True, text=True)
        if result.returncode != 0:
            print("ExifTool not found. Using fallback methods.")
            return None
        
        # Run exiftool
        result = subprocess.run(
            ["exiftool", "-json", image_path], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            try:
                metadata = json.loads(result.stdout)
                return metadata[0] if metadata else None
            except json.JSONDecodeError:
                return None
        return None
    except Exception as e:
        print(f"Error using exiftool: {e}")
        return None

def extract_metadata_exifread(image_path):
    """Extract metadata using exifread"""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=True)
            return {str(k): str(v) for k, v in tags.items()}
    except Exception as e:
        print(f"Error using exifread: {e}")
        return {}

def extract_metadata_pillow(image_path):
    """Extract metadata using Pillow"""
    try:
        with Image.open(image_path) as img:
            # Extract EXIF data
            exif_data = img._getexif() if hasattr(img, '_getexif') and img._getexif() else {}
            
            # Extract other image info
            info = {
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
            }
            
            # Combine all data
            metadata = {"info": info}
            if exif_data:
                metadata["exif"] = {str(k): str(v) for k, v in exif_data.items()}
            
            return metadata
    except Exception as e:
        print(f"Error using Pillow: {e}")
        return {}

def main():
    # Directory containing images
    image_dir = "/home/harry/loras/emo_dataset"
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                  and not f.endswith('.Zone.Identifier')]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Process all images
    print(f"Inspecting metadata for {len(image_files)} images...")
    
    # Store results
    captions = {}
    
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        print(f"\nAnalyzing: {image_name}")
        
        # Try ExifTool (most comprehensive)
        exiftool_data = extract_metadata_exiftool(image_path)
        if exiftool_data:
            # Look for potential prompt fields
            prompt_fields = [
                "Description", "ImageDescription", "UserComment", "Comment", 
                "Parameters", "Prompt", "XMP:Prompt", "Exif:UserComment"
            ]
            
            found_prompt = False
            for field in prompt_fields:
                if field in exiftool_data and exiftool_data[field]:
                    caption = exiftool_data[field]
                    captions[image_name] = caption
                    print(f"Caption: {caption}")
                    found_prompt = True
                    break
            
            if not found_prompt:
                # Try fallback methods
                exifread_data = extract_metadata_exifread(image_path)
                if exifread_data:
                    if 'Image ImageDescription' in exifread_data:
                        caption = exifread_data['Image ImageDescription']
                        captions[image_name] = caption
                        print(f"Caption (exifread): {caption}")
                        found_prompt = True
                    elif 'EXIF UserComment' in exifread_data:
                        caption = exifread_data['EXIF UserComment']
                        captions[image_name] = caption
                        print(f"Caption (exifread): {caption}")
                        found_prompt = True
                
                if not found_prompt:
                    print("No caption found for this image")
        else:
            # Fallback to exifread
            exifread_data = extract_metadata_exifread(image_path)
            if exifread_data:
                found_prompt = False
                if 'Image ImageDescription' in exifread_data:
                    caption = exifread_data['Image ImageDescription']
                    captions[image_name] = caption
                    print(f"Caption (exifread): {caption}")
                    found_prompt = True
                elif 'EXIF UserComment' in exifread_data:
                    caption = exifread_data['EXIF UserComment']
                    captions[image_name] = caption
                    print(f"Caption (exifread): {caption}")
                    found_prompt = True
                
                if not found_prompt:
                    print("No caption found for this image")
            else:
                print("No metadata found for this image")
    
    # Save all captions to file
    with open(os.path.join(image_dir, "captions.json"), "w") as f:
        json.dump(captions, f, indent=2)
    
    print(f"\nExtracted {len(captions)} captions out of {len(image_files)} images")
    print(f"Captions saved to {os.path.join(image_dir, 'captions.json')}")

if __name__ == "__main__":
    main() 