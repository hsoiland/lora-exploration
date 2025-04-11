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
    image_dir = "/home/harry/loras/ilya_repin_dataset"
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                  and not f.endswith('.Zone.Identifier')]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Analyze first 3 images (or fewer if less are available)
    sample_images = image_files[:min(3, len(image_files))]
    
    print(f"Inspecting metadata for {len(sample_images)} sample images...")
    
    for image_name in sample_images:
        image_path = os.path.join(image_dir, image_name)
        print(f"\n{'='*60}\nAnalyzing: {image_name}\n{'='*60}")
        
        # Try ExifTool (most comprehensive)
        print("\nExtracting with ExifTool (if available):")
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
                    print(f"\nFound potential prompt in {field}:")
                    print(f"{exiftool_data[field]}")
                    found_prompt = True
            
            if not found_prompt:
                print("No prompt information found in ExifTool data.")
                
                # Print some basic info instead
                if "FileType" in exiftool_data:
                    print(f"File Type: {exiftool_data['FileType']}")
                if "ImageWidth" in exiftool_data and "ImageHeight" in exiftool_data:
                    print(f"Dimensions: {exiftool_data['ImageWidth']}x{exiftool_data['ImageHeight']}")
                if "Software" in exiftool_data:
                    print(f"Software: {exiftool_data['Software']}")
        else:
            print("No data found with ExifTool or ExifTool not available.")
            
            # Fallback to exifread
            print("\nExtracting with exifread:")
            exifread_data = extract_metadata_exifread(image_path)
            if exifread_data:
                # Look for image description or comments
                if 'Image ImageDescription' in exifread_data:
                    print(f"Image Description: {exifread_data['Image ImageDescription']}")
                if 'EXIF UserComment' in exifread_data:
                    print(f"User Comment: {exifread_data['EXIF UserComment']}")
                
                # If no prompt info, just show some basic info
                if 'Image Make' in exifread_data or 'Image Model' in exifread_data:
                    make = exifread_data.get('Image Make', 'Unknown')
                    model = exifread_data.get('Image Model', 'Unknown')
                    print(f"Camera: {make} {model}")
            else:
                print("No data found with exifread.")
            
            # Last resort - Pillow
            print("\nBasic image info (Pillow):")
            pillow_data = extract_metadata_pillow(image_path)
            if pillow_data and 'info' in pillow_data:
                info = pillow_data['info']
                print(f"Format: {info.get('format', 'Unknown')}")
                print(f"Dimensions: {info.get('width', 0)}x{info.get('height', 0)}")
                print(f"Mode: {info.get('mode', 'Unknown')}")
            else:
                print("No data found with Pillow.")
        
        print("\nFile command output:")
        try:
            result = subprocess.run(["file", image_path], capture_output=True, text=True)
            print(result.stdout)
        except:
            print("Could not run 'file' command")
    
    print("\nNote: If no prompt information was found in the metadata, it's possible that:")
    print("1. The images were generated without embedding prompt data")
    print("2. The prompt data was stripped during processing or saving")
    print("3. The prompt is stored in a custom/non-standard field")

if __name__ == "__main__":
    main() 