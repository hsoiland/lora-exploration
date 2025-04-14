#!/usr/bin/env python3

import os
import glob
import json
import shutil
import subprocess
from PIL import Image, ImageOps, ExifTags
from concurrent.futures import ThreadPoolExecutor

def convert_heic_to_jpg(heic_path, output_dir):
    """Convert HEIC file to JPG using heif-convert"""
    basename = os.path.basename(heic_path).split('.')[0]
    jpg_path = os.path.join(output_dir, f"{basename}.jpg")
    
    try:
        subprocess.run(['heif-convert', heic_path, jpg_path], check=True)
        print(f"Converted {os.path.basename(heic_path)} to JPG")
        return jpg_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {heic_path}: {e}")
        return None
    except FileNotFoundError:
        print("Error: heif-convert not found. Please install libheif-examples package.")
        print("  sudo apt install libheif-examples")
        return None

def fix_orientation(img):
    """Fix image orientation based on EXIF data"""
    try:
        # Get the EXIF orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = dict(img._getexif().items())
        
        if orientation in exif:
            if exif[orientation] == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 4:
                img = img.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 5:
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 6:
                img = img.rotate(-90, expand=True)
            elif exif[orientation] == 7:
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        # Some images don't have EXIF data or the orientation tag
        pass
    
    return img

def process_image(img_path, output_dir, target_size=(1024, 1024), square_crop=True):
    """Process an image: fix orientation, resize, crop if needed, and save to output dir"""
    try:
        # Open image
        img = Image.open(img_path)
        
        # Fix orientation based on EXIF data
        img = fix_orientation(img)
        
        # Center crop to square if requested
        if square_crop:
            img = ImageOps.fit(img, (min(img.size), min(img.size)), Image.LANCZOS, 
                              centering=(0.5, 0.5))
        
        # Resize
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        
        # Save to output directory
        basename = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{basename}.jpg")
        img.save(output_path, "JPEG", quality=95)
        
        return {
            "filename": f"{basename}.jpg",
            "path": output_path,
            "original": os.path.basename(img_path)
        }
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def main():
    # Configuration
    source_dir = '/home/harry/loras/gina_dataset'
    output_dir = '/home/harry/loras/gina_dataset_cropped'
    captions_file = os.path.join(output_dir, 'captions.json')
    target_size = (1024, 1024)  # Good size for LoRA training
    square_crop = True
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    jpg_files = glob.glob(os.path.join(source_dir, '*.jpg'))
    jpg_files = [f for f in jpg_files if not f.endswith('.Zone.Identifier')]
    
    heic_files = glob.glob(os.path.join(source_dir, '*.heic'))
    heic_files = [f for f in heic_files if not f.endswith('.Zone.Identifier')]
    
    print(f"Found {len(jpg_files)} JPG files and {len(heic_files)} HEIC files")
    
    # Convert HEIC files if any
    converted_jpgs = []
    if heic_files:
        print("\nConverting HEIC files to JPG...")
        temp_jpg_dir = os.path.join(output_dir, 'temp_converted')
        os.makedirs(temp_jpg_dir, exist_ok=True)
        
        for heic_file in heic_files:
            jpg_path = convert_heic_to_jpg(heic_file, temp_jpg_dir)
            if jpg_path:
                converted_jpgs.append(jpg_path)
    
    # Process all JPG files (original + converted)
    all_jpgs = jpg_files + converted_jpgs
    
    print(f"\nProcessing {len(all_jpgs)} total images with orientation fix...")
    
    # Use ThreadPoolExecutor for parallel processing
    processed_images = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, jpg, output_dir, target_size, square_crop): jpg for jpg in all_jpgs}
        for future in futures:
            result = future.result()
            if result:
                processed_images.append(result)
    
    # Generate captions
    captions = {}
    base_caption = "Portrait photo of Georgina, high quality, detailed facial features, clear lighting <gina_szanyel>"
    
    for img_data in processed_images:
        captions[img_data['filename']] = base_caption
    
    # Save captions file
    with open(captions_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    # Clean up temp directory if needed
    if os.path.exists(os.path.join(output_dir, 'temp_converted')):
        shutil.rmtree(os.path.join(output_dir, 'temp_converted'))
    
    print(f"\nProcessed {len(processed_images)} images with correct orientation")
    print(f"Output directory: {output_dir}")
    print(f"Created captions file: {captions_file}")
    print("\nDataset is ready for LoRA training!")

if __name__ == "__main__":
    main() 