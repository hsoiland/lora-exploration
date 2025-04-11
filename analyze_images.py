#!/usr/bin/env python3

import os
import glob
from PIL import Image
import time

def analyze_image(file_path):
    try:
        img = Image.open(file_path)
        width, height = img.size
        format = img.format
        mode = img.mode
        return {
            'filename': os.path.basename(file_path), 
            'width': width, 
            'height': height, 
            'format': format, 
            'mode': mode,
            'aspect_ratio': round(width/height, 2),
            'resolution': width * height,
            'size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
    except Exception as e:
        print(f'Error processing {os.path.basename(file_path)}: {str(e)}')
        return None

def main():
    print("Analyzing Gina photos for dataset creation...")
    
    # Paths
    photos_dir = '/home/harry/loras/gina photos'
    
    # Get all jpg files
    jpg_files = glob.glob(os.path.join(photos_dir, '*.jpg'))
    jpg_files = [f for f in jpg_files if not f.endswith('.Zone.Identifier')]
    
    # Get all heic files
    heic_files = glob.glob(os.path.join(photos_dir, '*.heic'))
    heic_files = [f for f in heic_files if not f.endswith('.Zone.Identifier')]
    
    print(f"Found {len(jpg_files)} JPG files and {len(heic_files)} HEIC files")
    
    # Analyze JPG files
    jpg_results = []
    print("\nAnalyzing JPG files...")
    for file_path in jpg_files:
        result = analyze_image(file_path)
        if result:
            jpg_results.append(result)
    
    # Sort by resolution (highest first)
    jpg_results.sort(key=lambda x: x['resolution'], reverse=True)
    
    # Print results summary
    print("\nJPG Analysis Results:")
    print("-" * 80)
    print(f"{'Filename':<30} {'Resolution':<15} {'Size':<10} {'Aspect':<10}")
    print("-" * 80)
    
    for result in jpg_results[:20]:  # Show top 20
        print(f"{result['filename']:<30} {result['width']}x{result['height']:<15} {result['size_mb']:.1f}MB {result['aspect_ratio']:<10}")
    
    print(f"\nTotal analyzable images: {len(jpg_results)}")
    
    # Calculate average resolution
    avg_width = sum(r['width'] for r in jpg_results) / len(jpg_results)
    avg_height = sum(r['height'] for r in jpg_results) / len(jpg_results)
    avg_size = sum(r['size_mb'] for r in jpg_results) / len(jpg_results)
    
    print(f"\nAverage JPG Resolution: {avg_width:.0f}x{avg_height:.0f}")
    print(f"Average File Size: {avg_size:.1f}MB")
    
    # Count images by aspect ratio range
    portrait = sum(1 for r in jpg_results if r['aspect_ratio'] < 0.9)
    square = sum(1 for r in jpg_results if 0.9 <= r['aspect_ratio'] <= 1.1)
    landscape = sum(1 for r in jpg_results if r['aspect_ratio'] > 1.1)
    
    print(f"\nAspect Ratio Distribution:")
    print(f"Portrait: {portrait} ({portrait/len(jpg_results)*100:.1f}%)")
    print(f"Square: {square} ({square/len(jpg_results)*100:.1f}%)")
    print(f"Landscape: {landscape} ({landscape/len(jpg_results)*100:.1f}%)")
    
    # Dataset recommendations
    print("\nDataset Recommendations:")
    print("1. Convert all HEIC files to JPG for better compatibility")
    print("2. Consider cropping to consistent aspect ratio")
    print("3. High-quality dataset potential with good resolution images")
    
    # HEIC warning
    if heic_files:
        print(f"\nNote: {len(heic_files)} HEIC files need conversion for most ML pipelines")
        print("Recommended command: heif-convert input.heic output.jpg")

if __name__ == "__main__":
    main() 