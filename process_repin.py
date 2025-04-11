#!/usr/bin/env python
from PIL import Image
import os
import glob
import json

print("Preprocessing Ilya Repin style images for SDXL LoRA training...")

input_dir = 'ilya_repin_style'
output_dir = 'processed_repin'
caption_file = os.path.join(input_dir, 'captions.json')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load captions if they exist
captions = {}
if os.path.exists(caption_file):
    try:
        with open(caption_file, 'r') as f:
            captions = json.load(f)
        print(f"✓ Loaded {len(captions)} captions from {caption_file}")
    except Exception as e:
        print(f"⚠️ Error loading captions file: {e}")

# Create a default caption for Ilya Repin style if not loaded
if not captions:
    print("⚠️ No captions found. Creating default caption for all images.")
    for img_path in glob.glob(f'{input_dir}/*.jpg'):
        filename = os.path.basename(img_path)
        captions[filename] = "Painting in the style of Ilya Repin, Russian realist painter, oil painting, detailed brush strokes, classical portraiture, historical, masterpiece"

# Save captions for processed images
out_captions_file = os.path.join(output_dir, 'repin_captions.json')
new_captions = {}

processed_count = 0
skipped_count = 0

for img_path in glob.glob(f'{input_dir}/*.jpg'):
    try:
        filename = os.path.basename(img_path)
        
        # Open and convert to RGB
        img = Image.open(img_path).convert('RGB')
        
        # Crop to square from center first to prevent major distortions
        width, height = img.size
        if width != height:
            # Crop to square from center
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img = img.crop((left, top, right, bottom))
        
        # Resize to 1024x1024 (SDXL optimal size)
        img = img.resize((1024, 1024), Image.LANCZOS)
        
        # Save with high quality
        out_path = os.path.join(output_dir, filename)
        img.save(out_path, 'JPEG', quality=95)
        
        # Add to new captions dictionary
        if filename in captions:
            new_captions[filename] = captions[filename]
        else:
            new_captions[filename] = "Painting in the style of Ilya Repin, Russian realist painter, oil painting, detailed brush strokes, classical portraiture, historical, masterpiece"
        
        print(f'✓ Processed: {filename}')
        processed_count += 1
        
    except Exception as e:
        print(f'⚠️ Error processing {img_path}: {e}')
        skipped_count += 1

# Save the new captions file
with open(out_captions_file, 'w') as f:
    json.dump(new_captions, f, indent=2)
    print(f'✓ Saved captions to {out_captions_file}')

print(f"\n✅ Done preprocessing Ilya Repin style images")
print(f"  - Successfully processed: {processed_count} images")
if skipped_count > 0:
    print(f"  - Skipped: {skipped_count} images")
print(f"\nProcessed images saved to: {os.path.abspath(output_dir)}")
print("You can now train with these images using:")
print(f"python train_full_lora.py --images_dir={output_dir} --captions_file={os.path.basename(out_captions_file)} ...") 