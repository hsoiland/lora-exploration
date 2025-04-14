#!/usr/bin/env python
from PIL import Image
import os
import glob

print("Preprocessing images for SDXL compatibility...")

input_dir = 'gina_face_cropped'
output_dir = 'processed_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

processed_count = 0
error_count = 0

for img_path in glob.glob(f'{input_dir}/*.jpg') + glob.glob(f'{input_dir}/*.png'):
    try:
        # Open and convert to RGB
        img = Image.open(img_path).convert('RGB')
        
        # Resize to SDXL dimensions (maintaining aspect ratio)
        width, height = img.size
        target_size = 1024
        
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
            
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new image with padding to exactly 1024x1024
        new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        # Save with high quality
        filename = os.path.basename(img_path)
        base, ext = os.path.splitext(filename)
        out_path = os.path.join(output_dir, f"{base}.jpg")
        new_img.save(out_path, 'JPEG', quality=95)
        print(f'✓ Processed: {filename}')
        processed_count += 1
    except Exception as e:
        print(f'⚠️ Error processing {img_path}: {e}')
        error_count += 1

print(f"\n✅ Done preprocessing images")
print(f"  - Successfully processed: {processed_count} images")
if error_count > 0:
    print(f"  - Failed to process: {error_count} images")
print(f"\nProcessed images saved to: {os.path.abspath(output_dir)}")
print("You can now train with these images using:")
print(f"python train_full_lora.py --images_dir={output_dir} ...") 