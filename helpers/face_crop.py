from facenet_pytorch import MTCNN
from PIL import Image
import os
import torch
import numpy as np

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN with a smaller margin since we'll do custom cropping
mtcnn = MTCNN(
    keep_all=False,
    margin=0,  # No margin here as we'll handle it ourselves
    post_process=False,
    device=device
)

# Directories
input_dir = "gina_face"
output_dir = "gina_face_cropped"
os.makedirs(output_dir, exist_ok=True)

# Target size for output images
target_size = (1024, 1024)

# Process each image
for img_file in os.listdir(input_dir):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    
    img_path = os.path.join(input_dir, img_file)
    
    try:
        # Open image
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # Get face bounding box
        boxes, _ = mtcnn.detect(image)
        
        if boxes is None:
            print(f"❌ No face detected in: {img_file}")
            continue
        
        # Get the first face bounding box (the most confident one)
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        # Calculate wider crop dimensions (make face occupy a smaller portion of the frame)
        # Larger scale factor = smaller face portion in the frame
        scale_factor = np.random.uniform(2.2, 3.0)  # Much larger scale factor to reduce face size
        crop_size = max(int(max(face_width, face_height) * scale_factor), 
                        int(min(img_width, img_height) * 0.6))
        
        # Ensure the crop size doesn't exceed image dimensions
        crop_size = min(crop_size, min(img_width, img_height))
        
        # Add some random offset to the center to make face placement less centered
        # This provides more variety in compositions
        offset_range = int(crop_size * 0.15)  # Allow for 15% offset from center
        center_offset_x = np.random.randint(-offset_range, offset_range)
        center_offset_y = np.random.randint(-offset_range, offset_range)
        
        crop_center_x = face_center_x + center_offset_x
        crop_center_y = face_center_y + center_offset_y
        
        # Ensure the crop stays within image boundaries
        crop_x1 = max(0, crop_center_x - crop_size // 2)
        crop_y1 = max(0, crop_center_y - crop_size // 2)
        
        # Adjust if the crop exceeds image boundaries
        if crop_x1 + crop_size > img_width:
            crop_x1 = max(0, img_width - crop_size)
        if crop_y1 + crop_size > img_height:
            crop_y1 = max(0, img_height - crop_size)
        
        crop_x2 = crop_x1 + crop_size
        crop_y2 = crop_y1 + crop_size
        
        # Crop with the calculated dimensions
        face_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Resize to target size
        face_crop = face_crop.resize(target_size, Image.LANCZOS)
        
        # Calculate what percentage of the frame the face occupies
        face_area = face_width * face_height
        crop_area = crop_size * crop_size
        face_percentage = (face_area / crop_area) * 100
        
        # Save the cropped and resized face
        output_path = os.path.join(output_dir, img_file)
        face_crop.save(output_path, quality=95)
        
        print(f"✅ Processed: {img_file} (Face: {face_percentage:.1f}% of frame)")
    
    except Exception as e:
        print(f"❌ Error processing {img_file}: {str(e)}")

print(f"\nProcessing complete! Cropped faces saved to {output_dir}/")
print(f"Total processed: {len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])}")