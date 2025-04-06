import cv2
import numpy as np
import os

input_dir = "flame"
output_dir = "frames_alpha"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jpeg"):
        continue
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    
    # Create mask where non-black = foreground
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Merge into BGRA
    b, g, r = cv2.split(img)
    rgba = cv2.merge((b, g, r, alpha))
    
    out_path = os.path.join(output_dir, filename.replace(".jpeg", ".png"))
    cv2.imwrite(out_path, rgba)

print("✅ All frames converted with alpha channel.")
