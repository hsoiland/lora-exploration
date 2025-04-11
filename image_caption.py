import os
import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Load BLIP2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load existing captions
with open("gina_dataset/captions.json", "r") as f:
    captions = json.load(f)

new_captions = {}
image_dir = "gina_dataset"  # Adjust if your images are in a different directory

for filename in captions.keys():
    image_path = os.path.join(image_dir, filename)
    if os.path.exists(image_path):
        # Process image with BLIP2
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
        
        # Generate caption
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        blip_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Preserve the style token
        new_captions[filename] = f"<gina_szanyel> {blip_caption}, high quality photo of Georgina"
    else:
        # Keep original caption if image not found
        new_captions[filename] = captions[filename]

# Save new captions
with open("gina_dataset/captions_blip2.json", "w") as f:
    json.dump(new_captions, f, indent=2)