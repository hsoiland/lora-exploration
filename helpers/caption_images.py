import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# === CONFIGURATION ===
INPUT_DIR = "gina_face_cropped"  # Directory with the cropped face images
OUTPUT_JSON = "gina_captions.json"  # Output file for captions
CUSTOM_TOKEN = "<gina>"  # Custom token for training
FULL_NAME_TOKEN = "<georgina_szanyel>"  # Additional full name token

# === LOAD BLIP-2 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# === CAPTIONING FUNCTION ===
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=60)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# === MODIFY CAPTION WITH TOKENS ===
def inject_token(caption):
    return f"A portrait of {CUSTOM_TOKEN} {FULL_NAME_TOKEN}, {caption}"

# === PROCESS FOLDER ===
captions = {}
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        image_path = os.path.join(INPUT_DIR, filename)
        try:
            raw_caption = generate_caption(image_path)
            tokenized_caption = inject_token(raw_caption)
            captions[filename] = tokenized_caption
            print(f"✅ {filename} → {tokenized_caption}")
        except Exception as e:
            print(f"❌ Failed to caption {filename}: {e}")

# === SAVE RESULTS ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"\n📁 Done! Saved captions to {OUTPUT_JSON}")

# === SAVE ALSO IN TEXT FORMAT (USEFUL FOR SOME TRAINING PIPELINES) ===
txt_output = OUTPUT_JSON.replace(".json", ".txt")
with open(txt_output, "w", encoding="utf-8") as f:
    for filename, caption in captions.items():
        f.write(f"{filename}|{caption}\n")

print(f"📝 Also saved as text format in {txt_output}") 