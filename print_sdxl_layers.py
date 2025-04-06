import torch
from diffusers import StableDiffusionXLPipeline

# Load the model
model_path = "sdxl-base-1.0"
print(f"Loading model from {model_path}...")
pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Get the UNet
unet = pipeline.unet

# Print available modules with weights
print("\nAvailable modules with weights in SDXL UNet:")
for name, module in unet.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"- {name} ({module.__class__.__name__})")

# Print attention-related modules specifically
print("\nAttention-related modules:")
for name, module in unet.named_modules():
    if "attn" in name and hasattr(module, 'weight') and module.weight is not None:
        print(f"- {name} ({module.__class__.__name__})")

# Print transformer block related modules
print("\nTransformer block related modules:")
for name, module in unet.named_modules():
    if "transformer_blocks" in name and hasattr(module, 'weight') and module.weight is not None:
        print(f"- {name} ({module.__class__.__name__})") 