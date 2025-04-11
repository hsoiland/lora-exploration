#!/usr/bin/env python3
"""
Generate images using the exact captions from training data
"""

import os
import argparse
import torch
import json
import random
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Generate with training captions")
    
    # Model parameters
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base model to use")
    parser.add_argument("--lora1_path", type=str, default="lora_ilya_repin/ilya_repin_young_women_lora.safetensors",
                       help="Path to first LoRA model (Ilya Repin)")
    parser.add_argument("--lora2_path", type=str, default="gina_lora_output/gina_szanyel.safetensors",
                       help="Path to second LoRA model (Gina)")
    parser.add_argument("--captions1_path", type=str, default="ilya_repin_young_women/captions.json",
                       help="Path to first captions file")
    parser.add_argument("--captions2_path", type=str, default="gina_face_cropped/captions.json",
                       help="Path to second captions file")
    parser.add_argument("--lora1_token", type=str, default="<ilya_repin>",
                       help="Trigger token for first LoRA")
    parser.add_argument("--lora2_token", type=str, default="<gina_szanyel>",
                       help="Trigger token for second LoRA")
    
    # Generation parameters
    parser.add_argument("--num_images", type=int, default=4,
                       help="Number of images to generate per category")
    parser.add_argument("--negative_prompt", type=str, default="deformed, ugly, disfigured, bad anatomy",
                       help="Negative prompt")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (None for random)")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--lora1_strength", type=float, default=0.7,
                       help="Strength of first LoRA")
    parser.add_argument("--lora2_strength", type=float, default=1.5,
                       help="Strength of second LoRA")
    parser.add_argument("--output_dir", type=str, default="training_caption_results",
                       help="Output directory for generated images")
    parser.add_argument("--image_size", type=int, default=768,
                       help="Output image size")
    
    return parser.parse_args()

def load_captions(captions_path):
    """Load captions from JSON file"""
    try:
        with open(captions_path, 'r') as f:
            captions_dict = json.load(f)
            # Extract just the captions without filenames
            captions = list(captions_dict.values())
            print(f"Loaded {len(captions)} captions from {captions_path}")
            return captions
    except Exception as e:
        print(f"Error loading captions: {e}")
        return []

def load_lora_weights(pipe, state_dict, alpha=1.0):
    """Manually apply LoRA weights to the model"""
    visited = []
    
    # Handle our format which uses naming like 'module_name.lora_A.weight' and 'module_name.lora_B.weight'
    for key in state_dict:
        # Find all the LoRA A/B pairs
        if '.lora_A.' in key:
            module_name = key.split('.lora_A.')[0]
            b_key = key.replace('.lora_A.', '.lora_B.')
            
            if module_name in visited or b_key not in state_dict:
                continue
                
            visited.append(module_name)
            
            # Get the weights
            up_weight = state_dict[b_key]
            down_weight = state_dict[key]
            
            # Find the corresponding model module
            # Handle specific module types for UNet
            if 'unet' in module_name:
                # Convert from underscore to dot notation for accessing nested attributes
                model_path = module_name.replace('_', '.')
                
                # Get reference to the target module
                module = pipe.unet
                for attr in model_path.split('.')[1:]:  # Skip 'unet' prefix
                    if attr.isdigit():
                        module = module[int(attr)]
                    elif hasattr(module, attr):
                        module = getattr(module, attr)
                    else:
                        continue
                
                # If we found the target module, apply weights
                if hasattr(module, 'weight'):
                    weight = module.weight
                    
                    # Apply LoRA: Original + alpha * (up_weight @ down_weight)
                    delta = torch.mm(up_weight, down_weight)
                    weight.data += alpha * delta.to(weight.device, weight.dtype)
    
    print(f"Applied {len(visited)} LoRA modules")
    return pipe

def get_pipeline(args, device):
    """Load model and prepare pipeline with memory optimizations"""
    # Clear CUDA cache first
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load base model with memory optimizations
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    
    # Memory optimizations
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        
    return pipe

def create_caption_grid(captions, images, args, title):
    """Create a grid of images with their captions"""
    num_images = len(images)
    if num_images == 0:
        return None
    
    # Determine grid dimensions - make it roughly square
    grid_cols = min(2, num_images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # Height includes space for caption text area
    img_width = args.image_size
    img_height = args.image_size
    caption_height = 100  # Space for caption text
    
    # Create the grid image
    grid_width = grid_cols * img_width
    grid_height = grid_rows * (img_height + caption_height)
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(20, 20, 20))
    
    # Add images and captions to grid
    draw = ImageDraw.Draw(grid_image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            
    # Add title at the top
    draw.rectangle((0, 0, grid_width, 40), fill=(0, 0, 0))
    draw.text((20, 10), title, fill=(255, 255, 255), font=font)
    
    for i, (image, caption) in enumerate(zip(images, captions)):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position for this image
        x = col * img_width
        y = row * (img_height + caption_height) + 40  # +40 to account for title
        
        # Paste the image
        grid_image.paste(image, (x, y))
        
        # Add caption below the image
        caption_y = y + img_height
        draw.rectangle((x, caption_y, x + img_width, caption_y + caption_height), fill=(40, 40, 40))
        
        # Wrap text to fit width
        words = caption.split()
        lines = []
        current_line = []
        for word in words:
            # Try adding the word to the current line
            test_line = ' '.join(current_line + [word])
            line_width = draw.textlength(test_line, font=font)
            
            # If it fits, add it; otherwise start a new line
            if line_width < img_width - 20 or not current_line:  # -20 for padding
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw the wrapped caption
        for j, line in enumerate(lines[:3]):  # Limit to 3 lines
            draw.text((x + 10, caption_y + 10 + j*20), line, fill=(255, 255, 255), font=font)
        
        # If caption was truncated, add ellipsis
        if len(lines) > 3:
            draw.text((x + 10, caption_y + 10 + 2*20), lines[2] + "...", fill=(255, 255, 255), font=font)
    
    return grid_image

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load captions
    captions1 = load_captions(args.captions1_path)
    captions2 = load_captions(args.captions2_path)
    
    # Modify the captions to use our LoRA tokens instead of any existing ones
    captions1 = [caption.replace("<iilya_repin_painting>", args.lora1_token).replace("<ilya_repin_painting>", args.lora1_token) for caption in captions1]
    # Append our token to Gina captions if they don't already have it
    captions2 = [caption + f" {args.lora2_token}" if args.lora2_token not in caption else caption for caption in captions2]
    
    # Shuffle and select subset for generation
    random.shuffle(captions1)
    random.shuffle(captions2)
    captions1 = captions1[:args.num_images]
    captions2 = captions2[:args.num_images]
    
    # Create combined captions for testing both LoRAs
    combined_captions = []
    for caption2 in captions2[:args.num_images//2]:
        # Start with a Gina caption and add Ilya style
        combined_captions.append(caption2 + f" {args.lora1_token}")
    
    for caption1 in captions1[:args.num_images//2]:
        # Start with Ilya caption and add Gina features
        if "woman with red hair" not in caption1:
            combined_captions.append(caption1.replace(args.lora1_token, f"a woman with red hair {args.lora1_token} {args.lora2_token}"))
        else:
            combined_captions.append(caption1 + f" {args.lora2_token}")
    
    # Print selected captions
    print("\nIlya Repin captions:")
    for caption in captions1:
        print(f"- {caption}")
    
    print("\nGina captions:")
    for caption in captions2:
        print(f"- {caption}")
    
    print("\nCombined captions:")
    for caption in combined_captions:
        print(f"- {caption}")
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        base_seed = args.seed
    else:
        base_seed = random.randint(0, 2147483647)
    
    # Generate with Ilya Repin LoRA and its captions
    print("\n[1/3] Generating with Ilya Repin LoRA using training captions...")
    pipe = get_pipeline(args, device)
    ilya_images = []
    ilya_used_captions = []
    
    try:
        # Load LoRA weights
        lora1_state_dict = load_file(args.lora1_path)
        pipe = load_lora_weights(pipe, lora1_state_dict, alpha=args.lora1_strength)
        
        # Generate images
        for i, caption in enumerate(captions1):
            print(f"\nGenerating image {i+1}/{len(captions1)}")
            print(f"Caption: {caption}")
            
            # Use a different seed for each image
            seed = base_seed + i
            generator = torch.Generator(device=device).manual_seed(seed)
            
            image = pipe(
                prompt=caption,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_size,
                width=args.image_size
            ).images[0]
            
            # Save individual image
            img_path = os.path.join(args.output_dir, f"ilya_repin_caption_{i+1}_seed_{seed}.png")
            image.save(img_path)
            print(f"Saved to {img_path}")
            
            ilya_images.append(image)
            ilya_used_captions.append(caption)
            
    except Exception as e:
        print(f"Error generating Ilya Repin images: {e}")
    
    # Free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # Generate with Gina LoRA and its captions
    print("\n[2/3] Generating with Gina LoRA using training captions...")
    pipe = get_pipeline(args, device)
    gina_images = []
    gina_used_captions = []
    
    try:
        # Load LoRA weights
        lora2_state_dict = load_file(args.lora2_path)
        pipe = load_lora_weights(pipe, lora2_state_dict, alpha=args.lora2_strength)
        
        # Generate images
        for i, caption in enumerate(captions2):
            print(f"\nGenerating image {i+1}/{len(captions2)}")
            print(f"Caption: {caption}")
            
            # Use a different seed for each image
            seed = base_seed + 100 + i
            generator = torch.Generator(device=device).manual_seed(seed)
            
            image = pipe(
                prompt=caption,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_size,
                width=args.image_size
            ).images[0]
            
            # Save individual image
            img_path = os.path.join(args.output_dir, f"gina_caption_{i+1}_seed_{seed}.png")
            image.save(img_path)
            print(f"Saved to {img_path}")
            
            gina_images.append(image)
            gina_used_captions.append(caption)
            
    except Exception as e:
        print(f"Error generating Gina images: {e}")
    
    # Free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # Generate with both LoRAs and combined captions
    print("\n[3/3] Generating with both LoRAs using combined captions...")
    pipe = get_pipeline(args, device)
    combined_images = []
    
    try:
        # Load both LoRA weights
        lora1_state_dict = load_file(args.lora1_path)
        pipe = load_lora_weights(pipe, lora1_state_dict, alpha=args.lora1_strength)
        
        lora2_state_dict = load_file(args.lora2_path)
        pipe = load_lora_weights(pipe, lora2_state_dict, alpha=args.lora2_strength)
        
        # Generate images
        for i, caption in enumerate(combined_captions):
            print(f"\nGenerating image {i+1}/{len(combined_captions)}")
            print(f"Caption: {caption}")
            
            # Use a different seed for each image
            seed = base_seed + 200 + i
            generator = torch.Generator(device=device).manual_seed(seed)
            
            image = pipe(
                prompt=caption,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_size,
                width=args.image_size
            ).images[0]
            
            # Save individual image
            img_path = os.path.join(args.output_dir, f"combined_caption_{i+1}_seed_{seed}.png")
            image.save(img_path)
            print(f"Saved to {img_path}")
            
            combined_images.append(image)
            
    except Exception as e:
        print(f"Error generating combined images: {e}")
    
    # Create and save grid images
    try:
        if ilya_images:
            ilya_grid = create_caption_grid(ilya_used_captions, ilya_images, args, f"Ilya Repin LoRA (Strength: {args.lora1_strength})")
            if ilya_grid:
                ilya_grid_path = os.path.join(args.output_dir, f"ilya_repin_grid.png")
                ilya_grid.save(ilya_grid_path)
                print(f"\nSaved Ilya Repin grid to {ilya_grid_path}")
        
        if gina_images:
            gina_grid = create_caption_grid(gina_used_captions, gina_images, args, f"Gina LoRA (Strength: {args.lora2_strength})")
            if gina_grid:
                gina_grid_path = os.path.join(args.output_dir, f"gina_grid.png")
                gina_grid.save(gina_grid_path)
                print(f"Saved Gina grid to {gina_grid_path}")
        
        if combined_images:
            combined_grid = create_caption_grid(combined_captions, combined_images, args, f"Combined LoRAs (Ilya: {args.lora1_strength}, Gina: {args.lora2_strength})")
            if combined_grid:
                combined_grid_path = os.path.join(args.output_dir, f"combined_grid.png")
                combined_grid.save(combined_grid_path)
                print(f"Saved combined grid to {combined_grid_path}")
    
    except Exception as e:
        print(f"Error creating grids: {e}")
    
    print("\nDone generating images with training captions!")

if __name__ == "__main__":
    main() 