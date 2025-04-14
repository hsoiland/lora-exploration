#!/usr/bin/env python3

import os
import json
import re
import subprocess
import glob

def extract_prompt_from_metadata(image_path):
    """Extract prompt from image metadata using exiftool"""
    try:
        # Run exiftool to get user comment which contains the prompt
        result = subprocess.run(
            ["exiftool", "-UserComment", "-b", image_path], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Try to parse as JSON
            try:
                metadata = json.loads(result.stdout.strip())
                
                # Try to extract prompt from ComfyUI workflow structure
                for key, value in metadata.items():
                    if isinstance(value, dict) and "class_type" in value and value.get("class_type") == "smZ CLIPTextEncode":
                        if "inputs" in value and "text" in value["inputs"]:
                            return value["inputs"]["text"]
                
                # If we couldn't find it in the CLIPTextEncode node, look for prompt in any parameter named "text"
                prompt = extract_text_from_json(metadata)
                if prompt:
                    return prompt
                
                # Return the raw JSON if we can't find a specific prompt
                return "No clear prompt found in metadata: " + str(metadata)
            
            except json.JSONDecodeError:
                # Not JSON, so let's check if it's plain text
                # Look for text that might be a prompt using regex
                match = re.search(r"Oil painting portrait of (.+)Ilya Repin", result.stdout)
                if match:
                    return "Oil painting portrait of " + match.group(1) + "Ilya Repin" + result.stdout.split("Ilya Repin")[1]
                
                # If we can't parse it, return the raw text (limited to first 1000 chars)
                return result.stdout[:1000]
        
        # Try other metadata fields if UserComment is empty
        result = subprocess.run(
            ["exiftool", "-j", image_path], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            try:
                metadata = json.loads(result.stdout)
                if metadata and isinstance(metadata, list):
                    metadata = metadata[0]
                    
                    # Check various fields where prompt might be stored
                    for field in ["ImageDescription", "Description", "Comment", "XMP:Prompt", "Parameters"]:
                        if field in metadata and metadata[field]:
                            return metadata[field]
            except:
                pass
        
        return None
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None

def extract_text_from_json(data, key_hint="text"):
    """Recursively search for keys containing 'text' in nested JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_hint:
                return value
            elif isinstance(value, (dict, list)):
                result = extract_text_from_json(value, key_hint)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                result = extract_text_from_json(item, key_hint)
                if result:
                    return result
    return None

def main():
    # The directory containing the images
    image_dir = "/home/harry/loras/ilya_repin_dataset"
    output_json = os.path.join(image_dir, "original_prompts.json")
    
    # Get all image files
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_files:
        print(f"No JPG images found in {image_dir}")
        return
    
    print(f"Analyzing {len(image_files)} images for prompts...")
    
    # Extract prompts from each image
    prompts = {}
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"Processing {filename}...")
        
        prompt = extract_prompt_from_metadata(image_path)
        if prompt:
            prompts[filename] = prompt
            print(f"  Found prompt: {prompt[:100]}..." if len(prompt) > 100 else f"  Found prompt: {prompt}")
        else:
            print(f"  No prompt found in metadata")
    
    # Save the prompts to a JSON file
    with open(output_json, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\nExtracted {len(prompts)} prompts out of {len(image_files)} images")
    print(f"Saved prompts to {output_json}")

if __name__ == "__main__":
    main() 