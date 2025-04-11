#!/usr/bin/env python3

import os
import json
import shutil
import random

def main():
    # Define the subjects we want to focus on - young females only
    target_subjects = [
        "Young orphan girl wrapped in a tattered shawl, gazing upward",
        "Young girl clutching a broken toy, eyes wide with sorrow",
        "Ballet dancer backstage, sweat and grace mixed, resting",
        "Bride in traditional dress, veil glowing in soft light",
        "Servant girl gazing out the window, lost in thought",
        "Young female artist with paint-stained fingers, fierce eyes",
        "Dancer tying her shoes, bruised knees and burning will",
        "Young resistance fighter hiding a pistol beneath her coat"
    ]
    
    # Path to the generated prompts file
    prompts_file = "ilya_repin_prompts.json"
    
    # Output directory for selected images
    output_dir = "ilya_repin_selected"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the prompts mapping
    with open(prompts_file, 'r') as f:
        mapping = json.load(f)
    
    # Find all images matching our target subjects
    matching_images = []
    for filename, prompt in mapping.items():
        for subject in target_subjects:
            if subject in prompt:
                matching_images.append(filename)
                break  # Move to next image once a match is found
    
    print(f"Found {len(matching_images)} images matching the selected young female subjects")
    
    # If we have too many images, limit the selection
    max_images = 40
    selected_images = matching_images
    if len(matching_images) > max_images:
        selected_images = random.sample(matching_images, max_images)
    
    # Copy the selected images to the output directory
    source_dir = "/home/harry/loras/ilya_repin_style"
    for img in selected_images:
        source_path = os.path.join(source_dir, img)
        dest_path = os.path.join(output_dir, img)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"Copied {img}")
        else:
            print(f"Warning: Source file not found: {source_path}")
    
    # Create a captions file for the selected images
    captions = {}
    for img in selected_images:
        if img in mapping:
            captions[img] = mapping[img]
    
    # Save the captions
    captions_file = os.path.join(output_dir, "captions.json")
    with open(captions_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    # Print statistics about subject distribution
    subject_count = {subject: 0 for subject in target_subjects}
    for img in selected_images:
        prompt = mapping.get(img, "")
        for subject in target_subjects:
            if subject in prompt:
                subject_count[subject] += 1
                break
    
    print("\nSubject distribution in selected dataset:")
    for subject, count in subject_count.items():
        print(f"- {subject}: {count} images")
    
    print(f"\nCreated dataset with {len(selected_images)} images")
    print(f"Saved captions to {captions_file}")

if __name__ == "__main__":
    main() 