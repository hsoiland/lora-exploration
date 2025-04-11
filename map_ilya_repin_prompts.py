#!/usr/bin/env python3

import os
import re
import json
from collections import defaultdict

# Base prompt template
base_prompt = (
    "Oil painting portrait of [SUBJECT], depicted in the expressive realist style of Ilya Repin. "
    "Rich, textured brushstrokes. Deep emotional presence. Muted natural palette. "
    "Historical realism. Soft directional lighting or chiaroscuro. Dramatic, timeless atmosphere."
)

# List of female portrait subjects
subjects = [
    "Elderly fisherwoman holding a net, wind-worn face and kind eyes",
    "Young orphan girl wrapped in a tattered shawl, gazing upward",
    "Exiled revolutionary woman with bound wrists and defiant expression",
    "Village priestess in dark robes, lit by candlelight",
    "Widow clutching a letter, face streaked with tears",
    "Nurse dressing a soldier's wound, weary but resolute",
    "Peasant woman with wheat bundles, sun-kissed and strong",
    "Elder scholar woman under lamplight, spectacles low on nose",
    "Retired officer's widow in faded military finery, haunted eyes",
    "Street girl with a crust of bread, dirt-streaked cheeks",
    "Midwife near a hearth, hands worn, smile faint",
    "Blacksmith's wife in a soot-streaked apron, quiet strength",
    "Seamstress threading a needle by a window, focused expression",
    "Cossack matriarch in fur shawl, eyes sharp and wise",
    "Wandering mystic woman with wild curls and distant stare",
    "Young girl clutching a broken toy, eyes wide with sorrow",
    "Ballet dancer backstage, sweat and grace mixed, resting",
    "Dying poetess in bed, clutching her last journal",
    "Bride in traditional dress, veil glowing in soft light",
    "Servant girl gazing out the window, lost in thought",
    "Elder merchant woman counting coins, knowing smile",
    "Political prisoner in a stone cell, tired but proud",
    "Mother nursing an infant, gentle and still",
    "Young female artist with paint-stained fingers, fierce eyes",
    "Retired general's widow at dusk, medals in her hands",
    "Woodcutter's wife with axe resting on her shoulder",
    "Pilgrim woman holding a cross, face lined by travel",
    "Violinist mid-performance, eyes closed, rapt in sound",
    "Schoolteacher dusting her hands with chalk, calm resolve",
    "Railway worker woman with oil lamp, alert expression",
    "Mid-aged woman with spectacles, reading a political pamphlet",
    "Frail noble daughter wrapped in embroidered blankets",
    "Fortune teller adorned with coins and shawls, intense gaze",
    "Cloistered nun meditating, face bathed in candlelight",
    "Washerwoman by the river, sleeves rolled, eyes calm",
    "Dancer tying her shoes, bruised knees and burning will",
    "Young resistance fighter hiding a pistol beneath her coat"
]

def main():
    # Path to the image directory
    image_dir = "/home/harry/loras/ilya_repin_style"
    
    # Get all jpg files
    jpg_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Extract timestamps and group files by timestamp
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T(\d{2}\.\d{2}\.\d{2}))_(\d+)\.jpg')
    timestamp_groups = defaultdict(list)
    
    for filename in jpg_files:
        match = timestamp_pattern.match(filename)
        if match:
            full_timestamp = match.group(1)
            time_part = match.group(2)
            timestamp_groups[time_part].append(filename)
    
    # Sort timestamps to ensure consistent ordering
    sorted_timestamps = sorted(timestamp_groups.keys())
    
    # Create mapping dictionary
    mapping = {}
    
    for i, timestamp in enumerate(sorted_timestamps):
        # Get the subject for this timestamp (cycle through the list if needed)
        subject_index = i % len(subjects)
        subject = subjects[subject_index]
        
        # Create the full prompt
        full_prompt = base_prompt.replace("[SUBJECT]", subject)
        
        # Add mappings for all files with this timestamp
        for filename in timestamp_groups[timestamp]:
            mapping[filename] = full_prompt
    
    # Save to JSON file
    output_file = "ilya_repin_prompts.json"
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Generated {len(mapping)} prompt mappings, saved to {output_file}")
    print(f"Used {len(sorted_timestamps)} unique timestamps with {len(subjects)} subjects")

if __name__ == "__main__":
    main() 