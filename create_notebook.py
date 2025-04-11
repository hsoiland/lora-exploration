import nbformat as nbf
import os

# Create a new notebook with the right JSON structure
nb = nbf.v4.new_notebook()

# Add title and introduction
cells = [
    nbf.v4.new_markdown_cell('# LoRA Training Visualizations\n\nThis notebook displays visual explanations of how LoRA training works with diffusion models.')
]

# Add visualization cells for each diagram
if os.path.exists('lora_training_diagrams'):
    for i, filename in enumerate(['01_overall_flow.png', '02_unet_forward.png', '03_backprop.png', '04_complete_process.png']):
        if os.path.exists(f'lora_training_diagrams/{filename}'):
            # Add title for this section
            title = {
                '01_overall_flow.png': '## 1. Overall LoRA Training Process',
                '02_unet_forward.png': '## 2. UNet Forward Pass with LoRA',
                '03_backprop.png': '## 3. Backpropagation Through LoRA',
                '04_complete_process.png': '## 4. Complete SDXL Training Process'
            }.get(filename, f'## {i+1}. Visualization')
            
            cells.append(nbf.v4.new_markdown_cell(title))
            
            # Add code to display the image
            cells.append(nbf.v4.new_code_cell(f'''
from IPython.display import Image, display
display(Image('lora_training_diagrams/{filename}', width=1000))
'''))

# If no images were found, add a notice
if len(cells) == 1:
    cells.append(nbf.v4.new_markdown_cell('No visualization images found. Please run `visualize_lora_training.py` first.'))

# Add descriptions for each section
descriptions = {
    '01_overall_flow.png': '''
### Key Steps in the LoRA Training Process:

1. **Load Image & Caption**: Start with high-quality images and their captions
2. **Encode to Latent Space**: VAE compresses the image to a latent representation
3. **Add Random Noise**: Apply noise according to the diffusion schedule
4. **UNet Forward Pass**: Predict noise using original weights + LoRA adapters
5. **Compare with Original Noise**: Compare prediction to actual added noise
6. **Calculate Loss**: Mean Squared Error between prediction and target
7. **Backpropagate**: Calculate gradients for all parameters
8. **Update LoRA Only**: Only the small adapter matrices are modified
9. **Repeat**: Continue training on all images for multiple epochs

This approach is extremely parameter-efficient, training only ~1% of the weights.
''',
    '02_unet_forward.png': '''
### LoRA in Attention Layers:

The top-left diagram shows how LoRA adapters integrate within a single attention layer:
- Input splits into two paths: through the frozen weights and through LoRA matrices
- Matrix A maps to a low-dimensional representation (rank 16)
- Matrix B maps back to the original dimension
- Original output and LoRA output are combined

The top-right shows which modules get LoRA adapters:
- Query projections (to_q)
- Key projections (to_k)
- Value projections (to_v)
- Output projections (to_out.0)

The bottom diagrams show the overall architecture and noise prediction process.
''',
    '03_backprop.png': '''
### Backpropagation Through LoRA:

During backpropagation:
- Gradients flow from the loss function back through the network
- For original weights, gradients are calculated but NOT applied
- For LoRA matrices, gradients are both calculated and applied
- The mathematical formulas show exactly how these gradients are computed
- Only about 4% of the original parameter count needs to be trained
- Updates follow the standard formula: parameter -= learning_rate * gradient
''',
    '04_complete_process.png': '''
### Complete End-to-End Process:

This comprehensive diagram shows how all components work together:
1. **Preprocessing**: Images and captions are processed through encoders
2. **Latent Encoding**: The VAE compresses images to latent space
3. **Noise Addition**: Random noise is added according to the timestep
4. **UNet Prediction**: The UNet with LoRA predicts the noise
5. **Loss Calculation**: Predicted noise is compared to actual noise
6. **Parameter Updates**: Only LoRA weights are updated

The UNet details shown highlight the key LoRA configuration:
- Rank=16 (vs. original dimensions of 1024+)
- Scaling factor (α/r) = 2.0
- Only attention layers receive LoRA adapters
'''
}

# Add descriptions to the notebook
for i, filename in enumerate(['01_overall_flow.png', '02_unet_forward.png', '03_backprop.png', '04_complete_process.png']):
    if os.path.exists(f'lora_training_diagrams/{filename}'):
        # Find the position to insert the description (after the code cell)
        for j, cell in enumerate(cells):
            if cell.cell_type == 'code' and f'lora_training_diagrams/{filename}' in cell.source:
                cells.insert(j+1, nbf.v4.new_markdown_cell(descriptions.get(filename, 'Description not available.')))
                break

# Add cells to notebook
nb['cells'] = cells

# Add a conclusion
cells.append(nbf.v4.new_markdown_cell('''
## Using Your Trained LoRAs

After training, you can generate images by using the special tokens in your prompts:

```python
# Person-specific prompt
person_prompt = "<gina_szanyel> a portrait of a woman with red hair in a forest setting, high quality photo"

# Style-specific prompt
style_prompt = "<ilya_repin_painting> Oil painting portrait of a young woman, depicted in the expressive realist style"

# Combined prompt
combined_prompt = "<gina_szanyel> <ilya_repin_painting> portrait of a woman with red hair, oil painting style"
```

The small size of LoRA weights (typically 1-30MB) makes them easy to share and use with others.
'''))

# Write the notebook to file with proper JSON formatting
with open('lora_visualizations.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Created new notebook: lora_visualizations.ipynb') 