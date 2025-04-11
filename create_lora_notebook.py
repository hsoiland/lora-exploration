import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and introduction
cells = [
    nbf.v4.new_markdown_cell("""# LoRA Training and Backpropagation Explained

This notebook provides a visual explanation of how LoRA (Low-Rank Adaptation) works in fine-tuning diffusion models.""")
]

# Add section on LoRA basics
cells.append(nbf.v4.new_markdown_cell("""## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:

- Freezes the pre-trained model weights (keeping all knowledge)
- Injects trainable rank decomposition matrices into each layer
- Dramatically reduces the number of trainable parameters
- Results in small, portable "adapter" weights

This makes it possible to fine-tune billion-parameter models like SDXL on consumer hardware."""))

# Add code for simple parameter calculation
cells.append(nbf.v4.new_code_cell("""
# Parameter efficiency calculation
d = 768  # Hidden dimension (common in SDXL layers)
k = 768  # Same dimension for simplicity (square matrix)
r = 16   # LoRA rank

# Parameters in original matrix W
original_params = d * k

# Parameters in LoRA matrices A and B
lora_params = r * k + d * r

# Comparison
print(f"Original matrix parameters: {original_params:,}")
print(f"LoRA matrices parameters: {lora_params:,}")
print(f"Parameter reduction: {original_params / lora_params:.2f}x")
print(f"LoRA is {100 * lora_params / original_params:.2f}% of original size")
"""))

# Add mathematical explanation
cells.append(nbf.v4.new_markdown_cell("""## LoRA Architecture

In a neural network, each layer typically has a weight matrix $W$ that transforms inputs. LoRA decomposes the weight updates into two smaller matrices $A$ and $B$:

### Key Formula

The LoRA update can be expressed mathematically as:

$$W' = W + \\Delta W = W + BA \\cdot \\frac{\\alpha}{r}$$

Where:
- $W$ is the original weight matrix (frozen)
- $\\Delta W$ is the update (BA product)
- $B$ and $A$ are low-rank matrices (trainable)
- $r$ is the rank (typically 4-128)
- $\\alpha$ is a scaling factor (typically 1-128)"""))

# Add illustration of the concept using ASCII art (since we can't easily do diagrams)
cells.append(nbf.v4.new_markdown_cell("""## Visual Representation of LoRA

```
Original Model:                      With LoRA:
                                     
    Input                                Input
      |                                    |
      v                                   / \\
 [Frozen Weights]                   [Frozen W]   [Matrix A]
      |                                |            |
      v                                |            v
    Output                             |        [Matrix B]
                                       |            |
                                       |            v
                                       |      [Scale by α/r]
                                       |            |
                                       v            v
                                      Output + LoRA Output
                                            |
                                            v
                                       Final Output
```"""))

# Add training process explanation
cells.append(nbf.v4.new_markdown_cell("""## Training Process Flow

When training a LoRA for SDXL, the process works like this:

1. Load an image & its caption (e.g., from your Gina or Ilya Repin datasets)
2. Encode the image to latent space using the VAE
3. Add random noise to this latent
4. Have the UNet predict this noise (using LoRA modules)
5. Compare the predicted noise with the actual noise added
6. Calculate loss (MSE between predicted and actual noise)
7. Backpropagate through UNet but ONLY update the LoRA matrices A and B
8. Repeat for the next image

This approach maintains all the knowledge in the base model while adding your specific concept or style."""))

# Add backpropagation details
cells.append(nbf.v4.new_markdown_cell("""## Backpropagation Through LoRA

During backpropagation:

1. The loss gradient ∂L/∂y flows backward from the output
2. For original weights W, gradients are calculated but not applied (W is frozen)
3. For LoRA matrices:
   - ∂L/∂B = ∂L/∂y × ∂y/∂B = ∂L/∂y × (A×x)ᵀ × (α/r)
   - ∂L/∂A = ∂L/∂y × ∂y/∂A = Bᵀ × ∂L/∂y × xᵀ × (α/r)
4. Only A and B matrices are updated:
   - A_new = A - learning_rate × ∂L/∂A
   - B_new = B - learning_rate × ∂L/∂B

This focused update is why LoRA is so efficient for fine-tuning."""))

# Add PyTorch implementation
cells.append(nbf.v4.new_markdown_cell("""## PyTorch Implementation

Here's a simple PyTorch implementation of a LoRA layer:"""))

cells.append(nbf.v4.new_code_cell("""
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, alpha=32):
        super().__init__()
        self.original_layer = nn.Linear(in_dim, out_dim, bias=False)
        
        # Special initialization as in the LoRA paper
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # Scaling factor
        self.scale = alpha / rank
        
        # Freeze the original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output
        # The BA product creates the effective weight update
        lora_output = (self.lora_B @ self.lora_A) @ x.T
        lora_output = lora_output.T  # Transpose to match dimensions
        
        # Combined output
        return original_output + lora_output * self.scale


# Example usage
try:
    # Create a small example
    lora_layer = LoRALayer(768, 768, rank=16, alpha=32)
    test_input = torch.randn(1, 768)

    # Count parameters
    total_params = sum(p.numel() for p in lora_layer.parameters())
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Test forward pass
    output = lora_layer(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Error running the example: {e}")
    print("You may need to install PyTorch")
"""))

# Add application to datasets
cells.append(nbf.v4.new_markdown_cell("""## Application to Your Datasets

Your script `train_sdxl_lora_simple.py` is set up to train two LoRA adapters:

1. **Gina Dataset**: A person-specific LoRA with the `<gina_szanyel>` token
2. **Ilya Repin Dataset**: A style-specific LoRA with the `<ilya_repin_painting>` token

The training command sets these key parameters:
- `rank=16`: Good balance between capacity and efficiency
- `lora_alpha=32`: Scaling factor for the adapter matrices
- `image_size=1024`: SDXL's native resolution
- `num_train_epochs=20`: Sufficient training time

This approach allows creating both a subject-specific LoRA and a style-specific LoRA from your datasets."""))

# Add section on using trained LoRAs
cells.append(nbf.v4.new_markdown_cell("""## Using Your Trained LoRAs

After training, you can generate images by using the special tokens in your prompts:

```
# Person-specific prompt
"<gina_szanyel> a portrait of a woman with red hair in a forest setting, high quality photo"

# Style-specific prompt
"<ilya_repin_painting> Oil painting portrait of a young woman, depicted in the expressive realist style"

# Combined prompt
"<gina_szanyel> <ilya_repin_painting> portrait of a woman with red hair, oil painting style"
```

The small size of LoRA weights (typically 1-30MB) makes them easy to share and use with others."""))

# Conclusion and references
cells.append(nbf.v4.new_markdown_cell("""## References

1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
2. Diffusers library documentation for LoRA: https://huggingface.co/docs/diffusers/training/lora
3. PEFT library: https://github.com/huggingface/peft"""))

# Add all cells to the notebook
nb['cells'] = cells

# Create the notebook
with open('lora_training_complete.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Comprehensive LoRA notebook created successfully!") 