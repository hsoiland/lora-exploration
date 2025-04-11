import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory if it doesn't exist
os.makedirs('lora_training_diagrams', exist_ok=True)

# Set up color scheme
colors = {
    'latent': '#e6f7ff',
    'noise': '#ffebcc',
    'model': '#e6ffe6',
    'frozen': '#d4f1f9',
    'trainable': '#ffcccc',
    'loss': '#ffe6e6',
    'grad': '#f9ecff',
    'vae': '#f2d9e6',
    'unet': '#daeaf6',
    'prompt': '#fff8e1',
    'output': '#e8f5e9'
}

def draw_arrow(ax, start, end, color='black', width=1.5, style='->', connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, color=color, 
        linewidth=width, connectionstyle=connectionstyle
    )
    ax.add_patch(arrow)
    return arrow

def draw_box(ax, x, y, width, height, label, color='white', alpha=0.8, fontsize=10):
    box = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize)
    return box

# Figure 1: Overall LoRA Training Flow
plt.figure(figsize=(14, 10))
plt.suptitle('Detailed LoRA Training Process for SDXL', fontsize=16)

# Main flow
steps = [
    "Load Image & Caption", 
    "Encode to Latent Space (VAE)",
    "Add Random Noise (timestep t)",
    "UNet Forward Pass with LoRA",
    "Predict Noise",
    "Compare with Original Noise",
    "Calculate MSE Loss",
    "Backpropagate Gradients",
    "Update only LoRA A & B matrices",
    "Repeat for next image/batch"
]

colors_map = [
    colors['prompt'],
    colors['vae'], 
    colors['noise'],
    colors['unet'],
    colors['model'],
    colors['noise'],
    colors['loss'],
    colors['grad'],
    colors['trainable'],
    colors['prompt']
]

for i, (step, color) in enumerate(zip(steps, colors_map)):
    y_pos = 9 - i * 0.9
    plt.text(0.3, y_pos, f"{i+1}. {step}", fontsize=14, ha='left', va='center',
             bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.5'))
    
    if i < len(steps) - 1:
        draw_arrow(plt.gca(), (0.3, y_pos - 0.3), (0.3, y_pos - 0.6), width=2)

# Add explanatory notes
notes = [
    (7.5, 8.5, "Starts with high-quality images and descriptive captions"),
    (7.5, 7.5, "VAE compresses image to smaller latent representation"),
    (7.5, 6.5, "Random noise of varying strength added based on timestep"),
    (7.5, 5.5, "UNet has two paths: frozen weights and trainable LoRA"),
    (7.5, 4.5, "UNet's job is to predict the noise that was added"),
    (7.5, 3.5, "Compare prediction with the actual noise added"),
    (7.5, 2.5, "Loss = Mean Squared Error between noise and prediction"),
    (7.5, 1.5, "Gradients flow back, but only LoRA matrices are updated"),
    (7.5, 0.5, "Extremely parameter-efficient: only ~1% of weights trained")
]

for x, y, note in notes:
    plt.text(x, y, note, fontsize=12, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

plt.axis('off')
plt.tight_layout()
plt.savefig('lora_training_diagrams/01_overall_flow.png', dpi=300, bbox_inches='tight')

# Figure 2: Detailed UNet Forward Pass with LoRA
plt.figure(figsize=(14, 10))
plt.suptitle('UNet Forward Pass with LoRA', fontsize=16)

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# Diagram 1: LoRA architecture in one layer
ax1 = plt.subplot(gs[0, 0])
ax1.set_title('LoRA in a Single UNet Attention Layer', fontsize=14)

# Original weights path
draw_box(ax1, 0.2, 0.7, 0.6, 0.2, "Input Embedding", color=colors['prompt'])
draw_arrow(ax1, (0.5, 0.7), (0.5, 0.6))

draw_box(ax1, 0.2, 0.4, 0.6, 0.2, "Original Weights (Frozen)", color=colors['frozen'])
draw_box(ax1, 0.05, 0.2, 0.4, 0.1, "Original Output", color=colors['frozen'])
draw_arrow(ax1, (0.5, 0.4), (0.25, 0.3))

# LoRA path
draw_arrow(ax1, (0.5, 0.6), (0.7, 0.5), connectionstyle="arc3,rad=0.3")
draw_box(ax1, 0.5, 0.4, 0.4, 0.1, "Matrix A (Trainable)", color=colors['trainable'])
draw_arrow(ax1, (0.7, 0.4), (0.7, 0.3))
draw_box(ax1, 0.5, 0.2, 0.4, 0.1, "Matrix B (Trainable)", color=colors['trainable'])
draw_arrow(ax1, (0.7, 0.2), (0.7, 0.1))
draw_box(ax1, 0.5, 0, 0.4, 0.1, "LoRA Output × (α/r)", color=colors['trainable'])
draw_arrow(ax1, (0.5, 0.05), (0.45, 0.05))

# Combined output
draw_box(ax1, 0.2, 0, 0.25, 0.1, "Combined Output", color=colors['output'])
draw_arrow(ax1, (0.25, 0.2), (0.25, 0.1))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Diagram 2: Target modules in UNet
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('LoRA Target Modules in SDXL UNet', fontsize=14)

module_list = [
    "to_q (Query projection)",
    "to_k (Key projection)",
    "to_v (Value projection)",
    "to_out.0 (Output projection)",
    "ff.net.0.proj (Feed-forward)",
    "ff.net.2 (Feed-forward)"
]

for i, module in enumerate(module_list):
    if i < 4:  # First four are typically targeted
        color = colors['trainable']
    else:
        color = colors['frozen']
    
    draw_box(ax2, 0.1, 0.8 - i*0.12, 0.8, 0.1, module, color=color)

ax2.text(0.5, 0.2, "Rank-16 LoRA adapters added\nto attention modules only", 
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Diagram 3: Architecture diagram
ax3 = plt.subplot(gs[1, 0])
ax3.set_title('SDXL Architecture with LoRA', fontsize=14)

# Draw SDXL architecture with LoRA
components = [
    (0.1, 0.8, 0.8, 0.1, "Text Encoder 1 & 2 (CLIP)", colors['frozen']),
    (0.1, 0.65, 0.8, 0.1, "Text Embeddings", colors['prompt']),
    (0.1, 0.35, 0.8, 0.2, "UNet with LoRA Adapters", colors['unet']),
    (0.1, 0.2, 0.8, 0.1, "VAE Decoder (Frozen)", colors['frozen']),
    (0.1, 0.05, 0.8, 0.1, "Generated Image", colors['output']),
]

for x, y, w, h, label, color in components:
    draw_box(ax3, x, y, w, h, label, color=color)
    if y > 0.1:  # Don't draw arrow after the last component
        draw_arrow(ax3, (0.5, y), (0.5, y-0.05))

# Highlight LoRA in UNet
lora_highlight = """
UNet:
- Frozen base weights
- Small LoRA adapters
- Only ~1% parameters trained
- Preserves base knowledge
"""
ax3.text(1.05, 0.45, lora_highlight, fontsize=10, ha='left', va='center',
        bbox=dict(facecolor=colors['trainable'], alpha=0.7, boxstyle='round,pad=0.5'))
draw_arrow(ax3, (0.9, 0.45), (1.03, 0.45))

ax3.set_xlim(0, 1.5)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Diagram 4: Noise prediction process
ax4 = plt.subplot(gs[1, 1])
ax4.set_title('Noise Prediction Process', fontsize=14)

# Create a small visual of noise prediction process
draw_box(ax4, 0.1, 0.7, 0.3, 0.2, "Clean Latent\nz₀", color=colors['latent'])
draw_box(ax4, 0.6, 0.7, 0.3, 0.2, "Noise\nɛ", color=colors['noise'])

draw_arrow(ax4, (0.4, 0.8), (0.6, 0.8))
ax4.text(0.5, 0.9, "Add Noise", ha='center', va='center', fontsize=10)

draw_box(ax4, 0.35, 0.4, 0.3, 0.2, "Noisy Latent\nzₜ", color=colors['latent'])
draw_arrow(ax4, (0.25, 0.7), (0.4, 0.6), connectionstyle="arc3,rad=-0.3")
draw_arrow(ax4, (0.75, 0.7), (0.6, 0.6), connectionstyle="arc3,rad=0.3")

draw_box(ax4, 0.1, 0.1, 0.3, 0.2, "Predicted Noise\nɛ_θ", color=colors['model'])
draw_arrow(ax4, (0.45, 0.4), (0.25, 0.3))
ax4.text(0.35, 0.35, "UNet + LoRA", ha='center', va='center', fontsize=10)

draw_box(ax4, 0.6, 0.1, 0.3, 0.2, "Actual Noise\nɛ", color=colors['noise'])
draw_arrow(ax4, (0.75, 0.7), (0.75, 0.3), connectionstyle="arc3,rad=0")

# MSE Loss
draw_box(ax4, 0.35, -0.1, 0.3, 0.1, "MSE Loss", color=colors['loss'])
draw_arrow(ax4, (0.25, 0.1), (0.4, 0), connectionstyle="arc3,rad=-0.1")
draw_arrow(ax4, (0.75, 0.1), (0.6, 0), connectionstyle="arc3,rad=0.1")

ax4.set_xlim(0, 1)
ax4.set_ylim(-0.15, 1)
ax4.axis('off')

plt.tight_layout()
plt.savefig('lora_training_diagrams/02_unet_forward.png', dpi=300, bbox_inches='tight')

# Figure 3: Backpropagation Through LoRA
plt.figure(figsize=(14, 10))
plt.suptitle('Backpropagation Through LoRA', fontsize=16)

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# Diagram 1: Gradient flow overview
ax1 = plt.subplot(gs[0, 0])
ax1.set_title('Gradient Flow Overview', fontsize=14)

components = [
    (0.1, 0.8, 0.8, 0.1, "Loss Function (MSE)", colors['loss']),
    (0.1, 0.65, 0.8, 0.1, "UNet Output (Predicted Noise)", colors['model']),
    (0.1, 0.4, 0.35, 0.15, "Original Weights\n(Gradients Calculated\nbut Not Applied)", colors['frozen']),
    (0.55, 0.4, 0.35, 0.15, "LoRA Matrices\n(Gradients Applied)", colors['trainable']),
    (0.1, 0.2, 0.35, 0.1, "No Update", colors['frozen']),
    (0.55, 0.2, 0.35, 0.1, "Parameter Update", colors['trainable']),
]

for x, y, w, h, label, color in components:
    draw_box(ax1, x, y, w, h, label, color=color, fontsize=10)

# Arrows
draw_arrow(ax1, (0.5, 0.8), (0.5, 0.75))
ax1.text(0.65, 0.77, "∂L/∂y", fontsize=12)

draw_arrow(ax1, (0.3, 0.65), (0.3, 0.55), connectionstyle="arc3,rad=-0.1")
draw_arrow(ax1, (0.7, 0.65), (0.7, 0.55), connectionstyle="arc3,rad=0.1")

ax1.text(0.2, 0.6, "∂L/∂W", fontsize=12)
ax1.text(0.8, 0.6, "∂L/∂(BA)", fontsize=12)

draw_arrow(ax1, (0.3, 0.4), (0.3, 0.3))
draw_arrow(ax1, (0.7, 0.4), (0.7, 0.3))

# Add note
ax1.text(0.5, 0.05, "This is why LoRA is so efficient -\nonly a small fraction of parameters are updated",
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Diagram 2: Gradient calculation for LoRA
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('Gradient Calculation for LoRA Matrices', fontsize=14)

# Create math notation for LoRA gradient
math_text = r"""
$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial B} = \frac{\partial L}{\partial y} \cdot (A \cdot x)^T \cdot \frac{\alpha}{r}$

$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial A} = B^T \cdot \frac{\partial L}{\partial y} \cdot x^T \cdot \frac{\alpha}{r}$

Where:
- $L$ is the loss function
- $y$ is the model output
- $B$ and $A$ are LoRA matrices
- $x$ is the input to the layer
- $\alpha/r$ is the scaling factor
"""

ax2.text(0.5, 0.5, math_text, ha='center', va='center', fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Diagram 3: Parameter updates
ax3 = plt.subplot(gs[1, 0])
ax3.set_title('Parameter Updates During Training', fontsize=14)

# Create bar chart showing parameter counts
params = {'Original Model': 2500, 'LoRA': 100}
bars = ax3.bar(params.keys(), params.values(), color=[colors['frozen'], colors['trainable']])
ax3.set_ylabel('Millions of Parameters')
ax3.set_title('Trainable Parameters')

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 100,
            f'{int(height)}M',
            ha='center', va='bottom')

ax3.text(0.7, 1800, "Only ~1-4% of\nparameters are\ntrained in LoRA",
         fontsize=12, ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

# Diagram 4: Weight Update Process
ax4 = plt.subplot(gs[1, 1])
ax4.set_title('Weight Update Process', fontsize=14)

# Show the weight update process
updates = [
    (0.1, 0.8, 0.8, 0.1, "Original Weights W (Frozen)", colors['frozen']),
    (0.1, 0.6, 0.35, 0.1, "LoRA Matrix A", colors['trainable']),
    (0.55, 0.6, 0.35, 0.1, "LoRA Matrix B", colors['trainable']),
    (0.1, 0.4, 0.35, 0.1, "A - lr × ∂L/∂A", colors['trainable']),
    (0.55, 0.4, 0.35, 0.1, "B - lr × ∂L/∂B", colors['trainable']),
    (0.1, 0.2, 0.8, 0.1, "Effective Weight: W + BA × (α/r)", colors['output']),
]

for x, y, w, h, label, color in updates:
    draw_box(ax4, x, y, w, h, label, color=color, fontsize=10)

# Draw arrows
draw_arrow(ax4, (0.3, 0.6), (0.3, 0.5))
draw_arrow(ax4, (0.7, 0.6), (0.7, 0.5))
ax4.text(0.2, 0.55, "Update", fontsize=10)
ax4.text(0.8, 0.55, "Update", fontsize=10)

draw_arrow(ax4, (0.3, 0.4), (0.4, 0.3), connectionstyle="arc3,rad=-0.2")
draw_arrow(ax4, (0.7, 0.4), (0.6, 0.3), connectionstyle="arc3,rad=0.2")

# Add explanation
update_formula = """
For each training step:
1. Calculate gradients
2. Update A: A_new = A - learning_rate × ∂L/∂A
3. Update B: B_new = B - learning_rate × ∂L/∂B
4. Effective weights automatically update
"""

ax4.text(0.5, 0.05, update_formula, fontsize=10, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.savefig('lora_training_diagrams/03_backprop.png', dpi=300, bbox_inches='tight')

# Figure 4: Complete SDXL Training Process
plt.figure(figsize=(15, 12))
plt.suptitle('Complete SDXL Training Process with LoRA', fontsize=16)

# Create a diagram showing the entire training process for one batch
stages = [
    (0.1, 0.9, 0.2, 0.08, "Image & Caption", colors['prompt']),
    (0.4, 0.9, 0.2, 0.08, "Text Encoders", colors['frozen']),
    (0.7, 0.9, 0.2, 0.08, "Text Embeddings", colors['prompt']),
    
    (0.1, 0.75, 0.2, 0.08, "Image", colors['latent']),
    (0.4, 0.75, 0.2, 0.08, "VAE Encoder", colors['frozen']),
    (0.7, 0.75, 0.2, 0.08, "Latent z₀", colors['latent']),
    
    (0.1, 0.6, 0.2, 0.08, "Timestep t", colors['model']),
    (0.4, 0.6, 0.2, 0.08, "Noise Scheduler", colors['model']),
    (0.7, 0.6, 0.2, 0.08, "Noisy Latent zₜ", colors['latent']),
    
    (0.5, 0.45, 0.4, 0.1, "UNet with LoRA", colors['unet']),
    
    (0.2, 0.3, 0.2, 0.08, "Predicted Noise", colors['model']),
    (0.8, 0.3, 0.2, 0.08, "Original Noise", colors['noise']),
    
    (0.5, 0.15, 0.3, 0.08, "MSE Loss", colors['loss']),
    
    (0.5, 0.05, 0.4, 0.05, "Update LoRA Parameters", colors['trainable']),
]

# Draw boxes and connect with arrows
for x, y, w, h, label, color in stages:
    draw_box(plt.gca(), x, y, w, h, label, color=color)

# Connect with arrows
arrows = [
    ((0.3, 0.9), (0.4, 0.9)),  # Image & Caption to Text Encoders
    ((0.6, 0.9), (0.7, 0.9)),  # Text Encoders to Text Embeddings
    ((0.3, 0.75), (0.4, 0.75)),  # Image to VAE Encoder
    ((0.6, 0.75), (0.7, 0.75)),  # VAE Encoder to Latent
    ((0.3, 0.6), (0.4, 0.6)),  # Timestep to Noise Scheduler
    ((0.6, 0.6), (0.7, 0.6)),  # Noise Scheduler to Noisy Latent
    
    ((0.8, 0.9), (0.7, 0.5)),  # Text Embeddings to UNet
    ((0.8, 0.6), (0.7, 0.5)),  # Noisy Latent to UNet
    ((0.2, 0.6), (0.3, 0.5)),  # Timestep to UNet
    
    ((0.5, 0.45), (0.3, 0.3)),  # UNet to Predicted Noise
    ((0.9, 0.6), (0.9, 0.35)),  # Original Noise preserved
    
    ((0.3, 0.3), (0.4, 0.15)),  # Predicted Noise to Loss
    ((0.9, 0.3), (0.6, 0.15)),  # Original Noise to Loss
    
    ((0.5, 0.15), (0.5, 0.1)),  # Loss to Gradient
]

for start, end in arrows:
    draw_arrow(plt.gca(), start, end)

# Add explanations for each step
explanations = [
    (0.05, 0.95, "1. Preprocessing: Tokenize captions, load images"),
    (0.05, 0.8, "2. Encoding: Convert image to latent representation"),
    (0.05, 0.65, "3. Noise addition: Add noise according to diffusion schedule"),
    (0.05, 0.5, "4. Forward pass: UNet predicts noise with LoRA adapters"),
    (0.05, 0.35, "5. Loss calculation: Compare predicted vs. actual noise"),
    (0.05, 0.2, "6. Backpropagation: Calculate gradients for parameters"),
    (0.05, 0.1, "7. Update: Only LoRA matrices A and B are updated"),
]

for x, y, text in explanations:
    plt.text(x, y, text, fontsize=12, ha='left', va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

# Add UNet detail inset
unet_detail = """
UNet Details:
- Base weights: Frozen
- LoRA adapters: ~1% of params
- Attention layers: Query, Key, Value, Output
- LoRA rank: 16 (vs. 1024+ original dims)
- Scaling factor (α/r): 2.0
"""

plt.text(0.8, 0.45, unet_detail, fontsize=10, ha='left', va='center',
        bbox=dict(facecolor=colors['trainable'], alpha=0.7, boxstyle='round,pad=0.5'))

plt.axis('off')
plt.tight_layout()
plt.savefig('lora_training_diagrams/04_complete_process.png', dpi=300, bbox_inches='tight')

print("Visualizations created in the 'lora_training_diagrams' directory.") 