# src/__init__.py
# Import core functions to make them available from the package root
from .lora import inject_lora_into_model
from .rust_lora_self_attn import apply_lora_to_self_attention

# Import training functionality
try:
    from .lora_trainer import RustLoraTrainer, train_lora
except ImportError:
    pass  # Training functionality is optional 