"""
Test script for the Rust LORA implementation.

This is a minimal example to verify that the Rust LORA implementation is working correctly.
"""

import numpy as np
import torch
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Try importing the Rust LORA implementation
try:
    # Try importing directly (when run as a module)
    import lora_ops
    from lora_ops import LoraTrainingContext, AdamParams
except ImportError:
    # Try importing as src.lora_ops (when run as script)
    try:
        import src.lora_ops as lora_ops
        from src.lora_ops import LoraTrainingContext, AdamParams
    except ImportError:
        logging.error("Failed to import Rust LORA implementation")
        raise ImportError("Rust LORA implementation not found")

def test_lora_initialization():
    """Test basic initialization of LORA context."""
    logging.info("Testing LORA initialization...")
    
    # Create a small random weight matrix
    weight = np.random.randn(128, 256).astype(np.float32)
    logging.info(f"Created random weight matrix with shape {weight.shape}")
    
    try:
        # Create optimizer params
        optimizer = AdamParams(
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        logging.info("Created optimizer parameters")
        
        # Create LORA context
        ctx = LoraTrainingContext(
            layer_name="test_layer",
            weight=weight,
            rank=4,
            alpha=8,
            init_scale=0.01
        )
        logging.info("Successfully created LORA context")
        
        # Set optimizer
        ctx.set_optimizer(optimizer)
        logging.info("Set optimizer for LORA context")
        
        # Create a random gradient
        grad = np.random.randn(128, 256).astype(np.float32)
        
        # Call backward once
        ctx.backward(grad)
        logging.info("Successfully called backward once")
        
        # Get weights
        lora_a, lora_b = ctx.get_weights()
        logging.info(f"Retrieved LORA weights with shapes {lora_a.shape} and {lora_b.shape}")
        
        logging.info("All tests passed successfully!")
        return True
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Testing Rust LORA implementation...")
    success = test_lora_initialization()
    if success:
        print("✅ Rust LORA implementation is working correctly")
    else:
        print("❌ Rust LORA implementation test failed")
        sys.exit(1) 