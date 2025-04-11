import torch
import numpy as np
import time

try:
    import lora_ops
    print("✅ Successfully imported Rust LoRA module")
    has_rust = True
except ImportError:
    print("❌ Failed to import Rust LoRA module")
    has_rust = False

def batch_apply_lora_python(weight, lora_pairs, alpha=1.0):
    """
    Apply multiple LoRA transformations with Python
    
    Args:
        weight: Base weight matrix
        lora_pairs: List of (lora_a, lora_b) pairs to apply
        alpha: Scaling factor
    """
    result = weight.clone()
    
    for lora_a, lora_b in lora_pairs:
        # W += alpha * (BA)
        lora_delta = torch.matmul(lora_b, lora_a)
        result += alpha * lora_delta
    
    return result

def benchmark_batch_python(weight, lora_pairs, alpha=1.0, iterations=10):
    """Benchmark Python batch processing"""
    start_time = time.time()
    
    # Warmup
    result = batch_apply_lora_python(weight, lora_pairs, alpha)
    
    times = []
    for _ in range(iterations):
        start = time.time()
        result = batch_apply_lora_python(weight, lora_pairs, alpha)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return result, np.mean(times)

def batch_apply_lora_rust_manual(weight, lora_pairs, alpha=1.0):
    """
    Apply multiple LoRA transformations with Rust (manual batching)
    
    Uses individual Rust calls but without converting back to PyTorch in between
    """
    # Convert weight to NumPy
    weight_np = weight.detach().cpu().numpy()
    result_np = weight_np.copy()
    
    # Process each LoRA pair
    for lora_a, lora_b in lora_pairs:
        # Convert to NumPy
        lora_a_np = lora_a.detach().cpu().numpy()
        lora_b_np = lora_b.detach().cpu().numpy()
        
        # Apply LoRA (W + alpha * BA)
        try:
            # Call Rust implementation
            result_np = lora_ops.apply_lora(result_np, lora_a_np, lora_b_np, alpha)
        except Exception as e:
            print(f"Rust error: {e}")
            return None
    
    # Convert back to PyTorch tensor
    return torch.from_numpy(result_np)

def benchmark_batch_rust_manual(weight, lora_pairs, alpha=1.0, iterations=10):
    """Benchmark Rust manual batch processing"""
    if not has_rust:
        return None, 0
    
    # Warmup
    result = batch_apply_lora_rust_manual(weight, lora_pairs, alpha)
    
    times = []
    for _ in range(iterations):
        start = time.time()
        result = batch_apply_lora_rust_manual(weight, lora_pairs, alpha)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return result, np.mean(times)

def run_batch_benchmark():
    """Run LoRA batch processing benchmark"""
    print("\n=== LoRA Batch Processing Benchmark ===\n")
    
    # Matrix dimensions
    out_dim, in_dim = 1280, 1280  # SDXL size
    rank = 4
    
    # Create base weight matrix
    weight = torch.randn(out_dim, in_dim)
    
    # Create multiple LoRA pairs
    batch_sizes = [1, 5, 10, 20]
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Generate LoRA pairs
        lora_pairs = []
        for _ in range(batch_size):
            lora_a = torch.randn(rank, in_dim)  # Down projection
            lora_b = torch.randn(out_dim, rank)  # Up projection
            lora_pairs.append((lora_a, lora_b))
        
        # Python benchmark
        python_result, python_time = benchmark_batch_python(
            weight, lora_pairs, alpha=0.7, iterations=3
        )
        print(f"Python batch processing: {python_time:.6f} seconds")
        
        if has_rust:
            # Rust benchmark (manual batching)
            rust_result, rust_time = benchmark_batch_rust_manual(
                weight, lora_pairs, alpha=0.7, iterations=3
            )
            
            if rust_result is not None:
                print(f"Rust batch processing:   {rust_time:.6f} seconds")
                
                # Calculate speedup
                speedup = python_time / max(rust_time, 1e-8)
                if speedup > 1:
                    print(f"Rust is {speedup:.2f}x faster")
                else:
                    print(f"Python is {1/speedup:.2f}x faster")
                
                # Verify results match
                if python_result is not None and rust_result is not None:
                    diff = torch.abs(python_result - rust_result).mean().item()
                    print(f"Mean absolute difference: {diff:.8f}")
            else:
                print("Rust implementation failed")

if __name__ == "__main__":
    run_batch_benchmark() 