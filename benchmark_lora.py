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

def benchmark_python_lora(weight, lora_a, lora_b, alpha=1.0, iterations=10):
    """Benchmark the Python LoRA implementation"""
    times = []
    
    # Warmup
    lora_delta = torch.matmul(lora_b, lora_a)
    result = weight + alpha * lora_delta
    
    # Benchmark
    for _ in range(iterations):
        start_time = time.time()
        lora_delta = torch.matmul(lora_b, lora_a)
        result = weight + alpha * lora_delta
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    return result, np.mean(times), np.min(times), np.max(times)

def benchmark_rust_lora(weight, lora_a, lora_b, alpha=1.0, iterations=10):
    """Benchmark the Rust LoRA implementation"""
    if not has_rust:
        return None, 0, 0, 0
    
    # Convert to NumPy for Rust
    weight_np = weight.detach().cpu().numpy()
    lora_a_np = lora_a.detach().cpu().numpy()
    lora_b_np = lora_b.detach().cpu().numpy()
    
    # Warmup
    result_np = lora_ops.apply_lora(weight_np, lora_a_np, lora_b_np, alpha)
    
    times = []
    for _ in range(iterations):
        start_time = time.time()
        result_np = lora_ops.apply_lora(weight_np, lora_a_np, lora_b_np, alpha)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    result = torch.from_numpy(result_np)
    return result, np.mean(times), np.min(times), np.max(times)

def run_benchmark():
    """Run LoRA benchmarks with different matrix sizes"""
    print("\n=== LoRA Performance Benchmark ===\n")
    
    # Test various matrix sizes
    sizes = [
        # (out_features, in_features, rank)
        (128, 128, 4),   # Small
        (512, 512, 8),   # Medium
        (1280, 1280, 16) # Large (SDXL typical size)
    ]
    
    for out_dim, in_dim, rank in sizes:
        print(f"\nBenchmarking matrices: {out_dim}x{in_dim}, rank {rank}")
        
        # Create test matrices
        weight = torch.randn(out_dim, in_dim)
        lora_a = torch.randn(rank, in_dim)  # Down projection
        lora_b = torch.randn(out_dim, rank)  # Up projection
        alpha = 0.7
        
        # Python benchmark
        python_result, py_mean, py_min, py_max = benchmark_python_lora(
            weight, lora_a, lora_b, alpha, iterations=5
        )
        print(f"Python implementation:")
        print(f"  - Mean: {py_mean:.6f} seconds")
        print(f"  - Min:  {py_min:.6f} seconds")
        print(f"  - Max:  {py_max:.6f} seconds")
        
        if has_rust:
            # Rust benchmark
            rust_result, rust_mean, rust_min, rust_max = benchmark_rust_lora(
                weight, lora_a, lora_b, alpha, iterations=5
            )
            print(f"Rust implementation:")
            print(f"  - Mean: {rust_mean:.6f} seconds")
            print(f"  - Min:  {rust_min:.6f} seconds")
            print(f"  - Max:  {rust_max:.6f} seconds")
            
            # Compare results
            diff = torch.abs(python_result - rust_result).mean().item()
            print(f"Mean absolute difference: {diff:.8f}")
            
            # Calculate speedup/slowdown
            speedup = py_mean / max(rust_mean, 1e-8)
            if speedup > 1:
                print(f"Rust is {speedup:.2f}x faster than Python")
            else:
                print(f"Python is {1/speedup:.2f}x faster than Rust")
                print("Note: The overhead of Python/Rust interop may be significant for small matrices")
                print("      Rust would likely be faster for larger batches or when called from Rust code")

if __name__ == "__main__":
    run_benchmark() 