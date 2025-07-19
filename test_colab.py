#!/usr/bin/env python3
"""
Test script for Splash Attention CUDA kernel in Google Colab
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import time

def setup_colab_environment():
    """Setup the environment for CUDA compilation in Colab"""
    print("Setting up Google Colab environment...")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Install required packages
    os.system("pip install ninja")
    
    # Set environment variables for compilation
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'  # Common Colab GPU architectures
    
    return torch.cuda.is_available()

def compile_cuda_extension():
    """Compile the CUDA extension"""
    print("Compiling CUDA extension...")
    
    # Create the source directory if it doesn't exist
    os.makedirs('source', exist_ok=True)
    
    # The CUDA source should already be in source/adasplashattention.cu
    cuda_sources = ['source/adasplashattention.cu']
    
    # Check if source file exists
    if not os.path.exists('source/adasplashattention.cu'):
        print("ERROR: source/adasplashattention.cu not found!")
        print("Please upload the CUDA source file to the source/ directory")
        return None
    
    try:
        # Compile the extension
        splash_attention = load(
            name="splash_attention",
            sources=cuda_sources,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas=-v'],
            verbose=True
        )
        print("✅ CUDA extension compiled successfully!")
        return splash_attention
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return None

def reference_attention(Q, K, V, Q_idx, K_idx, sm_scale=1.0, alpha=1.5):
    """
    Reference implementation using standard PyTorch operations
    """
    B, H, NQ, d = Q.shape
    NK = K.size(2)
    
    # Create causal mask
    causal_mask = torch.zeros(B, H, NQ, NK, device=Q.device, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            for q in range(NQ):
                seq_q = Q_idx[b, h, q].item()
                for k in range(NK):
                    seq_k = K_idx[b, h, k].item()
                    causal_mask[b, h, q, k] = (seq_k <= seq_q)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-1, -2)) * sm_scale  # [B, H, NQ, NK]
    
    # Apply causal mask
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    # Apply entmax (simplified - using softmax for now as reference)
    if alpha == 1.0:
        # Standard softmax
        attn_weights = F.softmax(scores, dim=-1)
    else:
        # Simplified entmax approximation (for testing purposes)
        # In practice, you'd want a proper entmax implementation
        attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, V)
    
    return output

def create_test_tensors(B=2, H=4, NQ=64, NK=64, d=32, device='cuda'):
    """Create test tensors with reasonable values"""
    
    # Create Q, K, V tensors
    Q = torch.randn(B, H, NQ, d, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, NK, d, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, NK, d, device=device, dtype=torch.float32, requires_grad=True)
    
    # Create position indices for causal masking
    # Standard causal: each position can attend to previous positions
    Q_idx = torch.arange(NQ, device=device, dtype=torch.int32).view(1, 1, NQ)
    Q_idx = Q_idx.expand(B, H, NQ).contiguous()
    
    K_idx = torch.arange(NK, device=device, dtype=torch.int32).view(1, 1, NK)
    K_idx = K_idx.expand(B, H, NK).contiguous()
    
    return Q, K, V, Q_idx, K_idx

def test_forward_pass(splash_attention, Q, K, V, Q_idx, K_idx, sm_scale=1.0, alpha=1.5):
    """Test the forward pass"""
    print("\n🧪 Testing Forward Pass...")
    
    try:
        # Test CUDA kernel
        start_time = time.time()
        output_cuda = splash_attention.forward(Q, K, V, Q_idx, K_idx, sm_scale, alpha)
        cuda_time = time.time() - start_time
        
        print(f"✅ CUDA forward pass completed in {cuda_time:.4f}s")
        print(f"   Output shape: {output_cuda.shape}")
        print(f"   Output range: [{output_cuda.min().item():.4f}, {output_cuda.max().item():.4f}]")
        
        # Test reference implementation
        start_time = time.time()
        output_ref = reference_attention(Q, K, V, Q_idx, K_idx, sm_scale, alpha)
        ref_time = time.time() - start_time
        
        print(f"✅ Reference forward pass completed in {ref_time:.4f}s")
        print(f"   Speedup: {ref_time/cuda_time:.2f}x")
        
        # Compare outputs
        max_diff = torch.max(torch.abs(output_cuda - output_ref)).item()
        mean_diff = torch.mean(torch.abs(output_cuda - output_ref)).item()
        
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ Forward pass test PASSED")
        else:
            print("⚠️  Forward pass test FAILED - large differences detected")
        
        return output_cuda, output_ref
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return None, None

def test_backward_pass(splash_attention, Q, K, V, Q_idx, K_idx, sm_scale=1.0, alpha=1.5):
    """Test the backward pass"""
    print("\n🧪 Testing Backward Pass...")
    
    try:
        # Create tensors with gradients
        Q_test = Q.clone().detach().requires_grad_(True)
        K_test = K.clone().detach().requires_grad_(True)
        V_test = V.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = splash_attention.forward(Q_test, K_test, V_test, Q_idx, K_idx, sm_scale, alpha)
        
        # Create dummy gradient
        grad_output = torch.randn_like(output)
        
        # Backward pass
        start_time = time.time()
        gradients = splash_attention.backward(Q_test, K_test, V_test, Q_idx, K_idx, sm_scale, alpha, grad_output)
        backward_time = time.time() - start_time
        
        dQ, dK, dV = gradients
        
        print(f"✅ CUDA backward pass completed in {backward_time:.4f}s")
        print(f"   dQ shape: {dQ.shape}, range: [{dQ.min().item():.4f}, {dQ.max().item():.4f}]")
        print(f"   dK shape: {dK.shape}, range: [{dK.min().item():.4f}, {dK.max().item():.4f}]")
        print(f"   dV shape: {dV.shape}, range: [{dV.min().item():.4f}, {dV.max().item():.4f}]")
        
        # Check for NaN or inf values
        if torch.isnan(dQ).any() or torch.isnan(dK).any() or torch.isnan(dV).any():
            print("❌ NaN values detected in gradients")
        elif torch.isinf(dQ).any() or torch.isinf(dK).any() or torch.isinf(dV).any():
            print("❌ Inf values detected in gradients")
        else:
            print("✅ Backward pass test PASSED")
        
        return dQ, dK, dV
        
    except Exception as e:
        print(f"❌ Backward pass test failed: {e}")
        return None, None, None

def benchmark_performance(splash_attention, sizes_to_test=None):
    """Benchmark performance across different sizes"""
    if sizes_to_test is None:
        sizes_to_test = [
            (1, 1, 32, 32, 32),    # Small
            (2, 4, 64, 64, 64),    # Medium
            (4, 8, 128, 128, 64),  # Large
        ]
    
    print("\n📊 Performance Benchmark:")
    print("Size (B,H,NQ,NK,d) | CUDA Time | Memory Used")
    print("-" * 45)
    
    for B, H, NQ, NK, d in sizes_to_test:
        try:
            # Create test tensors
            Q, K, V, Q_idx, K_idx = create_test_tensors(B, H, NQ, NK, d)
            
            # Warm up
            _ = splash_attention.forward(Q, K, V, Q_idx, K_idx, 1.0, 1.5)
            torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            output = splash_attention.forward(Q, K, V, Q_idx, K_idx, 1.0, 1.5)
            torch.cuda.synchronize()
            
            end_time = time.time()
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            print(f"({B:2},{H:2},{NQ:3},{NK:3},{d:2})      | {(end_time-start_time)*1000:6.2f}ms | {memory_used:6.1f}MB")
            
        except Exception as e:
            print(f"({B:2},{H:2},{NQ:3},{NK:3},{d:2})      | FAILED    | {str(e)[:20]}")

def main():
    """Main test function"""
    print("🚀 Splash Attention CUDA Kernel Test")
    print("=" * 50)
    
    # Setup environment
    if not setup_colab_environment():
        print("❌ CUDA not available. This test requires a GPU runtime.")
        return
    
    # Compile extension
    splash_attention = compile_cuda_extension()
    if splash_attention is None:
        return
    
    # Create test tensors
    print("\n📝 Creating test tensors...")
    Q, K, V, Q_idx, K_idx = create_test_tensors(B=2, H=4, NQ=32, NK=32, d=64)
    print(f"✅ Created tensors: Q{Q.shape}, K{K.shape}, V{V.shape}")
    
    # Test forward pass
    output_cuda, output_ref = test_forward_pass(splash_attention, Q, K, V, Q_idx, K_idx)
    
    # Test backward pass
    if output_cuda is not None:
        dQ, dK, dV = test_backward_pass(splash_attention, Q, K, V, Q_idx, K_idx)
    
    # Benchmark performance
    benchmark_performance(splash_attention)
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main() 