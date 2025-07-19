import torch
import time

def analyze_splash_attention_sparsity():
    """Simple check to see if splash attention is actually achieving sparsity"""
    print("🔍 SPLASH ATTENTION SPARSITY ANALYSIS")
    print("=" * 50)
    
    # Test configuration
    B, H, NQ, NK, d = 1, 4, 1024, 1024, 64
    alpha = 1.5
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, NQ, d, device='cuda', dtype=torch.float32) * 0.1
    K = torch.randn(B, H, NK, d, device='cuda', dtype=torch.float32) * 0.1
    V = torch.randn(B, H, NK, d, device='cuda', dtype=torch.float32) * 0.1
    
    Q_idx = torch.arange(NQ, device='cuda', dtype=torch.int32).unsqueeze(0).expand(B*H, NQ).contiguous()
    K_idx = torch.arange(NK, device='cuda', dtype=torch.int32).unsqueeze(0).expand(B*H, NK).contiguous()
    
    sm_scale = 1.0 / (d ** 0.5)
    
    print(f"Configuration: {B}×{H}×{NQ}×{NK}×{d}")
    
    # Calculate theoretical complexity
    vanilla_ops = B * H * NQ * NK  # O(n²) attention matrix computation
    causal_ops = B * H * NQ * (NQ + 1) // 2  # Only lower triangular due to causality
    
    print(f"\n📊 THEORETICAL ANALYSIS:")
    print(f"   Full attention operations: {vanilla_ops:,}")
    print(f"   Causal attention operations: {causal_ops:,}")
    print(f"   Causal sparsity: {(1 - causal_ops/vanilla_ops)*100:.1f}%")
    
    # Splash attention theoretical sparsity
    BLOCK_M, BLOCK_N, K_KEEP = 32, 64, 8
    nQB = (NQ + BLOCK_M - 1) // BLOCK_M
    nKB = (NK + BLOCK_N - 1) // BLOCK_N
    
    # In the worst case, each query block attends to all key blocks
    # But only keeps K_KEEP elements per BLOCK_N in each block
    max_splash_ops = B * H * nQB * nKB * K_KEEP
    splash_sparsity_est = 1 - (max_splash_ops / causal_ops)
    
    print(f"\n⚡ SPLASH ATTENTION ANALYSIS:")
    print(f"   Query blocks: {nQB}, Key blocks: {nKB}")
    print(f"   Max splash operations: {max_splash_ops:,}")
    print(f"   Estimated additional sparsity: {splash_sparsity_est*100:.1f}%")
    print(f"   Total sparsity vs full: {(1 - max_splash_ops/vanilla_ops)*100:.1f}%")
    
    if splash_sparsity_est > 0.5:
        print(f"   ✅ Should provide significant speedup ({splash_sparsity_est:.1f}× fewer ops)")
    elif splash_sparsity_est > 0.2:
        print(f"   ⚡ Should provide moderate speedup ({splash_sparsity_est:.1f}× fewer ops)")
    else:
        print(f"   ❌ Limited sparsity benefit ({splash_sparsity_est:.1f}× fewer ops)")
    
    # Time baseline attention (simple implementation)
    print(f"\n⏱️ TIMING COMPARISON:")
    
    # Baseline: naive causal attention
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    causal_mask = torch.triu(torch.ones(NQ, NK, device=scores.device), diagonal=1).bool()
    scores.masked_fill_(causal_mask, -float('inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    baseline_output = torch.matmul(attn_weights, V)
    
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) * 1000
    
    # Splash attention
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    splash_output = fixed_original_splash.forward(Q, K, V, Q_idx, K_idx, sm_scale, alpha)
    
    torch.cuda.synchronize()
    splash_time = (time.perf_counter() - start) * 1000
    
    speedup = baseline_time / splash_time
    
    print(f"   Baseline time: {baseline_time:.2f} ms")
    print(f"   Splash time: {splash_time:.2f} ms")
    print(f"   Speedup: {speedup:.2f}× {'✅' if speedup > 1.0 else '❌'}")
    
    # Accuracy check
    mse_error = torch.nn.functional.mse_loss(splash_output, baseline_output).item()
    print(f"   MSE Error: {mse_error:.2e}")
    
    # Memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Baseline memory
    scores = torch.matmul(Q, K.transpose(-2, -1))
    baseline_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Splash memory
    _ = fixed_original_splash.forward(Q, K, V, Q_idx, K_idx, sm_scale, alpha)
    splash_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"\n💾 MEMORY USAGE:")
    print(f"   Baseline: {baseline_mem:.1f} MB")
    print(f"   Splash: {splash_mem:.1f} MB")
    print(f"   Ratio: {splash_mem/baseline_mem:.2f}× {'✅' if splash_mem < baseline_mem else '❌'}")
    
    # Diagnosis
    print(f"\n🎯 DIAGNOSIS:")
    
    if speedup < 0.5:
        print(f"   🚨 MAJOR ISSUE: Kernel is 2× slower than baseline!")
        print(f"      → Likely cause: Kernel overhead >> sparsity benefits")
        print(f"      → Solution: Check block sizes, kernel fusion, or try longer sequences")
    elif speedup < 1.0:
        print(f"   ⚠️  MINOR ISSUE: Kernel is slower than baseline")
        print(f"      → Likely cause: Overhead not yet compensated by sparsity")
        print(f"      → Solution: Try longer sequences (2K+) or optimize kernel")
    else:
        print(f"   ✅ SUCCESS: Kernel is faster than baseline!")
        print(f"      → Sparsity benefits are working")
    
    # Key insights
    theoretical_speedup = 1 / (1 - splash_sparsity_est) if splash_sparsity_est > 0 else 1
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   Theoretical max speedup: {theoretical_speedup:.2f}×")
    print(f"   Actual speedup: {speedup:.2f}×")
    print(f"   Efficiency: {speedup/theoretical_speedup*100:.1f}% of theoretical max")
    
    if speedup < theoretical_speedup * 0.5:
        print(f"   🔧 RECOMMENDATION: Kernel has high overhead - needs optimization")
    elif speedup < theoretical_speedup * 0.8:
        print(f"   ⚡ RECOMMENDATION: Good sparsity, but room for optimization")
    else:
        print(f"   🎉 RECOMMENDATION: Excellent implementation!")

def test_different_sparsity_levels():
    """Test how performance changes with different levels of sparsity"""
    print(f"\n🧪 TESTING DIFFERENT SPARSITY LEVELS")
    print("=" * 50)
    
    configs = [
        (512, "Small"),
        (1024, "Medium"), 
        (2048, "Large"),
        (4096, "XLarge"),
    ]
    
    print(f"{'Config':<8} {'Seq Len':<8} {'Baseline(ms)':<12} {'Splash(ms)':<12} {'Speedup':<10}")
    print("-" * 55)
    
    for seq_len, name in configs:
        try:
            B, H, d = 1, 4, 64
            
            torch.manual_seed(42)
            Q = torch.randn(B, H, seq_len, d, device='cuda', dtype=torch.float32) * 0.1
            K = torch.randn(B, H, seq_len, d, device='cuda', dtype=torch.float32) * 0.1
            V = torch.randn(B, H, seq_len, d, device='cuda', dtype=torch.float32) * 0.1
            
            Q_idx = torch.arange(seq_len, device='cuda', dtype=torch.int32).unsqueeze(0).expand(B*H, seq_len).contiguous()
            K_idx = torch.arange(seq_len, device='cuda', dtype=torch.int32).unsqueeze(0).expand(B*H, seq_len).contiguous()
            
            sm_scale = 1.0 / (d ** 0.5)
            alpha = 1.5
            
            # Quick baseline timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, -float('inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            baseline_output = torch.matmul(attn_weights, V)
            torch.cuda.synchronize()
            baseline_time = (time.perf_counter() - start) * 1000
            
            # Quick splash timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            splash_output = fixed_original_splash.forward(Q, K, V, Q_idx, K_idx, sm_scale, alpha)
            torch.cuda.synchronize()
            splash_time = (time.perf_counter() - start) * 1000
            
            speedup = baseline_time / splash_time
            
            print(f"{name:<8} {seq_len:<8} {baseline_time:<12.2f} {splash_time:<12.2f} {speedup:<10.2f}")
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{name:<8} {seq_len:<8} FAILED: {str(e)[:30]}")

def main():
    """Run the sparsity analysis"""
    try:
        # Quick kernel test
        test_tensor = torch.randn(1, 1, 4, 4, device='cuda')
        _ = fixed_original_splash.forward(
            test_tensor, test_tensor, test_tensor,
            torch.arange(4, device='cuda', dtype=torch.int32).unsqueeze(0),
            torch.arange(4, device='cuda', dtype=torch.int32).unsqueeze(0),
            1.0, 1.5
        )
        
        # Run analysis
        analyze_splash_attention_sparsity()
        test_different_sparsity_levels()
        
    except NameError:
        print("❌ 'fixed_original_splash' kernel not found!")
        print("Please load your kernel first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 