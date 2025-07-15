import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Build and load the CUDA extension
try:
    import splash_attention
except ImportError:
    # If not installed, build on-the-fly
    splash_attention = load(
        name="splash_attention",
        sources=["source/splash.cu"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
        verbose=True
    )

def test_splash_attention():
    # Test parameters
    B, H, N, d = 2, 4, 128, 64
    alpha = 1.5
    sm_scale = 1.0 / (d ** 0.5)
    
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(B, H, N, d, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, N, d, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, N, d, device=device, dtype=torch.float32, requires_grad=True)
    
    # Create position indices (for causal masking)
    Q_idx = torch.arange(N, device=device, dtype=torch.int32).expand(B*H, N)
    K_idx = torch.arange(N, device=device, dtype=torch.int32).expand(B*H, N)
    
    print("Testing forward pass...")
    try:
        # Test forward pass
        out = splash_attention.forward(Q, K, V, Q_idx, K_idx, sm_scale, alpha)
        print(f"Forward pass successful! Output shape: {out.shape}")
        print(f"Output stats: mean={out.mean():.6f}, std={out.std():.6f}")
        
        # Test backward pass
        print("Testing backward pass...")
        dOut = torch.randn_like(out)
        dQ, dK, dV = splash_attention.backward(Q, K, V, Q_idx, K_idx, sm_scale, alpha, dOut)
        
        # Store references to prevent deallocation before printing
        Q.grad = dQ
        K.grad = dK  
        V.grad = dV
        
        print("Backward pass successful!")
        print(f"Q.grad shape: {Q.grad.shape}")
        print(f"K.grad shape: {K.grad.shape}")  
        print(f"V.grad shape: {V.grad.shape}")
        
        # Check gradients are not NaN/Inf
        assert not torch.isnan(dQ).any(), "Q gradients contain NaN"
        assert not torch.isnan(dK).any(), "K gradients contain NaN"
        assert not torch.isnan(dV).any(), "V gradients contain NaN"
        assert not torch.isinf(dQ).any(), "Q gradients contain Inf"
        assert not torch.isinf(dK).any(), "K gradients contain Inf"
        assert not torch.isinf(dV).any(), "V gradients contain Inf"
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_splash_attention() 