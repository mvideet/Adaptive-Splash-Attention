import math
import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

minimal_attn = load(
    name="minimal_attn", sources=[os.path.join(os.path.dirname(__file__), "../source/main.cpp"), 
                                  os.path.join(os.path.dirname(__file__), "../source/splash.cu")], 
                                  extra_cuda_cflags=["-O2"]
)

def manual_sparse_attn(q, k, v, Q_idx, K_idx, sm_scale):
    """
    We'll replicate the causal mask logic:
        att[b,h,i,j] = -inf if Q_idx[b,h,i] < K_idx[b,h,j],
        else att[b,h,i,j] = (q[i] · k[j]) * sm_scale
    Then do softmax along the last dimension and multiply by v.
    """
    B, H, N, D = q.shape
    # att: (B, H, N, N)
    att = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    # shape(Q_idx) = (B, H, N) => expand to (B, H, N, 1), (B, H, 1, N)
    mask = Q_idx.unsqueeze(-1) >= K_idx.unsqueeze(-2)  # boolean
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    out = torch.matmul(att, v)  # (B, H, N, D)
    return out


def benchmark():
    B = 16  # batch_size    H = 12  # num heads
    H = 12  # num heads
    N = 64  # seq_len
    D = 48  # head_dim (adjusted to be divisible by H)
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    Q_idx = torch.arange(N, device=q.device).view(1, 1, N)
    Q_idx = Q_idx.expand(B, H, N).contiguous().int()

    K_idx = torch.arange(N, device=q.device).view(1, 1, N)
    K_idx = K_idx.expand(B, H, N).contiguous().int()

    print("=== Profiling manual attention ===")
    with torch.autograd.profiler.profile(use_device="cuda") as prof:
        manual_out = manual_sparse_attn(q, k, v, Q_idx, K_idx, sm_scale)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("=== Profiling splash attention ===")
    with torch.autograd.profiler.profile(use_device="cuda") as prof:
        minimal_out = minimal_attn.forward(q, k, v, Q_idx, K_idx, sm_scale)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    close_manual = torch.allclose(manual_out, minimal_out, rtol=1e-3, atol=1e-2)
    print("Comparison check (manual vs. minimal):", close_manual)


if __name__ == "__main__":
    benchmark()
