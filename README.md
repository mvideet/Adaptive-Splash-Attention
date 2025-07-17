# SplashAttention

A CUDA implementation of Sparse Flash Attention with α-entmax normalization for efficient transformer attention computation.

## Overview

SplashAttention is a sparse attention mechanism that significantly reduces the computational complexity of attention in transformers by focusing on only the most relevant key-value pairs. This implementation combines:

- **Sparse Attention**: Uses top-K selection to maintain only the most important attention connections
- **α-entmax Normalization**: Generalizes softmax with tunable sparsity parameter α
- **Block-Sparse Computation**: Efficient tiled CUDA kernels for memory-optimal processing
- **Causal Masking**: Support for autoregressive models with proper causality constraints

## Key Features

- 🚀 **High Performance**: Custom CUDA kernels optimized for modern GPUs
- 📊 **Memory Efficient**: Block-sparse computation reduces memory footprint
- 🎛️ **Configurable Sparsity**: Tunable K parameter and α-entmax for controllable sparsity
- 🔄 **Full Backward Pass**: Complete gradient computation for training
- 📏 **Flexible Dimensions**: Supports various sequence lengths and head dimensions

## Installation

### Prerequisites

- CUDA-capable GPU (Compute Capability 7.0+)
- CUDA Toolkit 11.0+
- PyTorch 1.12+
- Python 3.8+

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd splash_attention
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate splash_attention
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

## Usage

### Basic Example

```python
import torch
from splash_attention import SplashAttention

# Initialize SplashAttention layer
attention = SplashAttention(
    head_dim=64,
    alpha=1.5,      # α-entmax parameter (1.0 = softmax, >1.0 = sparse)
    k_keep=8,       # Number of top-K elements to keep
    sm_scale=0.125  # Scaling factor (typically 1/√d)
)

# Input tensors
batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# Position indices for causal masking
Q_idx = torch.arange(seq_len, device='cuda').unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
K_idx = torch.arange(seq_len, device='cuda').unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)

# Forward pass
output = attention(Q, K, V, Q_idx, K_idx)
```

### Running Benchmarks

```bash
# Run performance benchmarks
python benchmark/bench.py

# Or explore the interactive notebook
jupyter notebook benchmark/attention_benchmarks.ipynb
```

### Testing

```bash
# Run basic functionality tests
python test_splash.py
```

## Project Structure

```
splash_attention/
├── source/                     # CUDA source code
│   ├── adasplashattention.cu   # Main implementation (in progress)
│   ├── leaning_splash.cu       # Full working implementation
│   └── splash.cu               # Basic version
├── benchmark/                  # Performance evaluation
│   ├── attention_benchmarks.ipynb
│   └── bench.py
├── test_splash.py              # Unit tests
├── setup.py                    # Package configuration
├── environment.yaml            # Conda environment
└── README.md                   # This file
```

## Algorithm Details

### Sparse Attention Mechanism

1. **Top-K Selection**: For each query, compute attention scores with all keys and keep only the top-K highest scores
2. **α-entmax Normalization**: Apply α-entmax instead of softmax for controllable sparsity:
   ```
   p_i = max(0, ((α-1)s_i - τ)^(1/(α-1)))
   ```
3. **Block-Sparse Computation**: Organize computation in tiles to maximize GPU memory bandwidth
4. **Causal Masking**: Ensure autoregressive property by masking future positions

### Performance Characteristics

- **Complexity**: O(n·k) instead of O(n²) for traditional attention
- **Memory**: Sparse storage reduces memory requirements significantly
- **Speed**: Optimized CUDA kernels provide substantial speedup for long sequences

## Configuration Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|---------------|
| `alpha` | α-entmax sparsity parameter | 1.0-2.0 (1.0 = softmax) |
| `k_keep` | Number of top-K elements | 8-32 |
| `sm_scale` | Attention scaling factor | 1/√(head_dim) |
| `BLOCK_M` | Query block size | 32-64 |
| `BLOCK_N` | Key block size | 64-128 |

## Limitations & Notes

- **Learning Purpose**: This implementation is for educational use and experimentation
- **GPU Only**: Requires CUDA-capable hardware
- **Sequence Length**: Optimized for sequences up to 4K tokens
- **Precision**: Currently supports FP32 only

## Contributing

This is a learning project, but contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## References

- [FlashAttention](https://arxiv.org/abs/2205.14135) - Efficient attention computation
- [α-entmax](https://arxiv.org/abs/1905.05702) - Sparse attention normalization
- [Sparse Transformers](https://arxiv.org/abs/1904.10509) - Sparse attention patterns

## License

See LICENSE file for details.

## Acknowledgments

Built for learning purposes to understand modern attention mechanisms and CUDA optimization techniques.
