Some improvements:
- its not true splash attention
- LLM inference optimization techniques that are production grade
    - spec decoding
    - paged / compress KV cache
    - hash-based attention
 - cleanup and write better benchmarking code
 - Use templating to vary the kernel for different values of d