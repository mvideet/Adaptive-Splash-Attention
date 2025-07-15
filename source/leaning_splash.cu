// splash_attention_full.cu — CUDA/C++ PyTorch extension with full AdaSplash-style sparse α-entmax
// dynamic block masking, causal support, shared K/V tiling, normalized forward, and full backward

// === INCLUDES ===
#include <torch/extension.h>  // PyTorch C++ extension API for tensor operations
#include <cuda.h>             // Basic CUDA runtime definitions
#include <cuda_runtime.h>     // CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)
#include <vector>             // Standard C++ vector container
#include <float.h>            // Floating point constants like FLT_MAX

// === COMPILE-TIME CONSTANTS ===
// These #ifndef blocks allow overriding values during compilation with -DBLOCK_M=64
#ifndef BLOCK_M
#define BLOCK_M 32            // Number of queries processed per thread block (power of 2)
#endif
#ifndef BLOCK_N  
#define BLOCK_N 64            // Number of keys processed per tile (power of 2)
#endif
#ifndef D_MAX
#define D_MAX 128             // Maximum head dimension supported
#endif
#ifndef K_KEEP
#define K_KEEP 8              // Number of top-K attention scores to keep per query
#endif

#define EPS 1e-6f             // Small epsilon to prevent division by zero
#define MAX_HALLEY_ITERS 4    // Maximum iterations for Halley's method in entmax

// === CUDA ERROR CHECKING MACRO ===
// This macro wraps CUDA calls and automatically checks for errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error at ", __FILE__, ":", __LINE__, " - ", cudaGetErrorString(err)); \
} while(0)
// PyTorch's error checking with file/line info

// === COMPILE-TIME SAFETY ASSERTIONS ===
// These static_assert statements check constraints at compile time
static_assert(BLOCK_N > 0 && BLOCK_M > 0, "Block dimensions must be positive");
static_assert(BLOCK_N * D_MAX <= 8192, "Shared memory K tile too large (max ~8KB per tile)");
static_assert(2 * BLOCK_N * D_MAX <= 32768, "Shared memory K+V tiles too large (max ~32KB total)");
static_assert(BLOCK_M <= 1024, "BLOCK_M exceeds typical max threads per block");
static_assert(K_KEEP <= BLOCK_N, "K_KEEP should not exceed BLOCK_N");
static_assert(D_MAX >= 64, "D_MAX should be at least 64 for practical use");

// === DEVICE FUNCTION: IN-REGISTER TOP-K INSERTION ===
// __device__ means this function runs on GPU and can only be called from GPU code
// inline suggests the compiler should inline this function for performance
__device__ inline void insert_topk(float val,                    // Score value to potentially insert
                                   int idx,                      // Index associated with the score
                                   float (&vals)[K_KEEP],        // Reference to array of top-K values
                                   int (&inds)[K_KEEP]){         // Reference to array of top-K indices
    int j = K_KEEP - 1;          // Start from the smallest value in our top-K list
    
    // Shift elements right while val is larger than current element
    while(j >= 0 && val > vals[j]){                             // Compare with current element
        if(j < K_KEEP - 1){                                     // If not at the end of array
            vals[j+1] = vals[j];                                // Shift value one position right
            inds[j+1] = inds[j];                                // Shift index one position right
        }
        --j;                                                    // Move to next position left
    }
    ++j;                                                        // Adjust j to insertion position
    if(j < K_KEEP){                                            // If there's space in our top-K
        vals[j] = val;                                         // Insert the new value
        inds[j] = idx;                                         // Insert the corresponding index
    }
}

// === DEVICE FUNCTION: ENTMAX THRESHOLD COMPUTATION ===
// Computes α-entmax weights using Halley's method + bisection search
__device__ void entmax_threshold(const float* s,               // Input scores (sorted descending)
                                int k,                          // Number of elements in s
                                float alpha,                    // α parameter for entmax (α > 1)
                                float* p,                       // Output: entmax probabilities
                                float &tau,                     // Output: threshold parameter
                                bool full){                     // Whether to compute full derivatives
    
    const float inv_am1 = 1.f/(alpha-1.f);                    // Precompute 1/(α-1) for efficiency
    
    // Initialize bisection bounds for threshold τ
    float lo = (alpha-1.f)*s[k-1] - 1.f;                      // Lower bound (smallest score)
    float hi = (alpha-1.f)*s[0];                              // Upper bound (largest score)
    
    // Lambda function to evaluate entmax constraint equation f(τ) = Σp_i - 1
    auto eval = [&](float t,                                   // Current threshold τ
                   float &f,                                   // Output: constraint value f(τ)
                   float&fp,                                   // Output: first derivative f'(τ)  
                   float&fpp){                                 // Output: second derivative f''(τ)
        f = -1;                                                // Initialize f(τ) = -1 (target is 0)
        fp = fpp = 0;                                          // Initialize derivatives to zero
        
        for(int j=0;j<k;++j){                                  // Loop over all scores
            float u = (alpha-1.f)*s[j] - t;                   // Compute u_j = (α-1)s_j - τ
            if(u <= 0) break;                                 // If u_j ≤ 0, p_j = 0, skip rest
            
            float up = powf(u, inv_am1);                       // u^(1/(α-1)) = probability p_j
            f += up;                                           // Add to constraint sum
            
            if(full){                                          // If computing derivatives
                fp  += -inv_am1 * powf(u, (2.f-alpha)*inv_am1);     // ∂f/∂τ 
                fpp += inv_am1*(inv_am1+1.f-alpha) * powf(u, (3.f-2*alpha)*inv_am1); // ∂²f/∂τ²
            }
        }
    };
    
    tau = 0.5f*(lo + hi);                                      // Initialize τ as midpoint
    
    // Halley's method iterations (Newton's method with second derivative)
    for(int it=0; it<MAX_HALLEY_ITERS; ++it){
        float f, fp, fpp;                                      // Function and derivatives
        eval(tau, f, fp, fpp);                                 // Evaluate at current τ - the & means these are passed by reference and will be modified by eval()
        
        if(fabsf(f) < 1e-3f) break;                           // Converged if |f(τ)| < tolerance
        
        float tn = tau;                                        // Save current τ
        if(full){                                              // Use Halley's method if derivatives available
            float denom = 2.f*fp*fp - f*fpp;                  // Denominator for Halley update
            tn = tau - 2.f*f*fp / fmaxf(denom, EPS);          // Halley update step
        }
        
        if(!(tn >= lo && tn <= hi)) tn = 0.5f*(lo + hi);     // Fallback to bisection if out of bounds
        
        eval(tn, f, fp, fpp);                                  // Evaluate at new point
        if(f > 0) lo = tn; else hi = tn;                      // Update bisection bounds
        tau = tn;                                              // Update τ
    }
    
    // Compute final entmax probabilities
    float norm = 0;                                            // Normalization factor
    for(int j=0;j<k;++j){                                     // For each score
        float u = (alpha-1.f)*s[j] - tau;                     // Compute u_j
        float pj = (u > 0) ? powf(u, inv_am1) : 0;            // p_j = max(0, u_j^(1/(α-1)))
        p[j] = pj;                                             // Store probability
        norm += pj;                                            // Accumulate for normalization
    }
    norm = fmaxf(norm, EPS);                                  // Ensure norm > 0 to avoid division by zero
    for(int j=0;j<k;++j) p[j] /= norm;                        // Normalize probabilities to sum to 1
}

// === CUDA KERNEL: BUILD SPARSE ATTENTION MASK ===
// __global__ means this is a CUDA kernel that can be called from host (CPU) code
__global__ void build_mask_kernel(
    const float* Q,                                            // Queries [B*H, NQ, d] - input tensor
    const float* K,                                            // Keys [B*H, NK, d] - input tensor
    const int* Q_idx,                                          // Query position indices [B*H, NQ]
    const int* K_idx,                                          // Key position indices [B*H, NK]
    int B,                                                     // Batch size
    int H,                                                     // Number of attention heads
    int NQ,                                                    // Number of queries per head
    int NK,                                                    // Number of keys per head
    int d,                                                     // Head dimension
    float alpha,                                               // α parameter for entmax
    float sm_scale,                                            // Scaling factor (usually 1/√d)
    int8_t* M,                                                 // Output: block mask matrix [B*H, nQB, nKB]
    float* taus,                                               // Output: threshold values [B*H, NQ]
    int nQB,                                                   // Number of query blocks
    int nKB                                                    // Number of key blocks
){
    // === THREAD AND BLOCK INDEXING ===
    // Each thread processes one query; blockIdx and threadIdx are built-in CUDA variables
    int q = blockIdx.x*BLOCK_M + threadIdx.x;                 // Global query index: block_id * block_size + thread_in_block
    int bh = blockIdx.y;                                      // Batch*head index from Y dimension of grid
    if(q >= NQ) return;                                       // Exit if thread has no work (beyond NQ queries)
    
    // Get the sequence position of this query from Q_idx tensor
    // bh*NQ + q computes the linear index into Q_idx:
    //   bh = batch*head index, NQ = queries per head, q = query index within head
    // This position is used for causal masking to ensure each position only attends to previous positions
    int seq_q = Q_idx[bh*NQ + q];                             // Sequence position of this query (for causal masking)

    // Compute linear index to access query tensor Q
    // Same formula as above: bh*NQ + q flattens the [batch*head, query] indices into a single linear index
    // This will be multiplied by d later to get the exact offset into Q tensor
    int idx = bh*NQ + q;                                      // Linear index into query tensors
    
    // === LOAD QUERY INTO REGISTERS ===
    // Registers are the fastest memory on GPU (private per thread)
    float q_reg[D_MAX];                                        // Register array to store query vector
    #pragma unroll for(int t=0; t<d; ++t)                     // #pragma unroll tells compiler to unroll loop
        q_reg[t] = Q[idx*d + t];                              // Load query elements: q_reg[0], q_reg[1], ...
    
    // === INITIALIZE TOP-K BUFFERS ===
    float s_top[K_KEEP];                                       // Top-K attention scores (registers)
    int ind[K_KEEP];                                           // Indices of top-K elements (registers)
    #pragma unroll for(int i=0; i<K_KEEP; ++i){              // Initialize all elements
        s_top[i] = -FLT_MAX;                                  // Start with very negative scores
        ind[i] = -1;                                          // Invalid indices initially
    }
    
    // === SHARED MEMORY ALLOCATION ===
    // extern __shared__ declares dynamic shared memory (allocated at kernel launch)
    // Shared memory is fast memory shared among all threads in a block
    extern __shared__ float shmem[];                          // Dynamic shared memory array
    float* Ktile = shmem;                                     // Use shared memory for key tile
    
    // === TILED PROCESSING OF KEYS ===
    // Process keys in tiles of size BLOCK_N to fit in shared memory
    for(int start=0; start<NK; start+=BLOCK_N){              // Iterate over key tiles
        
        // === COOPERATIVE KEY LOADING ===
        // All threads in block cooperatively load one tile of keys into shared memory
        int tid = threadIdx.x;                                // Thread ID within block (0 to BLOCK_M-1)
        
        // Each thread loads multiple elements if BLOCK_N*d > BLOCK_M
        for(int x = tid; x < BLOCK_N*d; x += BLOCK_M){       // Stride by BLOCK_M to distribute work
            int col = x / d;                                  // Which key in the tile (0 to BLOCK_N-1)
            int dim = x % d;                                  // Which dimension (0 to d-1)
            int kn = start + col;                             // Global key index
            
            // Check bounds and causal constraint
            bool ok = (kn < NK &&                            // Key exists
                      dim < d &&                             // Dimension is valid (redundant but safe)
                      K_idx[bh*NK + kn] <= seq_q);           // Causal: key position ≤ query position
            
            // Load key element or zero if out of bounds/causal
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;   // Coalesced memory access
        }
        
        // === SYNCHRONIZATION ===
        // Wait for all threads to finish loading before proceeding
        __syncthreads();                                      // Block-level barrier synchronization
        
        // === COMPUTE ATTENTION SCORES ===
        // Each thread computes attention scores for all keys in current tile
        for(int j=0; j<BLOCK_N; ++j){                        // Loop over keys in tile
            int kn = start + j;                               // Global key index
            if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue; // Skip if out of bounds or violates causality
            
            // Compute dot product: score = q·k
            float acc = 0;                                    // Accumulator for dot product
            #pragma unroll for(int t=0; t<d; ++t)            // Unrolled loop for efficiency
                acc += q_reg[t] * Ktile[j*d + t];            // q[t] * k[t], summed over dimensions
            
            acc *= sm_scale;                                  // Apply scaling factor
            
            // Update top-K list with this score
            insert_topk(acc, kn, s_top, ind);                // Insert if score is among top-K
        }
        
        // Synchronize before loading next tile
        __syncthreads();                                      // Ensure all threads finish before next iteration
    }
    
    // === COMPUTE ENTMAX ON TOP-K SCORES ===
    float pbuf[K_KEEP];                                       // Buffer for entmax probabilities
    float tau;                                                // Threshold parameter
    entmax_threshold(s_top, K_KEEP, alpha, pbuf, tau, true); // Compute α-entmax weights
    taus[idx] = tau;                                          // Store threshold for backward pass
    
    // === WRITE BLOCK MASK ===
    // Mark which key blocks contain selected top-K elements
    int iQB = q / BLOCK_M;                                    // Which query block this thread belongs to
    int base = bh*nQB*nKB + iQB*nKB;                         // Base address in mask tensor
    
    for(int i=0; i<K_KEEP; ++i){                            // For each top-K element
        int kn = ind[i];                                      // Global key index
        if(kn < 0) continue;                                  // Skip invalid entries
        
        int jKB = kn / BLOCK_N;                              // Which key block contains this key
        M[base + jKB] = 1;                                   // Mark this block as needed
    }
}

// === CUDA KERNEL: BUILD LOOKUP TABLES ===
// Creates compressed sparse row (CSR) format lookup tables from block mask
__global__ void build_lookup_kernel(
    const int8_t* M,                                          // Input: block mask [B*H, nQB, nKB]
    int B, int H, int nQB, int nKB,                          // Tensor dimensions
    int* Qi_ptr,                                              // Output: query block pointers [B*H, nQB+1]
    int* Qi_idx,                                              // Output: active key blocks per query [B*H, nQB*nKB]
    int* Kj_ptr,                                              // Output: key block pointers [B*H, nKB+1]  
    int* Kj_idx                                               // Output: active query blocks per key [B*H, nQB*nKB]
){
    // One thread per batch*head (single threaded within each)
    int bh = blockIdx.x;                                      // Batch*head index from block ID
    if(bh >= B*H) return;                                    // Bounds check
    
    // === COMPUTE BASE ADDRESSES ===
    int baseM = bh * nQB * nKB;                              // Base address in mask tensor
    int bQi   = bh * (nQB + 1);                              // Base address in Qi_ptr
    int bQiI  = bh * (nQB * nKB);                            // Base address in Qi_idx
    int bKj   = bh * (nKB + 1);                              // Base address in Kj_ptr
    int bKjI  = bh * (nQB * nKB);                            // Base address in Kj_idx
    
    // === BUILD QUERY-TO-KEY LOOKUP (CSR FORMAT) ===
    Qi_ptr[bQi] = 0;                                         // First pointer is always 0
    
    // Count active key blocks for each query block
    for(int i=0;i<nQB;++i){                                  // For each query block
        int c=0;                                              // Counter for active key blocks
        for(int j=0;j<nKB;++j)                               // For each key block
            c += M[baseM + i*nKB + j];                       // Count if block is active
        Qi_ptr[bQi + i + 1] = Qi_ptr[bQi + i] + c;          // Cumulative count (CSR pointer)
    }
    
    // Fill indices of active key blocks for each query block
    for(int i=0;i<nQB;++i){                                  // For each query block
        int w = Qi_ptr[bQi + i];                             // Starting write position
        for(int j=0;j<nKB;++j)                               // For each key block
            if(M[baseM + i*nKB + j])                         // If block is active
                Qi_idx[bQiI + (w++)] = j;                    // Store key block index
    }
    
    // === BUILD KEY-TO-QUERY LOOKUP (TRANSPOSE) ===
    Kj_ptr[bKj] = 0;                                         // First pointer is always 0
    
    // Count active query blocks for each key block
    for(int j=0;j<nKB;++j){                                  // For each key block
        int c=0;                                              // Counter for active query blocks
        for(int i=0;i<nQB;++i)                               // For each query block
            c += M[baseM + i*nKB + j];                       // Count if block is active
        Kj_ptr[bKj + j + 1] = Kj_ptr[bKj + j] + c;          // Cumulative count
    }
    
    // Fill indices of active query blocks for each key block
    for(int j=0;j<nKB;++j){                                  // For each key block
        int w = Kj_ptr[bKj + j];                             // Starting write position
        for(int i=0;i<nQB;++i)                               // For each query block
            if(M[baseM + i*nKB + j])                         // If block is active
                Kj_idx[bKjI + (w++)] = i;                    // Store query block index
    }
}

// === CUDA KERNEL: SPARSE ATTENTION FORWARD PASS ===
__global__ void splash_forward_sparse(
    const float* Q, const float* K, const float* V,          // Input tensors
    const int* Q_idx, const int* K_idx,                      // Position indices
    const float* taus,                                        // Entmax thresholds from mask kernel
    const int* Qi_ptr, const int* Qi_idx,                    // Lookup tables from build_lookup_kernel
    float* Out,                                               // Output tensor [B*H, NQ, d]
    int B, int H, int NQ, int NK, int d,                     // Tensor dimensions
    float alpha, float sm_scale,                             // Attention parameters
    int nQB, int nKB                                         // Number of blocks
){
    // === THREAD INDEXING ===
    int q = blockIdx.x*BLOCK_M + threadIdx.x;                // Global query index
    int bh=blockIdx.y;                                        // Batch*head index
    if(q >= NQ) return;                                       // Bounds check
    
    // === LOAD QUERY DATA ===
    int seq_q = Q_idx[bh*NQ + q];                            // Query sequence position
    float tau = taus[bh*NQ + q];                             // Entmax threshold for this query
    const float* Qptr = Q + ((bh*NQ + q)*d);                 // Pointer to query vector
    
    // Load query into registers
    float qreg[D_MAX];                                        // Query vector in registers
    #pragma unroll for(int t=0; t<d; ++t)                    // Load all dimensions
        qreg[t] = Qptr[t];
    
    // === SHARED MEMORY SETUP ===
    extern __shared__ float sh[];                             // Dynamic shared memory
    float* Ktile = sh;                                        // First half for keys
    float* Vtile = sh + BLOCK_N * d;                         // Second half for values
    
    // === INITIALIZE ACCUMULATORS ===
    float accum[D_MAX];                                       // Output accumulator
    #pragma unroll for(int t=0; t<d; ++t)                    // Initialize to zero
        accum[t] = 0.f;
    float norm = 0.f;                                         // Normalization factor
    
    // === LOOKUP TABLE INDEXING ===
    int iQB = q / BLOCK_M;                                    // Query block index
    int off = bh*(nQB+1) + iQB;                              // Offset into Qi_ptr
    int offI= bh*(nQB*nKB);                                  // Offset into Qi_idx
    
    // === PROCESS ACTIVE KEY BLOCKS ===
    // Only process key blocks that were marked active in the mask
    for(int ptr = Qi_ptr[off]; ptr < Qi_ptr[off+1]; ++ptr){ // Iterate over active blocks
        int jKB = Qi_idx[offI + ptr];                        // Get active key block index
        int start = jKB * BLOCK_N;                           // Starting key index in this block
        
        // === COOPERATIVE TILE LOADING ===
        int tid = threadIdx.x;                                // Thread ID in block
        for(int x=tid; x < BLOCK_N*d; x += BLOCK_M){         // Distribute loading across threads
            int col = x / d, dim = x % d;                     // Key and dimension indices
            int kn = start + col;                             // Global key index
            
            // Check bounds and causal constraint
            bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q);
            
            // Load key and value elements
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;   // Load key or zero
            Vtile[x] = ok ? V[(bh*NK + kn)*d + dim] : 0.f;   // Load value or zero
        }
        __syncthreads();                                      // Wait for all threads to finish loading
        
        // === COMPUTE ATTENTION FOR THIS TILE ===
        for(int k=0; k<BLOCK_N; ++k){                        // For each key in tile
            int kn = start + k;                               // Global key index
            if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue; // Skip if invalid or violates causality
            
            // Compute attention score
            float s=0.f;                                      // Dot product accumulator
            #pragma unroll for(int t=0; t<d; ++t)            // Compute q·k
                s+=qreg[t]*Ktile[k*d + t];
            s *= sm_scale;                                    // Apply scaling
            
            // Compute entmax probability
            float u = (alpha-1.f)*s - tau;                   // u = (α-1)s - τ
            float p = (u > 0.f) ? powf(u, 1.f/(alpha-1.f)) : 0.f; // p = max(0, u^(1/(α-1)))
            
            norm += p;                                        // Accumulate normalization
            
            // Weighted accumulation of values
            #pragma unroll for(int t=0; t<d; ++t)            // For each dimension
                accum[t] += p * Vtile[k*d + t];              // output += p * value
        }
        __syncthreads();                                      // Synchronize before next tile
    }
    
    // === NORMALIZE AND WRITE OUTPUT ===
    float invN = 1.f / (norm + EPS);                         // Inverse normalization (avoid division by zero)
    float* OutPtr = Out + ((bh*NQ + q)*d);                   // Output pointer for this query
    #pragma unroll for(int t=0; t<d; ++t)                    // Write normalized output
        OutPtr[t] = accum[t] * invN;
}

// === CUDA KERNEL: SPARSE ATTENTION BACKWARD PASS ===
__global__ void splash_backward_sparse(
    const float* Q,const float* K,const float* V,            // Forward pass inputs
    const int* Q_idx,const int* K_idx,const float* taus,     // Indices and thresholds
    const int* Qi_ptr,const int* Qi_idx,const float* dOut,   // Lookup tables and output gradients
    float* dQ,float* dK,float* dV,                           // Output: input gradients
    int B,int H,int NQ,int NK,int d,                         // Tensor dimensions
    float alpha,float sm_scale,int nQB,int nKB               // Parameters
){
    // === THREAD INDEXING ===
    int q = blockIdx.x*BLOCK_M + threadIdx.x;                // Global query index
    int bh = blockIdx.y;                                      // Batch*head index
    if(q >= NQ) return;                                       // Bounds check
    
    // === LOAD FORWARD PASS DATA ===
    int seq_q = Q_idx[bh*NQ + q];                            // Query position
    const float* Qptr = Q + ((bh*NQ + q)*d);                 // Query pointer
    
    // Load query and initialize query gradient
    float qreg[D_MAX], dqreg[D_MAX];                         // Query and its gradient
    #pragma unroll for(int t=0; t<d; ++t){
        qreg[t]=Qptr[t];                                     // Load query
        dqreg[t]=0.f;                                        // Initialize gradient to zero
    }
    
    // === SHARED MEMORY SETUP ===
    extern __shared__ float sh[];                             // Dynamic shared memory
    float* Ktile = sh;                                        // Keys tile
    float* Vtile = sh + BLOCK_N*d;                           // Values tile
    
    // === INITIALIZE ACCUMULATORS ===
    float accum[D_MAX];                                       // Forward pass output accumulator
    float norm=0.f;                                           // Forward pass normalization
    #pragma unroll for(int t=0; t<d; ++t)                    // Initialize output accumulator
        accum[t]=0.f;
    
    // === LOOKUP TABLE INDEXING ===
    int iQB=q/BLOCK_M;                                        // Query block index
    int off = bh*(nQB+1) + iQB;                              // Offset into Qi_ptr
    int offI= bh*(nQB*nKB);                                  // Offset into Qi_idx
    
    // === FIRST PASS: RECOMPUTE FORWARD VALUES ===
    // Need to recompute forward pass values for gradient computation
    for(int ptr=Qi_ptr[off]; ptr<Qi_ptr[off+1]; ++ptr){     // For each active key block
        int jKB=Qi_idx[offI+ptr];                            // Key block index
        int start=jKB*BLOCK_N;                               // Starting key index
        
        // Load key/value tile cooperatively
        int tid=threadIdx.x;
        for(int x=tid; x<BLOCK_N*d; x+=BLOCK_M){             // Distribute work across threads
            int col=x/d, dim=x%d, kn=start+col;              // Key and dimension indices
            bool ok=(kn<NK && K_idx[bh*NK+kn]<=seq_q);       // Bounds and causal check
            Ktile[x]= ok? K[(bh*NK+kn)*d+dim] : 0.f;         // Load key
            Vtile[x]= ok? V[(bh*NK+kn)*d+dim] : 0.f;         // Load value
        }
        __syncthreads();                                      // Wait for loading
        
        // Recompute forward pass for this tile
        for(int k=0; k<BLOCK_N; ++k){                        // For each key in tile
            int kn=start+k;
            if(kn>=NK || K_idx[bh*NK+kn]>seq_q) continue;    // Skip invalid keys
            
            // Recompute attention score and probability
            float s=0;
            #pragma unroll for(int t=0; t<d; ++t)            // Dot product
                s+=qreg[t]*Ktile[k*d + t];
            s *= sm_scale;
            float u=(alpha-1.f)*s - taus[bh*NQ + q];         // Entmax intermediate
            float p=(u>0.f? powf(u,1.f/(alpha-1.f)) : 0.f);  // Entmax probability
            
            norm+=p;                                          // Accumulate normalization
            #pragma unroll for(int t=0; t<d; ++t)            // Accumulate output
                accum[t]+=p * Vtile[k*d + t];
        }
        __syncthreads();                                      // Synchronize before next tile
    }
    
    // === COMPUTE NORMALIZATION CONSTANTS ===
    float invN=1.f/(norm+EPS);                               // 1/norm for forward pass
    float invN2=-invN*invN;                                  // -1/norm² for gradient of normalization
    const float* dOp = dOut + ((bh*NQ + q)*d);              // Pointer to output gradient
    
    // === SECOND PASS: COMPUTE GRADIENTS ===
    for(int ptr=Qi_ptr[off]; ptr<Qi_ptr[off+1]; ++ptr){     // For each active key block
        int jKB=Qi_idx[offI+ptr], start=jKB*BLOCK_N;         // Key block and starting index
        
        // === RELOAD KEY/VALUE TILES ===
        // Need to reload because shared memory was overwritten
        int tid=threadIdx.x;
        for(int x=tid; x<BLOCK_N*d; x+=BLOCK_M){             // Cooperative loading
            int col=x/d, dim=x%d, kn=start+col;
            bool ok=(kn<NK && K_idx[bh*NK+kn]<=seq_q);
            Ktile[x]= ok? K[(bh*NK+kn)*d+dim] : 0.f;
            Vtile[x]= ok? V[(bh*NK+kn)*d+dim] : 0.f;
        }
        __syncthreads();                                      // Wait for loading
        
        // === COMPUTE GRADIENTS FOR THIS TILE ===
        for(int k=0; k<BLOCK_N; ++k){                        // For each key in tile
            int kn=start+k;
            if(kn>=NK || K_idx[bh*NK+kn]>seq_q) continue;    // Skip invalid keys
            
            // === RECOMPUTE FORWARD VALUES ===
            float s=0;                                        // Attention score
            #pragma unroll for(int t=0; t<d; ++t)            // Dot product
                s+=qreg[t]*Ktile[k*d + t];
            s *= sm_scale;
            float u=(alpha-1.f)*s - taus[bh*NQ + q];         // Entmax intermediate
            float p=(u>0.f? powf(u,1.f/(alpha-1.f)) : 0.f);  // Entmax probability
            
            // === GRADIENT W.R.T. VALUES (dV) ===
            float pb = p * invN;                              // Normalized probability
            for(int t=0; t<d; ++t)                           // For each dimension
                atomicAdd(&dV[(bh*NK + kn)*d + t], pb * dOp[t]); // dV += p * dOut (atomic for thread safety)
            
            // === GRADIENT W.R.T. PROBABILITIES (dp) ===
            float dp=0;                                       // Gradient w.r.t. probability p
            for(int t=0; t<d; ++t)                           // Direct term: ∂L/∂output * ∂output/∂p
                dp += Vtile[k*d + t] * dOp[t] * invN;
                
            // Normalization term: ∂L/∂output * ∂output/∂norm * ∂norm/∂p
            float accum_dot_dO = 0;
            for(int t=0; t<d; ++t)                           // Compute accum·dOut
                accum_dot_dO += accum[t] * dOp[t];
            dp += accum_dot_dO * invN2;                      // Add normalization gradient
            
            // === CHAIN RULE: dp/du * du/ds ===
            // Chain through entmax: p = u^(1/(α-1)) where u = (α-1)s - τ
            float grad_u = (u>0.f? (1.f/(alpha-1.f))*powf(u,(2.f-alpha)/(alpha-1.f))*dp : 0.f); // dp/du
            float grad_s = grad_u * (alpha-1.f);             // du/ds = (α-1)
            
            // === GRADIENTS W.R.T. QUERIES AND KEYS ===
            for(int t=0; t<d; ++t){                          // For each dimension
                // ∂s/∂q = k, ∂s/∂k = q, but need to include sm_scale
                dqreg[t] += grad_s * Ktile[k*d + t] * sm_scale;    // dQ gradient (accumulated)
                atomicAdd(&dK[(bh*NK + kn)*d + t], grad_s * qreg[t] * sm_scale); // dK gradient (atomic)
            }
        }
        __syncthreads();                                      // Synchronize before next tile
    }
    
    // === WRITE QUERY GRADIENT ===
    float* dQp = dQ + ((bh*NQ + q)*d);                       // Pointer to query gradient
    #pragma unroll for(int t=0; t<d; ++t)                    // Write accumulated gradient
        dQp[t] = dqreg[t];
}

// === HOST WRAPPER FUNCTION: FORWARD PASS ===
// This function runs on CPU and launches GPU kernels
torch::Tensor forward_splash(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,      // Input tensors from PyTorch
    torch::Tensor Q_idx, torch::Tensor K_idx,               // Position indices
    float sm_scale, float alpha                              // Attention parameters
) {
    // === INPUT VALIDATION ===
    TORCH_CHECK(Q.size(3) <= D_MAX, "Head dimension too large; increase D_MAX");
    
    // === QUERY GPU PROPERTIES ===
    int device;
    cudaGetDevice(&device);                                   // Get current GPU device
    int maxShm, maxThr;                                      // GPU limits
    cudaDeviceGetAttribute(&maxShm, cudaDevAttrMaxSharedMemoryPerBlock, device); // Max shared memory per block
    cudaDeviceGetAttribute(&maxThr, cudaDevAttrMaxThreadsPerBlock, device);      // Max threads per block
    
    // === RUNTIME VALIDATION ===
    TORCH_CHECK(BLOCK_M <= maxThr, "BLOCK_M (" #BLOCK_M ") exceeds device max threads per block");
    auto d = Q.size(3);                                      // Head dimension
    TORCH_CHECK(BLOCK_N * d * sizeof(float) <= maxShm,      // Check shared memory for K tile
        "Shared memory for K tile exceeds limit");
    TORCH_CHECK(2 * BLOCK_N * d * sizeof(float) <= maxShm,  // Check shared memory for K+V tiles
        "Shared memory for K+V tiles exceeds limit");
    
    // === EXTRACT TENSOR DIMENSIONS ===
    auto B = Q.size(0), H = Q.size(1), NQ = Q.size(2), NK = K.size(2); // Batch, heads, sequence lengths
    int nQB = (NQ + BLOCK_M - 1) / BLOCK_M;                 // Number of query blocks (ceiling division)
    int nKB = (NK + BLOCK_N - 1) / BLOCK_N;                 // Number of key blocks
    
    // === CREATE OUTPUT TENSORS ===
    auto opts8 = torch::TensorOptions().dtype(torch::kInt8).device(Q.device());    // Options for int8 tensors
    auto optsF = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());       // Options for float tensors
    
    // Intermediate tensors for sparse computation
    auto M = torch::zeros({B*H, nQB, nKB}, opts8);          // Block mask matrix
    auto taus = torch::zeros({B*H, NQ}, optsF);             // Entmax thresholds
    auto Qi_ptr = torch::zeros({B*H, nQB+1}, torch::dtype(torch::kInt32).device(Q.device())); // CSR pointers
    auto Qi_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device())); // CSR indices
    auto Kj_ptr = torch::zeros({B*H, nKB+1}, torch::dtype(torch::kInt32).device(Q.device())); // Transpose CSR
    auto Kj_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Out = torch::zeros_like(Q);                         // Output tensor
    
    // === EXTRACT RAW POINTERS ===
    // PyTorch tensors → raw CUDA pointers for kernel calls
    int8_t* M_p = M.data_ptr<int8_t>();                     // Block mask
    float* taus_p = taus.data_ptr<float>();                 // Thresholds
    int* Qidx_p = Q_idx.data_ptr<int>();                    // Query indices
    int* Kidx_p = K_idx.data_ptr<int>();                    // Key indices
    int32_t* Qi_pp = Qi_ptr.data_ptr<int32_t>();           // CSR pointers
    int32_t* Qi_ip = Qi_idx.data_ptr<int32_t>();           // CSR indices
    int32_t* Kj_pp = Kj_ptr.data_ptr<int32_t>();           // Transpose CSR
    int32_t* Kj_ip = Kj_idx.data_ptr<int32_t>();
    float* Qp = Q.data_ptr<float>();                         // Input tensors
    float* Kp = K.data_ptr<float>();
    float* Vp = V.data_ptr<float>();
    float* Outp = Out.data_ptr<float>();                     // Output tensor
    
    // === KERNEL LAUNCH CONFIGURATION ===
    dim3 grid1(nQB, B*H);                                   // Grid: nQB blocks in X, B*H blocks in Y
    dim3 block1(BLOCK_M);                                   // Block: BLOCK_M threads per block
    size_t shm1 = BLOCK_N * d * sizeof(float);             // Shared memory for K tile
    
    // === LAUNCH MASK BUILDING KERNEL ===
    build_mask_kernel<<<grid1, block1, shm1>>>(             // Launch kernel on GPU
        Qp, Kp, Qidx_p, Kidx_p,                            // Input pointers
        B, H, NQ, NK, d,                                    // Dimensions
        alpha, sm_scale,                                     // Parameters
        M_p, taus_p,                                        // Output pointers
        nQB, nKB                                            // Block dimensions
    );
    CUDA_CHECK(cudaDeviceSynchronize());                     // Wait for kernel completion + check errors
    
    // === LAUNCH LOOKUP TABLE KERNEL ===
    build_lookup_kernel<<<B*H, 32>>>(                       // One block per batch*head, 32 threads per block
        M_p, B, H, nQB, nKB,                               // Input: mask and dimensions
        Qi_pp, Qi_ip,                                       // Output: query→key lookup
        Kj_pp, Kj_ip                                        // Output: key→query lookup
    );
    CUDA_CHECK(cudaDeviceSynchronize());                     // Wait and check
    
    // === LAUNCH FORWARD SPARSE KERNEL ===
    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);         // Shared memory for K+V tiles
    splash_forward_sparse<<<grid1, block1, shm2>>>(         // Launch forward kernel
        Qp, Kp, Vp,                                         // Input tensors
        Qidx_p, Kidx_p, taus_p,                            // Indices and thresholds
        Qi_pp, Qi_ip,                                       // Lookup tables
        Outp,                                               // Output tensor
        B, H, NQ, NK, d,                                    // Dimensions
        alpha, sm_scale,                                     // Parameters
        nQB, nKB                                            // Block counts
    );
    CUDA_CHECK(cudaDeviceSynchronize());                     // Wait and check
    
    return Out;                                              // Return PyTorch tensor
}

// === HOST WRAPPER FUNCTION: BACKWARD PASS ===
std::vector<torch::Tensor> backward_splash(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,      // Forward inputs
    torch::Tensor Q_idx, torch::Tensor K_idx,               // Position indices
    float sm_scale, float alpha,                             // Parameters
    torch::Tensor dOut                                       // Output gradients from autograd
) {
    // === VALIDATION AND SETUP ===
    // Same validation and GPU setup as forward pass
    TORCH_CHECK(Q.size(3) <= D_MAX, "Head dimension too large; increase D_MAX");
    int device;
    cudaGetDevice(&device);
    int maxShm, maxThr;
    cudaDeviceGetAttribute(&maxShm, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&maxThr, cudaDevAttrMaxThreadsPerBlock, device);
    TORCH_CHECK(BLOCK_M <= maxThr, "BLOCK_M (" #BLOCK_M ") exceeds device max threads per block");
    auto d = Q.size(3);
    TORCH_CHECK(BLOCK_N * d * sizeof(float) <= maxShm,
        "Shared memory for K tile exceeds limit");
    TORCH_CHECK(2 * BLOCK_N * d * sizeof(float) <= maxShm,
        "Shared memory for K+V tiles exceeds limit");

    auto B = Q.size(0), H = Q.size(1), NQ = Q.size(2), NK = K.size(2);
    int nQB = (NQ + BLOCK_M - 1) / BLOCK_M;
    int nKB = (NK + BLOCK_N - 1) / BLOCK_N;

    // === CREATE TENSORS ===
    auto opts8 = torch::TensorOptions().dtype(torch::kInt8).device(Q.device());
    auto optsF = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    // Same intermediate tensors as forward pass
    auto M = torch::zeros({B*H, nQB, nKB}, opts8);
    auto taus = torch::zeros({B*H, NQ}, optsF);
    auto Qi_ptr = torch::zeros({B*H, nQB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Qi_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Kj_ptr = torch::zeros({B*H, nKB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Kj_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    
    // Output gradient tensors
    auto dQ = torch::zeros_like(Q);                          // Gradient w.r.t. queries
    auto dK = torch::zeros_like(K);                          // Gradient w.r.t. keys  
    auto dV = torch::zeros_like(V);                          // Gradient w.r.t. values

    // === EXTRACT POINTERS ===
    int8_t* M_p = M.data_ptr<int8_t>();
    float* taus_p = taus.data_ptr<float>();
    int* Qidx_p = Q_idx.data_ptr<int>();
    int* Kidx_p = K_idx.data_ptr<int>();
    int32_t* Qi_pp = Qi_ptr.data_ptr<int32_t>();
    int32_t* Qi_ip = Qi_idx.data_ptr<int32_t>();
    int32_t* Kj_pp = Kj_ptr.data_ptr<int32_t>();
    int32_t* Kj_ip = Kj_idx.data_ptr<int32_t>();
    float* Qp = Q.data_ptr<float>();
    float* Kp = K.data_ptr<float>();
    float* Vp = V.data_ptr<float>();
    float* dOp = dOut.data_ptr<float>();                     // Input: output gradients
    float* dQp = dQ.data_ptr<float>();                       // Output: input gradients
    float* dKp = dK.data_ptr<float>();
    float* dVp = dV.data_ptr<float>();

    // === LAUNCH KERNELS ===
    // Same kernel sequence as forward pass to rebuild mask and lookup tables
    dim3 grid1(nQB, B*H), block1(BLOCK_M);
    size_t shm1 = BLOCK_N * d * sizeof(float);
    build_mask_kernel<<<grid1, block1, shm1>>>(
        Qp, Kp, Qidx_p, Kidx_p,
        B, H, NQ, NK, d,
        alpha, sm_scale,
        M_p, taus_p,
        nQB, nKB
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    build_lookup_kernel<<<B*H, 32>>>(
        M_p, B, H, nQB, nKB,
        Qi_pp, Qi_ip,
        Kj_pp, Kj_ip
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch backward kernel
    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);
    splash_backward_sparse<<<grid1, block1, shm2>>>(
        Qp, Kp, Vp,                                         // Forward inputs
        Qidx_p, Kidx_p, taus_p,                            // Indices and thresholds
        Qi_pp, Qi_ip, dOp,                                  // Lookup tables and output gradients
        dQp, dKp, dVp,                                      // Output: input gradients
        B, H, NQ, NK, d,
        alpha, sm_scale,
        nQB, nKB
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return {dQ, dK, dV};                                     // Return gradient tensors
}

// === PYBIND11 MODULE DEFINITION ===
// This creates the Python module that can be imported
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {                  // TORCH_EXTENSION_NAME is set by PyTorch
    m.def("forward", &forward_splash, "Splash attention forward");  // Expose forward function to Python
    m.def("backward", &backward_splash, "Splash attention backward"); // Expose backward function to Python
}
