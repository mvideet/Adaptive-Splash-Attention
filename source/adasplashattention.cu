#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <float.h>
#include <cmath>

#ifndef BLOCK_M
#define BLOCK_M 32 // number of queries per thread block
#endif
#ifndef BLOCK_N
#define BLOCK_N 64 //number of keys per tile
#endif
#ifndef D_MAX
#define D_MAX 128 // maximum head dimension supported - make it larger if we have a larger hidden state
#endif
#ifndef K_KEEP
#define K_KEEP 8 //number of top-K attention scores to keep per query and keep it relatively sparse
#endif

#define EPS 1e-8f // increased epsilon for better stability
#define MAX_HALLEY_ITERS 5 // reduced iterations for stability

// wraps CUDA calls and automatically checks for errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error at ", __FILE__, ":", __LINE__, " - ", cudaGetErrorString(err)); \
} while(0)

static_assert(BLOCK_N > 0 && BLOCK_M > 0, "Block dimensions must be positive");
static_assert(BLOCK_N * D_MAX <= 8192, "Shared memory K tile too large"); //each block needs BLOCK_N * D_MAX floats & most GPUs have 48kb shared memory/block
static_assert(BLOCK_M <= 1024, "BLOCK_M exceeds max threads per block");
static_assert(K_KEEP <= BLOCK_N, "K_KEEP should not exceed BLOCK_N");
static_assert(D_MAX >= 64, "D_MAX should be at least 64 for practical use");

//inline says that the compiler should inline this function for performance
__device__ inline void insert_topk(float val, int idx, float (&vals)[K_KEEP], int (&inds)[K_KEEP]) {
    // vals stores the actual attention scores
    // inds stores which key each score corresponds to (actual pos in the sequence)
    int j = K_KEEP-1;
    while (j >= 0 && val > vals[j]) { // compare with current element
        if (j < K_KEEP-1) {
            vals[j+1] = vals[j]; // shift to the right and it discards the last from memory
            inds[j+1] = inds[j];
        }
        --j;
    }
    ++j;
    if (j < K_KEEP) {
        vals[j] = val;
        inds[j] = idx;
    }
}

// FIXED: More stable entmax implementation
__device__ void entmax_threshold(const float* s, int k, float alpha, float* p, float &tau, bool full) {
    // Clamp alpha to safe range to prevent numerical instability
    alpha = fmaxf(1.05f, fminf(alpha, 3.0f));
    
    const float inv_am1 = 1.0f / (alpha - 1.0f);
    
    // More conservative bounds
    float s_max = s[0];
    float s_min = s[k-1];
    
    float lo = (alpha - 1.0f) * s_min - 2.0f;  // More conservative lower bound
    float hi = (alpha - 1.0f) * s_max + 2.0f;  // More conservative upper bound

    tau = 0.5f * (lo + hi); // initialize tau as midpoint
    
    // Simplified bisection method (more stable than Halley's method)
    for (int it = 0; it < MAX_HALLEY_ITERS; ++it) {
        float f = -1.0f; // start at -1 and add each p_i
        float fp = 0.0f;
        float fpp = 0.0f;
        
        for (int j = 0; j < k; j++) {
            float u = (alpha - 1.0f) * s[j] - tau; // compute u_j = (α-1)s_j - τ
            if (u <= 0) break;
            
            // Use more stable power computation
            float up = powf(fmaxf(u, EPS), inv_am1); // u^(1/(α-1)) = probability p_j
            
            // Handle numerical issues
            if (isnan(up) || isinf(up)) {
                up = 0.0f;
            }
            
            f += up; // add to constraint sum
            if (full) {
                float derivative_term = -inv_am1 * powf(fmaxf(u, EPS), (2.0f - alpha) * inv_am1);
                if (isfinite(derivative_term)) {
                    fp += derivative_term; // ∂f/∂τ
                }
                float second_derivative_term = inv_am1 * (inv_am1 + 1.0f - alpha) * powf(fmaxf(u, EPS), (3.0f - 2*alpha) * inv_am1);
                if (isfinite(second_derivative_term)) {
                    fpp += second_derivative_term;
                }
            }
        }
        
        if (fabsf(f) < 1e-4f) break; // looser convergence criterion for stability
        
        float tn = tau; // save current tau
        if (full && fabsf(fp) > EPS) { // if derivatives are available and non-zero, use Halley's method
            float denom = 2.0f * fp * fp - f * fpp;
            if (fabsf(denom) > EPS) {
                tn = tau - 2.0f * f * fp / denom; // prevent division by 0
            }
        }
        
        if (!(tn >= lo && tn <= hi)) {
            tn = 0.5f * (lo + hi); // fallback to bisection if new tau is outside bounds
        }
        
        // Evaluate function at new tau
        f = -1.0f;
        for (int j = 0; j < k; j++) {
            float u = (alpha - 1.0f) * s[j] - tn;
            if (u <= 0) break;
            float up = powf(fmaxf(u, EPS), inv_am1);
            if (isfinite(up)) {
                f += up;
            }
        }
        
        if (f > 0) {
            lo = tn;             // If f>0, tau is too small, update lower bound
        } else {
            hi = tn;             // If f<=0, tau is too large, update upper bound  
        }
        tau = tn;             // Update tau for next iteration
        
        // Prevent tau from becoming extreme
        tau = fmaxf(tau, s_min * (alpha - 1.0f) - 10.0f);
        tau = fminf(tau, s_max * (alpha - 1.0f) + 10.0f);
    }
    
    // Compute probabilities with better numerical stability
    float norm = 0.0f;
    for (int j = 0; j < k; j++) {
        float u = (alpha - 1.0f) * s[j] - tau;   // Compute u_j after tau is optimized
        float pj = (u > EPS) ? powf(fmaxf(u, EPS), inv_am1) : 0.0f;
        
        // Handle numerical issues
        if (isnan(pj) || isinf(pj)) {
            pj = 0.0f;
        }
        
        p[j] = pj; //store probability
        norm += pj;
    }
    
    // Ensure normalization is stable
    norm = fmaxf(norm, EPS);
    for (int j = 0; j < k; j++) {
        p[j] /= norm;
        
        // Final safety check
        if (isnan(p[j]) || isinf(p[j])) {
            p[j] = 1.0f / k;  // Fallback to uniform distribution
        }
    }
}


__global__ void build_mask_kernel(
    const float* Q, const float* K, const int* Q_idx, const int* K_idx, 
    int B, int H, int NQ, int NK, int d, float alpha, float sm_scale, 
    int8_t* M, float* taus, int nQB, int nKB){
        // Calculate global query index by combining block and thread indices:
        // blockIdx.x gives the block index in x-dimension 
        // BLOCK_M is the thread block size
        // threadIdx.x gives the thread index within the block
        //NQ and NK are the number of queries and keys per head
        //Queries [B*H, NQ,d] input tensor
        //Keys [B*H, NK, d] input tensor
        //Q_idx = query position indices
        //Key_idx = key position indices [B*H, NK]
        //BLOCK_M is the number of queries processed per thread block
        //remember that there are many threads in a block that can use shared memory
        int q = blockIdx.x*BLOCK_M + threadIdx.x;//global query index
        int bh = blockIdx.y; //which attention head and batch item am i working on
        //each block of threads are tied to one attention head but many queries
        //will multiply across all keys in the batch/head
        if (q>=NQ){
            return;
        }
        int idx = bh*NQ+q;
        int seq_q = Q_idx[idx]; //global seq position of this query
        float q_reg[D_MAX]; //we will store the query vector that is D_head size 
        // Load query vector into registers for fast access
        // Q is a 1D array representing [B*H, NQ, d] tensor
        // idx*d points to start of this query's vector
        // Unroll loop for better performance
        #pragma unroll
        for(int t=0;t<d;++t)
            q_reg[t] = Q[idx*d+t];
        
        float s_top[K_KEEP]; //top-K attention scores - FIXED: use finite values
        int ind[K_KEEP]; //indices of top-K elements and initialize to -1
        #pragma unroll
        for(int i=0;i<K_KEEP;++i){
            s_top[i] = -1e6f; // FIXED: use finite value instead of -FLT_MAX
            ind[i] = -1;
        }
        extern __shared__ float shmem[]; //shared memory across the block
        float* Ktile = shmem; //use shared memory for key tile
        for (int start = 0; start<NK; start+=BLOCK_N){ //process keys in tiles of size BLOCK_N
            int tile_id = threadIdx.x; //thread ID within block
            //each thread loads multiple elements
            // We stride by BLOCK_M because that's the number of threads in the block
            // Each thread needs to handle multiple elements (BLOCK_N*d total elements)
            // So we distribute the work evenly across BLOCK_M threads by having each thread
            // process elements spaced BLOCK_M apart
            for(int x = tile_id; x < BLOCK_N*d; x += BLOCK_M){
                // Each thread processes elements at indices: tile_id, tile_id+BLOCK_M, tile_id+2*BLOCK_M, ...
                int col = x/d; //which key in the tile
                int dim = x%d; //which dimension of the key
                int kn = start + col; //global key index - we don't add dim since dim represents the vector dimension (0 to d-1), not the key position. Adding dim would incorrectly offset the key index.
                bool ok = (kn < NK &&                            // Key exists
                          dim < d &&                             // Dimension is valid
                          K_idx[bh*NK + kn] <= seq_q);          // Causal: key position ≤ query position
                
                // Load key element or zero if out of bounds/causal
                Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;  // Load key element if ok, otherwise 0
            // K is a 1D array that represents a 3D tensor [B*H, NK, d]
            // The indexing formula (bh*NK + kn)*d + dim flattens the 3D access:
            // bh*NK*d - moves to the start of the batch+head
            // kn*d - moves to the start of the key vector
            // dim - moves to the specific dimension
            }
        
            __syncthreads(); //wait for all threads to finish loading before proceeding

            for(int j = 0;j<BLOCK_N;j++){
                // Calculate global key index by adding tile offset (start) to local key index (j)
                int kn = start + j;  // start points to beginning of current tile, j is position within tile (0 to BLOCK_N-1)
                
                // Skip if key index is out of bounds (>= NK) or violates causal masking
                // K_idx[bh*NK + kn] gets the sequence position of key kn in batch/head bh:
                //   bh*NK - offset to start of current batch/head's key indices
                //   + kn - offset to specific key's position
                // Compare against seq_q (query's sequence position) for causal masking
                if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue;
                float sum = 0;
                #pragma unroll
                for(int t=0;t<d;t++)
                    sum+=q_reg[t]*Ktile[j*d+t];
                
                sum *=sm_scale;
                
                // FIXED: Clamp score to prevent extreme values
                sum = fmaxf(sum, -10.0f);
                sum = fminf(sum, 10.0f);
                
                insert_topk(sum, kn, s_top, ind);
            }            
            //compute attention scores for all the keys in the current tile.
            //we have a top-K buffer that stores the top-K attention scores and their corresponding key indices
            //we will use this buffer to compute the attention mask
            //we will use the entmax threshold to compute the attention mask
            //we will use the top-K buffer to compute the attention mask
            //we will use the entmax threshold to compute the attention mask
            __syncthreads(); //wait for all threads to finish before proceeding
        }
        
        float tau;
        float pbuf[K_KEEP]; // Buffer for entmax probabilities
    
        entmax_threshold(s_top, K_KEEP, alpha, pbuf, tau, true); // Compute α-entmax weights
        taus[idx] = tau; // Store threshold for backward pass
        

        int query_block = q/BLOCK_M; //which query block this is
        // Calculate base offset into mask tensor:
        // bh*nQB*nKB - moves to start of current batch+head's mask section
        //   bh - current batch*head index
        //   nQB - number of query blocks
        //   nKB - number of key blocks
        // query_block*nKB - moves to current query block's row
        //   query_block - index of current query block
        //   nKB - number of key blocks (stride between query block rows)
        int base = bh*nQB*nKB + query_block*nKB;
        for(int i = 0;i<K_KEEP;i++){
            if(ind[i] < 0) continue;
            int jKB = ind[i]/BLOCK_N; //which key block contains this key
            M[base + jKB] = 1; //mark this block as needed and we can't remove this from the mask
        }
    }


    // Q_idx maps local query indices to global sequence positions
    // Since queries are processed in BLOCK_M sized tiles, a query's local index q 
    // within a CUDA block may not match its original sequence position.
    // Q_idx[bh*NQ + q] gives the global sequence position for query q in batch/head bh,
    // which is needed for proper causal masking.
__global__ void build_lookup_kernel(const int8_t* M, int B, int H, int nQB, int nKB, int* Qi_ptr, int* Qi_idx){
    int bh = blockIdx.x; //batch*head index from block ID. This only uses 1D grid since it processes one batch*head per thread
    if(bh>=H*B) return;

    int baseM = bh * nQB * nKB;                              // Base address in mask tensor
    int bQi   = bh * (nQB + 1);  //CSR pointer array always has 1 extra entry (to mark the end):                             // Base address in Qi_ptr
    int bQiI  = bh * (nQB * nKB);   // Base address in Qi_idx array for this batch*head - shape is [B*H, nQB*nKB]
    Qi_ptr[bQi] = 0; //value is always 0 - shape is [B*H, nQB+1] because it is sparse

    //EG: Query attends to [keys] (0, [0, 3]), (1, [1]), (2, [0,2,3])
    //Qi_idx = [0, 3, 1, 0, 2, 3]
    //Qi_ptr = [0, 2, 3,6]
    //Fill in Qi_ptr
    for(int i = 0;i<nQB;i++){
        int c = 0;//active key blocks for this query block counter
        for(int j =0;j<nKB;j++){
            c += M[baseM+i*nKB+j];
        }
        Qi_ptr[bQi+i+1] = Qi_ptr[bQi+i]+c; //It updates the next pointer in the Qi_ptr array by adding the number of active key blocks (c) for the current query block i.

    }
    //Fill in Qi_idx
    for(int i = 0;i<nQB;i++){
        int w = Qi_ptr[bQi+i];  // Starting position in Qi_idx array where we'll store which key blocks this query block attends to
        for(int j =0;j<nKB;j++){
            if(M[baseM+i*nKB+j]){
                // Store the key block index j at position bQiI+w in Qi_idx array
                // bQiI: Base offset for this batch+head's Qi_idx section
                // w: Current write position within this query block's active key indices
                // w++ increments w after storing j, advancing to next write position
                // j: Index of the current key block that is active for this query block
                Qi_idx[bQiI+w++] = j;
            }
    }}
}

__global__ void splash_forward_sparse(
    const float* Q, const float* K, const float* V,          // Query, Key and Value tensors for attention computation
    const int* Q_idx, const int* K_idx,                      // Position indices [B*H, NQ] and [B*H, NK] for causal masking
    const float* taus,                                        // Entmax thresholds from mask kernel
    const int* Qi_ptr, const int* Qi_idx,                    // Lookup tables from build_lookup_kernel
    float* Out,                                               // Output tensor [B*H, NQ, d]
    int B, int H, int NQ, int NK, int d,                     // Tensor dimensions
    float alpha, float sm_scale,                             // Attention parameters
    int nQB, int nKB                                         // Number of blocks
){

    // Calculate global query index for a certain batch*head
    // - blockIdx.x * BLOCK_M: Offset to start of current thread block's queries
    // - threadIdx.x: Thread's position within block (0 to BLOCK_M-1)
    // This distributes queries across thread blocks and threads
    int q = blockIdx.x*BLOCK_M + threadIdx.x;

    // Get batch*head index from block's y-coordinate
    // Each block processes queries for one batch item and attention head
    int bh = blockIdx.y;

    // Exit if query index exceeds total number of queries
    if(q>=NQ) return;

    // Get sequence position for this query to enforce causal masking
    // Q_idx maps from query index to sequence position
    // bh*NQ + q: Offset into Q_idx array for this batch+head and query
    // We need sequence positions to ensure queries only attend to keys
    // at positions up to their own position (causal masking)
    int seq_q = Q_idx[bh*NQ + q];
    float tau = taus[bh*NQ + q];
    //create a pointer to the query vector that we are interested in 
    const float* Qptr = Q + ((bh*NQ + q)*d);

    //load the query into the registers
    float qreg[D_MAX];
    #pragma unroll
    for(int t=0;t<d;++t)
        qreg[t] = Qptr[t];
    //Get the shared memory set up for the keys and values
    extern __shared__ float sh[];
    float* Ktile = sh;
    // Point Vtile to second half of shared memory buffer, after the Ktile section
    // BLOCK_N * d bytes are allocated for Ktile, so Vtile starts after that offset
    float* Vtile = sh + BLOCK_N * d;

    //initialize the accumulator
    float accum[D_MAX];
    float norm = 0.f;  // Initialize normalization factor
    #pragma unroll
    for(int t=0; t<d; ++t)
        accum[t] = 0.f;

    //iQB maps the global query index q to its block index by div
    //off computes offset in the Qi_ptr array for the batch heads and query block
    //offI computes the offset into Qi_idx array
    int iQB = q/BLOCK_M;
    int offset = bh*(nQB+1) + iQB;
    int offsetI = bh*(nQB*nKB);

    // Process active key blocks that were marked as needed in the mask
    // Qi_ptr[offset] and Qi_ptr[offset+1] define the range of active key blocks for this query block
    // This implements sparse attention by only processing key blocks that passed the mask threshold
    for(int ptr = Qi_ptr[offset]; ptr < Qi_ptr[offset+1]; ptr++) {
        // ptr indexes into Qi_idx array which stores indices of active key blocks
        // offsetI + ptr gives the actual position in Qi_idx for this batch/head
        int jKB = Qi_idx[offsetI + ptr]; // Which block of keys are we processing
        
        // Calculate starting position of this key block
        // Each block contains BLOCK_N keys, so multiply block index by BLOCK_N
        int start = jKB * BLOCK_N;
        
        // Get thread ID within the block for parallel processing
        int tid = threadIdx.x;
        
        // Cooperatively load the key block into shared memory
        // Each thread loads elements spaced BLOCK_M apart
        for(int x = tid; x < BLOCK_N*d; x += BLOCK_M) {
            int col = x/d;  // Which key in the block (0 to BLOCK_N-1)
            int dim = x%d;  // Which dimension of the key vector (0 to d-1)
            int kn = start + col; // Global key index within the batch/head
            bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q); // Check bounds and causality
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;
            Vtile[x] = ok ? V[(bh*NK + kn)*d + dim] : 0.f;
        }
        __syncthreads();

        // Compute attention for this tile
        for(int k = 0; k < BLOCK_N; ++k) {
            int kn = start + k;
            if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue;
            
            float s = 0.f;
            #pragma unroll
            for(int t=0; t<d; ++t) {
                s += qreg[t]*Ktile[k*d + t];
            }
            s *= sm_scale;
            
            // FIXED: Clamp attention score to prevent extreme values
            s = fmaxf(s, -10.0f);
            s = fminf(s, 10.0f);
            
            float u = (alpha-1.f)*s - tau;
            float p = (u > 0) ? powf(fmaxf(u, EPS), 1.f/(alpha-1.f)) : 0.f;
            
            // FIXED: Add numerical stability checks
            if (isnan(p) || isinf(p)) {
                p = 0.0f;
            }
            
            norm += p;
            
            #pragma unroll
            for(int t=0; t<d; ++t) {
                accum[t] += p*Vtile[k*d + t];
            }
        }
        __syncthreads();
    }

    // Compute inverse normalization factor, adding small epsilon to avoid division by zero
    float invN = 1.f/(norm + EPS);

    // Get pointer to output location for this query
    float* OutPtr = Out + ((bh*NQ + q)*d);

    #pragma unroll
    for(int t=0; t<d; ++t) {
        float out_val = accum[t]*invN;
        
        // FIXED: Final safety check for output values
        if (isnan(out_val) || isinf(out_val)) {
            out_val = 0.0f;
        }
        
        OutPtr[t] = out_val;
    }
}


__global__ void splash_backward_sparse(
    const float* Q,          // [B,H,NQ,d] Query matrix from forward pass
    const float* K,          // [B,H,NK,d] Key matrix from forward pass  
    const float* V,          // [B,H,NK,d] Value matrix from forward pass
    const int* Q_idx,        // [B,H,NQ] Position indices for queries
    const int* K_idx,        // [B,H,NK] Position indices for keys
    const float* taus,       // [B,H,NQ] Threshold values for each query
    const int* Qi_ptr,       // [B,H,NQ+1] Pointers into Qi_idx for each query
    const int* Qi_idx,       // Indices of keys that each query attends to
    const float* dOut,       // [B,H,NQ,d] Output gradients from backpropagation
    float* dQ,              // [B,H,NQ,d] Output: Gradient w.r.t. queries
    float* dK,              // [B,H,NK,d] Output: Gradient w.r.t. keys
    float* dV,              // [B,H,NK,d] Output: Gradient w.r.t. values
    int B,                  // Batch size
    int H,                  // Number of attention heads
    int NQ,                 // Number of queries
    int NK,                 // Number of keys
    int d,                  // Hidden dimension size
    float alpha,            // Power term in attention formula
    float sm_scale,         // Scaling factor for attention scores
    int nQB,                // Number of query blocks
    int nKB                 // Number of key blocks
){
    // Calculate global query index:
    // blockIdx.x gives the block index in x dimension
    // BLOCK_M is the number of queries per block
    // threadIdx.x gives the thread index within the block
    // This spreads queries across thread blocks and threads
    int q = blockIdx.x*BLOCK_M + threadIdx.x; //which query are we working on
    int bh = blockIdx.y; //which batch*head are we working on
    if(q>=NQ) return; //if the query index is greater than the number of queries, return

    // Get sequence position for this query to enforce causal masking
    // Q_idx maps from query index to sequence position
    // bh*NQ + q: Offset into Q_idx array for this batch+head and query
    // We need sequence positions to ensure queries only attend to keys
    // at positions up to their own position (causal masking)
    int seq_q = Q_idx[bh*NQ + q]; //get the actual sequence position for this query
    const float* Qptr = Q + ((bh*NQ + q)*d); //get the pointer to the query
    // Load query and initialize query gradient
    float qreg[D_MAX], dqreg[D_MAX]; // Query and its gradient
    #pragma unroll
    for(int t=0; t<d; ++t) {
        qreg[t] = Qptr[t]; // Load query
        dqreg[t] = 0.f; // Initialize gradient to zero
    }

    extern __shared__ float sh[];
    float* Ktile = sh;
    float* Vtile = sh+BLOCK_N*d;

    float accum[D_MAX];
    float norm = 0.f;
    #pragma unroll
    for(int t= 0;t<d;++t)
        accum[t] = 0.f;
    
    int iQB = q/BLOCK_M; //<- which query block are we working on
    int offset = bh*(nQB+1) + iQB; //offset into the Qi_ptr array (csr format)
    int offsetI = bh*(nQB*nKB); //offset into the actual Qi_idx array - which key blocks are we working on

    // Declare variables at function scope to avoid redeclaration
    int start, tid;

    //recompute forward pass values
    for (int ptr = Qi_ptr[offset]; ptr < Qi_ptr[offset+1]; ++ptr){
        int jKB = Qi_idx[offsetI + ptr]; //which key block are we working on for the batch/head
        start = jKB*BLOCK_N;
        tid = threadIdx.x;//which thread are we working on = which key in the tile
        for(int x = tid;x<BLOCK_N*d;x+=BLOCK_M){
            // col represents the column index within the tile (0 to BLOCK_N-1)
            // x is the linearized index, d is the hidden dimension size
            // so x/d gives us which column we're processing
            int col = x/d;
            int dim = x%d;
            int kn = start+col; //loading the knth key in the head
            bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q);
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f; //the global key after adding the batch *NK and then which key it is in the head and then the number of dimensions + number of dimensions
            Vtile[x] = ok ? V[(bh*NK + kn)*d + dim] : 0.f;
        }
        __syncthreads();
        for(int k = 0;k<BLOCK_N;++k){
            int kn = start+k; //key index within the respective batch/head 
            if (kn >= NK || K_idx[bh*NK+kn] > seq_q) continue; //ensure autoregression
            float s= 0.f;//get ready to recompute the attention score
            #pragma unroll
            for(int t = 0;t<d;++t)
                s+=qreg[t]*Ktile[k*d+t]; //compute sum across a certain key
            s*=sm_scale;
            
            // FIXED: Clamp attention score for stability
            s = fmaxf(s, -10.0f);
            s = fminf(s, 10.0f);
            
            float u=(alpha-1.f)*s - taus[bh*NQ + q];         // Entmax intermediate
            float p=(u>0.f? powf(fmaxf(u, EPS), 1.f/(alpha-1.f)) : 0.f);  // Entmax probability
            
            // FIXED: Add numerical stability checks
            if (isnan(p) || isinf(p)) {
                p = 0.0f;
            }
            
            norm+=p; //accumulate normalization
            #pragma unroll
            for(int t = 0;t<d;++t)
                accum[t] += p*Vtile[k*d+t];

        }
        __syncthreads(); 
    }
    float invN = 1.f/(norm+EPS);
    // -1/norm² used for computing gradient of normalization term
    float invN2 = -invN*invN;
    // Pointer to output gradient for current query in batch/head
    const float* dOp = dOut + ((bh*NQ+q)*d);
    
    for(int ptr = Qi_ptr[offset]; ptr < Qi_ptr[offset+1]; ++ptr){
        int jKB = Qi_idx[offsetI + ptr];
        start = jKB*BLOCK_N;
        tid = threadIdx.x;
        for(int x = tid;x<BLOCK_N*d;x+=BLOCK_M){
            int col = x/d;
            int dim = x%d;
            int kn = start+col;
            bool ok = (kn < NK && K_idx[bh*NK+kn] <= seq_q);
            Ktile[x] = ok ? K[(bh*NK+kn)*d + dim] : 0.f;
            Vtile[x] = ok ? V[(bh*NK+kn)*d + dim] : 0.f;
        }
        __syncthreads();
        for(int k = 0;k<BLOCK_N;++k){
            int kn = start+k;
            if (kn >= NK || K_idx[bh*NK+kn] > seq_q) continue;
            float s= 0.f;
            #pragma unroll
            for(int t= 0;t<d;++t)
                s+=qreg[t]*Ktile[k*d+t];
            s*=sm_scale;
            
            // FIXED: Clamp attention score for stability
            s = fmaxf(s, -10.0f);
            s = fminf(s, 10.0f);
            
            float u=(alpha-1.f)*s - taus[bh*NQ + q];         // Entmax intermediate value: u = (α-1)s - τ
            float p=(u>0.f? powf(fmaxf(u, EPS), 1.f/(alpha-1.f)) : 0.f);  // Entmax probability: p = max(0, u^(1/(α-1)))
            
            // FIXED: Add numerical stability checks
            if (isnan(p) || isinf(p)) {
                p = 0.0f;
            }
            
            float pb = p*invN;
            // === GRADIENT W.R.T. VALUES (dV) ===
            // For each dimension t, compute gradient contribution to value tensor V
            #pragma unroll
            for(int t =0;t<d;++t){
                // dV[i] = sum_j (p_j * dOut[j])_i where:
                // - p_j is the normalized probability (pb) for key j
                // - dOut[j] is the gradient of loss w.r.t. output at position j
                // Using atomicAdd since multiple threads may update same dV location
                float dv_contrib = pb * dOp[t];
                
                // FIXED: Check for valid gradient contribution
                if (isfinite(dv_contrib)) {
                    atomicAdd(&dV[(bh*NK + kn)*d + t], dv_contrib);
                }
            }

            // === GRADIENT W.R.T. PROBABILITIES (dp) === 
            float dp=0.f;  // Initialize gradient w.r.t probability p
            for(int t= 0;t<d;++t)
            {
                // Direct gradient term: dp = sum_i (v_i * dOut_i) / norm
                // - v_i is the value vector (Vtile) 
                // - dOut_i is gradient of loss w.r.t. output
                // - Divided by norm (invN) for proper scaling
                dp += Vtile[k*d+t] * dOp[t] * invN;
            }

            // Compute dot product between accumulated values and output gradient
            float accum_dot_dO = 0.f;
            for(int t= 0;t<d; ++t){
                // accum_dot_dO = sum_i (accum_i * dOut_i)
                accum_dot_dO += accum[t] * dOp[t];
            }
            // Add normalization gradient term: -accum·dOut/norm²
            dp+=accum_dot_dO*invN2;

            // === CHAIN RULE THROUGH ENTMAX ===
            // grad_u = dp/du = dp * (1/(α-1)) * u^((2-α)/(α-1)) if u > 0, else 0
            float grad_u = 0.0f;
            if (u > EPS && fabsf(alpha - 1.0f) > EPS) {
                float exponent = (2.f-alpha)/(alpha-1.f);
                float power_term = powf(fmaxf(u, EPS), exponent);
                
                if (isfinite(power_term)) {
                    grad_u = (1.f/(alpha-1.f)) * power_term * dp;
                }
            }
            
            // grad_s = du/ds = (α-1)
            float grad_s = grad_u * (alpha-1.f);
            
            // FIXED: Check for valid gradient
            if (!isfinite(grad_s)) {
                grad_s = 0.0f;
            }

            // === GRADIENTS W.R.T. QUERY AND KEY ===
            #pragma unroll
            for(int t= 0;t<d;++t){
                // dQ += grad_s * K * sm_scale (accumulated in registers)
                float dq_contrib = grad_s * Ktile[k*d+t] * sm_scale;
                if (isfinite(dq_contrib)) {
                    dqreg[t] += dq_contrib;
                }
                
                // dK += grad_s * Q * sm_scale (atomic update to global memory)
                float dk_contrib = grad_s * qreg[t] * sm_scale;
                if (isfinite(dk_contrib)) {
                    atomicAdd(&dK[(bh*NK + kn)*d + t], dk_contrib);
                }
            }
        }
        __syncthreads(); // Synchronize before next iteration
    }
    // === WRITE QUERY GRADIENTS TO GLOBAL MEMORY ===
    float* dQp = dQ + ((bh*NQ + q)*d);                      // Pointer to query gradient
    #pragma unroll
    for(int t= 0;t<d;++t) {                    // Write accumulated gradients
        float dq_val = dqreg[t];
        
        // FIXED: Final safety check for gradients
        if (isnan(dq_val) || isinf(dq_val)) {
            dq_val = 0.0f;
        }
        
        dQp[t] = dq_val;
    }
}


torch::Tensor forward_splash(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Q_idx, torch::Tensor K_idx,
    float sm_scale, float alpha
){
    // === INPUT VALIDATION ===
    TORCH_CHECK(Q.size(3) <= D_MAX, "Head dimension too large; increase D_MAX");
    TORCH_CHECK(Q.device().is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(K.device() == Q.device(), "All tensors must be on same device");
    TORCH_CHECK(V.device() == Q.device(), "All tensors must be on same device");
    
    // FIXED: Clamp alpha to safe range to prevent numerical instability
    alpha = std::max(1.05f, std::min(alpha, 3.0f));
    
    int device;
    cudaGetDevice(&device);
    int maxShm, maxThr;
    cudaDeviceGetAttribute(&maxShm, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&maxThr, cudaDevAttrMaxThreadsPerBlock, device);
    TORCH_CHECK(BLOCK_M <= maxThr, "BLOCK_M (" #BLOCK_M ") exceeds device max threads per block");
    auto d = Q.size(3);
    TORCH_CHECK(BLOCK_N * d * sizeof(float) <= maxShm, "Shared memory for K tile exceeds limit");
    TORCH_CHECK(2 * BLOCK_N * d * sizeof(float) <= maxShm, "Shared memory for K+V tiles exceeds limit");
    auto B = Q.size(0), H = Q.size(1), NQ = Q.size(2), NK = K.size(2);
    int nQB = (NQ + BLOCK_M - 1) / BLOCK_M;
    int nKB = (NK + BLOCK_N - 1) / BLOCK_N;
    auto opts8 = torch::TensorOptions().dtype(torch::kInt8).device(Q.device());
    auto optsF = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto M = torch::zeros({B*H, nQB, nKB}, opts8);
    auto taus = torch::zeros({B*H, NQ}, optsF);
    auto Qi_ptr = torch::zeros({B*H, nQB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Qi_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Out = torch::zeros_like(Q);


    int8_t* M_p = M.data_ptr<int8_t>(); //block mask matrix
    float* taus_p = taus.data_ptr<float>(); //thresholds for each query
    int* Qidx_p = Q_idx.data_ptr<int>(); //query indices
    int* Kidx_p = K_idx.data_ptr<int>(); //key indices
    int32_t* Qi_pp = Qi_ptr.data_ptr<int32_t>(); //csr pointers for queries
    int32_t* Qi_ip = Qi_idx.data_ptr<int32_t>(); //csr indices for queries
    float* Qp = Q.data_ptr<float>(); //input query tensor
    float* Kp = K.data_ptr<float>(); //input key tensor
    float* Vp = V.data_ptr<float>(); //input value tensor
    float* Outp = Out.data_ptr<float>(); //output tensor

    dim3 grid1(nQB, B*H); //grid for building mask
    dim3 block1(BLOCK_M); //block for building mask
    size_t shm1 = BLOCK_N * d * sizeof(float); //shared memory for K tile
    
    // Launch mask building kernel
    build_mask_kernel<<<grid1, block1, shm1>>>(
        Qp, Kp, Qidx_p, Kidx_p, 
        B, H, NQ, NK, d, alpha, sm_scale, 
        M_p, taus_p, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());

    // Launch lookup table building kernel
    build_lookup_kernel<<<B*H, 32>>>(M_p, B, H, nQB, nKB, Qi_pp, Qi_ip);
    CUDA_CHECK(cudaGetLastError());

    // Launch sparse forward kernel
    size_t shm2 = 2 * BLOCK_N * d * sizeof(float); //shared memory for K+V tiles
    splash_forward_sparse<<<grid1, block1, shm2>>>(Qp, Kp, Vp, Qidx_p, Kidx_p, taus_p, Qi_pp, Qi_ip, Outp, B, H, NQ, NK, d, alpha, sm_scale, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());

    // FIXED: Add synchronization and final validation
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for NaN/Inf in output
    if (torch::isnan(Out).any().item<bool>()) {
        TORCH_WARN("NaN values detected in output, setting to zero");
        Out.masked_fill_(torch::isnan(Out), 0.0);
    }
    if (torch::isinf(Out).any().item<bool>()) {
        TORCH_WARN("Inf values detected in output, clamping");
        Out = torch::clamp(Out, -100.0, 100.0);
    }

    return Out;
}

std::vector<torch::Tensor> backward_splash(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Q_idx, torch::Tensor K_idx,
    float sm_scale, float alpha,
    torch::Tensor dOut
){
    TORCH_CHECK(Q.size(3) <= D_MAX, "Head dimension too large; increase D_MAX");
    TORCH_CHECK(Q.device().is_cuda(), "Tensors must be on CUDA device");
    
    // FIXED: Clamp alpha to safe range
    alpha = std::max(1.05f, std::min(alpha, 3.0f));
    
    // Get current CUDA device ID
    int device;
    cudaGetDevice(&device);

    // Query device limits for shared memory and threads per block
    int maxShm, maxThr;
    cudaDeviceGetAttribute(&maxShm, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&maxThr, cudaDevAttrMaxThreadsPerBlock, device);

    // Validate block size against device thread limit
    TORCH_CHECK(BLOCK_M <= maxThr, "BLOCK_M (" #BLOCK_M ") exceeds device max threads per block");

    // Get head dimension from input tensor
    auto d = Q.size(3);

    // Validate shared memory requirements for K and K+V tiles
    TORCH_CHECK(BLOCK_N * d * sizeof(float) <= maxShm, "Shared memory for K tile exceeds limit");
    TORCH_CHECK(2 * BLOCK_N * d * sizeof(float) <= maxShm, "Shared memory for K+V tiles exceeds limit");

    // Extract tensor dimensions
    auto B = Q.size(0),    // Batch size
         H = Q.size(1),    // Number of heads
         NQ = Q.size(2),   // Query sequence length
         NK = K.size(2);   // Key sequence length

    // Calculate number of blocks needed for queries and keys
    int nQB = (NQ + BLOCK_M - 1) / BLOCK_M;  // Ceiling division for query blocks
    int nKB = (NK + BLOCK_N - 1) / BLOCK_N;  // Ceiling division for key blocks
    auto opts8 = torch::TensorOptions().dtype(torch::kInt8).device(Q.device());
    auto optsF = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto M = torch::zeros({B*H, nQB, nKB}, opts8);
    auto taus = torch::zeros({B*H, NQ}, optsF);
    auto Qi_ptr = torch::zeros({B*H, nQB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Qi_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    int8_t* M_p = M.data_ptr<int8_t>();
    float* taus_p = taus.data_ptr<float>();
    int* Qidx_p = Q_idx.data_ptr<int>();
    int* Kidx_p = K_idx.data_ptr<int>();
    int32_t* Qi_pp = Qi_ptr.data_ptr<int32_t>();
    int32_t* Qi_ip = Qi_idx.data_ptr<int32_t>();
    float* Qp = Q.data_ptr<float>();
    float* Kp = K.data_ptr<float>();
    float* Vp = V.data_ptr<float>();
    float* dOp = dOut.data_ptr<float>();
    float* dQp = dQ.data_ptr<float>();
    float* dKp = dK.data_ptr<float>();
    float* dVp = dV.data_ptr<float>();

    dim3 grid1(nQB, B*H);
    dim3 block1(BLOCK_M);
    size_t shm1 = BLOCK_N * d * sizeof(float);
    
    // Rebuild mask and lookup tables (needed for backward pass)
    build_mask_kernel<<<grid1, block1, shm1>>>(
        Qp, Kp, Qidx_p, Kidx_p, 
        B, H, NQ, NK, d, alpha, sm_scale, 
        M_p, taus_p, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());

    build_lookup_kernel<<<B*H, 32>>>(M_p, B, H, nQB, nKB, Qi_pp, Qi_ip);
    CUDA_CHECK(cudaGetLastError());
    
    // Launch backward kernel
    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);
    splash_backward_sparse<<<grid1, block1, shm2>>>(Qp, Kp, Vp, Qidx_p, Kidx_p, taus_p, Qi_pp, Qi_ip, dOp, dQp, dKp, dVp, B, H, NQ, NK, d, alpha, sm_scale, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());
    
    // FIXED: Add synchronization and final validation
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check and fix gradients
    auto check_and_fix_gradients = [](torch::Tensor& grad, const std::string& name) {
        if (torch::isnan(grad).any().item<bool>()) {
            TORCH_WARN("NaN values detected in " + name + ", setting to zero");
            grad.masked_fill_(torch::isnan(grad), 0.0);
        }
        if (torch::isinf(grad).any().item<bool>()) {
            TORCH_WARN("Inf values detected in " + name + ", clamping");
            grad = torch::clamp(grad, -100.0, 100.0);
        }
    };
    
    check_and_fix_gradients(dQ, "dQ");
    check_and_fix_gradients(dK, "dK");
    check_and_fix_gradients(dV, "dV");

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_splash, "Splash attention forward");
    m.def("backward", &backward_splash, "Splash attention backward");
}






    
