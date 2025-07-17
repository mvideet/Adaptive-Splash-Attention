#include <cinttypes>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <float.h>




#ifndef BLOCK_M
#define BLOCK_M 32 // number of queries per thread block
#endif
#ifndef BLOCK_N
#define BLOCK_N 64 //number of keys per tile
#endif
#ifndef D_MAX
#define D_MAX 128 // maximum head dimension supported - make itlarger if we have a larger hidden state
#endif
#ifndef K_KEEP
#define K_KEEP 8 //number of top-K attention scores to keep per query and keep it relatively sparse
#endif

#define EPS 1e-6 // to prevent division by 0
#define MAX_HALLEY_ITERS 8 // number of max halley iterations for finding T via gradients

// wraps CUDA calls and automatically checks for errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error at ", __FILE__, ":", __LINE__, "-", cudaGetErrorString(err)); \
} while(0)

static_assert(BLOCK_N > 0 && BLOCK_M >0, "Block dimensions must be positive");
static_assert(BLOCK_N * D_MAX <=8192, "Shared memory K tile too large"); //each block needs BLOCK_N * D_MAX floats & most GPUs have 48kb shared mmeroy/block
static_assert(BLOCK_M<=1024, "BLOCK_M exceeds max threads per block");
static_assert(K_KEEP <=BLOCK_N, "K_KEEP should not exceed BLOCK_N");
static_assert(D_MAX>=64,"D_MAX should be at least 64 for practical use" );

//inline says that the compiler should inline this function for performacnce
__device__ inline void insert_topk(float val, int idx, float (&vals)[K_KEEP], int (&inds)[K_KEEP]) {
    // vals stores the actual attention scores
    // inds stores which key each score corresponds to (actual pos in the sequence)
    int j = K_KEEP-1;
    while (j>=0 && val > vals[j]) { // compare with current element
        if (j<K_KEEP-1) {
            vals[j+1] = vals[j]; // shift to the right and it discards the last from memory
            inds[j+1] = inds[j];
        }
        --j;
    }
    ++j;
    if (j<K_KEEP) {
        vals[j] = val;
        inds[j] = idx;
    }
}

__device__ void entmax_threshold(const float* s, int k, float alpha, float* p, float &tau, bool full){
const float inv_am1 = 1.f/(alpha-1.f);
//initialize bisection bounds
float lo = (alpha-1.f)*s[k-1] - 1.f; //smallest score
float hi = (alpha-1.f)*s[0]; //largest score

auto eval = [&](float t, float &f, float &fp, float &fpp){
    f = -1;//start at -1 and add each p_i in the loop below
    fp=fpp=0;
    for(int j = 0;j<k;j++){
        float u = (alpha-1.f)*s[j] - t; //compute u_j = (α-1)s_j - τ
        if(u<=0) break;
        float up = powf(u, inv_am1); //u^(1/(α-1)) = probability p_j
        f+=up; //add to constraint sum
        if(full){
            fp += -inv_am1 * powf(u, (2.f-alpha)*inv_am1); //∂f/∂τ
            fpp += inv_am1*(inv_am1+1.f-alpha) * powf(u, (3.f-2*alpha)*inv_am1);
        }
    }
};
tau = 0.5f*(lo+hi); //initialize tau as midpoint
//halley's method iterations
for(int it = 0; it<MAX_HALLEY_ITERS; ++it){
    float f, fp, fpp;
    eval(tau, f, fp, fpp);
    if(fabsf(f)<1e-3f) break;//converged
    float tn = tau; //save current tau
    if(full){//if derivatives are available, then use the updates, else resort to bisection
        float denom = 2.f*fp*fp - f*fpp;
        tn = tau-2.f*f*fp/fmaxf(denom,EPS); //prevent division by 0

    }
    if(!(tn>=lo && tn <=hi)) {
        tn = 0.5f*(lo+hi); //fallback to bisection if new tau is outside bounds
    }
    eval(tn, f, fp, fpp);  // Evaluate function and derivatives at new tau
    if(f>0){
        lo=tn;             // If f>0, tau is too small, update lower bound
    }
    else{
        hi=tn;             // If f<=0, tau is too large, update upper bound  
    }
    tau = tn;             // Update tau for next iteration
}
float norm = 0;
for(int j = 0;j<k;j++){
    float u = (alpha-1.f)*s[j] - tau;   // Compute u_j after tau is optimized
   float pj = (u>0)?powf(u, inv_am1):0;
   p[j] = pj; //store probability
   norm +=pj;
}
norm = fmaxf(norm, EPS);
for(int j = 0;j<k;j++){
    p[j] /= norm;
}
}


__global__ void build_mask_kernel(
    const float* Q, const float* K, const int* Q_idx, const int* K_idx, int B, int H, int NQ, int NK, int d, float alpha, float sm_scale, int8_t* M, float* taus, int nQB, int nKB){
        // Calculate global query index by combining block and thread indices:
        // blockIdx.x gives the block index in x-dimension 
        // BLOCK_M is the thread block size
        // threadIdx.x gives the thread index within the block
        //NQ and NK are the number of queries and keys per head
        //Queries [B*H, NQ,d] input tensor
        //Keys [B*H, NK, d] input tensor
        //Q_idx = query position indices
        //Key_idx = key position indicites [B*H, NK]
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
        
        #pragma unroll
        for(int t=0;t<d;++t)
            q_reg[t] = Q[idx*d+t];
        
        float s_top[K_KEEP]; //top-K attention scores and initialize to -infinity
        int ind[K_KEEP]; //indices of top-K elements and initialize to -1
        #pragma unroll
        for(int i=0;i<K_KEEP;++i){
            s_top[i] = -FLT_MAX;
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

__global__ void build_lookup_kernel(const int8_t* M, int B, int H, int nQB, int nKB, int* Qi_ptr, int* Qi_idx, int* Kj_ptr, int* Kj_idx){
    int bh = blockIdx.x; //batch*head index from block ID. THis only uses 1D grid since it processes one batch*head per thread
    if(bh>=H*B) return;

    int baseM = bh * nQB * nKB;                              // Base address in mask tensor
    int bQi   = bh * (nQB + 1);  //CSR pointer array always has 1 extra entry (to mark the end):                             // Base address in Qi_ptr
    int bQiI  = bh * (nQB * nKB);   // Base address in Qi_idx array for this batch*head - shape is [B*H, nQB*nKB]
    int bKj   = bh * (nKB + 1);    //CSR pointer always has 1 extra entry to mark the end                         
    int bKjI  = bh * (nQB * nKB);
    Qi_ptr[bQi] = 0; //value is always 0 - shape is [B*H, nQB+1] because it is sparse

    //EG: Query attentnds to [keys] (0, [0, 3]), (1, [1]), (2, [0,2,3])
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
    //Fill in Qi_indx
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
    Kj_ptr[bKj] = 0;//first pointer is always 0
    // Count active query blocks for each key block
    for(int j=0; j<nKB; j++) {
        int c = 0;  // Counter for active query blocks
        for(int i=0; i<nQB; i++) {
            c += M[baseM + i*nKB + j];  // Count if query block i attends to key block j
        }
        Kj_ptr[bKj + j + 1] = Kj_ptr[bKj + j] + c;  // Store cumulative count
    }

    // Fill indices of active query blocks for each key block
    for(int j=0; j<nKB; j++) {
        int w = Kj_ptr[bKj + j];  // Starting write position for this key block
        for(int i=0; i<nQB; i++) {
            if(M[baseM + i*nKB + j]) {  // If query block i attends to key block j
                Kj_idx[bKjI + w++] = i;  // Store query block index
            }
        }
    }
    //


}
