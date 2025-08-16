#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <float.h>
#include <cmath>

#ifndef BLOCK_M
#define BLOCK_M 32
#endif
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
#ifndef D_MAX
#define D_MAX 128
#endif
#ifndef K_KEEP
#define K_KEEP 8
#endif

#define EPS 1e-8f
#define MAX_HALLEY_ITERS 5

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error at ", __FILE__, ":", __LINE__, " - ", cudaGetErrorString(err)); \
} while(0)

static_assert(BLOCK_N > 0 && BLOCK_M > 0, "Block dimensions must be positive");
static_assert(BLOCK_N * D_MAX <= 8192, "Shared memory K tile too large");
static_assert(BLOCK_M <= 1024, "BLOCK_M exceeds max threads per block");
static_assert(K_KEEP <= BLOCK_N, "K_KEEP should not exceed BLOCK_N");
static_assert(D_MAX >= 64, "D_MAX should be at least 64 for practical use");

__device__ inline void insert_topk(float val, int idx, float (&vals)[K_KEEP], int (&inds)[K_KEEP]) {
    int j = K_KEEP-1;
    while (j >= 0 && val > vals[j]) {
        if (j < K_KEEP-1) {
            vals[j+1] = vals[j];
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

__device__ void entmax_threshold(const float* s, int k, float alpha, float* p, float &tau, bool full) {
    alpha = fmaxf(1.05f, fminf(alpha, 3.0f));
    
    const float inv_am1 = 1.0f / (alpha - 1.0f);
    
    float s_max = s[0];
    float s_min = s[k-1];
    
    float lo = (alpha - 1.0f) * s_min - 2.0f;
    float hi = (alpha - 1.0f) * s_max + 2.0f;

    tau = 0.5f * (lo + hi);
    
    for (int it = 0; it < MAX_HALLEY_ITERS; ++it) {
        float f = -1.0f;
        float fp = 0.0f;
        float fpp = 0.0f;
        
        for (int j = 0; j < k; j++) {
            float u = (alpha - 1.0f) * s[j] - tau;
            if (u <= 0) break;
            
            float up = powf(fmaxf(u, EPS), inv_am1);
            
            if (isnan(up) || isinf(up)) {
                up = 0.0f;
            }
            
            f += up;
            if (full) {
                float derivative_term = -inv_am1 * powf(fmaxf(u, EPS), (2.0f - alpha) * inv_am1);
                if (isfinite(derivative_term)) {
                    fp += derivative_term;
                }
                float second_derivative_term = inv_am1 * (inv_am1 + 1.0f - alpha) * powf(fmaxf(u, EPS), (3.0f - 2*alpha) * inv_am1);
                if (isfinite(second_derivative_term)) {
                    fpp += second_derivative_term;
                }
            }
        }
        
        if (fabsf(f) < 1e-4f) break;
        
        float tn = tau;
        if (full && fabsf(fp) > EPS) {
            float denom = 2.0f * fp * fp - f * fpp;
            if (fabsf(denom) > EPS) {
                tn = tau - 2.0f * f * fp / denom;
            }
        }
        
        if (!(tn >= lo && tn <= hi)) {
            tn = 0.5f * (lo + hi);
        }
        
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
            lo = tn;
        } else {
            hi = tn;
        }
        tau = tn;
        
        tau = fmaxf(tau, s_min * (alpha - 1.0f) - 10.0f);
        tau = fminf(tau, s_max * (alpha - 1.0f) + 10.0f);
    }
    
    float norm = 0.0f;
    for (int j = 0; j < k; j++) {
        float u = (alpha - 1.0f) * s[j] - tau;
        float pj = (u > EPS) ? powf(fmaxf(u, EPS), inv_am1) : 0.0f;
        
        if (isnan(pj) || isinf(pj)) {
            pj = 0.0f;
        }
        
        p[j] = pj;
        norm += pj;
    }
    
    norm = fmaxf(norm, EPS);
    for (int j = 0; j < k; j++) {
        p[j] /= norm;
        
        if (isnan(p[j]) || isinf(p[j])) {
            p[j] = 1.0f / k;
        }
    }
}

__global__ void build_mask_kernel(
    const float* Q, const float* K, const int* Q_idx, const int* K_idx, 
    int B, int H, int NQ, int NK, int d, float alpha, float sm_scale, 
    int8_t* M, float* taus, int nQB, int nKB){
        int q = blockIdx.x*BLOCK_M + threadIdx.x;
        int bh = blockIdx.y;
        if (q>=NQ){
            return;
        }
        int idx = bh*NQ+q;
        int seq_q = Q_idx[idx];
        float q_reg[D_MAX];
        #pragma unroll
        for(int t=0;t<d;++t)
            q_reg[t] = Q[idx*d+t];
        
        float s_top[K_KEEP];
        int ind[K_KEEP];
        #pragma unroll
        for(int i=0;i<K_KEEP;++i){
            s_top[i] = -1e6f;
            ind[i] = -1;
        }
        extern __shared__ float shmem[];
        float* Ktile = shmem;
        for (int start = 0; start<NK; start+=BLOCK_N){
            int tile_id = threadIdx.x;
            for(int x = tile_id; x < BLOCK_N*d; x += BLOCK_M){
                int col = x/d;
                int dim = x%d;
                int kn = start + col;
                bool ok = (kn < NK && dim < d && K_idx[bh*NK + kn] <= seq_q);
                Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;
            }
        
            __syncthreads();

            for(int j = 0;j<BLOCK_N;j++){
                int kn = start + j;
                if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue;
                float sum = 0;
                #pragma unroll
                for(int t=0;t<d;t++)
                    sum+=q_reg[t]*Ktile[j*d+t];
                
                sum *=sm_scale;
                sum = fmaxf(sum, -10.0f);
                sum = fminf(sum, 10.0f);
                
                insert_topk(sum, kn, s_top, ind);
            }            
            __syncthreads();
        }
        
        float tau;
        float pbuf[K_KEEP];
    
        entmax_threshold(s_top, K_KEEP, alpha, pbuf, tau, true);
        taus[idx] = tau;
        
        int query_block = q/BLOCK_M;
        int base = bh*nQB*nKB + query_block*nKB;
        for(int i = 0;i<K_KEEP;i++){
            if(ind[i] < 0) continue;
            int jKB = ind[i]/BLOCK_N;
            M[base + jKB] = 1;
        }
    }

__global__ void build_lookup_kernel(const int8_t* M, int B, int H, int nQB, int nKB, int* Qi_ptr, int* Qi_idx){
    int bh = blockIdx.x;
    if(bh>=H*B) return;

    int baseM = bh * nQB * nKB;
    int bQi   = bh * (nQB + 1);
    int bQiI  = bh * (nQB * nKB);
    Qi_ptr[bQi] = 0;

    for(int i = 0;i<nQB;i++){
        int c = 0;
        for(int j =0;j<nKB;j++){
            c += M[baseM+i*nKB+j];
        }
        Qi_ptr[bQi+i+1] = Qi_ptr[bQi+i]+c;
    }
    for(int i = 0;i<nQB;i++){
        int w = Qi_ptr[bQi+i];
        for(int j =0;j<nKB;j++){
            if(M[baseM+i*nKB+j]){
                Qi_idx[bQiI+w++] = j;
            }
    }}
}

__global__ void splash_forward_sparse(
    const float* Q, const float* K, const float* V,
    const int* Q_idx, const int* K_idx,
    const float* taus,
    const int* Qi_ptr, const int* Qi_idx,
    float* Out,
    int B, int H, int NQ, int NK, int d,
    float alpha, float sm_scale,
    int nQB, int nKB
){

    int q = blockIdx.x*BLOCK_M + threadIdx.x;
    int bh = blockIdx.y;

    if(q>=NQ) return;

    int seq_q = Q_idx[bh*NQ + q];
    float tau = taus[bh*NQ + q];
    const float* Qptr = Q + ((bh*NQ + q)*d);

    float qreg[D_MAX];
    #pragma unroll
    for(int t=0;t<d;++t)
        qreg[t] = Qptr[t];
    extern __shared__ float sh[];
    float* Ktile = sh;
    float* Vtile = sh + BLOCK_N * d;

    float accum[D_MAX];
    float norm = 0.f;
    #pragma unroll
    for(int t=0; t<d; ++t)
        accum[t] = 0.f;

    int iQB = q/BLOCK_M;
    int offset = bh*(nQB+1) + iQB;
    int offsetI = bh*(nQB*nKB);

    for(int ptr = Qi_ptr[offset]; ptr < Qi_ptr[offset+1]; ptr++) {
        int jKB = Qi_idx[offsetI + ptr];
        int start = jKB * BLOCK_N;
        int tid = threadIdx.x;
        
        for(int x = tid; x < BLOCK_N*d; x += BLOCK_M) {
            int col = x/d;
            int dim = x%d;
            int kn = start + col;
            bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q);
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;
            Vtile[x] = ok ? V[(bh*NK + kn)*d + dim] : 0.f;
        }
        __syncthreads();

        for(int k = 0; k < BLOCK_N; ++k) {
            int kn = start + k;
            if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue;
            
            float s = 0.f;
            #pragma unroll
            for(int t=0; t<d; ++t) {
                s += qreg[t]*Ktile[k*d + t];
            }
            s *= sm_scale;
            
            s = fmaxf(s, -10.0f);
            s = fminf(s, 10.0f);
            
            float u = (alpha-1.f)*s - tau;
            float p = (u > 0) ? powf(fmaxf(u, EPS), 1.f/(alpha-1.f)) : 0.f;
            
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

    float invN = 1.f/(norm + EPS);
    float* OutPtr = Out + ((bh*NQ + q)*d);

    #pragma unroll
    for(int t=0; t<d; ++t) {
        float out_val = accum[t]*invN;
        
        if (isnan(out_val) || isinf(out_val)) {
            out_val = 0.0f;
        }
        
        OutPtr[t] = out_val;
    }
}

__global__ void splash_backward_sparse(
    const float* Q,
    const float* K,
    const float* V,
    const int* Q_idx,
    const int* K_idx,
    const float* taus,
    const int* Qi_ptr,
    const int* Qi_idx,
    const float* dOut,
    float* dQ,
    float* dK,
    float* dV,
    int B,
    int H,
    int NQ,
    int NK,
    int d,
    float alpha,
    float sm_scale,
    int nQB,
    int nKB
){
    int q = blockIdx.x*BLOCK_M + threadIdx.x;
    int bh = blockIdx.y;
    if(q>=NQ) return;

    int seq_q = Q_idx[bh*NQ + q];
    const float* Qptr = Q + ((bh*NQ + q)*d);
    float qreg[D_MAX], dqreg[D_MAX];
    #pragma unroll
    for(int t=0; t<d; ++t) {
        qreg[t] = Qptr[t];
        dqreg[t] = 0.f;
    }

    extern __shared__ float sh[];
    float* Ktile = sh;
    float* Vtile = sh+BLOCK_N*d;

    float accum[D_MAX];
    float norm = 0.f;
    #pragma unroll
    for(int t= 0;t<d;++t)
        accum[t] = 0.f;
    
    int iQB = q/BLOCK_M;
    int offset = bh*(nQB+1) + iQB;
    int offsetI = bh*(nQB*nKB);

    int start, tid;

    for (int ptr = Qi_ptr[offset]; ptr < Qi_ptr[offset+1]; ++ptr){
        int jKB = Qi_idx[offsetI + ptr];
        start = jKB*BLOCK_N;
        tid = threadIdx.x;
        for(int x = tid;x<BLOCK_N*d;x+=BLOCK_M){
            int col = x/d;
            int dim = x%d;
            int kn = start+col;
            bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q);
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;
            Vtile[x] = ok ? V[(bh*NK + kn)*d + dim] : 0.f;
        }
        __syncthreads();
        for(int k = 0;k<BLOCK_N;++k){
            int kn = start+k;
            if (kn >= NK || K_idx[bh*NK+kn] > seq_q) continue;
            float s= 0.f;
            #pragma unroll
            for(int t = 0;t<d;++t)
                s+=qreg[t]*Ktile[k*d+t];
            s*=sm_scale;
            
            s = fmaxf(s, -10.0f);
            s = fminf(s, 10.0f);
            
            float u=(alpha-1.f)*s - taus[bh*NQ + q];
            float p=(u>0.f? powf(fmaxf(u, EPS), 1.f/(alpha-1.f)) : 0.f);
            
            if (isnan(p) || isinf(p)) {
                p = 0.0f;
            }
            
            norm+=p;
            #pragma unroll
            for(int t = 0;t<d;++t)
                accum[t] += p*Vtile[k*d+t];

        }
        __syncthreads(); 
    }
    float invN = 1.f/(norm+EPS);
    float invN2 = -invN*invN;
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
            
            s = fmaxf(s, -10.0f);
            s = fminf(s, 10.0f);
            
            float u=(alpha-1.f)*s - taus[bh*NQ + q];
            float p=(u>0.f? powf(fmaxf(u, EPS), 1.f/(alpha-1.f)) : 0.f);
            
            if (isnan(p) || isinf(p)) {
                p = 0.0f;
            }
            
            float pb = p*invN;
            #pragma unroll
            for(int t =0;t<d;++t){
                float dv_contrib = pb * dOp[t];
                
                if (isfinite(dv_contrib)) {
                    atomicAdd(&dV[(bh*NK + kn)*d + t], dv_contrib);
                }
            }

            float dp=0.f;
            for(int t= 0;t<d;++t)
            {
                dp += Vtile[k*d+t] * dOp[t] * invN;
            }

            float accum_dot_dO = 0.f;
            for(int t= 0;t<d; ++t){
                accum_dot_dO += accum[t] * dOp[t];
            }
            dp+=accum_dot_dO*invN2;

            float grad_u = 0.0f;
            if (u > EPS && fabsf(alpha - 1.0f) > EPS) {
                float exponent = (2.f-alpha)/(alpha-1.f);
                float power_term = powf(fmaxf(u, EPS), exponent);
                
                if (isfinite(power_term)) {
                    grad_u = (1.f/(alpha-1.f)) * power_term * dp;
                }
            }
            
            float grad_s = grad_u * (alpha-1.f);
            
            if (!isfinite(grad_s)) {
                grad_s = 0.0f;
            }

            #pragma unroll
            for(int t= 0;t<d;++t){
                float dq_contrib = grad_s * Ktile[k*d+t] * sm_scale;
                if (isfinite(dq_contrib)) {
                    dqreg[t] += dq_contrib;
                }
                
                float dk_contrib = grad_s * qreg[t] * sm_scale;
                if (isfinite(dk_contrib)) {
                    atomicAdd(&dK[(bh*NK + kn)*d + t], dk_contrib);
                }
            }
        }
        __syncthreads();
    }
    float* dQp = dQ + ((bh*NQ + q)*d);
    #pragma unroll
    for(int t= 0;t<d;++t) {
        float dq_val = dqreg[t];
        
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
    TORCH_CHECK(Q.size(3) <= D_MAX, "Head dimension too large; increase D_MAX");
    TORCH_CHECK(Q.device().is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(K.device() == Q.device(), "All tensors must be on same device");
    TORCH_CHECK(V.device() == Q.device(), "All tensors must be on same device");
    
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

    int8_t* M_p = M.data_ptr<int8_t>();
    float* taus_p = taus.data_ptr<float>();
    int* Qidx_p = Q_idx.data_ptr<int>();
    int* Kidx_p = K_idx.data_ptr<int>();
    int32_t* Qi_pp = Qi_ptr.data_ptr<int32_t>();
    int32_t* Qi_ip = Qi_idx.data_ptr<int32_t>();
    float* Qp = Q.data_ptr<float>();
    float* Kp = K.data_ptr<float>();
    float* Vp = V.data_ptr<float>();
    float* Outp = Out.data_ptr<float>();

    dim3 grid1(nQB, B*H);
    dim3 block1(BLOCK_M);
    size_t shm1 = BLOCK_N * d * sizeof(float);
    
    build_mask_kernel<<<grid1, block1, shm1>>>(
        Qp, Kp, Qidx_p, Kidx_p, 
        B, H, NQ, NK, d, alpha, sm_scale, 
        M_p, taus_p, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());

    build_lookup_kernel<<<B*H, 32>>>(M_p, B, H, nQB, nKB, Qi_pp, Qi_ip);
    CUDA_CHECK(cudaGetLastError());

    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);
    splash_forward_sparse<<<grid1, block1, shm2>>>(Qp, Kp, Vp, Qidx_p, Kidx_p, taus_p, Qi_pp, Qi_ip, Outp, B, H, NQ, NK, d, alpha, sm_scale, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    
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
    
    build_mask_kernel<<<grid1, block1, shm1>>>(
        Qp, Kp, Qidx_p, Kidx_p, 
        B, H, NQ, NK, d, alpha, sm_scale, 
        M_p, taus_p, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());

    build_lookup_kernel<<<B*H, 32>>>(M_p, B, H, nQB, nKB, Qi_pp, Qi_ip);
    CUDA_CHECK(cudaGetLastError());
    
    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);
    splash_backward_sparse<<<grid1, block1, shm2>>>(Qp, Kp, Vp, Qidx_p, Kidx_p, taus_p, Qi_pp, Qi_ip, dOp, dQp, dKp, dVp, B, H, NQ, NK, d, alpha, sm_scale, nQB, nKB);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
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






    
