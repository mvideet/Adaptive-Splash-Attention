// splash_attention_full.cu — CUDA/C++ PyTorch extension with full AdaSplash-style sparse α-entmax
// dynamic block masking, causal support, shared K/V tiling, normalized forward, and full backward

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <float.h>

// === Tunables ===
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

#define EPS 1e-6f
#define MAX_HALLEY_ITERS 4

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error at ", __FILE__, ":", __LINE__, " - ", cudaGetErrorString(err)); \
} while(0)

// Compile-time safety checks
static_assert(BLOCK_N > 0 && BLOCK_M > 0, "Block dimensions must be positive");
static_assert(BLOCK_N * D_MAX <= 8192, "Shared memory K tile too large (max ~8KB per tile)");
static_assert(2 * BLOCK_N * D_MAX <= 32768, "Shared memory K+V tiles too large (max ~32KB total)");
static_assert(BLOCK_M <= 1024, "BLOCK_M exceeds typical max threads per block");
static_assert(K_KEEP <= BLOCK_N, "K_KEEP should not exceed BLOCK_N");
static_assert(D_MAX >= 64, "D_MAX should be at least 64 for practical use");

// In-register top-K insertion
__device__ inline void insert_topk(float val,int idx,float (&vals)[K_KEEP],int (&inds)[K_KEEP]){
    int j = K_KEEP - 1;
    while(j >= 0 && val > vals[j]){
        if(j < K_KEEP - 1){ vals[j+1] = vals[j]; inds[j+1] = inds[j]; }
        --j;
    }
    ++j;
    if(j < K_KEEP){ vals[j] = val; inds[j] = idx; }
}

// Entmax threshold and weights (Halley + bisection)
__device__ void entmax_threshold(const float* s,int k,float alpha,float* p,float &tau,bool full){
    const float inv_am1 = 1.f/(alpha-1.f);
    float lo = (alpha-1.f)*s[k-1] - 1.f;
    float hi = (alpha-1.f)*s[0];
    auto eval = [&](float t,float &f,float&fp,float&fpp){
        f = -1; fp = fpp = 0;
        for(int j=0;j<k;++j){
            float u = (alpha-1.f)*s[j] - t;
            if(u <= 0) break;
            float up = powf(u, inv_am1);
            f += up;
            if(full){
                fp  += -inv_am1 * powf(u, (2.f-alpha)*inv_am1);
                fpp += inv_am1*(inv_am1+1.f-alpha) * powf(u, (3.f-2*alpha)*inv_am1);
            }
        }
    };
    tau = 0.5f*(lo + hi);
    for(int it=0; it<MAX_HALLEY_ITERS; ++it){
        float f, fp, fpp;
        eval(tau, f, fp, fpp);
        if(fabsf(f) < 1e-3f) break;
        float tn = tau;
        if(full){
            float denom = 2.f*fp*fp - f*fpp;
            tn = tau - 2.f*f*fp / fmaxf(denom, EPS);
        }
        if(!(tn >= lo && tn <= hi)) tn = 0.5f*(lo + hi);
        eval(tn, f, fp, fpp);
        if(f > 0) lo = tn; else hi = tn;
        tau = tn;
    }
    float norm = 0;
    for(int j=0;j<k;++j){
        float u = (alpha-1.f)*s[j] - tau;
        float pj = (u > 0) ? powf(u, inv_am1) : 0;
        p[j] = pj;
        norm += pj;
    }
    norm = fmaxf(norm, EPS);
    for(int j=0;j<k;++j) p[j] /= norm;
}

// Kernel: build mask bits and per-query tau with causal masking, pads out-of-range keys
__global__ void build_mask_kernel(
    const float* Q, const float* K,
    const int* Q_idx, const int* K_idx,
    int B, int H, int NQ, int NK, int d,
    float alpha, float sm_scale,
    int8_t* M, float* taus,
    int nQB, int nKB
){
    int q = blockIdx.x*BLOCK_M + threadIdx.x;
    int bh = blockIdx.y;
    if(q >= NQ) return;
    int seq_q = Q_idx[bh*NQ + q];
    int idx = bh*NQ + q;

    // load Q
    float q_reg[D_MAX];
    #pragma unroll for(int t=0; t<d; ++t)
        q_reg[t] = Q[idx*d + t];

    // top-K buffer
    float s_top[K_KEEP]; int ind[K_KEEP];
    #pragma unroll for(int i=0; i<K_KEEP; ++i){ s_top[i] = -FLT_MAX; ind[i] = -1; }

    extern __shared__ float shmem[];
    float* Ktile = shmem;

    // scan K in tiles, skipping causal
    for(int start=0; start<NK; start+=BLOCK_N){
        // load tile
        int tid = threadIdx.x;
        for(int x = tid; x < BLOCK_N*d; x += BLOCK_M){
            int col = x / d, dim = x % d;
            int kn = start + col;
            bool ok = (kn < NK && dim < d && K_idx[bh*NK + kn] <= seq_q);
            // bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q);
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;
        }
        __syncthreads();
        // compute dot, prune
        for(int j=0; j<BLOCK_N; ++j){
            int kn = start + j;
            if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue;
            float acc = 0;
            #pragma unroll for(int t=0; t<d; ++t)
                acc += q_reg[t] * Ktile[j*d + t];
            acc *= sm_scale;
            insert_topk(acc, kn, s_top, ind);
        }
        __syncthreads();
    }
    // compute full entmax on top-K
    float pbuf[K_KEEP], tau;
    entmax_threshold(s_top, K_KEEP, alpha, pbuf, tau, true);
    taus[idx] = tau;

    // write mask bits
    int iQB = q / BLOCK_M;
    int base = bh*nQB*nKB + iQB*nKB;
    for(int i=0; i<K_KEEP; ++i){
        int kn = ind[i];
        if(kn < 0) continue;
        int jKB = kn / BLOCK_N;
        M[base + jKB] = 1;
    }
}

// Kernel: build lookup tables
__global__ void build_lookup_kernel(
    const int8_t* M,
    int B, int H, int nQB, int nKB,
    int* Qi_ptr, int* Qi_idx,
    int* Kj_ptr, int* Kj_idx
){
    int bh = blockIdx.x;
    if(bh >= B*H) return;
    int baseM = bh * nQB * nKB;
    int bQi   = bh * (nQB + 1);
    int bQiI  = bh * (nQB * nKB);
    int bKj   = bh * (nKB + 1);
    int bKjI  = bh * (nQB * nKB);

    Qi_ptr[bQi] = 0;
    for(int i=0;i<nQB;++i){
        int c=0;
        for(int j=0;j<nKB;++j) c += M[baseM + i*nKB + j];
        Qi_ptr[bQi + i + 1] = Qi_ptr[bQi + i] + c;
    }
    for(int i=0;i<nQB;++i){
        int w = Qi_ptr[bQi + i];
        for(int j=0;j<nKB;++j) if(M[baseM + i*nKB + j]) Qi_idx[bQiI + (w++)] = j;
    }
    Kj_ptr[bKj] = 0;
    for(int j=0;j<nKB;++j){
        int c=0;
        for(int i=0;i<nQB;++i) c += M[baseM + i*nKB + j];
        Kj_ptr[bKj + j + 1] = Kj_ptr[bKj + j] + c;
    }
    for(int j=0;j<nKB;++j){
        int w = Kj_ptr[bKj + j];
        for(int i=0;i<nQB;++i) if(M[baseM + i*nKB + j]) Kj_idx[bKjI + (w++)] = i;
    }
}

// Kernel: forward sparse with shared K/V tiling and normalization
__global__ void splash_forward_sparse(
    const float* Q, const float* K, const float* V,
    const int* Q_idx, const int* K_idx, const float* taus,
    const int* Qi_ptr, const int* Qi_idx,
    float* Out,
    int B, int H, int NQ, int NK, int d,
    float alpha, float sm_scale,
    int nQB, int nKB
){
    int q = blockIdx.x*BLOCK_M + threadIdx.x;
    int bh=blockIdx.y;
    if(q >= NQ) return;

    int seq_q = Q_idx[bh*NQ + q];
    float tau = taus[bh*NQ + q];
    const float* Qptr = Q + ((bh*NQ + q)*d);
    float qreg[D_MAX];
    #pragma unroll for(int t=0; t<d; ++t) qreg[t] = Qptr[t];

    extern __shared__ float sh[];
    float* Ktile = sh;
    float* Vtile = sh + BLOCK_N * d;

    float accum[D_MAX];
    #pragma unroll for(int t=0; t<d; ++t) accum[t] = 0.f;
    float norm = 0.f;

    int iQB = q / BLOCK_M;
    int off = bh*(nQB+1) + iQB;
    int offI= bh*(nQB*nKB);

    for(int ptr = Qi_ptr[off]; ptr < Qi_ptr[off+1]; ++ptr){
        int jKB = Qi_idx[offI + ptr];
        int start = jKB * BLOCK_N;
        int tid = threadIdx.x;
        for(int x=tid; x < BLOCK_N*d; x += BLOCK_M){
            int col = x / d, dim = x % d;
            int kn = start + col;
            bool ok = (kn < NK && K_idx[bh*NK + kn] <= seq_q);
            Ktile[x] = ok ? K[(bh*NK + kn)*d + dim] : 0.f;
            Vtile[x] = ok ? V[(bh*NK + kn)*d + dim] : 0.f;
        }
        __syncthreads();
        for(int k=0; k<BLOCK_N; ++k){
            int kn = start + k;
            if(kn >= NK || K_idx[bh*NK + kn] > seq_q) continue;
            float s=0.f;
            #pragma unroll for(int t=0; t<d; ++t) s+=qreg[t]*Ktile[k*d + t];
            s *= sm_scale;
            float u = (alpha-1.f)*s - tau;
            float p = (u > 0.f) ? powf(u, 1.f/(alpha-1.f)) : 0.f;
            norm += p;
            #pragma unroll for(int t=0; t<d; ++t) accum[t] += p * Vtile[k*d + t];
        }
        __syncthreads();
    }
    float invN = 1.f / (norm + EPS);
    float* OutPtr = Out + ((bh*NQ + q)*d);
    #pragma unroll for(int t=0; t<d; ++t) OutPtr[t] = accum[t] * invN;
}

// Kernel: backward sparse with causal, shared tiling, entmax gradients
__global__ void splash_backward_sparse(
    const float* Q,const float* K,const float* V,
    const int* Q_idx,const int* K_idx,const float* taus,
    const int* Qi_ptr,const int* Qi_idx,const float* dOut,
    float* dQ,float* dK,float* dV,
    int B,int H,int NQ,int NK,int d,
    float alpha,float sm_scale,int nQB,int nKB
){
    int q = blockIdx.x*BLOCK_M + threadIdx.x;
    int bh = blockIdx.y;
    if(q >= NQ) return;
    int seq_q = Q_idx[bh*NQ + q];
    const float* Qptr = Q + ((bh*NQ + q)*d);
    float qreg[D_MAX], dqreg[D_MAX];
    #pragma unroll for(int t=0; t<d; ++t){ qreg[t]=Qptr[t]; dqreg[t]=0.f; }
    extern __shared__ float sh[];
    float* Ktile = sh;
    float* Vtile = sh + BLOCK_N*d;
    float accum[D_MAX]; float norm=0.f;
    #pragma unroll for(int t=0; t<d; ++t) accum[t]=0.f;
    int iQB=q/BLOCK_M;
    int off = bh*(nQB+1) + iQB;
    int offI= bh*(nQB*nKB);
    // first pass: compute pbuf, accum, norm
    for(int ptr=Qi_ptr[off]; ptr<Qi_ptr[off+1]; ++ptr){
        int jKB=Qi_idx[offI+ptr];
        int start=jKB*BLOCK_N;
        int tid=threadIdx.x;
        for(int x=tid; x<BLOCK_N*d; x+=BLOCK_M){
            int col=x/d, dim=x%d, kn=start+col;
            bool ok=(kn<NK && K_idx[bh*NK+kn]<=seq_q);
            Ktile[x]= ok? K[(bh*NK+kn)*d+dim] : 0.f;
            Vtile[x]= ok? V[(bh*NK+kn)*d+dim] : 0.f;
        }
        __syncthreads();
        for(int k=0; k<BLOCK_N; ++k){
            int kn=start+k;
            if(kn>=NK || K_idx[bh*NK+kn]>seq_q) continue;
            float s=0;
            #pragma unroll for(int t=0; t<d; ++t) s+=qreg[t]*Ktile[k*d + t];
            s *= sm_scale;
            float u=(alpha-1.f)*s - taus[bh*NQ + q];
            float p=(u>0.f? powf(u,1.f/(alpha-1.f)) : 0.f);
            norm+=p;
            #pragma unroll for(int t=0; t<d; ++t) accum[t]+=p * Vtile[k*d + t];
        }
        __syncthreads();
    }
    float invN=1.f/(norm+EPS), invN2=-invN*invN;
    const float* dOp = dOut + ((bh*NQ + q)*d);
    // second pass: gradients
    for(int ptr=Qi_ptr[off]; ptr<Qi_ptr[off+1]; ++ptr){
        int jKB=Qi_idx[offI+ptr], start=jKB*BLOCK_N;
        // reload K/V tiles for this block
        int tid=threadIdx.x;
        for(int x=tid; x<BLOCK_N*d; x+=BLOCK_M){
            int col=x/d, dim=x%d, kn=start+col;
            bool ok=(kn<NK && K_idx[bh*NK+kn]<=seq_q);
            Ktile[x]= ok? K[(bh*NK+kn)*d+dim] : 0.f;
            Vtile[x]= ok? V[(bh*NK+kn)*d+dim] : 0.f;
        }
        __syncthreads();
        for(int k=0; k<BLOCK_N; ++k){
            int kn=start+k;
            if(kn>=NK || K_idx[bh*NK+kn]>seq_q) continue;
            // recompute p for this k
            float s=0;
            #pragma unroll for(int t=0; t<d; ++t) s+=qreg[t]*Ktile[k*d + t];
            s *= sm_scale;
            float u=(alpha-1.f)*s - taus[bh*NQ + q];
            float p=(u>0.f? powf(u,1.f/(alpha-1.f)) : 0.f);
            // grad to V
            float pb = p * invN;
            for(int t=0; t<d; ++t)
                atomicAdd(&dV[(bh*NK + kn)*d + t], pb * dOp[t]);
            // grad to p
            float dp=0;
            for(int t=0; t<d; ++t) dp += Vtile[k*d + t] * dOp[t] * invN;
            float accum_dot_dO = 0;
            for(int t=0; t<d; ++t) accum_dot_dO += accum[t] * dOp[t];
            dp += accum_dot_dO * invN2;
            // chain through u->p (s, u, p already computed above)
            float grad_u = (u>0.f? (1.f/(alpha-1.f))*powf(u,(2.f-alpha)/(alpha-1.f))*dp : 0.f);
            float grad_s = grad_u * (alpha-1.f);
            // dQ, dK
            for(int t=0; t<d; ++t){
                dqreg[t] += grad_s * Ktile[k*d + t] * sm_scale;
                atomicAdd(&dK[(bh*NK + kn)*d + t], grad_s * qreg[t] * sm_scale);
            }
        }
        __syncthreads();
    }
    // write dQ
    float* dQp = dQ + ((bh*NQ + q)*d);
    #pragma unroll for(int t=0; t<d; ++t) dQp[t] = dqreg[t];
}

// Host wrapper: forward_splash
torch::Tensor forward_splash(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Q_idx, torch::Tensor K_idx,
    float sm_scale, float alpha
) {
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
    auto opts8 = torch::TensorOptions().dtype(torch::kInt8).device(Q.device());
    auto optsF = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto M = torch::zeros({B*H, nQB, nKB}, opts8);
    auto taus = torch::zeros({B*H, NQ}, optsF);
    auto Qi_ptr = torch::zeros({B*H, nQB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Qi_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Kj_ptr = torch::zeros({B*H, nKB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Kj_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Out = torch::zeros_like(Q);

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
    float* Outp = Out.data_ptr<float>();

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

    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);
    splash_forward_sparse<<<grid1, block1, shm2>>>(
        Qp, Kp, Vp,
        Qidx_p, Kidx_p, taus_p,
        Qi_pp, Qi_ip,
        Outp,
        B, H, NQ, NK, d,
        alpha, sm_scale,
        nQB, nKB
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return Out;
}

// Host wrapper: backward_splash
std::vector<torch::Tensor> backward_splash(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Q_idx, torch::Tensor K_idx,
    float sm_scale, float alpha,
    torch::Tensor dOut
) {
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

    auto opts8 = torch::TensorOptions().dtype(torch::kInt8).device(Q.device());
    auto optsF = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    auto M = torch::zeros({B*H, nQB, nKB}, opts8);
    auto taus = torch::zeros({B*H, NQ}, optsF);
    auto Qi_ptr = torch::zeros({B*H, nQB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Qi_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Kj_ptr = torch::zeros({B*H, nKB+1}, torch::dtype(torch::kInt32).device(Q.device()));
    auto Kj_idx = torch::zeros({B*H, nQB*nKB}, torch::dtype(torch::kInt32).device(Q.device()));
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

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
    float* dOp = dOut.data_ptr<float>();
    float* dQp = dQ.data_ptr<float>();
    float* dKp = dK.data_ptr<float>();
    float* dVp = dV.data_ptr<float>();

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

    size_t shm2 = 2 * BLOCK_N * d * sizeof(float);
    splash_backward_sparse<<<grid1, block1, shm2>>>(
        Qp, Kp, Vp,
        Qidx_p, Kidx_p, taus_p,
        Qi_pp, Qi_ip, dOp,
        dQp, dKp, dVp,
        B, H, NQ, NK, d,
        alpha, sm_scale,
        nQB, nKB
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return {dQ, dK, dV};
}

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_splash, "Splash attention forward");
    m.def("backward", &backward_splash, "Splash attention backward");
}
