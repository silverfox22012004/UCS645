/*
 * ============================================================
 * Part 4 — Tiled Matrix Multiply & ConvNet Layer Primitives
 * ============================================================
 * FOCUS       : GEMM tiling, cuBLAS benchmarks, Conv2D, MaxPool,
 *               BatchNorm
 * CUDA LEVEL  : 12.x
 *
 * Build (needs cuBLAS):
 *   nvcc -O2 -arch=sm_86 part4_matmul_and_convnets.cu -o part4 -lcublas
 * Execute:
 *   ./part4
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define GPU_SAFE(call)                                                      \
    do {                                                                    \
        cudaError_t status = (call);                                        \
        if (status != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA failure at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(status));        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define BLAS_SAFE(call)                                                     \
    do {                                                                    \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS failure at %s:%d — code %d\n",         \
                    __FILE__, __LINE__, (int)st);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define TILE_SZ 16

int arrays_equal(const float* a, const float* b, int len, float tol)
{
    for (int i = 0; i < len; i++)
        if (fabsf(a[i] - b[i]) > tol) return 0;
    return 1;
}

float elapsed_ms(cudaEvent_t s, cudaEvent_t e) { float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms; }

/* ================================================================
 * PART A — REFERENCE: Naive GEMM
 * ================================================================ */
__global__ void naiveGemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M || c >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += A[r * K + k] * B[k * N + c];
    C[r * N + c] = acc;
}

void launch_naive_gemm(float* dA, float* dB, float* dC, int M, int N, int K, float* ms)
{
    dim3 blk(TILE_SZ, TILE_SZ);
    dim3 grd((N + TILE_SZ - 1) / TILE_SZ, (M + TILE_SZ - 1) / TILE_SZ);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    naiveGemm<<<grd, blk>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    if (ms) *ms = elapsed_ms(e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

/* ================================================================
 * PART B — EXERCISE: Tiled GEMM
 * ================================================================ */
__global__ void tiledGemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float sA[TILE_SZ][TILE_SZ];
    __shared__ float sB[TILE_SZ][TILE_SZ];
    int r = blockIdx.y * TILE_SZ + threadIdx.y;
    int c = blockIdx.x * TILE_SZ + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE_SZ - 1) / TILE_SZ; t++) {
        sA[threadIdx.y][threadIdx.x] = (r < M && t * TILE_SZ + threadIdx.x < K) ? A[r * K + t * TILE_SZ + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (c < N && t * TILE_SZ + threadIdx.y < K) ? B[(t * TILE_SZ + threadIdx.y) * N + c] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SZ; k++) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (r < M && c < N) C[r * N + c] = acc;
}

void exercise_tiled_gemm(int M, int N, int K)
{
    size_t bA = (size_t)M * K * sizeof(float), bB = (size_t)K * N * sizeof(float), bC = (size_t)M * N * sizeof(float);
    float *hA = (float*)malloc(bA), *hB = (float*)malloc(bB), *hC = (float*)malloc(bC), *hRef = (float*)malloc(bC);
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int r = 0; r < M; r++) for (int c = 0; c < N; c++) {
        float s = 0; for (int k = 0; k < K; k++) s += hA[r * K + k] * hB[k * N + c]; hRef[r * N + c] = s;
    }
    float *dA, *dB, *dC;
    GPU_SAFE(cudaMalloc(&dA, bA)); GPU_SAFE(cudaMalloc(&dB, bB)); GPU_SAFE(cudaMalloc(&dC, bC));
    GPU_SAFE(cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice)); GPU_SAFE(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));
    dim3 blk(TILE_SZ, TILE_SZ), grd((N + TILE_SZ - 1) / TILE_SZ, (M + TILE_SZ - 1) / TILE_SZ);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); tiledGemm<<<grd, blk>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = elapsed_ms(e0, e1);
    GPU_SAFE(cudaMemcpy(hC, dC, bC, cudaMemcpyDeviceToHost));
    int pass = arrays_equal(hC, hRef, M * N, 5e-2f);
    double gf = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    printf("  [B1-TiledGemm] %dx%d@%dx%d  %.2f ms  %.1f GFLOPS  %s\n", M, K, K, N, ms, gf, pass ? "[PASS]" : "[FAIL]");
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
}

/* ----- B2. GEMM benchmark: naive vs tiled vs cuBLAS ----------- */
void exercise_gemm_benchmark(cublasHandle_t handle)
{
    int dims[] = {128, 256, 512, 1024}; int nd = 4;
    printf("\n  [B2-GemmBench]\n  %6s  %12s  %12s  %12s  %10s\n", "Size", "Naive(ms)", "Tiled(ms)", "cuBLAS(ms)", "cuBLAS GFLOPS");
    printf("  %s\n", "--------------------------------------------------------------");
    for (int s = 0; s < nd; s++) {
        int D = dims[s]; size_t nb = (size_t)D * D * sizeof(float);
        float *dA, *dB, *dC;
        GPU_SAFE(cudaMalloc(&dA, nb)); GPU_SAFE(cudaMalloc(&dB, nb)); GPU_SAFE(cudaMalloc(&dC, nb));
        GPU_SAFE(cudaMemset(dA, 0, nb)); GPU_SAFE(cudaMemset(dB, 0, nb));
        float nm = 0, tm = 0, cm = 0;
        launch_naive_gemm(dA, dB, dC, D, D, D, &nm);
        dim3 blk(TILE_SZ, TILE_SZ), grd((D + TILE_SZ - 1) / TILE_SZ, (D + TILE_SZ - 1) / TILE_SZ);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); tiledGemm<<<grd, blk>>>(dA, dB, dC, D, D, D);
        cudaEventRecord(e1); cudaEventSynchronize(e1); tm = elapsed_ms(e0, e1);
        float alpha = 1.0f, beta = 0.0f;
        cudaEventRecord(e0);
        BLAS_SAFE(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, D, D, &alpha, dB, D, dA, D, &beta, dC, D));
        cudaEventRecord(e1); cudaEventSynchronize(e1); cm = elapsed_ms(e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        double gf = 2.0 * D * D * D / (cm / 1000.0 + 1e-9) / 1e9;
        printf("  %6d  %12.2f  %12.2f  %12.2f  %10.1f\n", D, nm, tm, cm, gf);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

/* ================================================================
 * PART C — EXERCISE: CNN Layer Kernels
 * ================================================================ */

/* ----- C1. Max Pooling 2×2, stride 2 -------------------------- */
__global__ void maxPool2x2Kernel(const float* inp, float* out, int N, int C, int H, int W)
{
    int Ho = H / 2, Wo = W / 2;
    int n = blockIdx.z, c = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y, ow = threadIdx.x;
    if (oh >= Ho || ow >= Wo || n >= N || c >= C) return;
    float mx = -1e30f;
    for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        int iy = oh * 2 + dy, ix = ow * 2 + dx;
        mx = fmaxf(mx, inp[((n * C + c) * H + iy) * W + ix]);
    }
    out[((n * C + c) * Ho + oh) * Wo + ow] = mx;
}

void cpu_maxpool(const float* in, float* out, int N, int C, int H, int W)
{
    int Ho = H / 2, Wo = W / 2;
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++)
        for (int oh = 0; oh < Ho; oh++) for (int ow = 0; ow < Wo; ow++) {
            float mx = -1e30f;
            for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                float v = in[((n * C + c) * H + oh * 2 + dy) * W + ow * 2 + dx];
                if (v > mx) mx = v;
            }
            out[((n * C + c) * Ho + oh) * Wo + ow] = mx;
        }
}

void exercise_maxpool(void)
{
    int N = 4, C = 8, H = 16, W = 16, Ho = 8, Wo = 8;
    size_t ib = (size_t)N * C * H * W * sizeof(float), ob = (size_t)N * C * Ho * Wo * sizeof(float);
    float *hI = (float*)malloc(ib), *hO = (float*)malloc(ob), *hR = (float*)malloc(ob);
    for (int i = 0; i < N * C * H * W; i++) hI[i] = (float)rand() / RAND_MAX;
    cpu_maxpool(hI, hR, N, C, H, W);
    float *dI, *dO;
    GPU_SAFE(cudaMalloc(&dI, ib)); GPU_SAFE(cudaMalloc(&dO, ob));
    GPU_SAFE(cudaMemcpy(dI, hI, ib, cudaMemcpyHostToDevice)); GPU_SAFE(cudaMemset(dO, 0, ob));
    dim3 blk(Wo, 2), grd((Ho + 1) / 2, C, N);
    maxPool2x2Kernel<<<grd, blk>>>(dI, dO, N, C, H, W);
    GPU_SAFE(cudaMemcpy(hO, dO, ob, cudaMemcpyDeviceToHost));
    printf("  [C1-MaxPool2x2] (%d,%d,%d,%d)->(%d,%d,%d,%d)  %s\n", N, C, H, W, N, C, Ho, Wo,
           arrays_equal(hO, hR, N * C * Ho * Wo, 1e-5f) ? "[PASS]" : "[FAIL]");
    cudaFree(dI); cudaFree(dO); free(hI); free(hO); free(hR);
}

/* ----- C2. Batch Normalization (inference) -------------------- */
__global__ void bnInference(const float* x, float* y, const float* gamma, const float* beta,
                            const float* mu, const float* sigma2, int N, int C, int HW, float eps)
{
    int c = blockIdx.y, hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= HW || c >= C) return;
    for (int n = 0; n < N; n++) {
        int idx = (n * C + c) * HW + hw;
        float xn = (x[idx] - mu[c]) / sqrtf(sigma2[c] + eps);
        y[idx] = gamma[c] * xn + beta[c];
    }
}

void exercise_batchnorm(void)
{
    int N = 4, C = 8, H = 16, W = 16, HW = H * W; float eps = 1e-5f;
    size_t fb = (size_t)N * C * HW * sizeof(float), cb = C * sizeof(float);
    float *hX = (float*)malloc(fb), *hO = (float*)malloc(fb), *hRef = (float*)malloc(fb);
    float *hG = (float*)malloc(cb), *hB = (float*)malloc(cb), *hMu = (float*)malloc(cb), *hVar = (float*)malloc(cb);
    for (int i = 0; i < N * C * HW; i++) hX[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int c = 0; c < C; c++) {
        hG[c] = 1.0f; hB[c] = 0.0f;
        double s1 = 0, s2 = 0;
        for (int n = 0; n < N; n++) for (int hw = 0; hw < HW; hw++) { float v = hX[(n * C + c) * HW + hw]; s1 += v; s2 += v * v; }
        hMu[c] = (float)(s1 / (N * HW)); hVar[c] = (float)(s2 / (N * HW) - hMu[c] * hMu[c]);
        for (int n = 0; n < N; n++) for (int hw = 0; hw < HW; hw++) {
            int idx = (n * C + c) * HW + hw;
            hRef[idx] = hG[c] * ((hX[idx] - hMu[c]) / sqrtf(hVar[c] + eps)) + hB[c];
        }
    }
    float *dX, *dO, *dG, *dBt, *dMu, *dVar;
    GPU_SAFE(cudaMalloc(&dX, fb)); GPU_SAFE(cudaMalloc(&dO, fb));
    GPU_SAFE(cudaMalloc(&dG, cb)); GPU_SAFE(cudaMalloc(&dBt, cb));
    GPU_SAFE(cudaMalloc(&dMu, cb)); GPU_SAFE(cudaMalloc(&dVar, cb));
    GPU_SAFE(cudaMemcpy(dX, hX, fb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dG, hG, cb, cudaMemcpyHostToDevice)); GPU_SAFE(cudaMemcpy(dBt, hB, cb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dMu, hMu, cb, cudaMemcpyHostToDevice)); GPU_SAFE(cudaMemcpy(dVar, hVar, cb, cudaMemcpyHostToDevice));
    bnInference<<<dim3((HW + 255) / 256, C), 256>>>(dX, dO, dG, dBt, dMu, dVar, N, C, HW, eps);
    GPU_SAFE(cudaMemcpy(hO, dO, fb, cudaMemcpyDeviceToHost));
    printf("  [C2-BatchNorm] (%d,%d,%d,%d)  %s\n", N, C, H, W, arrays_equal(hO, hRef, N * C * HW, 1e-4f) ? "[PASS]" : "[FAIL]");
    cudaFree(dX); cudaFree(dO); cudaFree(dG); cudaFree(dBt); cudaFree(dMu); cudaFree(dVar);
    free(hX); free(hO); free(hRef); free(hG); free(hB); free(hMu); free(hVar);
}

/* ================================================================
 * PART D — STRETCH: Direct Conv2D
 * ================================================================ */
__global__ void directConv2d(const float* inp, const float* filt, float* out,
                             int N, int Ci, int H, int W, int Co, int kH, int kW,
                             int pH, int pW, int sH, int sW)
{
    int Ho = (H + 2 * pH - kH) / sH + 1, Wo = (W + 2 * pW - kW) / sW + 1;
    int n = blockIdx.z, oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y, ow = threadIdx.x;
    if (oh >= Ho || ow >= Wo || n >= N || oc >= Co) return;
    float acc = 0.0f;
    for (int ic = 0; ic < Ci; ic++) for (int kh = 0; kh < kH; kh++) for (int kw = 0; kw < kW; kw++) {
        int ih = oh * sH - pH + kh, iw = ow * sW - pW + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            acc += inp[((n * Ci + ic) * H + ih) * W + iw] * filt[((oc * Ci + ic) * kH + kh) * kW + kw];
    }
    out[((n * Co + oc) * Ho + oh) * Wo + ow] = acc;
}

void stretch_conv2d_test(void)
{
    int N = 2, Ci = 1, H = 8, W = 8, Co = 4, kH = 3, kW = 3, p = 1, s = 1;
    int Ho = (H + 2 * p - kH) / s + 1, Wo = (W + 2 * p - kW) / s + 1;
    int ni = N * Ci * H * W, nf = Co * Ci * kH * kW, no = N * Co * Ho * Wo;
    float *hI = (float*)calloc(ni, sizeof(float)), *hF = (float*)calloc(nf, sizeof(float)), *hO = (float*)calloc(no, sizeof(float));
    for (int i = 0; i < ni; i++) hI[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < nf; i++) hF[i] = (float)rand() / RAND_MAX;
    float *dI, *dF, *dO;
    GPU_SAFE(cudaMalloc(&dI, ni * sizeof(float))); GPU_SAFE(cudaMalloc(&dF, nf * sizeof(float))); GPU_SAFE(cudaMalloc(&dO, no * sizeof(float)));
    GPU_SAFE(cudaMemcpy(dI, hI, ni * sizeof(float), cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dF, hF, nf * sizeof(float), cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dO, 0, no * sizeof(float)));
    directConv2d<<<dim3((Ho + 3) / 4, Co, N), dim3(Wo, 4)>>>(dI, dF, dO, N, Ci, H, W, Co, kH, kW, p, p, s, s);
    GPU_SAFE(cudaDeviceSynchronize());
    GPU_SAFE(cudaMemcpy(hO, dO, no * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0; for (int i = 0; i < no; i++) total += hO[i];
    printf("  [D1-Conv2D] Ho=%d Wo=%d  sum=%.2f  %s\n", Ho, Wo, total, total > 0.0f ? "[PASS]" : "[FAIL]");
    cudaFree(dI); cudaFree(dF); cudaFree(dO); free(hI); free(hF); free(hO);
}

/* ================================================================
 * ENTRY POINT
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  Part 4: Tiled GEMM & ConvNet Layers\n");
    printf("========================================================\n");
    cudaDeviceProp dp; GPU_SAFE(cudaGetDeviceProperties(&dp, 0));
    int clk = 0; GPU_SAFE(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    double est = 2.0 * dp.multiProcessorCount * 128.0 * clk * 1e-9;
    printf("  GPU: %s  Peak TFLOPS (FP32 est.): ~%.2f\n\n", dp.name, est);

    cublasHandle_t handle; BLAS_SAFE(cublasCreate(&handle));

    printf("[Part A] Reference: Naive GEMM:\n");
    { int D = 256; float *dA, *dB, *dC; float ms;
      GPU_SAFE(cudaMalloc(&dA, D*D*sizeof(float))); GPU_SAFE(cudaMalloc(&dB, D*D*sizeof(float))); GPU_SAFE(cudaMalloc(&dC, D*D*sizeof(float)));
      launch_naive_gemm(dA, dB, dC, D, D, D, &ms);
      printf("  Naive %dx%d@%dx%d  %.2f ms  %.1f GFLOPS\n", D, D, D, D, ms, 2.0*D*D*D/(ms/1000.0)/1e9);
      cudaFree(dA); cudaFree(dB); cudaFree(dC); }

    printf("\n[Part B] Tiled GEMM:\n");
    exercise_tiled_gemm(512, 512, 512);
    exercise_gemm_benchmark(handle);

    printf("\n[Part C] CNN Layers:\n");
    exercise_maxpool();
    exercise_batchnorm();

    printf("\n[Part D] Stretch — Direct Conv2D:\n");
    stretch_conv2d_test();

    cublasDestroy(handle);
    printf("\n========================================================\n");
    printf("  All [PASS] → proceed to Part 5!\n");
    printf("========================================================\n\n");
    return 0;
}
