/*
 * ============================================================
 * Part 1 — GPU Fundamentals: Vector & Element-wise Operations
 * ============================================================
 * FOCUS       : GPU threading model, kernel dispatch, device memory
 * CUDA LEVEL  : 12.x
 *
 * Goals:
 *   1. Grasp the Thread → Block → Grid mapping
 *   2. Author and dispatch simple CUDA kernels
 *   3. Handle GPU memory lifecycle (alloc / transfer / free)
 *   4. Profile kernel execution via CUDA Events
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 part1_gpu_fundamentals.cu -o part1
 *   (swap sm_86 for your card's CC, e.g. sm_75 or sm_89)
 *
 * Execute:
 *   ./part1
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ── Wrapper macro for runtime error checking ── */
#define GPU_SAFE(call)                                                      \
    do {                                                                    \
        cudaError_t status = (call);                                        \
        if (status != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA failure at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(status));        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define BLOCK_DIM 256           /* threads per block */
#define NUM_ELEMS (1 << 20)     /* ~1 million elements */


/* ================================================================
 * PART A — REFERENCE IMPLEMENTATIONS (supplied as-is)
 *   Read through these before attempting the exercises below.
 * ================================================================ */

/* ----- A1. Pairwise vector sum (provided) --------------------- */
__global__ void pairwiseSum(const float* X, const float* Y, float* Z, int len)
{
    /* Every thread handles a single position */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
        Z[idx] = X[idx] + Y[idx];
}

/* ----- A2. Sequential baseline -------------------------------- */
void hostPairwiseSum(const float* X, const float* Y, float* Z, int len)
{
    for (int i = 0; i < len; i++)
        Z[i] = X[i] + Y[i];
}

/* Monotonic wall-clock timer (returns milliseconds) */
static double timer_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* Element-wise comparison within a tolerance */
int arrays_match(const float* a, const float* b, int len, float tol)
{
    for (int i = 0; i < len; i++)
        if (fabsf(a[i] - b[i]) > tol) return 0;
    return 1;
}

/* Benchmark the provided pairwise sum kernel */
void benchmark_pairwise_sum(int len)
{
    size_t nbytes = len * sizeof(float);

    /* Host-side buffers */
    float *hX = (float*)malloc(nbytes);
    float *hY = (float*)malloc(nbytes);
    float *hZ = (float*)malloc(nbytes);
    float *hZ_ref = (float*)malloc(nbytes);

    /* Fill with random values */
    for (int i = 0; i < len; i++) {
        hX[i] = (float)rand() / RAND_MAX;
        hY[i] = (float)rand() / RAND_MAX;
    }

    /* Sequential reference */
    double t_start = timer_ms();
    hostPairwiseSum(hX, hY, hZ_ref, len);
    double cpu_time = timer_ms() - t_start;

    /* Device-side buffers */
    float *dX, *dY, *dZ;
    GPU_SAFE(cudaMalloc(&dX, nbytes));
    GPU_SAFE(cudaMalloc(&dY, nbytes));
    GPU_SAFE(cudaMalloc(&dZ, nbytes));

    /* Transfer to device */
    GPU_SAFE(cudaMemcpy(dX, hX, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dY, hY, nbytes, cudaMemcpyHostToDevice));

    /* Grid/block dimensions */
    int tpb = BLOCK_DIM;
    int nblocks = (len + tpb - 1) / tpb;

    /* Event-based GPU timing */
    cudaEvent_t ev_begin, ev_end;
    GPU_SAFE(cudaEventCreate(&ev_begin));
    GPU_SAFE(cudaEventCreate(&ev_end));

    GPU_SAFE(cudaEventRecord(ev_begin));
    pairwiseSum<<<nblocks, tpb>>>(dX, dY, dZ, len);
    GPU_SAFE(cudaEventRecord(ev_end));
    GPU_SAFE(cudaEventSynchronize(ev_end));

    float gpu_time = 0.0f;
    GPU_SAFE(cudaEventElapsedTime(&gpu_time, ev_begin, ev_end));

    /* Retrieve result */
    GPU_SAFE(cudaMemcpy(hZ, dZ, nbytes, cudaMemcpyDeviceToHost));

    int pass = arrays_match(hZ, hZ_ref, len, 1e-4f);
    printf("  [A1-PairwiseSum] N=%d  CPU=%.1f ms  GPU=%.2f ms  Speedup=%.1fx  %s\n",
           len, cpu_time, gpu_time, cpu_time / gpu_time, pass ? "[PASS]" : "[FAIL]");

    /* Release resources */
    cudaFree(dX); cudaFree(dY); cudaFree(dZ);
    cudaEventDestroy(ev_begin); cudaEventDestroy(ev_end);
    free(hX); free(hY); free(hZ); free(hZ_ref);
}


/* ================================================================
 * PART B — HANDS-ON KERNELS
 *   Complete every TODO section.  Guidance is in the comments.
 * ================================================================ */

/* ----- B1. Scalar multiplication kernel -----------------------
 * Purpose : Scale every element by a constant k → R[i] = k * A[i]
 * Context : Mirrors learning-rate scaling during gradient descent.
 * ---------------------------------------------------------------- */
__global__ void scalarMul(const float* A, float* R, float k, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        /* ═══════════════════════════════════════════════════════
         * TODO (B1): Perform the scaling operation.
         *   HINT: R[idx] = k * A[idx];
         * ═══════════════════════════════════════════════════════ */
        R[idx] = A[idx] * k;
    }
}

void exercise_scalar_mul(int len)
{
    size_t nbytes = len * sizeof(float);
    float k = 3.14f;

    float *hA   = (float*)malloc(nbytes);
    float *hR   = (float*)malloc(nbytes);
    float *hRef = (float*)malloc(nbytes);
    for (int i = 0; i < len; i++) { hA[i] = (float)rand() / RAND_MAX; hRef[i] = hA[i] * k; }

    float *dA, *dR;
    GPU_SAFE(cudaMalloc(&dA, nbytes));
    GPU_SAFE(cudaMalloc(&dR, nbytes));
    GPU_SAFE(cudaMemcpy(dA, hA, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dR, 0, nbytes));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    scalarMul<<<nblk, tpb>>>(dA, dR, k, len);
    GPU_SAFE(cudaMemcpy(hR, dR, nbytes, cudaMemcpyDeviceToHost));

    int pass = arrays_match(hR, hRef, len, 1e-4f);
    printf("  [B1-ScalarMul] k=%.2f  %s\n", k, pass ? "[PASS]" : "[FAIL] -- check your kernel");

    cudaFree(dA); cudaFree(dR);
    free(hA); free(hR); free(hRef);
}


/* ----- B2. Element-wise squared error -------------------------
 * Purpose : R[i] = (A[i] - B[i])²
 * Context : Core of the mean squared error (MSE) loss function.
 * ---------------------------------------------------------------- */
__global__ void squaredError(const float* A, const float* B, float* R, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        /* ═══════════════════════════════════════════════════════
         * TODO (B2): Compute per-element squared error.
         *   HINT:
         *     float delta = A[idx] - B[idx];
         *     R[idx] = delta * delta;
         * ═══════════════════════════════════════════════════════ */
        float delta = A[idx] - B[idx];
        R[idx] = delta * delta;
    }
}

void exercise_squared_error(int len)
{
    size_t nbytes = len * sizeof(float);
    float *hA   = (float*)malloc(nbytes);
    float *hB   = (float*)malloc(nbytes);
    float *hR   = (float*)malloc(nbytes);
    float *hRef = (float*)malloc(nbytes);
    for (int i = 0; i < len; i++) {
        hA[i] = (float)rand() / RAND_MAX;
        hB[i] = (float)rand() / RAND_MAX;
        float d = hA[i] - hB[i];
        hRef[i] = d * d;
    }

    float *dA, *dB, *dR;
    GPU_SAFE(cudaMalloc(&dA, nbytes));
    GPU_SAFE(cudaMalloc(&dB, nbytes));
    GPU_SAFE(cudaMalloc(&dR, nbytes));
    GPU_SAFE(cudaMemcpy(dA, hA, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dB, hB, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dR, 0, nbytes));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    squaredError<<<nblk, tpb>>>(dA, dB, dR, len);
    GPU_SAFE(cudaMemcpy(hR, dR, nbytes, cudaMemcpyDeviceToHost));

    int pass = arrays_match(hR, hRef, len, 1e-4f);
    printf("  [B2-SquaredError] %s\n", pass ? "[PASS]" : "[FAIL] -- check your kernel");

    cudaFree(dA); cudaFree(dB); cudaFree(dR);
    free(hA); free(hB); free(hR); free(hRef);
}


/* ----- B3. Grid configuration calculator ----------------------
 * Purpose : Print the required (blocks, threads) for several N
 *           and verify that total_threads >= N.
 * ---------------------------------------------------------------- */
void exercise_grid_config(void)
{
    int test_sizes[] = {1, 100, 256, 257, 1024, 10000, 1 << 20};
    int n_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int tpb = BLOCK_DIM;

    printf("\n  [B3-GridConfig] threads_per_block=%d\n", tpb);
    printf("  %10s  %8s  %15s  %12s\n", "N", "blocks", "total_threads", "sufficient?");
    printf("  %s\n", "---------------------------------------------------");

    for (int t = 0; t < n_tests; t++) {
        int N = test_sizes[t];

        /* ═══════════════════════════════════════════════════════
         * TODO (B3): Calculate 'nblk' with ceiling division.
         *   HINT: nblk = (N + tpb - 1) / tpb;
         * ═══════════════════════════════════════════════════════ */
        int nblk = (N + tpb - 1) / tpb;

        int total = nblk * tpb;
        int ok = (total >= N);
        printf("  %10d  %8d  %15d  %12s\n", N, nblk, total, ok ? "[OK]" : "[FAIL]");
    }
}


/* ----- B4. PCIe transfer bandwidth profiling ------------------
 * Purpose : Measure H→D and D→H throughput in GB/s for a range
 *           of buffer sizes (1 – 512 MB).
 * Note    : PCIe bandwidth is frequently the training bottleneck!
 * ---------------------------------------------------------------- */
void exercise_transfer_bandwidth(void)
{
    int mb_sizes[] = {1, 8, 64, 256, 512};
    int n_sizes = sizeof(mb_sizes) / sizeof(mb_sizes[0]);

    printf("\n  [B4-TransferBandwidth]\n");
    printf("  %10s  %12s  %12s\n", "Size (MB)", "H2D (GB/s)", "D2H (GB/s)");
    printf("  %s\n", "----------------------------------------");

    for (int s = 0; s < n_sizes; s++) {
        int mb = mb_sizes[s];
        size_t nbytes = (size_t)mb * 1024 * 1024;

        float *h_buf, *d_buf;
        GPU_SAFE(cudaMallocHost(&h_buf, nbytes));   /* pinned memory */
        GPU_SAFE(cudaMalloc(&d_buf, nbytes));

        /* Fill buffer */
        for (size_t i = 0; i < nbytes / sizeof(float); i++)
            h_buf[i] = (float)i;

        cudaEvent_t ev0, ev1;
        GPU_SAFE(cudaEventCreate(&ev0));
        GPU_SAFE(cudaEventCreate(&ev1));
        float ms = 0.0f;

        /* ═══════════════════════════════════════════════════════
         * TODO (B4-H2D): Time the host → device transfer.
         *   HINT:
         *     cudaEventRecord(ev0);
         *     cudaMemcpy(d_buf, h_buf, nbytes, cudaMemcpyHostToDevice);
         *     cudaEventRecord(ev1);
         *     cudaEventSynchronize(ev1);
         *     cudaEventElapsedTime(&ms, ev0, ev1);
         *     float bw_h2d = (mb / 1024.0f) / (ms / 1000.0f);
         * ═══════════════════════════════════════════════════════ */
        GPU_SAFE(cudaEventRecord(ev0));
        GPU_SAFE(cudaMemcpy(d_buf, h_buf, nbytes, cudaMemcpyHostToDevice));
        GPU_SAFE(cudaEventRecord(ev1));
        GPU_SAFE(cudaEventSynchronize(ev1));
        GPU_SAFE(cudaEventElapsedTime(&ms, ev0, ev1));
        float bw_h2d = (mb / 1024.0f) / (ms / 1000.0f);

        /* ═══════════════════════════════════════════════════════
         * TODO (B4-D2H): Time the device → host transfer.
         *   HINT: Same approach, opposite copy direction.
         * ═══════════════════════════════════════════════════════ */
        GPU_SAFE(cudaEventRecord(ev0));
        GPU_SAFE(cudaMemcpy(h_buf, d_buf, nbytes, cudaMemcpyDeviceToHost));
        GPU_SAFE(cudaEventRecord(ev1));
        GPU_SAFE(cudaEventSynchronize(ev1));
        GPU_SAFE(cudaEventElapsedTime(&ms, ev0, ev1));
        float bw_d2h = (mb / 1024.0f) / (ms / 1000.0f);

        printf("  %10d  %12.1f  %12.1f\n", mb, bw_h2d, bw_d2h);

        cudaFree(d_buf);
        cudaFreeHost(h_buf);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }
}


/* ================================================================
 * PART C — ADVANCED STRETCH GOALS
 * ================================================================ */

/* ----- C1. Stretch: ReLU activation from scratch --------------
 * Implements: out[i] = max(0, x[i])
 * ---------------------------------------------------------------- */
__global__ void applyReLU(const float* inp, float* out, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        /* ═══════════════════════════════════════════════════════
         * STRETCH (C1): Implement ReLU activation.
         *   HINT: out[idx] = (inp[idx] > 0.0f) ? inp[idx] : 0.0f;
         * ═══════════════════════════════════════════════════════ */
        out[idx] = (inp[idx] > 0.0f) ? inp[idx] : 0.0f;
    }
}

void stretch_relu_test(int len)
{
    size_t nbytes = len * sizeof(float);
    float *hX   = (float*)malloc(nbytes);
    float *hOut = (float*)malloc(nbytes);
    float *hRef = (float*)malloc(nbytes);
    for (int i = 0; i < len; i++) {
        hX[i]  = ((float)rand() / RAND_MAX - 0.5f) * 8.0f;
        hRef[i] = hX[i] > 0.0f ? hX[i] : 0.0f;
    }

    float *dX, *dOut;
    GPU_SAFE(cudaMalloc(&dX,   nbytes));
    GPU_SAFE(cudaMalloc(&dOut, nbytes));
    GPU_SAFE(cudaMemcpy(dX, hX, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dOut, 0, nbytes));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    applyReLU<<<nblk, tpb>>>(dX, dOut, len);
    GPU_SAFE(cudaMemcpy(hOut, dOut, nbytes, cudaMemcpyDeviceToHost));

    int pass = arrays_match(hOut, hRef, len, 1e-5f);
    printf("  [C1-ReLU-Stretch] %s\n", pass ? "[PASS]" : "[FAIL]");

    cudaFree(dX); cudaFree(dOut);
    free(hX); free(hOut); free(hRef);
}


/* ----- C2. Stretch: Warp divergence analysis ------------------
 * Compare a deliberately divergent kernel against a predicated
 * (branch-free) equivalent and measure the overhead.
 * ---------------------------------------------------------------- */
__global__ void divergingKernel(float* arr, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        /* ═══════════════════════════════════════════════════════
         * STRETCH (C2): Intentional warp divergence.
         *   HINT: if (threadIdx.x & 1)
         *             arr[idx] += 1.0f;
         *         else
         *             arr[idx] *= 2.0f;
         * ═══════════════════════════════════════════════════════ */
        if (threadIdx.x % 2 == 0)
            arr[idx] = arr[idx] * 2.0f;
        else
            arr[idx] = arr[idx] + 1.0f;
    }
}

__global__ void predicatedKernel(float* arr, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        /* ═══════════════════════════════════════════════════════
         * STRETCH (C2): Branch-free version using predication.
         *   HINT:
         *     int is_even = 1 - (threadIdx.x & 1);
         *     arr[idx] = is_even * (arr[idx] * 2.0f) +
         *                (1 - is_even) * (arr[idx] + 1.0f);
         * ═══════════════════════════════════════════════════════ */
        int is_even = (threadIdx.x % 2 == 0);
        arr[idx] = is_even * (arr[idx] * 2.0f) + (1 - is_even) * (arr[idx] + 1.0f);
    }
}

void stretch_divergence_test(int len)
{
    size_t nbytes = len * sizeof(float);
    float *hArr = (float*)malloc(nbytes);
    for (int i = 0; i < len; i++) hArr[i] = (float)rand() / RAND_MAX;

    float *dDiv, *dPred;
    GPU_SAFE(cudaMalloc(&dDiv,  nbytes));
    GPU_SAFE(cudaMalloc(&dPred, nbytes));
    GPU_SAFE(cudaMemcpy(dDiv,  hArr, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dPred, hArr, nbytes, cudaMemcpyHostToDevice));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    int ITERS = 1000;

    cudaEvent_t ev0, ev1;
    GPU_SAFE(cudaEventCreate(&ev0));
    GPU_SAFE(cudaEventCreate(&ev1));
    float t_div, t_pred;

    GPU_SAFE(cudaEventRecord(ev0));
    for (int r = 0; r < ITERS; r++)
        divergingKernel<<<nblk, tpb>>>(dDiv, len);
    GPU_SAFE(cudaEventRecord(ev1));
    GPU_SAFE(cudaEventSynchronize(ev1));
    GPU_SAFE(cudaEventElapsedTime(&t_div, ev0, ev1));

    GPU_SAFE(cudaEventRecord(ev0));
    for (int r = 0; r < ITERS; r++)
        predicatedKernel<<<nblk, tpb>>>(dPred, len);
    GPU_SAFE(cudaEventRecord(ev1));
    GPU_SAFE(cudaEventSynchronize(ev1));
    GPU_SAFE(cudaEventElapsedTime(&t_pred, ev0, ev1));

    printf("  [C2-WarpDivergence] Divergent=%.2fms  BranchFree=%.2fms  "
           "Overhead=%.1fx\n", t_div, t_pred, t_div / (t_pred + 1e-6f));

    cudaFree(dDiv); cudaFree(dPred); free(hArr);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
}


/* ================================================================
 * ENTRY POINT
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  Part 1: GPU Fundamentals & Memory Transfers\n");
    printf("========================================================\n");

    /* Display device info */
    cudaDeviceProp devprop;
    GPU_SAFE(cudaGetDeviceProperties(&devprop, 0));
    printf("  GPU: %s  (SM %d.%d)  VRAM: %.0f MB\n\n",
           devprop.name, devprop.major, devprop.minor,
           devprop.totalGlobalMem / 1e6);

    printf("[Part A] Reference:\n");
    benchmark_pairwise_sum(NUM_ELEMS);

    printf("\n[Part B] Exercises:\n");
    exercise_scalar_mul(NUM_ELEMS);
    exercise_squared_error(NUM_ELEMS);
    exercise_grid_config();
    exercise_transfer_bandwidth();

    printf("\n[Part C] Stretch Goals:\n");
    stretch_relu_test(NUM_ELEMS);
    stretch_divergence_test(1 << 18);

    printf("\n========================================================\n");
    printf("  Part 1 finished! All [PASS] → proceed to Part 2.\n");
    printf("========================================================\n\n");
    return 0;
}
