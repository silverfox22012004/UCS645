/*
 * ============================================================
 * Part 2 — Shared Memory Operations & Parallel Reductions
 * ============================================================
 * FOCUS       : Shared memory tiling, block-level reduction,
 *               bank conflicts, atomic operations
 * CUDA LEVEL  : 12.x
 *
 * Goals:
 *   1. Move data from global → shared memory with proper sync
 *   2. Build a tree-style parallel reduction using __syncthreads()
 *   3. Observe and measure shared-memory bank conflict effects
 *   4. Safely accumulate via atomicAdd (histogram use-case)
 *   5. Exploit warp-level shuffles for register-only reductions
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 part2_shared_mem_ops.cu -o part2
 *
 * Execute:
 *   ./part2
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define GPU_SAFE(call)                                                      \
    do {                                                                    \
        cudaError_t status = (call);                                        \
        if (status != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA failure at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(status));        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define BLOCK_DIM  256
#define NUM_ELEMS  (1 << 20)

/* Floating-point comparison helper */
int arrays_equal(const float* a, const float* b, int len, float tol)
{
    for (int i = 0; i < len; i++)
        if (fabsf(a[i] - b[i]) > tol) return 0;
    return 1;
}


/* ================================================================
 * PART A — REFERENCE IMPLEMENTATIONS (provided)
 * ================================================================ */

/* ----- A1. Shared memory doubling demo ------------------------ */
__global__ void sharedDoubleDemo(const float* src, float* dst, int len)
{
    __shared__ float buf[256];
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    buf[threadIdx.x] = (gid < len) ? src[gid] : 0.0f;
    __syncthreads();              /* wait for all loads */
    if (gid < len)
        dst[gid] = buf[threadIdx.x] * 2.0f;
}

/* ----- A2. Block-level sum via tree reduction ----------------- */
__global__ void blockSumReduce(const float* data, float* partial, int len)
{
    __shared__ float shmem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    shmem[tid] = (gid < len) ? data[gid] : 0.0f;
    __syncthreads();

    /* Halve active threads at each level */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = shmem[0];
}

/* Two-pass reduction wrapper (GPU partial → CPU final) */
float reference_sum_reduce(const float* d_arr, int len)
{
    int tpb = BLOCK_DIM;
    int nblk = (len + tpb - 1) / tpb;
    float *d_part;
    GPU_SAFE(cudaMalloc(&d_part, nblk * sizeof(float)));

    blockSumReduce<<<nblk, tpb>>>(d_arr, d_part, len);

    float *h_part = (float*)malloc(nblk * sizeof(float));
    GPU_SAFE(cudaMemcpy(h_part, d_part, nblk * sizeof(float),
                        cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (int i = 0; i < nblk; i++) result += h_part[i];

    cudaFree(d_part);
    free(h_part);
    return result;
}

void test_reference_reduction(void)
{
    int len = NUM_ELEMS;
    float *h_data = (float*)malloc(len * sizeof(float));
    double cpu_total = 0.0;
    for (int i = 0; i < len; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
        cpu_total += h_data[i];
    }
    float *d_data;
    GPU_SAFE(cudaMalloc(&d_data, len * sizeof(float)));
    GPU_SAFE(cudaMemcpy(d_data, h_data, len * sizeof(float), cudaMemcpyHostToDevice));

    float gpu_total = reference_sum_reduce(d_data, len);
    printf("  [A2-BlockSum] GPU=%.2f  CPU=%.2f  Match: %s\n",
           gpu_total, (float)cpu_total,
           fabsf(gpu_total - (float)cpu_total) < 100.0f ? "[PASS]" : "[FAIL]");

    cudaFree(d_data);
    free(h_data);
}


/* ================================================================
 * PART B — EXERCISES
 * ================================================================ */

/* ----- B1. Shared-memory round-trip copy ----------------------
 * Load global → shared → global to practice the sync pattern.
 * ---------------------------------------------------------------- */
__global__ void sharedRoundTrip(const float* inp, float* out, int len)
{
    __shared__ float buf[256];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    /* ═══════════════════════════════════════════════════════════
     * TODO (B1-Step1): Load element into shared memory buffer.
     *   HINT: buf[threadIdx.x] = (gid < len) ? inp[gid] : 0.0f;
     * ═══════════════════════════════════════════════════════════ */
    buf[threadIdx.x] = (gid < len) ? inp[gid] : 0.0f;

    /* ═══════════════════════════════════════════════════════════
     * TODO (B1-Step2): Synchronise the block so every thread
     *   finishes its load before any thread reads from buf[].
     *   HINT: __syncthreads();
     * ═══════════════════════════════════════════════════════════ */
    __syncthreads();

    /* ═══════════════════════════════════════════════════════════
     * TODO (B1-Step3): Write the value back to global memory.
     *   HINT: if (gid < len) out[gid] = buf[threadIdx.x];
     * ═══════════════════════════════════════════════════════════ */
    if (gid < len) out[gid] = buf[threadIdx.x];
}

void exercise_shared_roundtrip(void)
{
    int len = 1 << 16;
    size_t nbytes = len * sizeof(float);
    float *hIn  = (float*)malloc(nbytes);
    float *hOut = (float*)malloc(nbytes);
    for (int i = 0; i < len; i++) hIn[i] = (float)i;

    float *dIn, *dOut;
    GPU_SAFE(cudaMalloc(&dIn,  nbytes));
    GPU_SAFE(cudaMalloc(&dOut, nbytes));
    GPU_SAFE(cudaMemcpy(dIn, hIn, nbytes, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dOut, 0, nbytes));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    sharedRoundTrip<<<nblk, tpb>>>(dIn, dOut, len);
    GPU_SAFE(cudaMemcpy(hOut, dOut, nbytes, cudaMemcpyDeviceToHost));

    int pass = arrays_equal(hOut, hIn, len, 1e-5f);
    printf("  [B1-SharedCopy] %s\n",
           pass ? "[PASS]" : "[FAIL] -- did you add __syncthreads()?");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut);
}


/* ----- B2. Block-level max reduction --------------------------
 * Locate the maximum value across N elements with shared memory.
 * Essential for softmax numerical stability (subtract-max trick).
 * ---------------------------------------------------------------- */
__global__ void blockMaxReduce(const float* data, float* partial, int len)
{
    __shared__ float shmem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    /* ═══════════════════════════════════════════════════════════
     * TODO (B2-Step1): Load data, using -1e30f for out-of-range
     *   (identity element for max).
     *   HINT: shmem[tid] = (gid < len) ? data[gid] : -1e30f;
     * ═══════════════════════════════════════════════════════════ */
    shmem[tid] = (gid < len) ? data[gid] : -1e30f;
    __syncthreads();

    /* ═══════════════════════════════════════════════════════════
     * TODO (B2-Step2): Tree reduction with fmaxf instead of +.
     *   HINT:
     *     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
     *         if (tid < s)
     *             shmem[tid] = fmaxf(shmem[tid], shmem[tid + s]);
     *         __syncthreads();
     *     }
     * ═══════════════════════════════════════════════════════════ */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shmem[tid] = fmaxf(shmem[tid], shmem[tid + s]);
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = shmem[0];
}

void exercise_max_reduce(void)
{
    int len = 1 << 18;
    float *h_data = (float*)malloc(len * sizeof(float));
    float cpu_max = -1e30f;
    for (int i = 0; i < len; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 100.0f;
        if (h_data[i] > cpu_max) cpu_max = h_data[i];
    }

    float *d_data;
    GPU_SAFE(cudaMalloc(&d_data, len * sizeof(float)));
    GPU_SAFE(cudaMemcpy(d_data, h_data, len * sizeof(float), cudaMemcpyHostToDevice));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    float *d_part;
    GPU_SAFE(cudaMalloc(&d_part, nblk * sizeof(float)));
    blockMaxReduce<<<nblk, tpb>>>(d_data, d_part, len);

    float *h_part = (float*)malloc(nblk * sizeof(float));
    GPU_SAFE(cudaMemcpy(h_part, d_part, nblk * sizeof(float),
                        cudaMemcpyDeviceToHost));
    float gpu_max = -1e30f;
    for (int b = 0; b < nblk; b++)
        if (h_part[b] > gpu_max) gpu_max = h_part[b];

    int pass = fabsf(gpu_max - cpu_max) < 0.01f;
    printf("  [B2-MaxReduce] GPU=%.4f  CPU=%.4f  %s\n",
           gpu_max, cpu_max, pass ? "[PASS]" : "[FAIL]");

    cudaFree(d_data); cudaFree(d_part);
    free(h_data); free(h_part);
}


/* ----- B3. Bank conflict timing experiment --------------------
 * Shared memory is banked (32 banks). Strided access patterns
 * cause conflicts. Profile several strides to observe the effect.
 * ---------------------------------------------------------------- */
__global__ void stridedAccess(float* data, float* result, int stride, int len)
{
    __shared__ float tile[1024];
    int tid = threadIdx.x;

    /* stride=1 is conflict-free; stride=32 maximises conflicts */
    tile[tid * stride % 1024] = (tid < len) ? data[tid] : 0.0f;
    __syncthreads();
    if (tid < len)
        result[tid] = tile[tid * stride % 1024] * 2.0f;
}

void exercise_bank_conflicts(void)
{
    int stride_vals[] = {1, 2, 4, 8, 16, 32};
    int n_vals = sizeof(stride_vals) / sizeof(stride_vals[0]);
    int len = 1024, REPEATS = 5000;

    float *d_data, *d_res;
    GPU_SAFE(cudaMalloc(&d_data, len * sizeof(float)));
    GPU_SAFE(cudaMalloc(&d_res,  len * sizeof(float)));
    GPU_SAFE(cudaMemset(d_data, 0, len * sizeof(float)));

    cudaEvent_t ev0, ev1;
    GPU_SAFE(cudaEventCreate(&ev0));
    GPU_SAFE(cudaEventCreate(&ev1));

    printf("\n  [B3-BankConflicts] (lower = better)\n");
    printf("  %8s  %12s  Notes\n", "Stride", "Time (us)");
    printf("  %s\n", "--------------------------------------");

    float baseline_us = -1.0f;
    for (int s = 0; s < n_vals; s++) {
        int st = stride_vals[s];

        /* ═══════════════════════════════════════════════════════
         * TODO (B3): Time the stridedAccess kernel.
         *   HINT:
         *     cudaEventRecord(ev0);
         *     for (int r = 0; r < REPEATS; r++)
         *         stridedAccess<<<1, len>>>(d_data, d_res, st, len);
         *     cudaEventRecord(ev1);
         *     cudaEventSynchronize(ev1);
         *     float ms; cudaEventElapsedTime(&ms, ev0, ev1);
         *     float us = ms * 1000.0f / REPEATS;
         * ═══════════════════════════════════════════════════════ */
        GPU_SAFE(cudaEventRecord(ev0));
        for (int r = 0; r < REPEATS; r++)
            stridedAccess<<<1, len>>>(d_data, d_res, st, len);
        GPU_SAFE(cudaEventRecord(ev1));
        GPU_SAFE(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        GPU_SAFE(cudaEventElapsedTime(&ms, ev0, ev1));
        float us = ms * 1000.0f / REPEATS;

        if (baseline_us < 0.0f) baseline_us = us;
        printf("  %8d  %12.2f  %s\n", st, us,
               st == 1  ? "(sequential — optimal)" :
               st == 32 ? "(32-way conflict — worst)" : "");
    }

    printf("  Baseline stride=1: %.2f us\n", baseline_us);
    cudaFree(d_data); cudaFree(d_res);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
}


/* ----- B4. Global-atomic histogram ----------------------------
 * Each thread atomically increments the bin for its data value.
 * atomicAdd prevents concurrent-write data races.
 * ---------------------------------------------------------------- */
__global__ void computeHistogram(const int* data, int* bins, int len, int n_bins)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < len) {
        /* ═══════════════════════════════════════════════════════
         * TODO (B4): Atomically bump the histogram bin.
         *   HINT: atomicAdd(&bins[data[gid]], 1);
         *   Why atomic? Multiple threads may target the same bin
         *   simultaneously — without atomics the count is wrong.
         * ═══════════════════════════════════════════════════════ */
        atomicAdd(&bins[data[gid]], 1);
    }
}

void exercise_histogram(void)
{
    int len = 1 << 18, n_bins = 256;
    int *h_data  = (int*)malloc(len * sizeof(int));
    int *h_bins  = (int*)calloc(n_bins, sizeof(int));
    int *h_ref   = (int*)calloc(n_bins, sizeof(int));

    for (int i = 0; i < len; i++) {
        h_data[i] = rand() % n_bins;
        h_ref[h_data[i]]++;
    }

    int *d_data, *d_bins;
    GPU_SAFE(cudaMalloc(&d_data, len * sizeof(int)));
    GPU_SAFE(cudaMalloc(&d_bins, n_bins * sizeof(int)));
    GPU_SAFE(cudaMemcpy(d_data, h_data, len * sizeof(int), cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(d_bins, 0, n_bins * sizeof(int)));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    computeHistogram<<<nblk, tpb>>>(d_data, d_bins, len, n_bins);
    GPU_SAFE(cudaMemcpy(h_bins, d_bins, n_bins * sizeof(int),
                        cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int b = 0; b < n_bins; b++)
        if (h_bins[b] != h_ref[b]) { pass = 0; break; }

    printf("  [B4-Histogram] N=%d bins=%d  %s\n",
           len, n_bins,
           pass ? "[PASS]" : "[FAIL] -- did you use atomicAdd?");

    cudaFree(d_data); cudaFree(d_bins);
    free(h_data); free(h_bins); free(h_ref);
}


/* ================================================================
 * PART C — STRETCH: Warp-Level Primitives
 * ================================================================ */

/* ----- C1. Warp shuffle sum -----------------------------------
 * __shfl_down_sync passes register values directly between warp
 * lanes — faster than shared memory (no smem or barrier needed).
 * ---------------------------------------------------------------- */
__global__ void warpShuffleSum(const float* data, float* out, int len)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < len) ? data[gid] : 0.0f;

    /* ═══════════════════════════════════════════════════════════
     * STRETCH (C1): Butterfly reduction across the warp.
     *   HINT:
     *     for (int off = 16; off > 0; off >>= 1)
     *         val += __shfl_down_sync(0xffffffff, val, off);
     *   After the loop, lane 0 holds the warp sum.
     *   Use atomicAdd to merge results across warps.
     * ═══════════════════════════════════════════════════════════ */
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);

    if (threadIdx.x % 32 == 0) atomicAdd(out, val);
}

void stretch_warp_sum(void)
{
    int len = 32;   /* single warp */
    float *h_data = (float*)malloc(len * sizeof(float));
    float expected = 0.0f;
    for (int i = 0; i < len; i++) { h_data[i] = (float)i; expected += h_data[i]; }

    float *d_data, *d_out;
    GPU_SAFE(cudaMalloc(&d_data, len * sizeof(float)));
    GPU_SAFE(cudaMalloc(&d_out, sizeof(float)));
    GPU_SAFE(cudaMemcpy(d_data, h_data, len * sizeof(float), cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(d_out, 0, sizeof(float)));

    warpShuffleSum<<<1, 32>>>(d_data, d_out, len);

    float gpu_sum;
    GPU_SAFE(cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    int pass = fabsf(gpu_sum - expected) < 0.01f;
    printf("  [C1-WarpSum] GPU=%.1f  CPU=%.1f  %s\n",
           gpu_sum, expected, pass ? "[PASS]" : "[FAIL]");

    cudaFree(d_data); cudaFree(d_out);
    free(h_data);
}


/* ----- C2. Shared-memory histogram (reduced contention) -------
 * Each block keeps a private histogram in smem, then merges into
 * global memory via atomicAdd — far less contention.
 * ---------------------------------------------------------------- */
__global__ void histogramSmem(const int* data, int* bins,
                              int len, int n_bins)
{
    extern __shared__ int local_bins[];  /* dynamic smem = n_bins */

    /* ═══════════════════════════════════════════════════════════
     * STRETCH (C2): Two-phase histogram.
     *   Phase 1 — zero local_bins for this block:
     *     for (int b = threadIdx.x; b < n_bins; b += blockDim.x)
     *         local_bins[b] = 0;
     *     __syncthreads();
     *   Phase 2 — accumulate into shared (low contention):
     *     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     *     if (gid < len) atomicAdd(&local_bins[data[gid]], 1);
     *     __syncthreads();
     *   Phase 3 — flush to global:
     *     for (int b = threadIdx.x; b < n_bins; b += blockDim.x)
     *         atomicAdd(&bins[b], local_bins[b]);
     * ═══════════════════════════════════════════════════════════ */
    for (int b = threadIdx.x; b < n_bins; b += blockDim.x)
        local_bins[b] = 0;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < len)
        atomicAdd(&local_bins[data[gid]], 1);
    __syncthreads();

    for (int b = threadIdx.x; b < n_bins; b += blockDim.x)
        atomicAdd(&bins[b], local_bins[b]);
}

void stretch_smem_histogram(void)
{
    int len = 1 << 20, n_bins = 256;
    int *h_data  = (int*)malloc(len * sizeof(int));
    int *h_bins  = (int*)calloc(n_bins, sizeof(int));
    int *h_ref   = (int*)calloc(n_bins, sizeof(int));
    for (int i = 0; i < len; i++) { h_data[i] = rand() % n_bins; h_ref[h_data[i]]++; }

    int *d_data, *d_bins;
    GPU_SAFE(cudaMalloc(&d_data, len * sizeof(int)));
    GPU_SAFE(cudaMalloc(&d_bins, n_bins * sizeof(int)));
    GPU_SAFE(cudaMemcpy(d_data, h_data, len * sizeof(int), cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(d_bins, 0, n_bins * sizeof(int)));

    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;
    int smem_sz = n_bins * sizeof(int);
    histogramSmem<<<nblk, tpb, smem_sz>>>(d_data, d_bins, len, n_bins);
    GPU_SAFE(cudaMemcpy(h_bins, d_bins, n_bins * sizeof(int),
                        cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int b = 0; b < n_bins; b++)
        if (h_bins[b] != h_ref[b]) { pass = 0; break; }

    printf("  [C2-SmemHistogram] N=%d bins=%d  %s\n",
           len, n_bins, pass ? "[PASS]" : "[FAIL]");

    cudaFree(d_data); cudaFree(d_bins);
    free(h_data); free(h_bins); free(h_ref);
}


/* ================================================================
 * ENTRY POINT
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  Part 2: Shared Memory Operations & Reductions\n");
    printf("========================================================\n");

    cudaDeviceProp devprop;
    GPU_SAFE(cudaGetDeviceProperties(&devprop, 0));
    printf("  GPU: %s  Shared mem/block: %zu KB\n\n",
           devprop.name, devprop.sharedMemPerBlock / 1024);

    printf("[Part A] Reference:\n");
    test_reference_reduction();

    printf("\n[Part B] Exercises:\n");
    exercise_shared_roundtrip();
    exercise_max_reduce();
    exercise_bank_conflicts();
    exercise_histogram();

    printf("\n[Part C] Stretch Goals:\n");
    stretch_warp_sum();
    stretch_smem_histogram();

    printf("\n========================================================\n");
    printf("  All [PASS] → proceed to Part 3!\n");
    printf("========================================================\n\n");
    return 0;
}
