/*
 * ============================================================
 * Part 3 — Neural Network Building Blocks: Activations & Loss
 * ============================================================
 * FOCUS       : Activation functions, Softmax, Cross-Entropy, Adam
 * CUDA LEVEL  : 12.x
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 part3_nn_building_blocks.cu -o part3 -lm
 * Execute:
 *   ./part3
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
#define NUM_ELEMS  (1 << 18)

int arrays_equal(const float* a, const float* b, int len, float tol)
{
    for (int i = 0; i < len; i++)
        if (fabsf(a[i] - b[i]) > tol) return 0;
    return 1;
}

/* ================================================================
 * PART A — REFERENCE: ReLU and Softmax (provided)
 * ================================================================ */

__global__ void reluForward(const float* x, float* y, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) y[idx] = fmaxf(0.0f, x[idx]);
}

/* Row-wise numerically-stable softmax (one thread per sample) */
__global__ void stableSoftmax(const float* logits, float* probs, int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = logits + n * C;
    float* out = probs + n * C;
    float peak = -1e30f;
    for (int c = 0; c < C; c++) peak = fmaxf(peak, row[c]);
    float denom = 0.0f;
    for (int c = 0; c < C; c++) denom += expf(row[c] - peak);
    for (int c = 0; c < C; c++) out[c] = expf(row[c] - peak) / denom;
}

void test_softmax_reference(void)
{
    int N = 4, C = 10;
    float *h_log = (float*)malloc(N * C * sizeof(float));
    float *h_prob = (float*)malloc(N * C * sizeof(float));
    for (int i = 0; i < N * C; i++) h_log[i] = (float)rand() / RAND_MAX;

    float *d_log, *d_prob;
    GPU_SAFE(cudaMalloc(&d_log, N * C * sizeof(float)));
    GPU_SAFE(cudaMalloc(&d_prob, N * C * sizeof(float)));
    GPU_SAFE(cudaMemcpy(d_log, h_log, N * C * sizeof(float), cudaMemcpyHostToDevice));

    stableSoftmax<<<(N + 255) / 256, 256>>>(d_log, d_prob, N, C);
    GPU_SAFE(cudaMemcpy(h_prob, d_prob, N * C * sizeof(float), cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int n = 0; n < N; n++) {
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += h_prob[n * C + c];
        if (fabsf(s - 1.0f) > 1e-5f) { pass = 0; break; }
    }
    printf("  [A2-Softmax] Row sums = 1.0: %s\n", pass ? "[PASS]" : "[FAIL]");
    cudaFree(d_log); cudaFree(d_prob);
    free(h_log); free(h_prob);
}

/* ================================================================
 * PART B — EXERCISES: Activation Kernels
 * ================================================================ */

/* ----- B1. Sigmoid -------------------------------------------- */
__global__ void sigmoidKernel(const float* x, float* y, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

void exercise_sigmoid(int len)
{
    size_t nb = len * sizeof(float);
    float *hX = (float*)malloc(nb), *hY = (float*)malloc(nb), *hRef = (float*)malloc(nb);
    for (int i = 0; i < len; i++) {
        hX[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        hRef[i] = 1.0f / (1.0f + expf(-hX[i]));
    }
    float *dX, *dY;
    GPU_SAFE(cudaMalloc(&dX, nb)); GPU_SAFE(cudaMalloc(&dY, nb));
    GPU_SAFE(cudaMemcpy(dX, hX, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dY, 0, nb));
    sigmoidKernel<<<(len + 255) / 256, 256>>>(dX, dY, len);
    GPU_SAFE(cudaMemcpy(hY, dY, nb, cudaMemcpyDeviceToHost));
    printf("  [B1-Sigmoid] %s\n", arrays_equal(hY, hRef, len, 1e-5f) ? "[PASS]" : "[FAIL]");
    cudaFree(dX); cudaFree(dY); free(hX); free(hY); free(hRef);
}

/* ----- B2. Hyperbolic Tangent --------------------------------- */
__global__ void hypTanKernel(const float* x, float* y, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        y[idx] = tanhf(x[idx]);
    }
}

void exercise_tanh(int len)
{
    size_t nb = len * sizeof(float);
    float *hX = (float*)malloc(nb), *hY = (float*)malloc(nb), *hRef = (float*)malloc(nb);
    for (int i = 0; i < len; i++) {
        hX[i] = ((float)rand() / RAND_MAX - 0.5f) * 6.0f;
        hRef[i] = tanhf(hX[i]);
    }
    float *dX, *dY;
    GPU_SAFE(cudaMalloc(&dX, nb)); GPU_SAFE(cudaMalloc(&dY, nb));
    GPU_SAFE(cudaMemcpy(dX, hX, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dY, 0, nb));
    hypTanKernel<<<(len + 255) / 256, 256>>>(dX, dY, len);
    GPU_SAFE(cudaMemcpy(hY, dY, nb, cudaMemcpyDeviceToHost));
    printf("  [B2-Tanh] %s\n", arrays_equal(hY, hRef, len, 1e-5f) ? "[PASS]" : "[FAIL]");
    cudaFree(dX); cudaFree(dY); free(hX); free(hY); free(hRef);
}

/* ----- B3. Leaky ReLU ----------------------------------------- */
__global__ void leakyReluKernel(const float* x, float* y, float slope, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        y[idx] = fmaxf(x[idx], slope * x[idx]);
    }
}

void exercise_leaky_relu(int len, float slope)
{
    size_t nb = len * sizeof(float);
    float *hX = (float*)malloc(nb), *hY = (float*)malloc(nb), *hRef = (float*)malloc(nb);
    for (int i = 0; i < len; i++) {
        hX[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        hRef[i] = hX[i] > 0.0f ? hX[i] : slope * hX[i];
    }
    float *dX, *dY;
    GPU_SAFE(cudaMalloc(&dX, nb)); GPU_SAFE(cudaMalloc(&dY, nb));
    GPU_SAFE(cudaMemcpy(dX, hX, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dY, 0, nb));
    leakyReluKernel<<<(len + 255) / 256, 256>>>(dX, dY, slope, len);
    GPU_SAFE(cudaMemcpy(hY, dY, nb, cudaMemcpyDeviceToHost));
    printf("  [B3-LeakyReLU] alpha=%.2f  %s\n", slope, arrays_equal(hY, hRef, len, 1e-5f) ? "[PASS]" : "[FAIL]");
    cudaFree(dX); cudaFree(dY); free(hX); free(hY); free(hRef);
}

/* ----- B4. ReLU Backward (gradient gate) ---------------------- */
__global__ void reluGradGate(const float* gradOut, const float* fwdInput,
                             float* gradIn, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        gradIn[idx] = (fwdInput[idx] > 0.0f) ? gradOut[idx] : 0.0f;
    }
}

void exercise_relu_backward(int len)
{
    size_t nb = len * sizeof(float);
    float *hX = (float*)malloc(nb), *hGO = (float*)malloc(nb);
    float *hGI = (float*)malloc(nb), *hRef = (float*)malloc(nb);
    for (int i = 0; i < len; i++) {
        hX[i]  = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        hGO[i] = (float)rand() / RAND_MAX;
        hRef[i] = hX[i] > 0.0f ? hGO[i] : 0.0f;
    }
    float *dX, *dGO, *dGI;
    GPU_SAFE(cudaMalloc(&dX, nb)); GPU_SAFE(cudaMalloc(&dGO, nb)); GPU_SAFE(cudaMalloc(&dGI, nb));
    GPU_SAFE(cudaMemcpy(dX, hX, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dGO, hGO, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dGI, 0, nb));
    reluGradGate<<<(len + 255) / 256, 256>>>(dGO, dX, dGI, len);
    GPU_SAFE(cudaMemcpy(hGI, dGI, nb, cudaMemcpyDeviceToHost));
    printf("  [B4-ReLUBackward] %s\n", arrays_equal(hGI, hRef, len, 1e-5f) ? "[PASS]" : "[FAIL]");
    cudaFree(dX); cudaFree(dGO); cudaFree(dGI);
    free(hX); free(hGO); free(hGI); free(hRef);
}

/* ================================================================
 * PART C — EXERCISES: Loss Functions
 * ================================================================ */

/* ----- C1. Binary Cross-Entropy ------------------------------- */
__global__ void binaryCELoss(const float* pred, const float* truth,
                             float* loss, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float p = fminf(fmaxf(pred[idx], 1e-7f), 1.0f - 1e-7f);
        loss[idx] = -(truth[idx] * logf(p) + (1.0f - truth[idx]) * logf(1.0f - p));
    }
}

void exercise_bce_loss(int len)
{
    size_t nb = len * sizeof(float);
    float *hP = (float*)malloc(nb), *hT = (float*)malloc(nb);
    float *hL = (float*)malloc(nb), *hRef = (float*)malloc(nb);
    for (int i = 0; i < len; i++) {
        hP[i] = (float)rand() / RAND_MAX;
        hT[i] = (rand() % 2) ? 1.0f : 0.0f;
        float pc = fminf(fmaxf(hP[i], 1e-7f), 1.0f - 1e-7f);
        hRef[i] = -(hT[i] * logf(pc) + (1.0f - hT[i]) * logf(1.0f - pc));
    }
    float *dP, *dT, *dL;
    GPU_SAFE(cudaMalloc(&dP, nb)); GPU_SAFE(cudaMalloc(&dT, nb)); GPU_SAFE(cudaMalloc(&dL, nb));
    GPU_SAFE(cudaMemcpy(dP, hP, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dT, hT, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dL, 0, nb));
    binaryCELoss<<<(len + 255) / 256, 256>>>(dP, dT, dL, len);
    GPU_SAFE(cudaMemcpy(hL, dL, nb, cudaMemcpyDeviceToHost));
    printf("  [C1-BCE-Loss] %s\n", arrays_equal(hL, hRef, len, 1e-4f) ? "[PASS]" : "[FAIL]");
    cudaFree(dP); cudaFree(dT); cudaFree(dL);
    free(hP); free(hT); free(hL); free(hRef);
}

/* ----- C2. Categorical Cross-Entropy (log-sum-exp) ------------ */
__global__ void categoricalCELoss(const float* logits, const int* labels,
                                  float* loss, int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = logits + n * C;
    float peak = -1e30f;
    for (int c = 0; c < C; c++) peak = fmaxf(peak, row[c]);
    float denom = 0.0f;
    for (int c = 0; c < C; c++) denom += expf(row[c] - peak);
    int lbl = labels[n];
    loss[n] = -(row[lbl] - peak) + logf(denom);
}

void exercise_cross_entropy(int N, int C)
{
    size_t lb = (size_t)N * C * sizeof(float);
    float *hLog = (float*)malloc(lb);
    int *hLbl = (int*)malloc(N * sizeof(int));
    float *hL = (float*)malloc(N * sizeof(float)), *hRef = (float*)malloc(N * sizeof(float));
    for (int n = 0; n < N; n++) {
        float mx = -1e30f, se = 0.0f;
        for (int c = 0; c < C; c++) {
            hLog[n * C + c] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
            if (hLog[n * C + c] > mx) mx = hLog[n * C + c];
        }
        for (int c = 0; c < C; c++) se += expf(hLog[n * C + c] - mx);
        hLbl[n] = rand() % C;
        hRef[n] = -(hLog[n * C + hLbl[n]] - mx) + logf(se);
    }
    float *dLog; int *dLbl; float *dL;
    GPU_SAFE(cudaMalloc(&dLog, lb)); GPU_SAFE(cudaMalloc(&dLbl, N * sizeof(int)));
    GPU_SAFE(cudaMalloc(&dL, N * sizeof(float)));
    GPU_SAFE(cudaMemcpy(dLog, hLog, lb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dLbl, hLbl, N * sizeof(int), cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dL, 0, N * sizeof(float)));
    categoricalCELoss<<<(N + 255) / 256, 256>>>(dLog, dLbl, dL, N, C);
    GPU_SAFE(cudaMemcpy(hL, dL, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  [C2-CrossEntropy] N=%d C=%d  %s\n", N, C, arrays_equal(hL, hRef, N, 1e-4f) ? "[PASS]" : "[FAIL]");
    cudaFree(dLog); cudaFree(dLbl); cudaFree(dL);
    free(hLog); free(hLbl); free(hL); free(hRef);
}

/* ================================================================
 * PART D — STRETCH: Fused Adam Optimizer
 * ================================================================ */
__global__ void fusedAdamStep(float* w, const float* g, float* m, float* v,
                              float lr, float b1, float b2, float eps,
                              float b1_pow_t, float b2_pow_t, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        m[idx] = b1 * m[idx] + (1.0f - b1) * g[idx];
        v[idx] = b2 * v[idx] + (1.0f - b2) * g[idx] * g[idx];
        float mc = m[idx] / (1.0f - b1_pow_t);
        float vc = v[idx] / (1.0f - b2_pow_t);
        w[idx] -= lr * mc / (sqrtf(vc) + eps);
    }
}

void stretch_adam_test(int len)
{
    size_t nb = len * sizeof(float);
    float *hW = (float*)malloc(nb), *hG = (float*)malloc(nb);
    float *hM = (float*)calloc(len, sizeof(float)), *hV = (float*)calloc(len, sizeof(float));
    for (int i = 0; i < len; i++) { hW[i] = (float)rand() / RAND_MAX; hG[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f; }

    float *dW, *dG, *dM, *dV;
    GPU_SAFE(cudaMalloc(&dW, nb)); GPU_SAFE(cudaMalloc(&dG, nb));
    GPU_SAFE(cudaMalloc(&dM, nb)); GPU_SAFE(cudaMalloc(&dV, nb));
    GPU_SAFE(cudaMemcpy(dW, hW, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemcpy(dG, hG, nb, cudaMemcpyHostToDevice));
    GPU_SAFE(cudaMemset(dM, 0, nb)); GPU_SAFE(cudaMemset(dV, 0, nb));

    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int tpb = BLOCK_DIM, nblk = (len + tpb - 1) / tpb;

    for (int t = 1; t <= 5; t++) {
        float bp1 = powf(b1, t), bp2 = powf(b2, t);
        fusedAdamStep<<<nblk, tpb>>>(dW, dG, dM, dV, lr, b1, b2, eps, bp1, bp2, len);
        for (int i = 0; i < len; i++) {
            hM[i] = b1 * hM[i] + (1.0f - b1) * hG[i];
            hV[i] = b2 * hV[i] + (1.0f - b2) * hG[i] * hG[i];
            float mc = hM[i] / (1.0f - bp1), vc = hV[i] / (1.0f - bp2);
            hW[i] -= lr * mc / (sqrtf(vc) + eps);
        }
    }
    float *hWgpu = (float*)malloc(nb);
    GPU_SAFE(cudaMemcpy(hWgpu, dW, nb, cudaMemcpyDeviceToHost));
    printf("  [D1-Adam] 5 steps  %s\n", arrays_equal(hWgpu, hW, len, 1e-5f) ? "[PASS]" : "[FAIL]");
    cudaFree(dW); cudaFree(dG); cudaFree(dM); cudaFree(dV);
    free(hW); free(hG); free(hM); free(hV); free(hWgpu);
}

/* ================================================================
 * ENTRY POINT
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  Part 3: Neural Network Building Blocks\n");
    printf("========================================================\n");
    cudaDeviceProp dp;
    GPU_SAFE(cudaGetDeviceProperties(&dp, 0));
    printf("  GPU: %s\n\n", dp.name);

    printf("[Part A] Reference:\n");
    test_softmax_reference();

    printf("\n[Part B] Activation Kernels:\n");
    exercise_sigmoid(NUM_ELEMS);
    exercise_tanh(NUM_ELEMS);
    exercise_leaky_relu(NUM_ELEMS, 0.01f);
    exercise_relu_backward(NUM_ELEMS);

    printf("\n[Part C] Loss Functions:\n");
    exercise_bce_loss(NUM_ELEMS);
    exercise_cross_entropy(512, 10);

    printf("\n[Part D] Stretch — Adam Optimizer:\n");
    stretch_adam_test(1 << 16);

    printf("\n========================================================\n");
    printf("  All [PASS] → proceed to Part 4!\n");
    printf("========================================================\n\n");
    return 0;
}
