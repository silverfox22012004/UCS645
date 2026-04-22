%%writefile array_sum.cu
// file: array_sum.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void arraySumKernel(float* d_arr, float* d_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_arr[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    float *h_arr, *h_out;
    float *d_arr, *d_out;

    h_arr = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_arr[i] = 1.0f;
    }

    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("Threads,Blocks,Time(ms)\n");  // CSV format for graph

    for (int i = 0; i < numConfigs; i++) {
        int threadsPerBlock = blockSizes[i];
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        h_out = (float*)malloc(blocksPerGrid * sizeof(float));
        cudaMalloc((void**)&d_out, blocksPerGrid * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        arraySumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_arr, d_out, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        arraySumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_arr, d_out, n);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(h_out, d_out, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

        // Print CSV line
        printf("%d,%d,%f\n", threadsPerBlock, blocksPerGrid, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_out);
        free(h_out);
    }

    cudaFree(d_arr);
    free(h_arr);

    return 0;
}