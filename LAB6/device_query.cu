%%writefile device_info.cu
// file: device_info.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>   // 🔥 REQUIRED for CUDA APIs

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        printf("  CUDA Capability: %d.%d\n",
               deviceProp.major, deviceProp.minor);

        printf("  Global Memory: %.0f MB (%llu bytes)\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f,
               (unsigned long long)deviceProp.totalGlobalMem);

        printf("  Constant Memory: %zu bytes\n",
               deviceProp.totalConstMem);

        printf("  Shared Memory per Block: %zu bytes\n",
               deviceProp.sharedMemPerBlock);

        printf("  Warp Size: %d\n", deviceProp.warpSize);

        printf("  Max Threads per SM: %d\n",
               deviceProp.maxThreadsPerMultiProcessor);

        printf("  Max Threads per Block: %d\n",
               deviceProp.maxThreadsPerBlock);

        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);

        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
    }

    return 0;
}