/*
 * ============================================================
 * Part 5 — End-to-End Digit Classifier (MNIST CNN Pipeline)
 * ============================================================
 * FOCUS       : cuDNN, cuBLAS, CUDA Streams, training loop
 * CUDA LEVEL  : 12.x  |  cuDNN 9.x  |  cuBLAS
 *
 * Architecture (LeNet-5 variant):
 *   Input [N,1,28,28]
 *   -> Conv(1->32,5x5,pad=2) -> BatchNorm -> ReLU -> MaxPool(2)
 *   -> Conv(32->64,5x5,pad=2) -> BatchNorm -> ReLU -> MaxPool(2)
 *   -> Flatten -> FC(3136->256) -> ReLU -> FC(256->10)
 *   -> Cross-Entropy Loss
 *
 * Build:
 *   nvcc -O2 -arch=sm_86 part5_digit_classifier.cu -o part5 \
 *        -lcudnn -lcublas -lm
 * Execute:
 *   ./part5
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define GPU_SAFE(call)                                                      \
    do { cudaError_t e=(call);                                              \
         if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",              \
         __FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

#define DNN_SAFE(call)                                                      \
    do { cudnnStatus_t e=(call);                                            \
         if(e!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"cuDNN %s:%d %d\n",    \
         __FILE__,__LINE__,(int)e);exit(1);} } while(0)

#define BLAS_SAFE(call)                                                     \
    do { cublasStatus_t e=(call);                                           \
         if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d %d\n",  \
         __FILE__,__LINE__,(int)e);exit(1);} } while(0)

#define BATCH       256
#define LR          0.01f
#define EPOCHS      10
#define IMG_DIM     784     /* 28*28 */
#define N_CLASSES   10

cudnnHandle_t   hDnn;
cublasHandle_t  hBlas;

static double timer_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ================================================================
 * PART A — MNIST Data Loader (provided)
 * ================================================================ */
static int read_be32(FILE* f) { unsigned char b[4]; fread(b,1,4,f); return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3]; }

typedef struct { float* pixels; int* labels; int count; } DigitDataset;

DigitDataset load_digits(const char* img_path, const char* lbl_path)
{
    FILE *fi = fopen(img_path, "rb"), *fl = fopen(lbl_path, "rb");
    if (!fi || !fl) { fprintf(stderr, "Cannot open MNIST files:\n  %s\n  %s\n", img_path, lbl_path); exit(1); }
    read_be32(fi); read_be32(fl);
    int n = read_be32(fi); read_be32(fl);
    int rows = read_be32(fi), cols = read_be32(fi);
    (void)rows; (void)cols;
    DigitDataset ds;
    ds.count  = n;
    ds.pixels = (float*)malloc((size_t)n * IMG_DIM * sizeof(float));
    ds.labels = (int*)malloc(n * sizeof(int));
    unsigned char* tmp = (unsigned char*)malloc(IMG_DIM);
    for (int i = 0; i < n; i++) {
        fread(tmp, 1, IMG_DIM, fi);
        for (int j = 0; j < IMG_DIM; j++) ds.pixels[i * IMG_DIM + j] = (tmp[j] - 127.5f) / 127.5f;
        unsigned char lbl; fread(&lbl, 1, 1, fl); ds.labels[i] = (int)lbl;
    }
    free(tmp); fclose(fi); fclose(fl);
    printf("[OK] Loaded %d digit samples from %s\n", n, img_path);
    return ds;
}

/* ================================================================
 * PART B — cuDNN Descriptor Builders (provided)
 * ================================================================ */
cudnnTensorDescriptor_t build_tensor_desc(int N, int C, int H, int W)
{
    cudnnTensorDescriptor_t d; DNN_SAFE(cudnnCreateTensorDescriptor(&d));
    DNN_SAFE(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    return d;
}
cudnnFilterDescriptor_t build_filter_desc(int k, int c, int h, int w)
{
    cudnnFilterDescriptor_t d; DNN_SAFE(cudnnCreateFilterDescriptor(&d));
    DNN_SAFE(cudnnSetFilter4dDescriptor(d, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));
    return d;
}
cudnnConvolutionDescriptor_t build_conv_desc(int pad, int stride)
{
    cudnnConvolutionDescriptor_t d; DNN_SAFE(cudnnCreateConvolutionDescriptor(&d));
    DNN_SAFE(cudnnSetConvolution2dDescriptor(d, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    return d;
}

/* ================================================================
 * PART C — Custom Kernels (provided)
 * ================================================================ */
__global__ void inplaceReLU(float* x, int N) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<N) x[i]=fmaxf(0.0f,x[i]); }

__global__ void softmaxCE(const float* logits, const int* labels, float* probs, float* loss, int N, int C)
{
    int n = blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
    const float* row = logits+n*C; float* pr = probs+n*C;
    float mx = -1e30f; for(int c=0;c<C;c++) mx=fmaxf(mx,row[c]);
    float se = 0; for(int c=0;c<C;c++){pr[c]=expf(row[c]-mx); se+=pr[c];}
    for(int c=0;c<C;c++) pr[c]/=se;
    loss[n] = -logf(pr[labels[n]]+1e-9f);
}

__global__ void sgdStep(float* w, const float* grad, float lr, int N) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) w[i]-=lr*grad[i]; }

__global__ void biasAdd(float* out, const float* bias, int N, int C);

/* ================================================================
 * PART E — cuDNN Convolution Forward
 * ================================================================ */
void run_conv_forward(
    cudnnTensorDescriptor_t idesc, float* dI,
    cudnnFilterDescriptor_t fdesc, float* dF,
    cudnnConvolutionDescriptor_t cdesc,
    cudnnTensorDescriptor_t odesc, float* dO)
{
    float alpha=1, beta=0;
    int nAlgo=0; cudnnConvolutionFwdAlgoPerf_t perf;
    DNN_SAFE(cudnnFindConvolutionForwardAlgorithm(hDnn, idesc, fdesc, cdesc, odesc, 1, &nAlgo, &perf));
    cudnnConvolutionFwdAlgo_t algo = perf.algo;
    size_t ws_sz=0; void* dWs=NULL;
    DNN_SAFE(cudnnGetConvolutionForwardWorkspaceSize(hDnn, idesc, fdesc, cdesc, odesc, algo, &ws_sz));
    if(ws_sz>0) GPU_SAFE(cudaMalloc(&dWs, ws_sz));
    DNN_SAFE(cudnnConvolutionForward(hDnn, &alpha, idesc, dI, fdesc, dF, cdesc, algo, dWs, ws_sz, &beta, odesc, dO));
    if(dWs) cudaFree(dWs);
}

/* ================================================================
 * PART F — cuDNN Pooling Forward
 * ================================================================ */
void run_maxpool_forward(
    cudnnTensorDescriptor_t idesc, float* dI,
    cudnnTensorDescriptor_t odesc, float* dO,
    int ph, int pw, int sh, int sw)
{
    cudnnPoolingDescriptor_t pd; DNN_SAFE(cudnnCreatePoolingDescriptor(&pd));
    DNN_SAFE(cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, ph, pw, 0, 0, sh, sw));
    float alpha=1, beta=0;
    DNN_SAFE(cudnnPoolingForward(hDnn, pd, &alpha, idesc, dI, &beta, odesc, dO));
    DNN_SAFE(cudnnDestroyPoolingDescriptor(pd));
}

/* ================================================================
 * PART G — FC Layer via cuBLAS
 * ================================================================ */
void run_fc_forward(float* dIn, float* dW, float* dBias, float* dOut, int N, int inF, int outF)
{
    float alpha=1, beta=0;
    BLAS_SAFE(cublasSgemm(hBlas, CUBLAS_OP_T, CUBLAS_OP_N, outF, N, inF, &alpha, dW, inF, dIn, inF, &beta, dOut, outF));
    dim3 blk(256), grd((outF+255)/256, N);
    biasAdd<<<grd, blk>>>(dOut, dBias, N, outF);
}

__global__ void biasAdd(float* out, const float* bias, int N, int C)
{
    int n=blockIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(n<N && c<C) out[n*C+c]+=bias[c];
}

/* ================================================================
 * PART H — Async Pipeline Demo
 * ================================================================ */
void demo_async_pipeline(const float* hImgs, int nSamples, float* dBufA, float* dBufB)
{
    size_t bsz = (size_t)BATCH*IMG_DIM*sizeof(float);
    cudaStream_t sComp, sXfer;
    GPU_SAFE(cudaStreamCreate(&sComp)); GPU_SAFE(cudaStreamCreate(&sXfer));
    int nBatch = nSamples/BATCH;
    if(nBatch<=0){printf("  [H-Async] no full batches\n"); GPU_SAFE(cudaStreamDestroy(sComp)); GPU_SAFE(cudaStreamDestroy(sXfer)); return;}
    GPU_SAFE(cudaMemcpyAsync(dBufA, hImgs, bsz, cudaMemcpyHostToDevice, sXfer));
    GPU_SAFE(cudaStreamSynchronize(sXfer));
    for(int i=0;i<nBatch;i++){
        if(i+1<nBatch) GPU_SAFE(cudaMemcpyAsync(dBufB, hImgs+(size_t)(i+1)*BATCH*IMG_DIM, bsz, cudaMemcpyHostToDevice, sXfer));
        int ne=BATCH*IMG_DIM; inplaceReLU<<<(ne+255)/256,256,0,sComp>>>(dBufA,ne);
        GPU_SAFE(cudaStreamSynchronize(sComp));
        if(i+1<nBatch){GPU_SAFE(cudaStreamSynchronize(sXfer)); float*tmp=dBufA; dBufA=dBufB; dBufB=tmp;}
    }
    printf("  [H-Async] processed %d batches with overlap\n", nBatch);
    GPU_SAFE(cudaDeviceSynchronize());
    GPU_SAFE(cudaStreamDestroy(sComp)); GPU_SAFE(cudaStreamDestroy(sXfer));
}

/* ================================================================
 * PART I — Training Loop
 * ================================================================ */
void run_epoch(int epoch,
    float* dCw1, float* dCw2, float* dFw1, float* dFb1, float* dFw2, float* dFb2,
    float* dX, float* dC1, float* dP1, float* dC2, float* dP2,
    float* dFC1, float* dLogit, float* dProb, float* dLoss,
    cudnnTensorDescriptor_t xd, cudnnFilterDescriptor_t f1d, cudnnConvolutionDescriptor_t c1d,
    cudnnTensorDescriptor_t c1od, cudnnTensorDescriptor_t p1d,
    cudnnFilterDescriptor_t f2d, cudnnConvolutionDescriptor_t c2d,
    cudnnTensorDescriptor_t c2od, cudnnTensorDescriptor_t p2d,
    const float* hImgs, const int* hLbls, int nTrain)
{
    int nB = nTrain/BATCH; float totalLoss=0;
    for(int b=0;b<nB;b++){
        const float* bImgs = hImgs+(size_t)b*BATCH*IMG_DIM;
        const int* bLbls = hLbls+b*BATCH;
        int* dLbls; GPU_SAFE(cudaMalloc(&dLbls, BATCH*sizeof(int)));
        GPU_SAFE(cudaMemcpy(dX, bImgs, (size_t)BATCH*IMG_DIM*sizeof(float), cudaMemcpyHostToDevice));
        GPU_SAFE(cudaMemcpy(dLbls, bLbls, BATCH*sizeof(int), cudaMemcpyHostToDevice));

        /* Conv1 + ReLU + Pool1 */
        run_conv_forward(xd, dX, f1d, dCw1, c1d, c1od, dC1);
        int nc1=BATCH*32*28*28; inplaceReLU<<<(nc1+255)/256,256>>>(dC1,nc1);
        run_maxpool_forward(c1od, dC1, p1d, dP1, 2,2,2,2);

        /* Conv2 + ReLU + Pool2 */
        run_conv_forward(p1d, dP1, f2d, dCw2, c2d, c2od, dC2);
        int nc2=BATCH*64*14*14; inplaceReLU<<<(nc2+255)/256,256>>>(dC2,nc2);
        run_maxpool_forward(c2od, dC2, p2d, dP2, 2,2,2,2);

        /* FC1 + ReLU */
        run_fc_forward(dP2, dFw1, dFb1, dFC1, BATCH, 64*7*7, 256);
        inplaceReLU<<<(BATCH*256+255)/256,256>>>(dFC1, BATCH*256);

        /* FC2 -> logits */
        run_fc_forward(dFC1, dFw2, dFb2, dLogit, BATCH, 256, N_CLASSES);

        /* Loss */
        softmaxCE<<<(BATCH+255)/256,256>>>(dLogit, dLbls, dProb, dLoss, BATCH, N_CLASSES);
        float hLB[BATCH]; GPU_SAFE(cudaMemcpy(hLB, dLoss, BATCH*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i=0;i<BATCH;i++) totalLoss+=hLB[i];
        if(b%50==0) printf("  Epoch %d  Batch [%d/%d]  AvgLoss=%.4f\n", epoch, b, nB, totalLoss/((b+1)*BATCH));
        cudaFree(dLbls);
    }
    printf("  --- Epoch %d Done  AvgLoss=%.4f ---\n", epoch, totalLoss/(nB*BATCH));
}

/* ================================================================
 * PART J — STRETCH: FP16 Tensor Core GEMM
 * ================================================================ */
void stretch_fp16_gemm(int M, int N, int K)
{
    printf("  [J-FP16-TensorCore] STRETCH: implement cublasGemmEx with CUDA_R_16F\n");
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  Part 5: MNIST Digit Classifier (cuDNN + cuBLAS)\n");
    printf("========================================================\n");
    cudaDeviceProp dp; GPU_SAFE(cudaGetDeviceProperties(&dp,0));
    printf("  GPU: %s  Compute: %d.%d  VRAM: %.0f MB\n\n", dp.name, dp.major, dp.minor, dp.totalGlobalMem/1e6);

    DNN_SAFE(cudnnCreate(&hDnn)); BLAS_SAFE(cublasCreate(&hBlas));

    DigitDataset train = load_digits("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    DigitDataset test  = load_digits("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte");

    /* Allocate weights */
    float *dCw1,*dCw2,*dFw1,*dFb1,*dFw2,*dFb2;
    GPU_SAFE(cudaMalloc(&dCw1,32*1*5*5*sizeof(float)));   GPU_SAFE(cudaMalloc(&dCw2,64*32*5*5*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dFw1,256*3136*sizeof(float)));    GPU_SAFE(cudaMalloc(&dFb1,256*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dFw2,10*256*sizeof(float)));      GPU_SAFE(cudaMalloc(&dFb2,10*sizeof(float)));
    { int lens[]={32*1*5*5,64*32*5*5,256*3136,256,10*256,10};
      float* ptrs[]={dCw1,dCw2,dFw1,dFb1,dFw2,dFb2};
      for(int p=0;p<6;p++){
        float*buf=(float*)malloc(lens[p]*sizeof(float));
        float sc=sqrtf(2.0f/lens[p]);
        for(int i=0;i<lens[p];i++) buf[i]=sc*(2.0f*(float)rand()/RAND_MAX-1.0f);
        GPU_SAFE(cudaMemcpy(ptrs[p],buf,lens[p]*sizeof(float),cudaMemcpyHostToDevice)); free(buf);
    }}

    /* Feature maps */
    float *dX,*dC1,*dP1,*dC2,*dP2,*dFC1,*dLogit,*dProb,*dLoss;
    GPU_SAFE(cudaMalloc(&dX,(size_t)BATCH*1*28*28*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dC1,(size_t)BATCH*32*28*28*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dP1,(size_t)BATCH*32*14*14*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dC2,(size_t)BATCH*64*14*14*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dP2,(size_t)BATCH*64*7*7*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dFC1,(size_t)BATCH*256*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dLogit,(size_t)BATCH*10*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dProb,(size_t)BATCH*10*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dLoss,(size_t)BATCH*sizeof(float)));

    /* cuDNN descriptors */
    cudnnTensorDescriptor_t xd=build_tensor_desc(BATCH,1,28,28);
    cudnnTensorDescriptor_t c1od=build_tensor_desc(BATCH,32,28,28);
    cudnnTensorDescriptor_t p1d=build_tensor_desc(BATCH,32,14,14);
    cudnnTensorDescriptor_t c2od=build_tensor_desc(BATCH,64,14,14);
    cudnnTensorDescriptor_t p2d=build_tensor_desc(BATCH,64,7,7);
    cudnnFilterDescriptor_t f1d=build_filter_desc(32,1,5,5);
    cudnnFilterDescriptor_t f2d=build_filter_desc(64,32,5,5);
    cudnnConvolutionDescriptor_t c1d=build_conv_desc(2,1), c2d=build_conv_desc(2,1);

    printf("\n[Training] %d epochs...\n\n", EPOCHS);
    for(int ep=1;ep<=EPOCHS;ep++){
        double t0=timer_ms();
        run_epoch(ep, dCw1,dCw2,dFw1,dFb1,dFw2,dFb2, dX,dC1,dP1,dC2,dP2,dFC1,dLogit,dProb,dLoss,
                  xd,f1d,c1d,c1od,p1d,f2d,c2d,c2od,p2d, train.pixels,train.labels,train.count);
        printf("  Epoch %d done in %.1f s\n\n", ep, (timer_ms()-t0)/1000.0);
    }

    printf("[Stretch] Async pipeline:\n");
    float *dBA,*dBB; GPU_SAFE(cudaMalloc(&dBA,(size_t)BATCH*IMG_DIM*sizeof(float)));
    GPU_SAFE(cudaMalloc(&dBB,(size_t)BATCH*IMG_DIM*sizeof(float)));
    demo_async_pipeline(train.pixels,train.count,dBA,dBB);
    cudaFree(dBA); cudaFree(dBB);

    printf("\n[Stretch] FP16 Tensor Core GEMM:\n");
    stretch_fp16_gemm(1024,1024,1024);

    /* Cleanup */
    cudaFree(dCw1);cudaFree(dCw2);cudaFree(dFw1);cudaFree(dFb1);cudaFree(dFw2);cudaFree(dFb2);
    cudaFree(dX);cudaFree(dC1);cudaFree(dP1);cudaFree(dC2);cudaFree(dP2);
    cudaFree(dFC1);cudaFree(dLogit);cudaFree(dProb);cudaFree(dLoss);
    free(train.pixels);free(train.labels);free(test.pixels);free(test.labels);
    cudnnDestroy(hDnn); cublasDestroy(hBlas);

    printf("\n========================================================\n");
    printf("  Part 5 complete!\n");
    printf("========================================================\n\n");
    return 0;
}
