// Compile for GTX1060
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o MatrixTrace MatrixTrace.cu

#include <stdio.h>
#include <stdlib.h>

// Host and device pointers
float* h_matrix = NULL;     // Host: full matrix (flattened)
float* h_diag = NULL;       // Host: diagonal elements
float* h_blockSums = NULL;  // Host: partial sums from each block

float* d_diag = NULL;       // Device: diagonal elements
float* d_blockSums = NULL;  // Device: partial sums from each block

// Function to fill an array with random floats in (-1,1)
void RandomInit(float* data, int n);

// CUDA kernel: computes the sum of elements in diag (i.e., trace)
__global__ void MatrixTrace(const float* diag, float* blockSums, int numElements) {
    extern __shared__ float cache[]; // Shared memory for reduction

    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int localIdx = threadIdx.x;

    float sum = 0.0f;
    // Stride loop to sum up diagonal elements assigned to this thread
    while (globalIdx < numElements) {
        sum += diag[globalIdx];
        globalIdx += blockDim.x * gridDim.x;
    }
    cache[localIdx] = sum;

    __syncthreads();

    // Parallel reduction within the block (threadsPerBlock must be a power of 2)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localIdx < stride) {
            cache[localIdx] += cache[localIdx + stride];
        }
        __syncthreads();
    }

    // First thread in block writes the block's sum to global memory
    if (localIdx == 0) {
        blockSums[blockIdx.x] = cache[0];
    }
}

int main(void) {
    int gpuID;
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU ID: ");
    scanf("%d", &gpuID);
    printf("%d\n", gpuID);

    err = cudaSetDevice(gpuID);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gpuID);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gpuID);

    printf("Trace of a Matrix: tr(A)\n");
    int matrixSize;
    printf("Enter the size of the matrix: ");
    scanf("%d", &matrixSize);
    printf("%d\n", matrixSize);

    int threadsPerBlock;
    printf("Enter the number (2^m) of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        exit(1);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);
    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647!\n");
        exit(1);
    }

    // Allocate host memory
    int matrixBytes = matrixSize * matrixSize * sizeof(float);
    int diagBytes = matrixSize * sizeof(float);
    int blockSumsBytes = blocksPerGrid * sizeof(float);

    h_matrix = (float*)malloc(matrixBytes);
    h_diag = (float*)malloc(diagBytes);
    h_blockSums = (float*)malloc(blockSumsBytes);

    // Initialize matrix and extract diagonal
    RandomInit(h_matrix, matrixSize * matrixSize);
    for (int i = 0; i < matrixSize; i++) {
        h_diag[i] = h_matrix[i * matrixSize + i];
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&d_diag, diagBytes);
    cudaMalloc((void**)&d_blockSums, blockSumsBytes);

    // Copy diagonal to device
    cudaMemcpy(d_diag, h_diag, diagBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float inputTime;
    cudaEventElapsedTime(&inputTime, start, stop);
    printf("Input time for GPU: %f (ms)\n", inputTime);

    // Launch kernel
    cudaEventRecord(start, 0);
    int sharedMemSize = threadsPerBlock * sizeof(float);
    MatrixTrace<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_diag, d_blockSums, matrixSize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpuComputeTime;
    cudaEventElapsedTime(&gpuComputeTime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gpuComputeTime);
    printf("GPU Gflops: %f\n", matrixSize / (1000000.0 * gpuComputeTime));

    // Copy result back to host
    cudaEventRecord(start, 0);
    cudaMemcpy(h_blockSums, d_blockSums, blockSumsBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_diag);
    cudaFree(d_blockSums);

    // Final reduction on host
    double traceGPU = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        traceGPU += (double)h_blockSums[i];
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float outputTime;
    cudaEventElapsedTime(&outputTime, start, stop);
    printf("Output time for GPU: %f (ms)\n", outputTime);

    float totalGPUTime = inputTime + gpuComputeTime + outputTime;
    printf("Total time for GPU: %f (ms)\n", totalGPUTime);

    // CPU reference calculation
    cudaEventRecord(start, 0);
    double traceCPU = 0.0;
    for (int i = 0; i < matrixSize; i++) {
        traceCPU += (double)h_matrix[i * matrixSize + i];
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cpuTime;
    cudaEventElapsedTime(&cpuTime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cpuTime);
    printf("CPU Gflops: %f\n", matrixSize / (1000000.0 * cpuTime));
    printf("Speed up of GPU = %f\n", cpuTime / totalGPUTime);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check result
    printf("Check result:\n");
    double diff = fabs(traceCPU - traceGPU) / traceCPU;
    printf("|traceCPU - traceGPU| / traceCPU = %20.15e\n", diff);
    printf("traceGPU   = %20.15e\n", traceGPU);
    printf("traceCPU   = %20.15e\n", traceCPU);
    printf("\n");

    // Free host memory
    free(h_matrix);
    free(h_diag);
    free(h_blockSums);

    cudaDeviceReset();
}

// Fill array with random floats in (-1,1)
void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
        // data[i] = 1.0f; // Uncomment to set all elements to one for debugging
    }
}