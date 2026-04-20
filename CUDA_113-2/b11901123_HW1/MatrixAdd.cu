// Compile for GTX1060
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o MatrixAdd MatrixAdd.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function prototypes
void InitializeRandom(float* data, int n);
bool setupGPU(int* gpuID);
bool allocateMemory(float** host_A, float** host_B, float** host_C, float** host_Ref,
                  float** dev_A, float** dev_B, float** dev_C, int matrixSize);
void releaseResources(float* host_A, float* host_B, float* host_C, float* host_Ref,
                    float* dev_A, float* dev_B, float* dev_C);
bool configureBlocksAndThreads(int matrixSize, int* threadBlockSize, 
                             dim3* threadsPerBlock, dim3* numBlocks);
void runGPUComputation(float* host_A, float* host_B, float* host_C,
                      float* dev_A, float* dev_B, float* dev_C,
                      dim3 numBlocks, dim3 threadsPerBlock,
                      int matrixSize, int memorySize);
void runCPUComputation(float* host_A, float* host_B, float* host_Ref, int matrixSize);
void compareResults(float* host_C, float* host_Ref, int matrixSize);

// Device code - GPU kernel function
__global__ void MatrixReciprocalSum(const float* matA, const float* matB, float* matC, int matSize)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (row < matSize && col < matSize)
        matC[row * matSize + col] = 1.0 / matA[row * matSize + col] + 1.0 / matB[row * matSize + col];
    
    __syncthreads();
}

// Host code
int main()
{
    int gpuID;
    if (!setupGPU(&gpuID)) {
        return 1;
    }
    
    printf("Matrix Reciprocal Sum: C = 1/A + 1/B\n");
    
    // Get matrix size
    int matrixSize;
    printf("Enter the size of the matrices: ");
    scanf("%d", &matrixSize);        
    printf("%d\n", matrixSize);
    
    // Check memory requirements
    int maxMemory = 1024*1024*1024;     // 1 Gigabyte
    if (2*matrixSize*matrixSize > maxMemory) {     // each float takes 4 bytes
        printf("The size of these 3 matrices cannot be fitted into 6 Gbyte\n");
        return 2;
    }
    int memorySize = matrixSize * matrixSize * sizeof(float);
    
    // Allocate memory
    float *host_A, *host_B, *host_C, *host_Ref;
    float *dev_A, *dev_B, *dev_C;
    
    if (!allocateMemory(&host_A, &host_B, &host_C, &host_Ref,
                       &dev_A, &dev_B, &dev_C, matrixSize)) {
        return 3;
    }
    
    // Initialize the input matrices with random numbers
    InitializeRandom(host_A, matrixSize * matrixSize);
    InitializeRandom(host_B, matrixSize * matrixSize);
    
    // Configure blocks and threads
    int threadBlockSize;
    dim3 threadsPerBlock, numBlocks;
    
    if (!configureBlocksAndThreads(matrixSize, &threadBlockSize, &threadsPerBlock, &numBlocks)) {
        releaseResources(host_A, host_B, host_C, host_Ref, dev_A, dev_B, dev_C);
        return 4;
    }
    
    // Run GPU computation
    runGPUComputation(host_A, host_B, host_C, dev_A, dev_B, dev_C, 
                     numBlocks, threadsPerBlock, matrixSize, memorySize);
    
    // Run CPU computation for comparison
    runCPUComputation(host_A, host_B, host_Ref, matrixSize);
    
    // Compare results
    compareResults(host_C, host_Ref, matrixSize);
    
    // Free allocated memory
    releaseResources(host_A, host_B, host_C, host_Ref, dev_A, dev_B, dev_C);
    
    cudaDeviceReset();
    return 0;
}

// Initializes an array with random float entries between 0.0 and 1.0
void InitializeRandom(float* data, int n)
{
    for(int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}

// Setup GPU device
bool setupGPU(int* gpuID)
{
    printf("Enter the GPU ID: ");
    scanf("%d", gpuID);
    printf("%d\n", *gpuID);
    
    cudaError_t errorCode = cudaSetDevice(*gpuID);
    if (errorCode != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", *gpuID);
        return false;
    }
    printf("Set GPU with device ID = %d\n", *gpuID);
    return true;
}

// Allocate memory on host and device
bool allocateMemory(float** host_A, float** host_B, float** host_C, float** host_Ref,
                  float** dev_A, float** dev_B, float** dev_C, int matrixSize)
{
    int memorySize = matrixSize * matrixSize * sizeof(float);
    
    // Allocate input matrices host_A and host_B in host memory
    cudaMallocHost((void**)host_A, memorySize);
    cudaMallocHost((void**)host_B, memorySize);
    cudaMallocHost((void**)host_C, memorySize);
    
    // Allocate matrices in device memory    
    cudaMalloc((void**)dev_A, memorySize);
    cudaMalloc((void**)dev_B, memorySize);
    cudaMalloc((void**)dev_C, memorySize);
    
    // Allocate reference matrix
    *host_Ref = (float*)malloc(memorySize);
    
    return true;
}

// Release allocated resources
void releaseResources(float* host_A, float* host_B, float* host_C, float* host_Ref,
                    float* dev_A, float* dev_B, float* dev_C)
{
    // Free host memory
    if (host_A) cudaFreeHost(host_A);
    if (host_B) cudaFreeHost(host_B);
    if (host_C) cudaFreeHost(host_C);
    if (host_Ref) free(host_Ref);
    
    // Free device memory
    if (dev_A) cudaFree(dev_A);
    if (dev_B) cudaFree(dev_B);
    if (dev_C) cudaFree(dev_C);
}

// Configure blocks and threads for GPU execution
bool configureBlocksAndThreads(int matrixSize, int* threadBlockSize, 
                             dim3* threadsPerBlock, dim3* numBlocks)
{
config_loop:
    printf("Enter the block size: ");
    scanf("%d", threadBlockSize);
    printf("%d\n", *threadBlockSize);

    *threadsPerBlock = dim3(*threadBlockSize, *threadBlockSize);
    if (threadsPerBlock->x * threadsPerBlock->y > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        goto config_loop;
    }

    *numBlocks = dim3((matrixSize + *threadBlockSize - 1) / *threadBlockSize, 
                    (matrixSize + *threadBlockSize - 1) / *threadBlockSize);
    printf("The number of blocks is %d\n", numBlocks->x * numBlocks->y);
    // 2^31 - 1 = 2147483647
    if (numBlocks->x * numBlocks->y > 2147483647) {
        printf("The number of blocks must be less than 2147483647!\n");
        goto config_loop;
    }
    
    return true;
}

// Run GPU computation and measure performance
void runGPUComputation(float* host_A, float* host_B, float* host_C,
                      float* dev_A, float* dev_B, float* dev_C,
                      dim3 numBlocks, dim3 threadsPerBlock,
                      int matrixSize, int memorySize)
{
    // Create the timer
    cudaEvent_t timerStart, timerStop;
    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);

    // Start the timer for input operations
    cudaEventRecord(timerStart, 0);

    // Copy matrices from host memory to device memory
    cudaMemcpy(dev_A, host_A, memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, memorySize, cudaMemcpyHostToDevice);
    
    // Stop the timer
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    float inputTime;
    cudaEventElapsedTime(&inputTime, timerStart, timerStop);
    printf("Input time for GPU: %f (ms)\n", inputTime);

    // Start the timer for computation
    cudaEventRecord(timerStart, 0);

    MatrixReciprocalSum<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, matrixSize);
    
    // Stop the timer
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    float gpuComputeTime;
    cudaEventElapsedTime(&gpuComputeTime, timerStart, timerStop);
    printf("Processing time for GPU: %f (ms)\n", gpuComputeTime);
    printf("GPU Gflops: %f\n", 3*matrixSize*matrixSize/(1000000.0*gpuComputeTime));
    
    // Start the timer for output operations
    cudaEventRecord(timerStart, 0);

    // Copy result from device memory to host memory
    cudaMemcpy(host_C, dev_C, memorySize, cudaMemcpyDeviceToHost);
    
    // Stop the timer
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    float outputTime;
    cudaEventElapsedTime(&outputTime, timerStart, timerStop);
    printf("Output time for GPU: %f (ms)\n", outputTime);

    float totalGpuTime = inputTime + gpuComputeTime + outputTime;
    printf("Total time for GPU: %f (ms)\n", totalGpuTime);
    
    // Destroy the timer
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
}

// Run CPU computation for comparison
void runCPUComputation(float* host_A, float* host_B, float* host_Ref, int matrixSize)
{
    // Create timer events
    cudaEvent_t timerStart, timerStop;
    cudaEventCreate(&timerStart);
    cudaEventCreate(&timerStop);
    
    // Start the timer
    cudaEventRecord(timerStart, 0);

    // Compute reference solution on CPU
    for (int i = 0; i < matrixSize * matrixSize; ++i) 
        host_Ref[i] = 1.0 / host_A[i] + 1.0 / host_B[i];
    
    // Stop the timer
    cudaEventRecord(timerStop, 0);
    cudaEventSynchronize(timerStop);

    float cpuTime;
    cudaEventElapsedTime(&cpuTime, timerStart, timerStop);
    printf("Processing time for CPU: %f (ms)\n", cpuTime);
    printf("CPU Gflops: %f\n", 3*matrixSize*matrixSize/(1000000.0*cpuTime));
    
    // Destroy timer events
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
}

// Compare GPU and CPU results
void compareResults(float* host_C, float* host_Ref, int matrixSize)
{
    printf("Check result:\n");
    double errorSum = 0.0; 
    double difference;
    
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        difference = abs(host_Ref[i] - host_C[i]);
        errorSum += difference*difference; 
    }
    
    errorSum = sqrt(errorSum);
    printf("norm(host_C - host_Ref)=%20.15e\n\n", errorSum);
}