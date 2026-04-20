// Compile for GTX1060
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -Xcompiler -fopenmp -o DotProduct DotProduct.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

// Global memory pointers and computation variables
float* host_vector_x;        // First input vector on host
float* host_vector_y;        // Second input vector on host
double cpu_dot_result = 0.0; // CPU reference computation result
double* gpu_partial_sums;    // Array to store partial sums from each GPU

// Function prototypes
void initialize_random_vectors(float* vector, int vector_size);

__global__ void compute_dot_product_kernel(const float* vec_a, const float* vec_b, 
                                         float* block_sums, int elements_count)
{
    // Shared memory for intra-block reduction
    extern __shared__ float shared_cache[];
    
    // Calculate global thread index and cache position
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cache_position = threadIdx.x;
    
    // Initialize partial sum for this thread
    float thread_sum = 0.0;
    
    // Grid-stride loop to handle vectors larger than grid size
    while (global_idx < elements_count) {
        thread_sum += vec_a[global_idx] * vec_b[global_idx];
        global_idx += blockDim.x * gridDim.x;
    }
    
    // Store thread result in shared memory
    shared_cache[cache_position] = thread_sum;
    __syncthreads();
    
    // Perform tree-based reduction in shared memory
    // Note: block size must be power of 2 for this reduction
    int reduction_stride = blockDim.x / 2;
    while (reduction_stride != 0) {
        if (cache_position < reduction_stride) {
            shared_cache[cache_position] += shared_cache[cache_position + reduction_stride];
        }
        __syncthreads();
        reduction_stride /= 2;
    }
    
    // Thread 0 writes block result to global memory
    if (cache_position == 0) {
        block_sums[blockIdx.x] = shared_cache[0];
    }
}

int main(void)
{
    printf("=== Multi-GPU Vector Dot Product Calculator ===\n\n");
    
    // Configuration variables
    int vector_length, num_gpus, current_thread_id = 0;
    int* device_ids;
    long memory_limit = 1024 * 1024 * 1024;  // 1GB memory limit
    
    // Get number of GPUs to use
    printf("Specify number of GPUs to utilize: ");
    scanf("%d", &num_gpus);
    printf("Using %d GPU(s)\n", num_gpus);
    
    // Allocate and read GPU device IDs
    device_ids = (int*)malloc(sizeof(int) * num_gpus);
    int devices_read = 0;
    printf("Enter GPU device IDs (space-separated): ");
    for(int gpu_idx = 0; gpu_idx < num_gpus; gpu_idx++) {
        scanf("%d", &device_ids[gpu_idx]);
        printf("%d ", device_ids[gpu_idx]);
        devices_read++;
        if(getchar() == '\n') break;
    }
    printf("\n");
    
    // Validate input
    if(devices_read != num_gpus) {
        fprintf(stderr, "Error: Expected %d GPU device IDs\n", num_gpus);
        exit(1);
    }
    
    // Get vector size
    printf("Enter vector size (number of elements): ");
    scanf("%d", &vector_length);
    printf("Vector size: %d elements\n", vector_length);
    
    // Check memory requirements
    if (3 * vector_length * sizeof(float) > memory_limit) {
        printf("Error: Vector size exceeds memory limit of 1GB\n");
        exit(1);
    }
    
    // Configure CUDA execution parameters
    int threads_per_block;
    printf("Enter threads per block (must be power of 2): ");
    scanf("%d", &threads_per_block);
    printf("Threads per block: %d\n", threads_per_block);
    
    if(threads_per_block > 1024) {
        printf("Error: Threads per block cannot exceed 1024\n");
        exit(1);
    }
    
    int blocks_per_grid;
    printf("Enter blocks per grid: ");
    scanf("%d", &blocks_per_grid);
    printf("Blocks per grid: %d\n", blocks_per_grid);
    
    if(blocks_per_grid > 2147483647) {
        printf("Error: Number of blocks exceeds maximum limit\n");
        exit(1);
    }
    
    // Memory allocation sizes
    int vector_bytes = vector_length * sizeof(float);
    int block_results_bytes = blocks_per_grid * sizeof(float);
    
    // Allocate host memory using pinned memory for faster transfers
    gpu_partial_sums = (double*)malloc(num_gpus * sizeof(double));
    cudaMallocHost((void**)&host_vector_x, vector_bytes);
    cudaMallocHost((void**)&host_vector_y, vector_bytes);
    
    // Initialize vectors with random values
    printf("\nInitializing vectors with random values...\n");
    initialize_random_vectors(host_vector_x, vector_length);
    initialize_random_vectors(host_vector_y, vector_length);
    
    // CUDA timing events
    cudaEvent_t timer_start, timer_stop;
    float data_transfer_in_time, gpu_computation_time, data_transfer_out_time;
    
    // Set up OpenMP for multi-GPU execution
    omp_set_num_threads(num_gpus);
    
    printf("\nStarting multi-GPU computation...\n");
    
    // Parallel execution across multiple GPUs
    #pragma omp parallel private(current_thread_id)
    {
        // Per-thread variables
        float *host_block_results;
        float *device_vec_x, *device_vec_y, *device_block_sums;
        
        current_thread_id = omp_get_thread_num();
        cudaSetDevice(device_ids[current_thread_id]);
        
        // Synchronize before starting timing
        #pragma omp barrier
        
        // Initialize timing (only master thread)
        if(current_thread_id == 0) {
            cudaEventCreate(&timer_start);
            cudaEventCreate(&timer_stop);
            cudaEventRecord(timer_start, 0);
        }
        
        // Allocate host memory for block results
        host_block_results = (float*)malloc(block_results_bytes);
        
        // Allocate device memory for this GPU's portion
        cudaMalloc((void**)&device_vec_x, vector_bytes / num_gpus);
        cudaMalloc((void**)&device_vec_y, vector_bytes / num_gpus);
        cudaMalloc((void**)&device_block_sums, block_results_bytes);
        
        // Copy data segments to respective GPUs
        int elements_per_gpu = vector_length / num_gpus;
        int start_offset = elements_per_gpu * current_thread_id;
        
        cudaMemcpy(device_vec_x, host_vector_x + start_offset, 
                  vector_bytes / num_gpus, cudaMemcpyHostToDevice);
        cudaMemcpy(device_vec_y, host_vector_y + start_offset, 
                  vector_bytes / num_gpus, cudaMemcpyHostToDevice);
        
        #pragma omp barrier
        
        // Record data transfer time
        if(current_thread_id == 0) {
            cudaEventRecord(timer_stop, 0);
            cudaEventSynchronize(timer_stop);
            cudaEventElapsedTime(&data_transfer_in_time, timer_start, timer_stop);
            printf("Data transfer to GPUs: %.2f ms\n", data_transfer_in_time);
            fflush(stdout);
        }
        
        #pragma omp barrier
        
        // Start computation timing
        if(current_thread_id == 0) {
            cudaEventRecord(timer_start, 0);
        }
        
        // Launch kernel with appropriate shared memory size
        int shared_memory_bytes = threads_per_block * sizeof(float);
        compute_dot_product_kernel<<<blocks_per_grid, threads_per_block, shared_memory_bytes>>>
            (device_vec_x, device_vec_y, device_block_sums, elements_per_gpu);
        
        // Ensure kernel completion
        cudaDeviceSynchronize();
        
        #pragma omp barrier
        
        // Record computation time
        if(current_thread_id == 0) {
            cudaEventRecord(timer_stop, 0);
            cudaEventSynchronize(timer_stop);
            cudaEventElapsedTime(&gpu_computation_time, timer_start, timer_stop);
            printf("GPU computation time: %.2f ms\n", gpu_computation_time);
            printf("GPU performance: %.2f GFLOPS\n", 
                   2.0 * vector_length / (1000000.0 * gpu_computation_time));
            fflush(stdout);
        }
        
        #pragma omp barrier
        
        // Start output timing
        if(current_thread_id == 0) {
            cudaEventRecord(timer_start, 0);
        }
        
        // Transfer results back and perform final reduction
        cudaMemcpy(host_block_results, device_block_sums, 
                  block_results_bytes, cudaMemcpyDeviceToHost);
        
        // Sum up all block results for this GPU
        gpu_partial_sums[current_thread_id] = 0.0;
        for(int block_idx = 0; block_idx < blocks_per_grid; block_idx++) {
            gpu_partial_sums[current_thread_id] += (double)host_block_results[block_idx];
        }
        
        // Clean up device memory
        cudaFree(device_vec_x);
        cudaFree(device_vec_y);
        cudaFree(device_block_sums);
        free(host_block_results);
        
        #pragma omp barrier
        
        // Record output time
        if(current_thread_id == 0) {
            cudaEventRecord(timer_stop, 0);
            cudaEventSynchronize(timer_stop);
            cudaEventElapsedTime(&data_transfer_out_time, timer_start, timer_stop);
            printf("Data transfer from GPUs: %.2f ms\n", data_transfer_out_time);
            fflush(stdout);
        }
    }
    
    // Calculate total GPU time and final result
    float total_gpu_time = data_transfer_in_time + gpu_computation_time + data_transfer_out_time;
    printf("Total GPU execution time: %.2f ms\n", total_gpu_time);
    
    // Combine results from all GPUs
    double final_gpu_result = 0.0;
    for(int gpu_idx = 0; gpu_idx < num_gpus; gpu_idx++) {
        final_gpu_result += gpu_partial_sums[gpu_idx];
    }
    
    // Compute CPU reference solution for verification
    printf("\nComputing CPU reference solution...\n");
    cudaEventRecord(timer_start, 0);
    
    for(int i = 0; i < vector_length; i++) {
        cpu_dot_result += (double)(host_vector_x[i] * host_vector_y[i]);
    }
    
    cudaEventRecord(timer_stop, 0);
    cudaEventSynchronize(timer_stop);
    
    float cpu_computation_time;
    cudaEventElapsedTime(&cpu_computation_time, timer_start, timer_stop);
    printf("CPU computation time: %.2f ms\n", cpu_computation_time);
    printf("CPU performance: %.2f GFLOPS\n", 
           2.0 * vector_length / (1000000.0 * cpu_computation_time));
    printf("GPU speedup factor: %.2fx\n", cpu_computation_time / total_gpu_time);
    
    // Clean up timing resources
    cudaEventDestroy(timer_start);
    cudaEventDestroy(timer_stop);
    
    // Verify results
    printf("\n=== RESULTS VERIFICATION ===\n");
    double relative_error = fabs((cpu_dot_result - final_gpu_result) / cpu_dot_result);
    printf("Relative error: %.15e\n", relative_error);
    printf("GPU result:     %.15e\n", final_gpu_result);
    printf("CPU result:     %.15e\n", cpu_dot_result);
    
    if(relative_error < 1e-10) {
        printf("✓ Results match within acceptable tolerance\n");
    } else {
        printf("✗ Warning: Results differ significantly\n");
    }
    
    // Clean up host memory
    cudaFreeHost(host_vector_x);
    cudaFreeHost(host_vector_y);
    free(gpu_partial_sums);
    
    // Reset all GPUs
    for(int gpu_idx = 0; gpu_idx < num_gpus; gpu_idx++) {
        cudaSetDevice(device_ids[gpu_idx]);
        cudaDeviceReset();
    }
    
    printf("\nExecution completed successfully.\n");
    return 0;
}

/**
 * Initialize vector with random floating-point values in range [-1, 1]
 * @param vector: Pointer to the vector to initialize
 * @param size: Number of elements in the vector
 */
void initialize_random_vectors(float* vector, int size)
{
    for (int i = 0; i < size; ++i) {
        vector[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
    }
}