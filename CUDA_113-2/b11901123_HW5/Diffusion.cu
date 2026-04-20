// Compile for GTX1060
// nvcc -arch=compute_61 -code=sm_61 -O3 -Xcompiler -fopenmp -o Diffusion Diffusion.cu -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

// Global arrays for temperature field
float* host_field_A;     // Temperature field buffer A
float* host_field_B;     // Temperature field buffer B  
float* host_error_sum;   // Error accumulation array
float* final_solution;   // Final temperature distribution
float** gpu_field_A;     // GPU temperature field A
float** gpu_field_B;     // GPU temperature field B
float** gpu_error_sum;   // GPU error reduction array

// Simulation parameters
const int MAX_ITERATIONS = 10000000;
const double CONVERGENCE_THRESHOLD = 1.0e-10;
const float TOP_TEMPERATURE = 400.0f;     // Top boundary temperature (K)
const float BOUNDARY_TEMPERATURE = 273.0f; // Other boundaries temperature (K)

__global__ void thermalDiffusionKernel(float* current_temp, float* left_neighbor, 
                                      float* right_neighbor, float* bottom_neighbor,
                                      float* top_neighbor, float* next_temp, 
                                      float* error_array)
{
    extern __shared__ float shared_memory[];
    
    float top_val, left_val, center_val, right_val, bottom_val;
    float temperature_diff = 0.0f;
    int grid_position, boundary_skip;

    // Calculate grid dimensions and thread position
    int subdomain_width = blockDim.x * gridDim.x;
    int subdomain_height = blockDim.y * gridDim.y;
    int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
    int shared_index = threadIdx.x + threadIdx.y * blockDim.x;

    grid_position = thread_x + thread_y * subdomain_width;
    boundary_skip = 0;
    temperature_diff = 0.0;
    
    // Initialize neighbor values
    bottom_val = left_val = right_val = top_val = 0.0;
    center_val = current_temp[grid_position];
    
    // Handle left boundary conditions
    if (thread_x == 0) {
        if (left_neighbor != NULL) {
            left_val = left_neighbor[(subdomain_width-1) + thread_y * subdomain_width];
            right_val = current_temp[grid_position + 1];
        } else {
            boundary_skip = 1; // Left edge of entire domain
        }
    }
    // Handle right boundary conditions  
    else if (thread_x == subdomain_width - 1) {
        if (right_neighbor != NULL) {
            left_val = current_temp[grid_position - 1];
            right_val = right_neighbor[thread_y * subdomain_width];
        } else {
            boundary_skip = 1; // Right edge of entire domain
        }
    }
    // Interior points in x-direction
    else {
        left_val = current_temp[grid_position - 1];
        right_val = current_temp[grid_position + 1];
    }

    // Handle bottom boundary conditions
    if (thread_y == 0) {
        if (bottom_neighbor != NULL) {
            bottom_val = bottom_neighbor[thread_x + (subdomain_height-1) * subdomain_width];
            top_val = current_temp[grid_position + subdomain_width];
        } else {
            boundary_skip = 1; // Bottom edge of entire domain
        }
    }
    // Handle top boundary conditions
    else if (thread_y == subdomain_height - 1) {
        if (top_neighbor != NULL) {
            bottom_val = current_temp[grid_position - subdomain_width];
            top_val = top_neighbor[thread_x];
        } else {
            boundary_skip = 1; // Top edge of entire domain
        }
    }
    // Interior points in y-direction
    else {
        bottom_val = current_temp[grid_position - subdomain_width];
        top_val = current_temp[grid_position + subdomain_width];
    }

    // Apply Jacobi iteration for interior points
    if (boundary_skip == 0) {
        next_temp[grid_position] = 0.25f * (bottom_val + left_val + right_val + top_val);
        temperature_diff = next_temp[grid_position] - center_val;
    }

    // Store squared error in shared memory for reduction
    shared_memory[shared_index] = temperature_diff * temperature_diff;
    __syncthreads();

    // Parallel reduction to compute block error sum
    int reduction_size = blockDim.x * blockDim.y / 2;
    while (reduction_size != 0) {
        if (shared_index < reduction_size) {
            shared_memory[shared_index] += shared_memory[shared_index + reduction_size];
        }
        __syncthreads();
        reduction_size /= 2;
    }

    // Store block error sum to global memory
    int block_index = blockIdx.x + gridDim.x * blockIdx.y;
    if (shared_index == 0) {
        error_array[block_index] = shared_memory[0];
    }
}

void setupBoundaryConditions(float* temperature_grid, int domain_width, int domain_height) {
    printf("Setting up thermal boundary conditions...\n");
    printf("  Top edge: %.1f K\n", TOP_TEMPERATURE);
    printf("  Other edges: %.1f K\n", BOUNDARY_TEMPERATURE);
    
    for (int y = 0; y < domain_height; ++y) {
        for (int x = 0; x < domain_width; ++x) {
            if (y == domain_height - 1) {
                // Top edge - high temperature
                temperature_grid[x + y * domain_width] = TOP_TEMPERATURE;
            } else if (y == 0 || x == 0 || x == domain_width - 1) {
                // Left, right, and bottom edges - low temperature
                temperature_grid[x + y * domain_width] = BOUNDARY_TEMPERATURE;
            } else {
                // Interior points - initial guess
                temperature_grid[x + y * domain_width] = 0.0f;
            }
        }
    }
}

int main(void) {
    // System configuration
    volatile bool buffer_toggle = true;
    int thread_id = 0;
    int num_gpus, *gpu_devices;
    int domain_width, domain_height;           // Total domain dimensions
    int subdomain_width, subdomain_height;     // Per-GPU subdomain dimensions
    int gpu_grid_x, gpu_grid_y;               // GPU grid partitioning
    int block_size_x, block_size_y;           // CUDA block dimensions
    int num_blocks_x, num_blocks_y;           // Number of blocks per GPU
    int shared_mem_size;                      // Shared memory per block
    int iteration_count = 0;                  // Iteration counter
    float computation_time, total_time;       // Timing variables
    float data_transfer_in, data_transfer_out;
    double floating_point_ops, convergence_error;
    cudaEvent_t timer_start, timer_stop;

    // Get system configuration
    printf("=== Multi-GPU Heat Diffusion Solver ===\n");
    printf("Solving 2D thermal equilibrium problem\n\n");
    
    printf("Enter GPU grid configuration (NGx NGy): ");
    scanf("%d %d", &gpu_grid_x, &gpu_grid_y);
    printf("GPU grid: %d x %d\n", gpu_grid_x, gpu_grid_y);
    
    num_gpus = gpu_grid_x * gpu_grid_y;
    gpu_devices = (int*)malloc(sizeof(int) * num_gpus);
    
    for (int i = 0; i < num_gpus; i++) {
        printf("Enter GPU device ID %d: ", i);
        scanf("%d", &gpu_devices[i]);
        printf("Using GPU device %d\n", gpu_devices[i]);
    }

    printf("\nEnter domain dimensions (Nx Ny): ");
    scanf("%d %d", &domain_width, &domain_height);
    printf("Domain size: %d x %d\n", domain_width, domain_height);
    
    // Validate domain partitioning
    if (domain_width % gpu_grid_x != 0) {
        printf("ERROR: Domain width must be divisible by GPU grid width\n");
        exit(1);
    }
    if (domain_height % gpu_grid_y != 0) {
        printf("ERROR: Domain height must be divisible by GPU grid height\n");
        exit(1);
    }
    
    subdomain_width = domain_width / gpu_grid_x;
    subdomain_height = domain_height / gpu_grid_y;
    printf("Subdomain per GPU: %d x %d\n", subdomain_width, subdomain_height);

    printf("\nEnter CUDA block size (tx ty): ");
    scanf("%d %d", &block_size_x, &block_size_y);
    printf("Block size: %d x %d\n", block_size_x, block_size_y);
    
    dim3 cuda_threads(block_size_x, block_size_y);
    
    // Calculate grid dimensions
    num_blocks_x = domain_width / block_size_x;
    num_blocks_y = domain_height / block_size_y;
    
    if (num_blocks_x * block_size_x != domain_width || 
        num_blocks_y * block_size_y != domain_height) {
        printf("ERROR: Domain dimensions must be divisible by block size\n");
        exit(1);
    }
    
    if ((num_blocks_x / gpu_grid_x > 65535) || (num_blocks_y / gpu_grid_y > 65535)) {
        printf("ERROR: Grid size exceeds CUDA limits\n");
        exit(1);
    }
    
    dim3 cuda_blocks(num_blocks_x / gpu_grid_x, num_blocks_y / gpu_grid_y);
    printf("Blocks per GPU: %d x %d\n", cuda_blocks.x, cuda_blocks.y);

    // Initialize memory
    convergence_error = 10 * CONVERGENCE_THRESHOLD;
    buffer_toggle = true;
    
    int total_elements = domain_width * domain_height;
    int memory_size = total_elements * sizeof(float);
    int error_array_size = num_blocks_x * num_blocks_y * sizeof(float);
    
    host_field_A = (float*)malloc(memory_size);
    host_field_B = (float*)malloc(memory_size);
    host_error_sum = (float*)malloc(error_array_size);
    final_solution = (float*)malloc(memory_size);

    // Setup initial conditions
    memset(host_field_A, 0, memory_size);
    memset(host_field_B, 0, memory_size);
    setupBoundaryConditions(host_field_A, domain_width, domain_height);
    setupBoundaryConditions(host_field_B, domain_width, domain_height);

    // Allocate GPU memory
    printf("\nAllocating GPU memory...\n");
    shared_mem_size = block_size_x * block_size_y * sizeof(float);
    
    gpu_field_A = (float**)malloc(num_gpus * sizeof(float*));
    gpu_field_B = (float**)malloc(num_gpus * sizeof(float*));
    gpu_error_sum = (float**)malloc(num_gpus * sizeof(float*));

    omp_set_num_threads(num_gpus);
    #pragma omp parallel private(thread_id)
    {
        int gpu_x, gpu_y;
        thread_id = omp_get_thread_num();
        gpu_x = thread_id % gpu_grid_x;
        gpu_y = thread_id / gpu_grid_x;
        cudaSetDevice(gpu_devices[thread_id]);

        // Enable peer-to-peer access
        int right_gpu = ((gpu_x + 1) % gpu_grid_x) + gpu_y * gpu_grid_x;
        cudaDeviceEnablePeerAccess(gpu_devices[right_gpu], 0);
        int left_gpu = ((gpu_x + gpu_grid_x - 1) % gpu_grid_x) + gpu_y * gpu_grid_x;
        cudaDeviceEnablePeerAccess(gpu_devices[left_gpu], 0);
        int top_gpu = gpu_x + ((gpu_y + 1) % gpu_grid_y) * gpu_grid_x;
        cudaDeviceEnablePeerAccess(gpu_devices[top_gpu], 0);
        int bottom_gpu = gpu_x + ((gpu_y + gpu_grid_y - 1) % gpu_grid_y) * gpu_grid_x;
        cudaDeviceEnablePeerAccess(gpu_devices[bottom_gpu], 0);

        // Initialize timing
        if (thread_id == 0) {
            cudaEventCreate(&timer_start);
            cudaEventCreate(&timer_stop);
            cudaEventRecord(timer_start, 0);
        }

        // Allocate device memory
        cudaMalloc((void**)&gpu_field_A[thread_id], memory_size / num_gpus);
        cudaMalloc((void**)&gpu_field_B[thread_id], memory_size / num_gpus);
        cudaMalloc((void**)&gpu_error_sum[thread_id], error_array_size / num_gpus);

        // Transfer initial data to GPU
        for (int i = 0; i < subdomain_height; i++) {
            float *host_ptr = host_field_A + gpu_x * subdomain_width + 
                             (gpu_y * subdomain_height + i) * domain_width;
            float *device_ptr = gpu_field_A[thread_id] + i * subdomain_width;
            cudaMemcpy(device_ptr, host_ptr, subdomain_width * sizeof(float), 
                      cudaMemcpyHostToDevice);
        }
        
        for (int i = 0; i < subdomain_height; i++) {
            float *host_ptr = host_field_B + gpu_x * subdomain_width + 
                             (gpu_y * subdomain_height + i) * domain_width;
            float *device_ptr = gpu_field_B[thread_id] + i * subdomain_width;
            cudaMemcpy(device_ptr, host_ptr, subdomain_width * sizeof(float), 
                      cudaMemcpyHostToDevice);
        }

        #pragma omp barrier

        if (thread_id == 0) {
            cudaEventRecord(timer_stop, 0);
            cudaEventSynchronize(timer_stop);
            cudaEventElapsedTime(&data_transfer_in, timer_start, timer_stop);
            printf("Data transfer to GPU: %.2f ms\n", data_transfer_in);
        }
    }

    // Main computation loop
    cudaEventRecord(timer_start, 0);
    printf("\nSolving thermal equilibrium...\n");
    printf("Convergence threshold: %.2e\n", CONVERGENCE_THRESHOLD);
    
    while ((convergence_error > CONVERGENCE_THRESHOLD) && (iteration_count < MAX_ITERATIONS)) {
        #pragma omp parallel private(thread_id)
        {
            int gpu_x, gpu_y;
            thread_id = omp_get_thread_num();
            gpu_x = thread_id % gpu_grid_x;
            gpu_y = thread_id / gpu_grid_x;
            cudaSetDevice(gpu_devices[thread_id]);

            float **current_field, **next_field;
            float *left_boundary, *right_boundary, *top_boundary, *bottom_boundary;
            float *current_gpu, *next_gpu;
            
            current_field = (buffer_toggle) ? gpu_field_A : gpu_field_B;
            next_field = (buffer_toggle) ? gpu_field_B : gpu_field_A;
            current_gpu = current_field[thread_id];
            next_gpu = next_field[thread_id];
            
            // Set up boundary pointers for inter-GPU communication
            left_boundary = (gpu_x == 0) ? NULL : current_field[gpu_x - 1 + gpu_y * gpu_grid_x];
            right_boundary = (gpu_x == gpu_grid_x - 1) ? NULL : current_field[gpu_x + 1 + gpu_y * gpu_grid_x];
            bottom_boundary = (gpu_y == 0) ? NULL : current_field[gpu_x + (gpu_y - 1) * gpu_grid_x];
            top_boundary = (gpu_y == gpu_grid_y - 1) ? NULL : current_field[gpu_x + (gpu_y + 1) * gpu_grid_x];

            // Launch thermal diffusion kernel
            thermalDiffusionKernel<<<cuda_blocks, cuda_threads, shared_mem_size>>>(
                current_gpu, left_boundary, right_boundary, bottom_boundary,
                top_boundary, next_gpu, gpu_error_sum[thread_id]);
            cudaDeviceSynchronize();

            // Copy error data back to host
            cudaMemcpy(host_error_sum + num_blocks_x * num_blocks_y / num_gpus * thread_id, 
                      gpu_error_sum[thread_id], error_array_size / num_gpus, 
                      cudaMemcpyDeviceToHost);
        }

        // Compute global error
        convergence_error = 0.0;
        for (int i = 0; i < num_blocks_x * num_blocks_y; i++) {
            convergence_error += host_error_sum[i];
        }
        convergence_error = sqrt(convergence_error);

        iteration_count++;
        buffer_toggle = !buffer_toggle;
    }
    
    printf("Convergence achieved!\n");
    printf("Final error: %.15e\n", convergence_error);
    
    cudaEventRecord(timer_stop, 0);
    cudaEventSynchronize(timer_stop);
    cudaEventElapsedTime(&computation_time, timer_start, timer_stop);
    
    floating_point_ops = 7.0 * (domain_width - 2) * (domain_height - 2) * iteration_count;
    printf("Computation time: %.2f ms\n", computation_time);
    printf("Performance: %.2f GFlops\n", floating_point_ops / (1000000.0 * computation_time));

    // Transfer final results back to host
    cudaEventRecord(timer_start, 0);
    printf("\nTransferring results...\n");
    
    #pragma omp parallel private(thread_id)
    {
        int gpu_x, gpu_y;
        thread_id = omp_get_thread_num();
        gpu_x = thread_id % gpu_grid_x;
        gpu_y = thread_id / gpu_grid_x;
        cudaSetDevice(gpu_devices[thread_id]);

        float* final_gpu_field = (buffer_toggle) ? gpu_field_B[thread_id] : gpu_field_A[thread_id];
        
        for (int i = 0; i < subdomain_height; i++) {
            float *host_ptr = final_solution + gpu_x * subdomain_width + 
                             (gpu_y * subdomain_height + i) * domain_width;
            float *device_ptr = final_gpu_field + i * subdomain_width;
            cudaMemcpy(host_ptr, device_ptr, subdomain_width * sizeof(float), 
                      cudaMemcpyDeviceToHost);
        }
        
        // Clean up GPU memory
        cudaFree(gpu_field_A[thread_id]);
        cudaFree(gpu_field_B[thread_id]);
        cudaFree(gpu_error_sum[thread_id]);
    }

    cudaEventRecord(timer_stop, 0);
    cudaEventSynchronize(timer_stop);
    cudaEventElapsedTime(&data_transfer_out, timer_start, timer_stop);
    total_time = data_transfer_in + computation_time + data_transfer_out;
    
    printf("Data transfer from GPU: %.2f ms\n", data_transfer_out);
    printf("Total execution time: %.2f ms\n", total_time);

    // Cleanup
    printf("\nCleaning up resources...\n");
    free(host_field_A);
    free(host_field_B);
    free(host_error_sum);
    free(final_solution);
    free(gpu_field_A);
    free(gpu_field_B);
    free(gpu_error_sum);
    free(gpu_devices);
    
    cudaEventDestroy(timer_start);
    cudaEventDestroy(timer_stop);

    #pragma omp parallel private(thread_id)
    {
        thread_id = omp_get_thread_num();
        cudaSetDevice(gpu_devices[thread_id]);
        cudaDeviceReset();
    }

    printf("=== Heat Diffusion Solver Complete ===\n");
    return 0;
}