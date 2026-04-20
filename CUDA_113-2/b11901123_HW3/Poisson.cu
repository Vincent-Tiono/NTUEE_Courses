// Compile for GTX1060
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o Poisson Poisson.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

// Memory allocation for simulation data
typedef struct {
    float *host_phi_current;      // Current potential on host
    float *host_phi_next;         // Next iteration potential on host
    float *host_gpu_result;       // GPU result copied back to host
    float *host_residual_sums;    // Residual sums for convergence checking
    float *host_charge_density;   // Source charge distribution

    float *device_phi_current;    // Current potential on device
    float *device_phi_next;       // Next iteration potential on device
    float *device_residual_sums;  // Block-wise residual sums
    float *device_charge_density; // Source charge distribution on device
} PoissonSolverData;

// Configuration parameters
typedef struct {
    int max_iterations;
    double convergence_tolerance;
    int grid_size_x, grid_size_y, grid_size_z;
    int thread_count_x, thread_count_y, thread_count_z;
    int block_count_x, block_count_y, block_count_z;
    int computation_mode;         // 0=CPU, 1=GPU, 2=Both
    int gpu_device_id;
    char output_filename[128];
} PoissonSolverConfig;

/**
 * CUDA kernel for solving the Poisson equation using finite difference method
 * 
 * @param phi_current Current potential field
 * @param phi_next Next iteration potential field
 * @param residual_sums Sum of squared residuals per block
 * @param grid_size_x Number of grid points in x direction
 * @param grid_size_y Number of grid points in y direction
 * @param grid_size_z Number of grid points in z direction
 * @param update_phi_next Whether to update phi_next (true) or phi_current (false)
 * @param texture_current Texture for current potential field
 * @param texture_next Texture for next potential field
 * @param texture_charge Texture for charge density field
 */
__global__ void poissonSolverKernel(
    float* phi_current, 
    float* phi_next, 
    float* residual_sums,
    int grid_size_x, 
    int grid_size_y, 
    int grid_size_z, 
    bool update_phi_next,
    cudaTextureObject_t texture_current, 
    cudaTextureObject_t texture_next, 
    cudaTextureObject_t texture_charge
) {
    // Shared memory for parallel reduction of residuals
    extern __shared__ float residual_cache[];
    
    // Neighbor values in 6 directions
    float phi_left, phi_right, phi_back, phi_front, phi_bottom, phi_top;
    float charge_density;
    float residual;
    
    // Grid indices and memory locations
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_z = blockDim.z * blockIdx.z + threadIdx.z;
    int linear_idx = idx_x + idx_y * grid_size_x + idx_z * grid_size_x * grid_size_y;
    
    // Cache index for residual reduction
    int cache_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Handle boundary conditions - keep zeros at domain boundaries
    if (idx_x == 0 || idx_x >= grid_size_x - 1 || 
        idx_y == 0 || idx_y >= grid_size_y - 1 || 
        idx_z == 0 || idx_z >= grid_size_z - 1) {
        residual = 0.0f;
    } else {
        // Linear indices for neighbors
        int idx_left = linear_idx - 1;
        int idx_right = linear_idx + 1;
        int idx_back = linear_idx - grid_size_x;
        int idx_front = linear_idx + grid_size_x;
        int idx_bottom = linear_idx - grid_size_x * grid_size_y;
        int idx_top = linear_idx + grid_size_x * grid_size_y;

        // Get charge density from texture
        charge_density = tex1Dfetch<float>(texture_charge, linear_idx);
        
        if (update_phi_next) {
            // Read neighbor values from current potential using texture memory
            phi_left = tex1Dfetch<float>(texture_current, idx_left);
            phi_right = tex1Dfetch<float>(texture_current, idx_right);
            phi_back = tex1Dfetch<float>(texture_current, idx_back);
            phi_front = tex1Dfetch<float>(texture_current, idx_front);
            phi_bottom = tex1Dfetch<float>(texture_current, idx_bottom);
            phi_top = tex1Dfetch<float>(texture_current, idx_top);
            float phi_center = tex1Dfetch<float>(texture_current, linear_idx);
            
            // Update next potential field using 7-point stencil 
            phi_next[linear_idx] = (phi_bottom + phi_back + phi_left + 
                                   phi_right + phi_front + phi_top - 
                                   charge_density) / 6.0f;
            
            // Calculate residual as the difference between new and old potential
            residual = phi_next[linear_idx] - phi_center;
        } else {
            // Read neighbor values from next potential using texture memory
            phi_left = tex1Dfetch<float>(texture_next, idx_left);
            phi_right = tex1Dfetch<float>(texture_next, idx_right);
            phi_back = tex1Dfetch<float>(texture_next, idx_back);
            phi_front = tex1Dfetch<float>(texture_next, idx_front);
            phi_bottom = tex1Dfetch<float>(texture_next, idx_bottom);
            phi_top = tex1Dfetch<float>(texture_next, idx_top);
            float phi_center = tex1Dfetch<float>(texture_next, linear_idx);
            
            // Update current potential field using 7-point stencil
            phi_current[linear_idx] = (phi_bottom + phi_back + phi_left + 
                                      phi_right + phi_front + phi_top - 
                                      charge_density) / 6.0f;
            
            // Calculate residual as the difference between new and old potential
            residual = phi_current[linear_idx] - phi_center;
        }
    }

    // Store squared residual in shared memory
    residual_cache[cache_idx] = residual * residual;
    __syncthreads();

    // Parallel reduction to sum residuals within each block
    int active_threads = blockDim.x * blockDim.y * blockDim.z / 2;
    while (active_threads > 0) {
        if (cache_idx < active_threads) {
            residual_cache[cache_idx] += residual_cache[cache_idx + active_threads];
        }
        __syncthreads();
        active_threads /= 2;
    }

    // Store the residual sum for this block
    if (cache_idx == 0) {
        int block_idx = blockIdx.x + 
                       gridDim.x * blockIdx.y + 
                       gridDim.x * gridDim.y * blockIdx.z;
        residual_sums[block_idx] = residual_cache[0];
    }
}

/**
 * Initialize solution data with zeros and a point charge at the center
 */
void initializePoissonData(PoissonSolverData *data, PoissonSolverConfig *config) {
    // Calculate memory sizes
    int total_grid_points = config->grid_size_x * config->grid_size_y * config->grid_size_z;
    size_t field_size_bytes = total_grid_points * sizeof(float);
    size_t block_data_size_bytes = config->block_count_x * 
                                 config->block_count_y * 
                                 config->block_count_z * sizeof(float);
    
    // Allocate pinned host memory for better transfer performance
    cudaMallocHost((void**)&data->host_phi_current, field_size_bytes);
    cudaMallocHost((void**)&data->host_phi_next, field_size_bytes);
    cudaMallocHost((void**)&data->host_gpu_result, field_size_bytes);
    cudaMallocHost((void**)&data->host_residual_sums, block_data_size_bytes);
    cudaMallocHost((void**)&data->host_charge_density, field_size_bytes);
   
    // Initialize potentials to zero (including boundary conditions)
    memset(data->host_phi_current, 0, field_size_bytes);
    memset(data->host_phi_next, 0, field_size_bytes);

    // Initialize charge density (point charge at center)
    memset(data->host_charge_density, 0, field_size_bytes);
    // Set a negative point charge at center (-1.0 for source term in Poisson equation)
    int center_x = config->grid_size_x / 2;
    int center_y = config->grid_size_y / 2;
    int center_z = config->grid_size_z / 2;
    int center_idx = center_x + 
                   center_y * config->grid_size_x + 
                   center_z * config->grid_size_x * config->grid_size_y;
    data->host_charge_density[center_idx] = -1.0f;
}

/**
 * Solve Poisson equation on GPU using CUDA
 */
float solvePoissonGPU(PoissonSolverData *data, PoissonSolverConfig *config) {
    // Timing events for performance analysis
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float setup_time, computation_time, cleanup_time, total_time;

    // Start measuring setup time
    cudaEventRecord(start_event, 0);

    // Calculate sizes
    int total_grid_points = config->grid_size_x * config->grid_size_y * config->grid_size_z;
    size_t field_size_bytes = total_grid_points * sizeof(float);
    size_t block_data_size_bytes = config->block_count_x * 
                                 config->block_count_y * 
                                 config->block_count_z * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&data->device_phi_next, field_size_bytes);
    cudaMalloc((void**)&data->device_phi_current, field_size_bytes);
    cudaMalloc((void**)&data->device_residual_sums, block_data_size_bytes);
    cudaMalloc((void**)&data->device_charge_density, field_size_bytes);

    // Setup texture objects for faster memory access
    cudaTextureObject_t texture_current, texture_next, texture_charge;
    struct cudaResourceDesc res_desc;
    struct cudaTextureDesc tex_desc;

    // Create texture for current potential field
    memset(&tex_desc, 0, sizeof(tex_desc));
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = data->device_phi_current;
    res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
    res_desc.res.linear.sizeInBytes = field_size_bytes;
    cudaCreateTextureObject(&texture_current, &res_desc, &tex_desc, NULL);

    // Create texture for next potential field
    memset(&tex_desc, 0, sizeof(tex_desc));
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = data->device_phi_next;
    res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
    res_desc.res.linear.sizeInBytes = field_size_bytes;
    cudaCreateTextureObject(&texture_next, &res_desc, &tex_desc, NULL);

    // Create texture for charge density field
    memset(&tex_desc, 0, sizeof(tex_desc));
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = data->device_charge_density;
    res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
    res_desc.res.linear.sizeInBytes = field_size_bytes;
    cudaCreateTextureObject(&texture_charge, &res_desc, &tex_desc, NULL);

    // Copy data to device
    cudaMemcpy(data->device_phi_next, data->host_phi_next, 
              field_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(data->device_phi_current, data->host_phi_current, 
              field_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(data->device_charge_density, data->host_charge_density, 
              field_size_bytes, cudaMemcpyHostToDevice);
    
    // End setup timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&setup_time, start_event, stop_event);
    printf("GPU Setup Time: %.2f ms\n", setup_time);

    // Start computation timer
    cudaEventRecord(start_event, 0);

    // Define thread block and grid dimensions
    dim3 threads(config->thread_count_x, config->thread_count_y, config->thread_count_z);
    dim3 blocks(config->block_count_x, config->block_count_y, config->block_count_z);
    
    // Calculate shared memory size for residual reduction
    int shared_mem_size = config->thread_count_x * 
                        config->thread_count_y * 
                        config->thread_count_z * sizeof(float);
  
    // Solver variables
    double residual_norm = 10.0 * config->convergence_tolerance; // Initial residual
    int iteration = 0;
    bool update_phi_next = true;
  
    // Main iteration loop
    while ((residual_norm > config->convergence_tolerance) && 
           (iteration < config->max_iterations)) {
        // Launch Poisson solver kernel
        poissonSolverKernel<<<blocks, threads, shared_mem_size>>>(
            data->device_phi_current, 
            data->device_phi_next, 
            data->device_residual_sums,
            config->grid_size_x, 
            config->grid_size_y, 
            config->grid_size_z, 
            update_phi_next,
            texture_current, 
            texture_next, 
            texture_charge
        );
        
        // Copy residual sums back to host
        cudaMemcpy(data->host_residual_sums, data->device_residual_sums, 
                  block_data_size_bytes, cudaMemcpyDeviceToHost);
        
        // Calculate total residual norm
        residual_norm = 0.0;
        for (int i = 0; i < config->block_count_x * 
                           config->block_count_y * 
                           config->block_count_z; i++) {
            residual_norm += data->host_residual_sums[i];
        }
        residual_norm = sqrt(residual_norm);
        
        // Increment iteration counter and swap update direction
        iteration++;
        update_phi_next = !update_phi_next;
    }
 
    printf("Final residual (GPU): %.15e\n", residual_norm);
    printf("Iterations completed (GPU): %d\n", iteration);

    // Stop computation timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&computation_time, start_event, stop_event);
    printf("GPU Computation Time: %.2f ms\n", computation_time);
    
    // Calculate computational performance
    double flops = 7.0 * (config->grid_size_x - 2) * 
                 (config->grid_size_y - 2) * 
                 (config->grid_size_z - 2) * iteration;
    printf("GPU Performance: %.2f Gflops\n", flops / (1000000.0 * computation_time));

    // Start cleanup timer
    cudaEventRecord(start_event, 0);
    
    // Copy final result back to host
    if (update_phi_next) {
        // Last update was to phi_current, so it has the latest data
        cudaMemcpy(data->host_gpu_result, data->device_phi_current, 
                  field_size_bytes, cudaMemcpyDeviceToHost);
    } else {
        // Last update was to phi_next, so it has the latest data
        cudaMemcpy(data->host_gpu_result, data->device_phi_next, 
                  field_size_bytes, cudaMemcpyDeviceToHost);
    }

    // Clean up GPU resources
    cudaDestroyTextureObject(texture_current);
    cudaDestroyTextureObject(texture_next);
    cudaDestroyTextureObject(texture_charge);
    cudaFree(data->device_phi_next);
    cudaFree(data->device_phi_current);
    cudaFree(data->device_residual_sums);
    cudaFree(data->device_charge_density);

    // End cleanup timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&cleanup_time, start_event, stop_event);
    printf("GPU Cleanup Time: %.2f ms\n", cleanup_time);

    // Calculate total time
    total_time = setup_time + computation_time + cleanup_time;
    printf("Total GPU Time: %.2f ms\n", total_time);
    
    // Save GPU results to file
    FILE *gpu_output;
    snprintf(config->output_filename, sizeof(config->output_filename), 
            "GPU_32.npy");
    gpu_output = fopen(config->output_filename, "w");
    fwrite(data->host_gpu_result, sizeof(float), total_grid_points, gpu_output);
    fclose(gpu_output);
    
    return total_time;
}

/**
 * Solve Poisson equation on CPU
 */
float solvePoissonCPU(PoissonSolverData *data, PoissonSolverConfig *config) {
    // Reset variables for CPU computation
    double residual_norm = 10.0 * config->convergence_tolerance;
    int iteration = 0;
    bool update_phi_next = true;
    
    // Calculate sizes
    int total_grid_points = config->grid_size_x * config->grid_size_y * config->grid_size_z;
    size_t field_size_bytes = total_grid_points * sizeof(float);
    
    // Reset potential fields to zero
    memset(data->host_phi_current, 0, field_size_bytes);
    memset(data->host_phi_next, 0, field_size_bytes);

    // Create timing events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Start CPU timer
    cudaEventRecord(start_event, 0);
    
    // Temporary variables for neighbors and residuals
    float phi_bottom, phi_back, phi_left, phi_right, phi_front, phi_top;
    float charge, residual;
    int idx, idx_bottom, idx_back, idx_left, idx_right, idx_front, idx_top;

    // Main iteration loop
    while ((residual_norm > config->convergence_tolerance) && 
           (iteration < config->max_iterations)) {
        if (update_phi_next) {
            residual_norm = 0.0;
            for (int z = 0; z < config->grid_size_z; z++) {
                for (int y = 0; y < config->grid_size_y; y++) {
                    for (int x = 0; x < config->grid_size_x; x++) {
                        // Skip boundary points (keep as zero)
                        if (x == 0 || x == config->grid_size_x-1 || 
                            y == 0 || y == config->grid_size_y-1 || 
                            z == 0 || z == config->grid_size_z-1) {
                            continue;
                        }
                        
                        // Calculate linear indices
                        idx = x + y * config->grid_size_x + 
                             z * config->grid_size_x * config->grid_size_y;
                        idx_left = idx - 1;
                        idx_right = idx + 1;
                        idx_back = idx - config->grid_size_x;
                        idx_front = idx + config->grid_size_x;
                        idx_bottom = idx - config->grid_size_x * config->grid_size_y;
                        idx_top = idx + config->grid_size_x * config->grid_size_y;
                        
                        // Get neighbor values
                        phi_bottom = data->host_phi_current[idx_bottom];
                        phi_back = data->host_phi_current[idx_back];
                        phi_left = data->host_phi_current[idx_left];
                        phi_right = data->host_phi_current[idx_right];
                        phi_front = data->host_phi_current[idx_front];
                        phi_top = data->host_phi_current[idx_top];
                        charge = data->host_charge_density[idx];
                        
                        // Update next potential using 7-point stencil
                        data->host_phi_next[idx] = (phi_bottom + phi_back + phi_left + 
                                                   phi_right + phi_front + phi_top - 
                                                   charge) / 6.0f;
                        
                        // Calculate residual
                        residual = data->host_phi_next[idx] - data->host_phi_current[idx];
                        residual_norm += residual * residual;
                    }
                }
            }
        } else {
            residual_norm = 0.0;
            for (int z = 0; z < config->grid_size_z; z++) {
                for (int y = 0; y < config->grid_size_y; y++) {
                    for (int x = 0; x < config->grid_size_x; x++) {
                        // Skip boundary points
                        if (x == 0 || x == config->grid_size_x-1 || 
                            y == 0 || y == config->grid_size_y-1 || 
                            z == 0 || z == config->grid_size_z-1) {
                            continue;
                        }
                        
                        // Calculate linear indices
                        idx = x + y * config->grid_size_x + 
                             z * config->grid_size_x * config->grid_size_y;
                        idx_left = idx - 1;
                        idx_right = idx + 1;
                        idx_back = idx - config->grid_size_x;
                        idx_front = idx + config->grid_size_x;
                        idx_bottom = idx - config->grid_size_x * config->grid_size_y;
                        idx_top = idx + config->grid_size_x * config->grid_size_y;
                        
                        // Get neighbor values
                        phi_bottom = data->host_phi_next[idx_bottom];
                        phi_back = data->host_phi_next[idx_back];
                        phi_left = data->host_phi_next[idx_left];
                        phi_right = data->host_phi_next[idx_right];
                        phi_front = data->host_phi_next[idx_front];
                        phi_top = data->host_phi_next[idx_top];
                        charge = data->host_charge_density[idx];
                        
                        // Update current potential using 7-point stencil
                        data->host_phi_current[idx] = (phi_bottom + phi_back + phi_left + 
                                                     phi_right + phi_front + phi_top - 
                                                     charge) / 6.0f;
                        
                        // Calculate residual
                        residual = data->host_phi_current[idx] - data->host_phi_next[idx];
                        residual_norm += residual * residual;
                    }
                }
            }
        }
        
        // Switch update direction for next iteration
        update_phi_next = !update_phi_next;
        iteration++;
        residual_norm = sqrt(residual_norm);
    }

    printf("Final residual (CPU): %.15e\n", residual_norm);
    printf("Iterations completed (CPU): %d\n", iteration);

    // Stop CPU timer
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start_event, stop_event);
    printf("CPU Computation Time: %.2f ms\n", cpu_time);
    
    // Calculate performance metrics
    double flops = 7.0 * (config->grid_size_x - 2) * 
                 (config->grid_size_y - 2) * 
                 (config->grid_size_z - 2) * iteration;
    printf("CPU Performance: %.2f Gflops\n", flops / (1000000.0 * cpu_time));

    // Save CPU results to file
    FILE *cpu_output;
    snprintf(config->output_filename, sizeof(config->output_filename), 
            "CPU_32.npy");
    cpu_output = fopen(config->output_filename, "w");
    
    // Write the most recent potential field
    if (update_phi_next) {
        // Last update was to phi_current
        fwrite(data->host_phi_current, sizeof(float), total_grid_points, cpu_output);
    } else {
        // Last update was to phi_next
        fwrite(data->host_phi_next, sizeof(float), total_grid_points, cpu_output);
    }
    
    fclose(cpu_output);
    
    // Clean up timing events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return cpu_time;
}

/**
 * Free allocated memory
 */
void cleanupPoissonData(PoissonSolverData *data) {
    cudaFreeHost(data->host_phi_next);
    cudaFreeHost(data->host_phi_current);
    cudaFreeHost(data->host_gpu_result);
    cudaFreeHost(data->host_residual_sums);
    cudaFreeHost(data->host_charge_density);
}

/**
 * Main function
 */
int main(void) {
    PoissonSolverData data;
    PoissonSolverConfig config;
    float gpu_time = 0.0f, cpu_time = 0.0f;
    
    // Set default solver parameters
    config.max_iterations = 10000000;
    config.convergence_tolerance = 1.0e-10;

    // Get GPU device ID
    printf("Enter the GPU ID (0/1): ");
    scanf("%d", &config.gpu_device_id);
    printf("%d\n", config.gpu_device_id);

    // Initialize the selected GPU
    cudaError_t cuda_status = cudaSetDevice(config.gpu_device_id);
    if (cuda_status != cudaSuccess) {
        printf("Error: Cannot select GPU with device ID = %d\n", config.gpu_device_id);
        return 1;
    }
    printf("Using GPU with device ID = %d\n", config.gpu_device_id);

    printf("Solving 3D Poisson equation with Dirichlet boundary conditions\n");

    // Get grid dimensions
    printf("Enter the size (nx, ny, nz) of the 3D lattice: ");
    scanf("%d %d %d", &config.grid_size_x, &config.grid_size_y, &config.grid_size_z);
    printf("%d %d %d\n", config.grid_size_x, config.grid_size_y, config.grid_size_z);

    // Get thread block dimensions
    printf("Enter the number of threads (tx, ty, tz) per block: ");
    scanf("%d %d %d", &config.thread_count_x, &config.thread_count_y, &config.thread_count_z);
    printf("%d %d %d\n", config.thread_count_x, config.thread_count_y, config.thread_count_z);
    
    // Validate thread dimensions
    int threads_per_block = config.thread_count_x * config.thread_count_y * config.thread_count_z;
    if (threads_per_block > 1024) {
        printf("Error: The number of threads per block must be less than 1024!\n");
        return 1;
    }
    
    // Calculate block dimensions
    config.block_count_x = (config.grid_size_x + config.thread_count_x - 1) / config.thread_count_x;
    config.block_count_y = (config.grid_size_y + config.thread_count_y - 1) / config.thread_count_y;
    config.block_count_z = (config.grid_size_z + config.thread_count_z - 1) / config.thread_count_z;
    
    // Validate grid dimensions
    if ((config.block_count_x > 65535) || (config.block_count_y > 65535) || (config.block_count_z > 65535)) {
        printf("Error: The grid size exceeds the CUDA limit of 65535!\n");
        return 1;
    }
    
    printf("Grid dimensions: (%d, %d, %d)\n", 
           config.block_count_x, config.block_count_y, config.block_count_z);

    // Choose computation mode
    printf("Compute using CPU/GPU/both (0/1/2)? ");
    scanf("%d", &config.computation_mode);
    printf("%d\n", config.computation_mode);
    fflush(stdout);
    
    // Initialize problem data
    initializePoissonData(&data, &config);
    
    // Create timing events
    cudaEvent_t global_start, global_stop;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_stop);
    
    // Start global timer
    cudaEventRecord(global_start, 0);
    
    // Execute selected computation mode
    if (config.computation_mode == 1 || config.computation_mode == 2) {
        // GPU computation
        gpu_time = solvePoissonGPU(&data, &config);
        printf("\n");
    }
    
    if (config.computation_mode == 0 || config.computation_mode == 2) {
        // CPU computation
        cpu_time = solvePoissonCPU(&data, &config);
        
        // Print speedup if both CPU and GPU were used
        if (config.computation_mode == 2) {
            printf("GPU Speedup: %.2f\n", cpu_time / gpu_time);
        }
    }
    
    // Stop global timer
    cudaEventRecord(global_stop, 0);
    cudaEventSynchronize(global_stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, global_start, global_stop);
    printf("Total Program Time: %.2f ms\n", total_time);
    
    // Clean up resources
    cleanupPoissonData(&data);
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);
    
    // Reset device for clean exit
    cudaDeviceReset();
    return 0;
}