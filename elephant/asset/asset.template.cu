/**
 * CUDA implementation of ASSET.joint_probability_matrix function (refer to
 * Python documentation).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define L {{L}}
#define N {{N}}
#define D {{D}}

#if D > N
#error "D must be less or equal N"
#endif

#define min_macros(a,b)   ((a) < (b) ? (a) : (b))

#define ASSET_DEBUG       {{ASSET_DEBUG}}
#define ULL               unsigned long long


/**
 * The maximum number of threads per block.
 * This number must be in range [1, 1024].
 * The effective number of threads will be set dynamically
 * at runtime to match the tile (width L) of a block.
 */
#define N_THREADS         {{N_THREADS}}

/**
 * To reduce branch divergence in 'next_sequence_sorted' function
 * within a warp (threads in a warp take different branches),
 * each thread runs CWR_LOOPS of 'combinations_with_replacement'.
 */
#define CWR_LOOPS         {{CWR_LOOPS}}

#define L_BLOCK_SUPREMUM  min_macros(N_THREADS, L)

typedef {{precision}} asset_float;

__constant__ asset_float log_factorial[N + 1];
__constant__ asset_float logK;
__constant__ ULL ITERATIONS_TODO;
__constant__ ULL L_BLOCK;
__constant__ ULL L_NUM_BLOCKS;
__constant__ ULL iteration_table[D][N];  /* Maps the iteration ID to the entries
                                            of a sequence_sorted array */

/**
 * Compute capabilities lower than 6.0 don't have hardware support for
 * double-precision atomicAdd. This software implementation is taken from
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    ULL* address_as_ull = (ULL*)address;
    ULL old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/**
 * Builds the next sequence_sorted, given the absolute iteration ID.
 * The time complexity is O(N+D), not O(N*D).
 *
 * @param sequence_sorted the output sequence_sorted array of size D
 * @param iteration       the global iteration ID
 */
__device__ void next_sequence_sorted(int *sequence_sorted, ULL iteration) {
    int row, element = N - 1;
    for (row = D - 1; row >= 0; row--) {
        while (element > row && iteration < iteration_table[row][element]) {
            element--;
        }
        iteration -= iteration_table[row][element];
        sequence_sorted[D - 1 - row] = element + 1;
    }
}


/**
 * Set 'sequence_sorted' to the next valid sequence of indices in-place.
 */
__device__ void combinations_with_replacement(int *sequence_sorted) {
    int increment_id = D - 1;
    while (increment_id > 0 && sequence_sorted[increment_id - 1] == sequence_sorted[increment_id]) {
      sequence_sorted[increment_id] = D - increment_id;
      increment_id--;
    }
    sequence_sorted[increment_id]++;
}


/**
 * CUDA kernel that computes P_total - the joint survival probabilities matrix.
 *
 * @param P_out           P_total output array of size L
 * @param log_du_device   input log_du flattened matrix of size L*(D+1)
 */
__global__ void jsf_uniform_orderstat_3d_kernel(asset_float *P_out, const float *log_du_device) {
    unsigned int i;
    ULL row;

    // the row shift of log_du and P_total in the number of elements, between 0 and L
    const ULL l_shift = (blockIdx.x % L_NUM_BLOCKS) * L_BLOCK;

    // account for the last block width that can be less than L_BLOCK
    const ULL block_width = (L - l_shift < L_BLOCK) ? (L - l_shift) : L_BLOCK;

    extern __shared__ float shared_mem[];
    asset_float *P_total = (asset_float*) shared_mem;  // L_BLOCK floats
    float *log_du = (float*)&P_total[L_BLOCK];       // L_BLOCK * (D + 1) floats

    for (row = threadIdx.x; row < block_width; row += blockDim.x) {
        P_total[row] = 0;
        for (i = 0; i <= D; i++) {
            log_du[row * (D + 1) + i] = log_du_device[(row + l_shift) * (D + 1) + i];
        }
    }

    __syncthreads();

    int di[D + 1];
    int sequence_sorted[D];
    asset_float P_thread[L_BLOCK_SUPREMUM];
    for (row = 0; row < block_width; row++) {
        P_thread[row] = 0;
    }

    const ULL burnout = (blockIdx.x / L_NUM_BLOCKS) * blockDim.x * CWR_LOOPS + threadIdx.x * CWR_LOOPS;
    const ULL stride = (gridDim.x / L_NUM_BLOCKS) * blockDim.x * CWR_LOOPS;

    ULL iteration, cwr_loop;
    for (iteration = burnout; iteration < ITERATIONS_TODO; iteration += stride) {
        next_sequence_sorted(sequence_sorted, iteration);

        for (cwr_loop = 0; (cwr_loop < CWR_LOOPS) && (sequence_sorted[0] != N + 1); cwr_loop++) {
            int prev = N;
            for (i = 0; i < D; i++) {
                di[i] = prev - sequence_sorted[i];
                prev = sequence_sorted[i];
            }
            di[D] = sequence_sorted[D - 1];

            asset_float sum_log_di_factorial = 0.f;
            for (i = 0; i <= D; i++) {
                sum_log_di_factorial += log_factorial[di[i]];
            }

            asset_float colsum;
            const asset_float colsum_base = logK - sum_log_di_factorial;
            const float *log_du_row = log_du;
            for (row = 0; row < block_width; row++) {
                colsum = colsum_base;
                for (i = 0; i <= D; i++) {
                    if (di[i] != 0) {
                        colsum += di[i] * log_du_row[i];
                    }
                }
                P_thread[row] += exp(colsum);
                log_du_row += D + 1;
            }

            combinations_with_replacement(sequence_sorted);
        }
    }

    for (row = threadIdx.x; row < block_width + threadIdx.x; row++) {
        // Reduce atomicAdd conflicts by adding threadIdx.x to each row
        atomicAdd(P_total + row % block_width, P_thread[row % block_width]);
    }

    __syncthreads();

    for (row = threadIdx.x; row < block_width; row += blockDim.x) {
        atomicAdd(P_out + row + l_shift, P_total[row]);
    }
}


/**
 * Creates a flattened matrix (D-1)*N that will be used
 * to map the iteration ID to a sequence_sorted array.
 */
ULL create_iteration_table() {
    ULL *m = (ULL*) calloc(D * N, sizeof(ULL));
    unsigned int row, col;
    for (col = 0; col < N; col++) {
        m[col] = col;
    }
    for (row = 1; row < D; row++) {
        ULL sum = 0;
        for (col = row + 1; col < N; col++) {
            sum += m[(row - 1) * N + col];
            m[row * N + col] = sum;
        }
    }

    ULL it_todo = 1;
    double it_todo_double = 1.0;
    for (row = 0; row < D; row++) {
        it_todo += m[row * N + N-1];
        it_todo_double += m[row * N + N-1];
    }

    // check for the integer overflow;
    // values greater than ULONG_MAX are not supported by CUDA
    assert(it_todo_double <= ULONG_MAX);

    gpuErrchk( cudaMemcpyToSymbol(iteration_table, m, sizeof(ULL) * D * N) );

    gpuErrchk( cudaMemcpyToSymbol((const void*) &ITERATIONS_TODO, (const void*) &it_todo, sizeof(ULL)) );

    free(m);

    return it_todo;
}


// For debugging purposes only
void print_constants() {
    int i, col;
    printf(">>> iteration_table\n");
    ULL iteration_table_host[D * N];
    cudaMemcpyFromSymbol(iteration_table_host, iteration_table, sizeof(ULL) * D * N);
    int row;
    for (row = 0; row < D; row++) {
        for (col = 0; col < N; col++) {
            printf("%10llu ", iteration_table_host[row * N + col]);
        }
        printf("\n");
    }
    printf("\n");

    ULL it_todo_host;
    cudaMemcpyFromSymbol((void*)&it_todo_host, (const void*)&ITERATIONS_TODO, sizeof(ULL));
    printf(">>> ITERATIONS_TODO = %llu\n", it_todo_host);

    ULL l_block;
    cudaMemcpyFromSymbol((void*)&l_block, (const void*)&L_BLOCK, sizeof(ULL));
    printf(">>> L_BLOCK = %llu\n", l_block);

    ULL l_num_blocks;
    cudaMemcpyFromSymbol((void*)&l_num_blocks, (const void*)&L_NUM_BLOCKS, sizeof(ULL));
    printf(">>> L_NUM_BLOCKS = %llu\n", l_num_blocks);

    asset_float logK_host;
    cudaMemcpyFromSymbol((void*)&logK_host, (const void*)&logK, sizeof(asset_float));
    printf(">>> logK = %f\n\n", logK_host);

    asset_float log_factorial_host[N + 1];
    cudaMemcpyFromSymbol(log_factorial_host, log_factorial, sizeof(asset_float) * (N+1));
    printf(">>> log_factorial\n");
    for (i = 0; i <= N; i++) {
        printf("%f ", log_factorial_host[i]);
    }
    printf("\n\n");
}


/**
 * ASSET jsf_uniform_orderstat_3d host function to calculate P_total.
 * The result of a calculation is saved in P_total_host array.
 *
 * @param P_total_host a pointer to P_total array to be calculated
 * @param log_du_host  input flattened L*(D+1) matrix of log_du values
 */
void jsf_uniform_orderstat_3d(asset_float *P_total_host, FILE *log_du_file) {
    float *log_du_device;
    gpuErrchk( cudaMalloc((void**)&log_du_device, sizeof(float) * L * (D + 1)) );

    float *log_du_host;

#if L * (D + 1) < 100000000LLU
    // For arrays of size <100 Mb, allocate host memory for log_du
    log_du_host = (float*) malloc(sizeof(float) * L * (D + 1));
    fread(log_du_host, sizeof(float), L * (D + 1), log_du_file);
    gpuErrchk( cudaMemcpyAsync(log_du_device, log_du_host, sizeof(float) * L * (D + 1), cudaMemcpyHostToDevice) );
#else
    // Use P_total buffer to read log_du and copy batches to a GPU card
    log_du_host = (float*) P_total_host;
    ULL col;
    for (col = 0; col <= D; col++) {
        fread(log_du_host, sizeof(float), L, log_du_file);
        // Wait till the copy finishes before filling the buffer with a next chunk.
        gpuErrchk( cudaMemcpy(log_du_device + col * L, log_du_host, sizeof(float) * L, cudaMemcpyHostToDevice) );
    }
#endif

    fclose(log_du_file);

    asset_float *P_total_device;

    // Initialize P_total_device with zeros.
    // Note that values other than 0x00 or 0xFF (NaN) won't work
    // with cudaMemset when the data type is float or double.
    gpuErrchk( cudaMalloc((void**)&P_total_device, sizeof(asset_float) * L) );
    gpuErrchk( cudaMemsetAsync(P_total_device, 0, sizeof(asset_float) * L) );

    ULL it_todo = create_iteration_table();

    asset_float logK_host = 0.f;
    asset_float log_factorial_host[N + 1] = {0.f};

    int i;
    for (i = 1; i <= N; i++) {
        logK_host += log((asset_float) i);
        log_factorial_host[i] = logK_host;
    }

    gpuErrchk( cudaMemcpyToSymbol((const void*) &logK, (const void*) &logK_host, sizeof(asset_float)) );
    gpuErrchk( cudaMemcpyToSymbol(log_factorial, log_factorial_host, sizeof(asset_float) * (N + 1)) );

    cudaDeviceProp device_prop;
    gpuErrchk( cudaGetDeviceProperties(&device_prop, 0) );
    const ULL max_l_block = device_prop.sharedMemPerBlock / (sizeof(asset_float) * (D + 2));

    /**
     * It's not necessary to match N_THREADS with the final L_BLOCK. Alternatively,
     * the desired L_BLOCK can be another parameter specified by the user. But
     * the optimal L_BLOCK on average matches N_THREADS, therefore, to avoid
     * the user thinking too much, we take care of the headache by setting
     * L_BLOCK = N_THREADS.
     */
    unsigned int n_threads = (unsigned int) min_macros(N_THREADS, min_macros(max_l_block, device_prop.maxThreadsPerBlock));
    if (n_threads > device_prop.warpSize) {
        // It's more efficient to make the number of threads
        // a multiple of the warp size (32).
        n_threads -= n_threads % device_prop.warpSize;
    }
    const ULL l_block = min_macros(n_threads, L);
    gpuErrchk( cudaMemcpyToSymbol((const void*) &L_BLOCK, (const void*) &l_block, sizeof(ULL)) );

    const ULL l_num_blocks = (ULL) ceil(L * 1.f / l_block);
    gpuErrchk( cudaMemcpyToSymbol((const void*) &L_NUM_BLOCKS, (const void*) &l_num_blocks, sizeof(ULL)) );

    ULL grid_size = (ULL) ceil(it_todo * 1.f / (n_threads * CWR_LOOPS));
    grid_size = min_macros(grid_size, device_prop.maxGridSize[0]);
    if (grid_size > l_num_blocks) {
        // make grid_size divisible by l_num_blocks
        grid_size -= grid_size % l_num_blocks;
    } else {
        // grid_size must be at least l_num_blocks
        grid_size = l_num_blocks;
    }

    printf(">>> it_todo=%llu, grid_size=%llu, L_BLOCK=%llu, N_THREADS=%u\n\n", it_todo, grid_size, l_block, n_threads);

    // Wait for asynchronous memory copies to finish.
    gpuErrchk( cudaDeviceSynchronize() );

    if (log_du_host != (float*) P_total_host) {
        // the memory has been allocated
        free(log_du_host);
    }

#if ASSET_DEBUG
    print_constants();
#endif

    // Executing the kernel
    const ULL shared_mem_used = sizeof(asset_float) * l_block + sizeof(float) * l_block * (D + 1);
    jsf_uniform_orderstat_3d_kernel<<<grid_size, n_threads, shared_mem_used>>>(P_total_device, log_du_device);

    // Check for invalid launch argument.
    gpuErrchk( cudaPeekAtLastError() );

    // Transfer data back to host memory.
    // If the exit code is non-zero, the kernel failed to complete the task.
    cudaError_t cuda_completed_status = cudaMemcpy(P_total_host, P_total_device, sizeof(asset_float) * L, cudaMemcpyDeviceToHost);

    cudaFree(P_total_device);
    cudaFree(log_du_device);

    gpuErrchk( cuda_completed_status );
}


int main(int argc, char* argv[]) {
    // compile command: nvcc -o asset.o asset.cu
    // (run after you fill the template keys L, N, D, etc.)
    if (argc != 3) {
        fprintf(stderr, "Usage: ./asset.o /path/to/log_du.dat /path/to/P_total_output.dat\n");
        return 1;
    }
    char *log_du_path = argv[1];
    char *P_total_path = argv[2];

    FILE *log_du_file = fopen(log_du_path, "rb");

    if (log_du_file == NULL) {
        fprintf(stderr, "File '%s' not found\n", log_du_path);
        return 1;
    }

    asset_float *P_total = (asset_float*) malloc(sizeof(asset_float) * L);

    jsf_uniform_orderstat_3d(P_total, log_du_file);

    FILE *P_total_file = fopen(P_total_path, "wb");
    if (P_total_file == NULL) {
        free(P_total);
        fprintf(stderr, "Could not open '%s' for writing.\n", P_total_path);
        return 1;
    }
    fwrite(P_total, sizeof(asset_float), L, P_total_file);
    fclose(P_total_file);

    free(P_total);

    return 0;
}
