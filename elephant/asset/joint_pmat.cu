#define L {{L}}
#define N {{N}}
#define D {{D}}

#define L_BLOCK          {{L_BLOCK}}
#define L_NUM_BLOCKS     {{L_NUM_BLOCKS}}
#define ITERATIONS_TODO  {{ITERATIONS_TODO}}
#define logK             {{logK}}

#if D > N
#error "D must be less or equal N"
#endif

#define ULL               unsigned long long

/**
 * To reduce branch divergence in 'next_sequence_sorted' function
 * within a warp (threads in a warp take different branches),
 * each thread runs CWR_LOOPS of 'combinations_with_replacement'.
 */
#define CWR_LOOPS          {{CWR_LOOPS}}

typedef {{precision}} asset_float;

__constant__ asset_float log_factorial[N + 1];
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

    __shared__ asset_float P_total[L_BLOCK];
    __shared__ float log_du[L_BLOCK * (D + 1)];

    for (row = threadIdx.x; row < block_width; row += blockDim.x) {
        P_total[row] = 0;
        for (i = 0; i <= D; i++) {
            log_du[row * (D + 1) + i] = log_du_device[(row + l_shift) * (D + 1) + i];
        }
    }

    __syncthreads();

    int di[D + 1];
    int sequence_sorted[D];
    asset_float P_thread[L_BLOCK];
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
