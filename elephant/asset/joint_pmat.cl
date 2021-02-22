// Enable support for double floating-point precision, if needed.
#if {{ASSET_ENABLE_DOUBLE_SUPPORT}}
  #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#endif

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

/**
 * OpenCL spec. defines unsigned long as uint64.
 */
#define ULL               unsigned long

/**
 * Convert float or double to uint32 or uint64 accordingly.
 */
#define ATOMIC_UINT       {{ATOMIC_UINT}}

/**
 * To reduce branch divergence in 'next_sequence_sorted' function
 * within a warp (threads in a warp take different branches),
 * each thread runs CWR_LOOPS of 'combinations_with_replacement'.
 */
#define CWR_LOOPS         {{CWR_LOOPS}}

typedef {{precision}} asset_float;

__constant asset_float log_factorial[] = {{log_factorial}};
__constant ULL iteration_table[] = {{iteration_table}};

void atomicAdd_global(__global asset_float* source, const asset_float operand)
{
    union {
        ATOMIC_UINT intVal;
        asset_float floatVal;
    } newVal;
    union {
        ATOMIC_UINT intVal;
        asset_float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atom_cmpxchg((volatile global ATOMIC_UINT *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

void atomicAdd_local(__local asset_float* source, const asset_float operand)
{
    union {
        ATOMIC_UINT intVal;
        asset_float floatVal;
    } newVal;

    union {
        ATOMIC_UINT intVal;
        asset_float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atom_cmpxchg((volatile local ATOMIC_UINT *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


/**
 * Builds the next sequence_sorted, given the absolute iteration ID.
 * The time complexity is O(N+D), not O(N*D).
 *
 * @param sequence_sorted the output sequence_sorted array of size D
 * @param iteration       the global iteration ID
 */
void next_sequence_sorted(int *sequence_sorted, ULL iteration) {
    int row, element = N - 1;
    for (row = D - 1; row >= 0; row--) {
        while (element > row && iteration < iteration_table[row * N + element]) {
            element--;
        }
        iteration -= iteration_table[row * N + element];
        sequence_sorted[D - 1 - row] = element + 1;
    }
}


/**
 * Set 'sequence_sorted' to the next valid sequence of indices in-place.
 */
void combinations_with_replacement(int *sequence_sorted) {
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
__kernel void jsf_uniform_orderstat_3d_kernel(__global asset_float *P_out, __global const float *log_du_device) {
    unsigned int i;
    ULL row;

    const int threadIdx_x = get_local_id(0);
    const int blockDim_x = get_local_size(0);

    // blockIdx_x and gridDim_x are upperbounded by 2^31 - 1.
    const ULL blockIdx_x = get_group_id(0);
    const ULL gridDim_x = get_num_groups(0);

    // the row shift of log_du and P_total in the number of elements, between 0 and L
    const ULL l_shift = (blockIdx_x % L_NUM_BLOCKS) * L_BLOCK;

    // account for the last block width that can be less than L_BLOCK
    const ULL block_width = (L - l_shift < L_BLOCK) ? (L - l_shift) : L_BLOCK;

    __local asset_float P_total[L_BLOCK];
    __local float log_du[L_BLOCK * (D + 1)];

    for (row = threadIdx_x; row < block_width; row += blockDim_x) {
        P_total[row] = 0;
        for (i = 0; i <= D; i++) {
            log_du[row * (D + 1) + i] = log_du_device[(row + l_shift) * (D + 1) + i];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int di[D + 1];
    int sequence_sorted[D];
    asset_float P_thread[L_BLOCK];
    for (row = 0; row < block_width; row++) {
        P_thread[row] = 0;
    }

    const ULL burnout = (blockIdx_x / L_NUM_BLOCKS) * blockDim_x * CWR_LOOPS + threadIdx_x * CWR_LOOPS;
    const ULL stride = (gridDim_x / L_NUM_BLOCKS) * blockDim_x * CWR_LOOPS;

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
            for (row = 0; row < block_width; row++) {
                colsum = colsum_base;
                for (i = 0; i <= D; i++) {
                    if (di[i] != 0) {
                        colsum += di[i] * log_du[row * (D + 1) + i];
                    }
                }
                P_thread[row] += exp(colsum);
            }

            combinations_with_replacement(sequence_sorted);
        }
    }

    for (row = threadIdx_x; row < block_width + threadIdx_x; row++) {
        // Reduce atomicAdd conflicts by adding threadIdx_x to each row
        atomicAdd_local(P_total + row % block_width, P_thread[row % block_width]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (row = threadIdx_x; row < block_width; row += blockDim_x) {
        atomicAdd_global(P_out + row + l_shift, P_total[row]);
    }
}
