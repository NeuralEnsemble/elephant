#define FILT_SIZE     {{FILT_SIZE}}
#define N_LARGEST     {{N_LARGEST}}
#define PMAT_COLS     {{PMAT_COLS}}
#define Y_OFFSET      {{Y_OFFSET}}
#define NONZERO_SIZE  {{NONZERO_SIZE}}
#define SYMMETRIC     {{SYMMETRIC}}

#define min_macros(a,b)   ((a) < (b) ? (a) : (b))

#define IT_TODO       {{IT_TODO}}

__constant__ unsigned int filt_rows[NONZERO_SIZE];
__constant__ unsigned int filt_cols[NONZERO_SIZE];


__global__ void pmat_neighbors(float *lmat, const float *pmat) {
    const unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid > IT_TODO) {
        return;
    }

    const unsigned long long y = gid / (PMAT_COLS - FILT_SIZE + 1);
    const unsigned long long x = gid - y * (PMAT_COLS - FILT_SIZE + 1);

    if (SYMMETRIC && (x > (y + Y_OFFSET))) {
        return;
    }

    float largest[N_LARGEST + 1];
    unsigned int i, j;
    unsigned long long pos;
    float tmp;
    for (i = 0; i < NONZERO_SIZE; i++) {
        pos = PMAT_COLS * (y + filt_rows[i]) + x + filt_cols[i];
        largest[min_macros(i, N_LARGEST)] = pmat[pos];
        // insertion sort
        for (j = min_macros(i, N_LARGEST); (j > 0) && (largest[j] > largest[j - 1]); j--) {
            // swap
            tmp = largest[j];
            largest[j] = largest[j - 1];
            largest[j - 1] = tmp;
        }
    }

    // lmat is already shifted by FILT_SIZE/2 in the first axis (Y)
    pos = y * PMAT_COLS * N_LARGEST + (x + FILT_SIZE / 2) * N_LARGEST;
    for (i = 0; i < N_LARGEST; i++) {
        lmat[pos + i] = largest[N_LARGEST - 1 - i];
    }

}
