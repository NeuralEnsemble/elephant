#define L             {{L}}
#define N_LARGEST     {{N_LARGEST}}
#define PMAT_ROWS     {{PMAT_ROWS}}
#define PMAT_COLS     {{PMAT_COLS}}
#define NONZERO_SIZE  {{NONZERO_SIZE}}
#define SYMMETRIC     {{SYMMETRIC}}

#define min_macros(a,b)   ((a) < (b) ? (a) : (b))


__constant unsigned int filt_rows[] = {{filt_rows}};
__constant unsigned int filt_cols[] = {{filt_cols}};


__kernel void pmat_neighbors(__global float *lmat, __global const float *pmat) {
    const unsigned long gid = get_global_id(0);

#if SYMMETRIC
    const unsigned long y0 = (unsigned long) ceil( (-3 + sqrt((float) (9 + 8 * gid))) / 2 );
    const unsigned long y = y0 + L;
    const unsigned long x = (unsigned long) (gid - (y0 * (y0 + 1) / 2));
#else
    const unsigned long y = gid / (PMAT_COLS - L + 1);
    const unsigned long x = gid - y * (PMAT_COLS - L + 1);
#endif

    float buffer[N_LARGEST + 1];
    unsigned int i, j;
    unsigned long pos;
    float tmp;
    for (i = 0; i < NONZERO_SIZE; i++) {
        pos = PMAT_COLS * (y + filt_rows[i]) + x + filt_cols[i];
        buffer[min_macros(i, N_LARGEST)] = pmat[pos];
        // insertion sort
        for (j = min_macros(i, N_LARGEST); (j > 0) && (buffer[j] > buffer[j - 1]); j--) {
            // swap
            tmp = buffer[j];
            buffer[j] = buffer[j - 1];
            buffer[j - 1] = tmp;
        }
    }

    pos = (y + L / 2) * PMAT_COLS * N_LARGEST + (x + L / 2) * N_LARGEST;
    for (i = 0; i < N_LARGEST; i++) {
        lmat[pos + i] = buffer[N_LARGEST - 1 - i];
    }

}
