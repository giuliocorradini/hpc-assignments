#pragma once

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Default data type is double, default size is 512. */
#include "gramschmidt.h"


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void facilities_print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                        DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                        DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j;

    fprintf(stderr, "A\n");

      for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++)
            fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
        fprintf(stderr, "\n");
        }


    fprintf(stderr, "Q\n");

      for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++)
            fprintf(stderr, DATA_PRINTF_MODIFIER, Q[i][j]);
        fprintf(stderr, "\n");
        }

    fprintf(stderr, "R\n");

      for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nj; j++)
            fprintf(stderr, DATA_PRINTF_MODIFIER, R[i][j]);
        fprintf(stderr, "\n");
        }

}


static void facilities_print_matrix(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
        fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
    printf("\n");
    }
}

/* Kernel golden standard. Unoptimized */
static void facilities_kernel_gs(int ni, int nj,
                               DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                               DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j, k;

    DATA_TYPE nrm;

      for (k = 0; k < _PB_NJ; k++)
  {
    nrm = 0;

    for (i = 0; i < _PB_NI; i++)
      nrm += A[i][k] * A[i][k];

    R[k][k] = sqrt(nrm);

    for (i = 0; i < _PB_NI; i++)
      Q[i][k] = A[i][k] / R[k][k];

    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q[i][k] * A[i][j];

      for (i = 0; i < _PB_NI; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}


#include <assert.h>

static void facilities_compare_results(int ni, int nj, DATA_TYPE POLYBENCH_2D(X, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(X_cmp, NI, NJ, ni, nj)) {
    #ifdef COMPARE_RESULTS
    for (int i=0; i<ni; i++) {
        for (int j=0; j<nj; j++) {
            if (abs(X[i][j] - X_cmp[i][j]) > 10e-2) {
                   fprintf(stderr, "Different value at (%d, %d)\n", i, j); 
                   fprintf(stderr, "%.5f %.5f\n", X[i][j], X_cmp[i][j]); 
                    assert(0);
            }
        }
    }
    #endif
}
