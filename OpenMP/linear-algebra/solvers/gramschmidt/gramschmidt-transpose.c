#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"

#define NTHREADS 4

#include "transpose.h"  //< depends on NTHREADS definiton

/* Array initialization. */
static void init_array(int ni, int nj,
                       DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                       DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      A[i][j] = ((DATA_TYPE)(i+1) * (j+1)) / ni;
      Q[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
    }
  for (i = 0; i < nj; i++)
    for (j = 0; j < nj; j++)
      R[i][j] = ((DATA_TYPE)i * (j + 2)) / nj;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                        DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                        DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if (i % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
  for (i = 0; i < nj; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, R[i][j]);
      if (i % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, Q[i][j]);
      if (i % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, NJ, NJ, nj, nj);
  POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A_T, DATA_TYPE, NJ, NI, nj, ni);
  POLYBENCH_2D_ARRAY_DECL(Q_T, DATA_TYPE, NJ, NI, nj, ni);

  /* Initialize array(s). */
  init_array(ni, nj,
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(R),
             POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt(ni, nj,
                     POLYBENCH_ARRAY(A),
                     POLYBENCH_ARRAY(R),
                     POLYBENCH_ARRAY(Q),
                     POLYBENCH_ARRAY(A_T),
                     POLYBENCH_ARRAY(Q_T));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);
  POLYBENCH_FREE_ARRAY(A_T);
  POLYBENCH_FREE_ARRAY(Q_T);

  return 0;
}
