#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"

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
      A[i][j] = ((DATA_TYPE)i * j) / ni;
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

    fprintf(stderr, "A\n");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if (j % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
    fprintf(stderr, "R\n");
  for (i = 0; i < nj; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, R[i][j]);
      if (j % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
    fprintf(stderr, "Q\n");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, Q[i][j]);
      if (j % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gramschmidt(int ni, int nj,
                               DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                               DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  for (k = 0; k < _PB_NJ; k++)
  {
    // Consideriamo la colonna k-esima di A
    nrm = 0;

    //  Calcoliamo la norma di A^(k)
    // reduction
    for (i = 0; i < _PB_NI; i++)
      nrm += A[i][k] * A[i][k];

    //  che viene salvata in nel k-esimo elemento diagonale di R
    R[k][k] = sqrt(nrm);

    // la k-esima colonna di Q è la normalizzazione della k-esima colonna di A
    // R[k][k] è una very busy expression
    // qui puoi fare un parallel for su più thread (con anche le SIMD)
    for (i = 0; i < _PB_NI; i++)
      Q[i][k] = A[i][k] / R[k][k];

    // Per ogni colonna successiva alla k-esima (definita nell'outer loop)
    // anche questo possiamo parallelizzarlo su più thread, perché ogni colonna è tratta indipendentemente
    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
      //reduction    
      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q[i][k] * A[i][j];

      // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
      // reduction
      for (i = 0; i < _PB_NI; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}

/* Gold standard for numerical accuracy */
static void kernel_gramschmidt_gs(int ni, int nj,
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

static void golden_standard_difference(
    int ni, int nj,
                               DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(A_gs, NJ, NJ, nj, nj),
                               DATA_TYPE POLYBENCH_2D(R, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(R_gs, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(Q, NJ, NJ, nj, nj),
                               DATA_TYPE POLYBENCH_2D(Q_gs, NI, NJ, ni, nj)
) {
  /* Compare results against golden standard */
  for (int i=0; i<ni; i++) {
    for(int j=0; j<nj; j++) {
        A[i][j] -= A_gs[i][j];
        R[i][j] -= R_gs[i][j];
    }
  }

  for (int i=0; i<nj; i++) { 
    for(int j=0; j<nj; j++) {
        Q[i][j] -= Q_gs[i][j];
    }
  }
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

  /* Declaration for golden standard */
  POLYBENCH_2D_ARRAY_DECL(A_gs, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(R_gs, DATA_TYPE, NJ, NJ, nj, nj);
  POLYBENCH_2D_ARRAY_DECL(Q_gs, DATA_TYPE, NI, NJ, ni, nj);

  /* Initialize array(s). */
  init_array(ni, nj,
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(R),
             POLYBENCH_ARRAY(Q));


  init_array(ni, nj,
             POLYBENCH_ARRAY(A_gs),
             POLYBENCH_ARRAY(R_gs),
             POLYBENCH_ARRAY(Q_gs));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt(ni, nj,
                     POLYBENCH_ARRAY(A),
                     POLYBENCH_ARRAY(R),
                     POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));
  /* Run golden standard */
  kernel_gramschmidt_gs(ni, nj, POLYBENCH_ARRAY(A_gs), POLYBENCH_ARRAY(R_gs), POLYBENCH_ARRAY(Q_gs));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));
//  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A_gs), POLYBENCH_ARRAY(R_gs), POLYBENCH_ARRAY(Q_gs)));

  golden_standard_difference(
    ni, nj,
    POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_gs),
    POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(R_gs),
    POLYBENCH_ARRAY(Q), POLYBENCH_ARRAY(Q_gs)
  );

  printf("Difference against golden standard.\n"); 
  //polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  POLYBENCH_FREE_ARRAY(A_gs);
  POLYBENCH_FREE_ARRAY(R_gs);
  POLYBENCH_FREE_ARRAY(Q_gs);
  return 0;
}
