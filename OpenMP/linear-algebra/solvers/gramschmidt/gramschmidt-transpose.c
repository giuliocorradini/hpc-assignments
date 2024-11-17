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

/********************* 

IMPLEMENTAZIONI KERNEL

**********************/

/**Funzione per trasformare la matrice in trasposta */
static void transponi_matrice(int ni, int nj,
                              DATA_TYPE M[_PB_NI][_PB_NJ],
                              DATA_TYPE M_T[_PB_NJ][_PB_NI])
{
  //read M by row so we avoid caches miss
  #pragma omp parallel for simd num_threads(NTHREADS) schedule(static) collapse(2)
  for(int i = 0; i < _PB_NI; i++){
   for(int j = 0; j < _PB_NJ; j++){
      M_T[j][i] = M[i][j];
   }
  }

}

/*Main computational kernel. The whole function will be timed,
  including the call and return. */

/*Si è osservato che l'algoritmo carica i dati iterando sulle righe della matrice, dunque può convenire 
cambiare le matricei A e Q con le loro trasposte, a patto che la creazione di tali matrici valga l'ottimizzazzione
ottenuta*/
static void kernel_gramschmidt(int ni, int nj,
                              DATA_TYPE POLYBENCH_2D(A, NJ, NI, nj, ni),
                              DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                              DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  DATA_TYPE A_T[_PB_NJ][_PB_NI];
  DATA_TYPE Q_T[_PB_NJ][_PB_NI];

  transponi_matrice(_PB_NI,_PB_NJ,A, A_T);
  transponi_matrice(_PB_NI,_PB_NJ,Q, Q_T);

  for (k = 0; k < _PB_NJ; k++)
  {
   // Consideriamo la colonna k-esima di A
    nrm = 0;

    //  Calcoliamo la norma di A^(k)
    #pragma omp parallel for simd reduction(+:nrm) 
    for (i = 0; i < _PB_NI; i++)
      nrm += A_T[k][i] * A_T[k][i];

    //  che viene salvata in nel k-esimo elemento diagonale di R
    R[k][k] = sqrt(nrm);

    #pragma omp parallel for simd num_threads(NTHREADS) schedule(static)
    for (i = 0; i < _PB_NI; i++)
      Q[k][i] = A[k][i] / R[k][k];


    #pragma omp parallel for schedule(static) num_threads(NTHREADS)
    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
      //reduction
      #pragma omp parallel for simd reduction(+ : R[k][j])
      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q[k][i] * A[j][i];

      // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
      #pragma omp parallel for simd schedule(static) num_threads(NTHREADS)
      for (i = 0; i < _PB_NI; i++)
        A[j][i] = A[j][i] - Q[k][i] * R[k][j];
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
                     POLYBENCH_ARRAY(Q));

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

  return 0;
}
