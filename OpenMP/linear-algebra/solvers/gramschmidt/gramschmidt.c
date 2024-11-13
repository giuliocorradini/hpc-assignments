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

//0 = sequenziale, 1 = parallelo, 2 = offloading
#ifndef VERSION
#define VERSION 0
#endif
//#define OFFLOADING

/*Main computational kernel. The whole function will be timed,
  including the call and return. */
#if VERSION == 0
static void kernel_gramschmidt(int ni, int nj,
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

#elif VERSION == 1
//parallel version
static void kernel_gramschmidt(int ni, int nj,
                              DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                              DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                              DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  for (k = 0; k < _PB_NJ; k++)
  {
    nrm = 0;
    
    #pragma omp parallel for reduction(+:nrm) num_threads(4) schedule(static) shared(A) private(i)
    for (i = 0; i < _PB_NI; i++)
      nrm += A[i][k] * A[i][k];

    R[k][k] = sqrt(nrm);

    #pragma omp parallel for num_threads(4) schedule(static) shared(A) private(i)
    for (i = 0; i < _PB_NI; i++)
      Q[i][k] = A[i][k] / R[k][k];

    #pragma omp parallel for private(i) num_threads(4) schedule(static)
    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      #pragma omp parallel for reduction(+:R[k][j]) num_threads(4) schedule(static) shared(A,Q) private(i)
      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q[i][k] * A[i][j];

      #pragma omp parallel for num_threads(4) schedule(static) shared(Q,R) private(i) 
      for (i = 0; i < _PB_NI; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}

#elif VERSION == 2
//let's try offloading
#pragma omp declare target
static void kernel_gramschmidt(int ni, int nj,
                              DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                              DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                              DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  for (k = 0; k < _PB_NJ; k++)
  {
    #pragma omp target data map(to: A[0:_PB_NI][k:_PB_NJ]) map(from:R[0:_PB_NJ][0:_PB_NJ],Q[0:_PB_NI][k:_PB_NJ])
    #pragma omp target teams num_teams(_PB_NI/omp_get_max_threads()) thread_limit(_PB_NI/omp_get_max_threads())
    {
      nrm = 0;
      #pragma omp distribute parallel for reduction(+:nrm) shared(A)
      for (i = 0; i < _PB_NI; i++)
        nrm += A[i][k] * A[i][k];
    
      R[k][k] = sqrt(nrm);
    
      #pragma omp distribute parallel for reduction(+:nrm) shared(A)
      for (i = 0; i < _PB_NI; i++)
        Q[i][k] = A[i][k] / R[k][k];

    }

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

#elif VERSION == 3
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

  for (k = 0; k < _PB_NJ; k++)
  {
    nrm = 0;
    
    #pragma omp parallel for reduction(+:nrm) num_threads(4) schedule(static) shared(A) private(i)
    for (j = 0; j < _PB_NJ; j++)
      nrm += A[k][j] * A[k][j];

    R[k][k] = sqrt(nrm);

    #pragma omp parallel for num_threads(4) schedule(static) shared(A) private(i)
    for (i = 0; i < _PB_NI; i++)
      Q[i][k] = A[i][k] / R[k][k];

    #pragma omp parallel for private(i) num_threads(4) schedule(static)
    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      #pragma omp parallel for reduction(+:R[k][j]) num_threads(4) schedule(static) shared(A,Q) private(i)
      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q[i][k] * A[i][j];

      #pragma omp parallel for num_threads(4) schedule(static) shared(Q,R) private(i) 
      for (i = 0; i < _PB_NI; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
}

#endif

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
