#pragma once
#include <polybench.h>
#include "gramschmidt.h"

/**Funzione per trasformare la matrice in trasposta */
static void transpose_matrix(int ni, int nj,
   DATA_TYPE POLYBENCH_2D(M, NI, NJ, ni, nj),
   DATA_TYPE POLYBENCH_2D(M_T, NJ, NI, nj, ni))
{
    // Transpose the matrix M, to read its columns into cache as rows

    #pragma omp parallel for simd num_threads(NTHREADS) schedule(static) collapse(2)
      for(int i = 0; i < _PB_NI; i++){
       for(int j = 0; j < _PB_NJ; j++){
          M_T[j][i] = M[i][j];
       }
      }

}

#ifndef OPTIMIZATION
static void kernel_gramschmidt(int ni, int nj,
                               DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                               DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj),
                               DATA_TYPE POLYBENCH_2D(A_T, NJ, NI, ni, nj),
                               DATA_TYPE POLYBENCH_2D(Q_T, NJ, NI, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  transpose_matrix(ni, nj, A, A_T);
  transpose_matrix(ni, nj, Q, Q_T);

  for (k = 0; k < _PB_NJ; k++)
  {
    nrm = 0;

    for (i = 0; i < _PB_NI; i++)
      nrm += A_T[k][i] * A_T[k][i];

    R[k][k] = sqrt(nrm);

    for (i = 0; i < _PB_NI; i++)
      Q_T[k][i] = A_T[k][i] / R[k][k];

    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q_T[k][i] * A_T[j][i];

      for (i = 0; i < _PB_NI; i++)
        A_T[j][i] = A_T[j][i] - Q_T[k][i] * R[k][j];
    }
  }
}
#elif OPTIMIZATION == static
static void kernel_gramschmidt(int ni, int nj,
                              DATA_TYPE POLYBENCH_2D(A, NJ, NI, nj, ni),
                              DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                              DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj),
                              DATA_TYPE POLYBENCH_2D(A_T, NJ, NI, ni, nj),
                              DATA_TYPE POLYBENCH_2D(Q_T, NJ, NI, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  transpose_matrix(ni, nj, A, A_T);
  transpose_matrix(ni, nj, Q, Q_T);

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
      Q_T[k][i] = A_T[k][i] / R[k][k];


    #pragma omp parallel for schedule(static) num_threads(NTHREADS)
    for (j = k + 1; j < _PB_NJ; j++)
    {
      R[k][j] = 0;

      // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
      //reduction
      for (i = 0; i < _PB_NI; i++)
        R[k][j] += Q_T[k][i] * A_T[j][i];

      // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
      for (i = 0; i < _PB_NI; i++)
        A_T[j][i] = A_T[j][i] - Q_T[k][i] * R[k][j];
    }
  }

}
#elif OPTIMIZATION == workerthreads
static void kernel_gramschmidt(int ni, int nj,
                              DATA_TYPE POLYBENCH_2D(A, NJ, NI, nj, ni),
                              DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                              DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj),
                              DATA_TYPE POLYBENCH_2D(A_T, NJ, NI, ni, nj),
                              DATA_TYPE POLYBENCH_2D(Q_T, NJ, NI, ni, nj))
{
  int i, j, k;

  DATA_TYPE nrm;

  transpose_matrix(ni, nj, A, A_T);
  transpose_matrix(ni, nj, Q, Q_T);

   DATA_TYPE nrm;

  #pragma omp parallel num_threads(NTHREADS) private(i, j, k)        //spawn a group of threads
  {
      #pragma omp single
      for (k = 0; k < _PB_NJ; k++)
      {
         nrm = 0;

         #pragma omp taskloop shared(nrm) 
         for (i = 0; i < _PB_NI; i++)
               nrm += A_T[k][i] * A_T[k][i];

         R[k][k] = sqrt(nrm);

         #pragma omp taskloop
         for (i = 0; i < _PB_NI; i++)
               Q_T[k][i] = A_T[k][i] / R[k][k];

         #pragma omp taskloop
         for (j = k + 1; j < _PB_NJ; j++)
         {
         R[k][j] = 0;

         for (i = 0; i < _PB_NI; i++)
            R[k][j] += Q_T[k][i] * A_T[j][i];

         for (i = 0; i < _PB_NI; i++)
            A_T[j][i] = A_T[j][i] - Q_T[k][i] * R[k][j];
         }
      }
   }
  
}
#endif //OPTIMIZATION