#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"

#define NTHREADS (4)

/* Array initialization. */
static void init_array(int ni, int nj,
    DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
    DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
    int i, j;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / ni;
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

static void print_matrix(int ni, int nj,
    DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++)
            fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
        printf("\n");
    }
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

    for (k = 0; k < _PB_NJ; k++) {
        // Consideriamo la colonna k-esima di A
        nrm = 0;

        //  Calcoliamo la norma di A^(k)
        #pragma omp parallel for simd reduction(+ : nrm)
        for (i = 0; i < _PB_NI; i++)
            nrm += A[i][k] * A[i][k];

        //  che viene salvata in nel k-esimo elemento diagonale di R
        R[k][k] = sqrt(nrm);

        // la k-esima colonna di Q è la normalizzazione della k-esima colonna di A
        // R[k][k] è una very busy expression
        // qui puoi fare un parallel for su più thread (con anche le SIMD)
        // Possiamo dividere il lavoro per righe (ovvero dividiamo lo spazio di iterazione di i)
        //  usiamo più linee di cache (nei vari processori) ma non dovremmo avere problemi di snooping
        #pragma omp parallel for simd num_threads(NTHREADS) schedule(static)
        for (i = 0; i < _PB_NI; i++)
            Q[i][k] = A[i][k] / R[k][k];

        // Per ogni colonna successiva alla k-esima (definita nell'outer loop)
        // anche questo possiamo parallelizzarlo su più thread, perché ogni colonna è tratta indipendentemente
        #pragma omp parallel for schedule(static) num_threads(NTHREADS) shared(R, Q, A)
        for (j = k + 1; j < _PB_NJ; j++) {
            R[k][j] = 0;

            // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
            //reduction
            //    sto lavorando per colonne... non il top per la cache
            for (i = 0; i < _PB_NI; i++)
                R[k][j] += Q[i][k] * A[i][j];

            // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
            // reduction
            for (i = 0; i < _PB_NI; i++)
                A[i][j] = A[i][j] - Q[i][k] * R[k][j];
        }
    }
}

#include "facilities.h"
int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, NJ, NJ, nj, nj);
    POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, NI, NJ, ni, nj);

    /* Declaration for golden standard */

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

    #ifdef COMPARE_RESULTS
    POLYBENCH_2D_ARRAY_DECL(A2, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(R2, DATA_TYPE, NJ, NJ, nj, nj);
    POLYBENCH_2D_ARRAY_DECL(Q2, DATA_TYPE, NI, NJ, ni, nj);

    init_array(ni, nj,
        POLYBENCH_ARRAY(A2),
        POLYBENCH_ARRAY(R2),
        POLYBENCH_ARRAY(Q2));

    facilities_kernel_gs(ni, nj,
        POLYBENCH_ARRAY(A2),
        POLYBENCH_ARRAY(R2),
        POLYBENCH_ARRAY(Q2));

    facilities_compare_results(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A2));
    facilities_compare_results(nj, nj, POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(R2));
    facilities_compare_results(ni, nj, POLYBENCH_ARRAY(Q), POLYBENCH_ARRAY(Q2));
    #endif

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(R);
    POLYBENCH_FREE_ARRAY(Q);

    return 0;
}
