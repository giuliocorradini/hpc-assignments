#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <iostream>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

using namespace std;

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"
#include "matrix.h"

extern "C"
{
#include "utils.h"
}

/* Array initialization. */
static void init_array(int ni, int nj, Arr2D &A, Arr2D &R, Arr2D &Q) {
    int i, j;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            A[i][j] = ((DATA_TYPE)i * j) / ni;
            Q[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
        }
    for (i = 0; i < nj; i++)
        for (j = 0; j < nj; j++)
            R[i][j] = ((DATA_TYPE)i * (j + 2)) / nj;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, Arr2D &A, Arr2D &R, Arr2D &Q) {
    int i, j;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            cout << A[i][j];
            if (i % 20 == 0)
                cout << endl;
        }
    cout << endl;
    for (i = 0; i < nj; i++)
        for (j = 0; j < nj; j++) {
            cout << R[i][j];
            if (i % 20 == 0)
                cout << endl;
        }
    cout << endl;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            cout << Q[i][j];
            if (i % 20 == 0)
                cout << endl;
        }
    cout << endl;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gramschmidt(int ni, int nj, Arr2D &A, Arr2D &R, Arr2D &Q) {
    int i, j, k;

    DATA_TYPE nrm;

    for (k = 0; k < nj; k++) {
        // Consideriamo la colonna k-esima di A
        nrm = 0;

        //  Calcoliamo la norma di A^(k)
        for (i = 0; i < ni; i++)
            nrm += A[i][k] * A[i][k];

        //  che viene salvata in nel k-esimo elemento diagonale di R
        R[k][k] = sqrt(nrm);

        // la k-esima colonna di Q è la normalizzazione della k-esima colonna di A
        // R[k][k] è una very busy expression
        for (i = 0; i < ni; i++)
            Q[i][k] = A[i][k] / R[k][k];

        // Per ogni colonna successiva alla k-esima (definita nell'outer loop)
        for (j = k + 1; j < nj; j++) {
            R[k][j] = 0;

            // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
            for (i = 0; i < ni; i++)
                R[k][j] += Q[i][k] * A[i][j];

            // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
            for (i = 0; i < ni; i++)
                A[i][j] = A[i][j] - Q[i][k] * R[k][j];
        }
    }
}

/**********************************************

CUDA KERNELS

**********************************************/
__global__ void compute_norma_and_q(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ b, DATA_TYPE *__restrict__ c, int ni, int nj)
{

}
int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;

    Arr2D A(ni, nj);
    Arr2D R(nj, nj);
    Arr2D Q(ni, nj);

    /* Initialize array(s). */
    init_array(ni, nj, A, R, Q);

    struct timespec rt[2];
    double wt;


    clock_gettime(CLOCK_REALTIME, rt + 0);
    kernel_gramschmidt(ni, nj, A, R, Q);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));
    
    //CUDA VERSION
    //Reinizializza matrici
    init_array(ni, nj, A, R, Q);

    DATA_TYPE *a =A.arr;
    DATA_TYPE *r =R.arr; 
    DATA_TYPE *q =Q.arr; 

    //allocazione memoria A,R,Q (R probabilmente non serve, si può rispatmiare spazio)
    cudaMallocHost((void **)&a, sizeof(DATA_TYPE) * ni * nj);
    cudaMallocHost((void **)&r, sizeof(DATA_TYPE) * nj * nj);
    cudaMallocHost((void **)&q, sizeof(DATA_TYPE) * ni * nj);

    //allocazione memoria GPU
    DATA_TYPE *d_a, *d_r, *d_q;
    gpuErrchk(cudaMalloc((void **)&d_a, sizeof(DATA_TYPE) * ni * nj));
    gpuErrchk(cudaMalloc((void **)&d_r, sizeof(DATA_TYPE) * nj * nj));
    gpuErrchk(cudaMalloc((void **)&d_q, sizeof(DATA_TYPE) * ni * nj));

    //Memory movement
    gpuErrchk(cudaMemcpy(d_a, a, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_r, r, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_q, q, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((ni + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (nj + (BLOCK_SIZE)-1) / (BLOCK_SIZE));
    compute_norma_and_q<<<dimGrid, dimBlock>>>(d_a, d_r, d_q, ni, nj);
    gpuErrchk(cudaPeekAtLastError());

    //TODO Remove if unecessary
    gpuErrchk(cudaMemcpy(a, d_a, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(r, d_r, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(q, d_q, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_REALTIME, rt + 0);

    //DO PARALELIZE
    
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt  (GPU) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));

    

    #ifdef PRINT_DEBUG
    print_array(ni, nj, A, R, Q);
    #endif

    //FREE HOST MEMORY
    cudaFreeHost(a);
    cudaFreeHost(r);
    cudaFreeHost(q);
    //FREE GPU MEMORY
    cudaFree(a);
    cudaFree(r);
    cudaFree(q);

    return 0;
}
