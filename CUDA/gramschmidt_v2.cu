#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

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
#define NTHREADS 4

using namespace std;

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"
#include "matrix.h"

extern "C"
{
#include "utils.h"
}

/**Funzione per trasformare la matrice in trasposta */
static void transpose_matrix(int ni, int nj, DATA_TYPE* M, DATA_TYPE* M_T)
{
    // Transpose the matrix M, to read its columns into cache as rows
#pragma omp parallel for simd num_threads(NTHREADS) schedule(static) collapse(2)
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            M_T[j * nj + i] = M[i * ni + j];
        }
    }
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
            nrm += A[k][i] * A[k][i];

        //  che viene salvata in nel k-esimo elemento diagonale di R
        R[k][k] = sqrt(nrm);

        // la k-esima colonna di Q è la normalizzazione della k-esima colonna di A
        // R[k][k] è una very busy expression
        for (i = 0; i < ni; i++)
            Q[k][i] = A[k][i] / R[k][k];

        // Per ogni colonna successiva alla k-esima (definita nell'outer loop)
        for (j = k + 1; j < nj; j++) {
            R[k][j] = 0;

            // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
            for (i = 0; i < ni; i++)
                R[k][j] += Q[k][i] * A[j][i];

            // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
            for (i = 0; i < ni; i++)
                A[j][i] = A[j][i] - Q[k][i] * R[k][j];
        }
    }
}

/**********************************************

CUDA IMPLEMENTATION

**********************************************/
__global__ void calcolo_norma_a(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, int ni, int nj, int k) {
    
    __shared__ DATA_TYPE partial_norm;

    if(blockIdx.y==0 && threadIdx.y==0){
        partial_norm = 0;
    }

    int a_row = blockDim.y*blockIdx.y + threadIdx.y;
    if(a_row < ni){
        DATA_TYPE value = a[a_row*nj + k];
        value *= value;
        atomicAdd(&r[k*ni+k], value);
    }
}


int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;

    Arr2D A(ni, nj);
    Arr2D R(nj, nj);
    Arr2D Q(ni, nj);
    Arr2D A_T(ni, nj);
    Arr2D Q_T(ni, nj);

    /* Initialize array(s). */
    init_array(ni, nj, A, R, Q);

    struct timespec rt[2];
    double wt;


    DATA_TYPE *ab = A.arr;
    DATA_TYPE *qb = Q.arr;

    DATA_TYPE *a =A_T.arr;
    DATA_TYPE *q =Q_T.arr; 

    //transpongo qu per ottimizzare i tempo in memoria
    transpose_matrix(ni, nj, a, ab);
    transpose_matrix(ni, nj, q, qb);

    clock_gettime(CLOCK_REALTIME, rt + 0);

    kernel_gramschmidt(ni, nj, A_T, R, Q_T);

    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));
    
    //CUDA VERSION
    //Reinizializza matrici
    init_array(ni, nj, A, R, Q);

    DATA_TYPE *r =R.arr; 
    a = A.arr;
    q = Q.arr;

    //allocazione memoria A,R,Q (R probabilmente non serve, si può rispatmiare spazio)
    cudaMallocHost((void **)&a, sizeof(DATA_TYPE) * ni * nj);
    cudaMallocHost((void **)&r, sizeof(DATA_TYPE) * nj * nj);
    cudaMallocHost((void **)&q, sizeof(DATA_TYPE) * ni * nj);

    //allocazione memoria GPU
    DATA_TYPE *d_a, *d_r, *d_q;
    gpuErrchk(cudaMalloc((void **)&d_a, sizeof(DATA_TYPE) * ni * nj));
    gpuErrchk(cudaMalloc((void **)&d_r, sizeof(DATA_TYPE) * nj * nj));
    gpuErrchk(cudaMalloc((void **)&d_q, sizeof(DATA_TYPE) * ni * nj));

    //READY, STEADY, RUN!!!
    clock_gettime(CLOCK_REALTIME, rt + 0);

    gpuErrchk(cudaMemcpy(d_a, a, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_r, r, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_q, q, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice));
    //compute the factorization

    int num_blocks;
    for (int k = 0; k < nj; k++) {
        //KERNEL PER CALCOLO DI NORM A - DIM BLOCK limitata a 32*1
        num_blocks = (ni + BLOCK_SIZE - 1) / BLOCK_SIZE;
        calcolo_norma_a<<<num_blocks, BLOCK_SIZE>>>(d_a, d_r, ni, nj, k);
        gpuErrchk(cudaPeekAtLastError());
        //DOPO che tutti i tread hanno scritto su A setto la radice
        //cudaMemset(r[k*ni + k], sqrt(r[k*ni + k]), sizeof(DATA_TYPE));

    
    
    }
        /* compute_a<<<dimGrid, dimBlock>>>(d_a, d_r, d_q, ni, nj, k);
        gpuErrchk(cudaPeekAtLastError()); */
        //RIPORTO A nell'HOST

    //MEMORY BACK TO HOST
    gpuErrchk(cudaMemcpy(a, d_a, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(r, d_r, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(q, d_q, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));

    
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt  (GPU) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));

   
    
    #ifdef PRINT_DEBUG
    //ritraspongo le matrici in caso si voglia stamparle
    transpose_matrix(ni, nj, A_T, A);
    transpose_matrix(ni, nj, Q_T, Q);
    
    print_array(ni, nj, A, R, Q);
    #endif

    //FREE HOST MEMORY
    cudaFreeHost(a);
    cudaFreeHost(r);
    cudaFreeHost(q);
    //FREE GPU MEMORY
    cudaFree(d_a);
    cudaFree(d_r);
    cudaFree(d_q);

    return 0;
}