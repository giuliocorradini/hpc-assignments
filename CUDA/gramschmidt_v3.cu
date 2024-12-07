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
            R[i][j] = 0.0f;//((DATA_TYPE)i * (j + 2)) / nj;
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

CUDA IMPLEMENTATION

**********************************************/
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void norma_a(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, DATA_TYPE *__restrict__ q, int ni, int nj, int k) {

    cg::thread_block cta = cg::this_thread_block();

    int a_row = blockDim.y*blockIdx.y + threadIdx.y;
    if(a_row < ni){
        DATA_TYPE value = a[a_row*nj + k];
        value *= value;
        atomicAdd(&r[k*ni+k], value);

        //sync della grid
        cg::sync(cta);
        //un thread, quando tutti gli altri hanno finito di scrivere, calcola la norma
        if(blockIdx.y==0 && blockIdx.x==0 && threadIdx.y==0 && threadIdx.x==0){
            r[k*ni+k] = sqrt(r[k*ni+k]);
        }

    }

}

__global__ void init_col_k_q(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, DATA_TYPE *__restrict__ q, int ni, int nj, int k) {

    int a_row = blockDim.y*blockIdx.y + threadIdx.y;
    if(a_row < ni){
        q[a_row*nj + k] = a[a_row*nj + k] / r[k*ni+k];
    }
}
__global__ void dot_product_a_q(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, DATA_TYPE *__restrict__ q, int ni, int nj, int k) {

    //r è inizializzzata a matrice nulla
    DATA_TYPE partial_sum = 0;
    int a_row = blockDim.y*blockIdx.y + threadIdx.y;
    int a_col = blockDim.x*blockIdx.x + threadIdx.x;

    if(a_row < ni && a_col > k){
        for(int j = k; j<nj; j++){
            partial_sum += a[a_row*nj + a_col] * q[a_row*ni + k];
        }

        r[k*ni+a_col] = partial_sum;  
    }

}

__global__ void update_a(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, DATA_TYPE *__restrict__ q, int ni, int nj, int k) {
    
    int a_row = blockDim.y*blockIdx.y + threadIdx.y;
    //offset dovuto a k per tenere conto del restringimento della grid
    int a_col = (k/blockDim.x) + blockDim.x*blockIdx.x + threadIdx.x;

    if(a_col > k && a_row < ni){
        //è per chiarezza, il compilatore poi propaga il valore
        DATA_TYPE result = a[a_row*ni + a_col] - q[a_row*ni + k] * r[k*ni+a_col];
        //aggiorno il valore, non c'è concorrenza stavolta
        a[a_row*ni + a_col] = result;
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

    /* Initialize array(s). */
    init_array(ni, nj, A, R, Q);

    struct timespec rt[2];
    double wt;

    DATA_TYPE *a =A.arr;
    DATA_TYPE *q =Q.arr; 
    DATA_TYPE *r =R.arr; 

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
        //uso un thread per riga
        num_blocks = (ni + BLOCK_SIZE - 1) / BLOCK_SIZE;
        norma_a<<<num_blocks, BLOCK_SIZE>>>(d_a, d_r, d_q, ni, nj, k);
        gpuErrchk(cudaPeekAtLastError());
        
        //INIZIALIZZO COLONNA k DI Q - La grid ha la stessa dimensione
        init_col_k_q<<<num_blocks, BLOCK_SIZE>>>(d_a, d_r, d_q, ni, nj, k);
        gpuErrchk(cudaPeekAtLastError());
        
        //DOPO che tutti i tread hanno scritto su A setto la radice
        //serve un thread per colonna
        num_blocks = (nj + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dot_product_a_q<<<num_blocks, BLOCK_SIZE>>>(d_a, d_r, d_q, ni, nj, k);
        gpuErrchk(cudaPeekAtLastError());

        //AVENDO IN R IL PRODOTTO SCALARE POSSO AGGIORNARE A, stavolta con un kernel parallelo
        //le dimensioni sono le stesse dell'operazione precedente
        //la griglia si restringe con l'aumentare di k per creare solo i thread necessari
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid((nj + BLOCK_SIZE - 1 - k)/BLOCK_SIZE, ((ni + BLOCK_SIZE - 1)/BLOCK_SIZE));
        update_a<<<dimGrid, dimBlock>>>(d_a, d_r, d_q, ni, nj, k);
        gpuErrchk(cudaPeekAtLastError());
    
    }

    //MEMORY BACK TO HOST
    gpuErrchk(cudaMemcpy(a, d_a, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(r, d_r, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(q, d_q, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost));

    
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt  (GPU) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));

   
    
    #ifdef PRINT_DEBUG
    //ritraspongo le matrici in caso si voglia stamparle
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
