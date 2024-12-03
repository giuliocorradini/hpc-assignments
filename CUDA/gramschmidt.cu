#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
using namespace std;

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"
#include "matrix.h"

extern "C"
{
#include "utils.h"
}

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif


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

/**
 *  Computes the normalization of the k-st column of A using a reduction in shared memory.
 *  The kernel is launched once, with a single block of dimension BLOCK_DIM.
 *
 *  Returns the norm of A^(k) in R[k][k]
 */
__global__ void column_norm(Arr2D &A, int k, DATA_TYPE *nrm) {
    __shared__ DATA_TYPE norm[BLOCK_DIM];

    //IMPROVEMENT: bring A^(k) in shmem

    int SUBCOL_DIM = floordiv(A.y, blockDim.x);

    for (int ly=threadIdx.x; ly < threadIdx.x + SUBCOL_DIM; ly++) {
        if (ly < A.y)
            norm[threadIdx.x] += A[ly][k] * A[ly][k];
        else
            norm[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int b=blockDim.x / 2; b>0; b >>= 1) {  //  funnel pattern of reduction
        if (threadIdx.x < b)
            norm[threadIdx.x] += norm[threadIdx.x + b];
        __syncthreads();
    }

    *nrm = norm[0];
}

__global__ void copy_to_q(Arr2D &A, Arr2D &Q, int k, DATA_TYPE *norm) {
    //  Q^(k) <- normalized A^(k)
    int SUBCOL = blockIdx.x * blockDim.x;
    int tid = threadIdx.x + SUBCOL;

    if (tid < A.y)
       Q[tid][k] = A[tid][k] / *norm;
}

/**
 *  Update R (lower triangular matrix) by updating the block [0, k-1] x [k, y]
 */
__device__ void update_with_basis(Arr2D &A, Arr2D &Q, Arr2D &R, int k) {
    /*R[k][threadIdx.x] = 0;    //TODO: recompute taking offset into account
    
    //Mi porto in shared memory la k-esima colonna di Q, che mi serve (volendo la posso tenere anche dal prcedente passo)
    __shared__ DATA_TYPE qk[blockDim.x];
    qk[threadIdx.x] = Q[threadIdx.x + blockIdx.x * blockDim.x][k];

    //in base al threadID, faccio gemm tra Q e A e salvo in R[k][j]
    __shared__ r_kj_partial[blockDim.x];  
    for (int ly = threadIdx.y; ly < blockDim.y; ly++) {
        r_kj_partial[threadIdx.x] += Q[ly][threadIdx.x] * A[ly][threadIdx.x];
    }
    atomicAdd(&R[k][threadIdx.x], r_kj_partial[threadIdx.x]);*/
}

__device__ void update_a(Arr2D &A, Arr2D &Q, Arr2D &R, int k) {
/*    for (int ly=threadIdx.y; ly<blockIdx.y; ly++) {
        if (ly + blockIdx.y * blockDim.y < A.y)
            A[ly + blockIdx.y * blockDim.y][column] -= Q[ly + blockIdx.y * blockDim.y][k] * R[k][threadIdx.x + blockIdx.x * blockDim.x];*/
}

/**
 *  Host function for gramschmidt computation. Kernels are launched from host with VRAM resident data
 */
void cu_gramschmidt(Arr2D &A, Arr2D &R, Arr2D &Q) {
    /*for (k=0; k<A.x; k++) {
        normalize_column<<<BLOCK_DIM, 1>>>(A, Q, k, norm);

        dim3 edge_blocking(BLOCK_DIM, BLOCK_DIM);
        dim3 column_edge_grid(floordiv(A.x-k, BLOCK_DIM), floordiv(A.y, BLOCK_DIM));
        update_with_basis<<<edge_blocking, column_edge_grid>>>(A, Q, R, k);

        update_a<<<edge_blocking, column_edge_blocking>>>(A, Q, R, k);
    }*/
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;

    HostArr2D A(ni, nj);
    HostArr2D R(nj, nj);
    HostArr2D Q(ni, nj);

    /* Initialize array(s). */
    init_array(ni, nj, A, R, Q);

    struct timespec rt[2];
    double wt;

/*    clock_gettime(CLOCK_REALTIME, rt + 0);
    kernel_gramschmidt(ni, nj, A, R, Q);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));*/

    DeviceArr2D dA(ni, nj);
    DeviceArr2D dR(nj, nj);
    DeviceArr2D dQ(ni, nj);

    dA = A;

    DATA_TYPE *nrm;
    cudaMalloc(&nrm, sizeof(DATA_TYPE));
    column_norm<<<32, 1>>>(dA, 1, nrm);
    copy_to_q<<<BLOCK_DIM, floordiv(dA.y, BLOCK_DIM)>>>(dA, dQ, 1, nrm);
    
    A = dA;
    Q = dQ;
    for (int y=0; y<A.y; y++) {
        cout << A[y][1] << "\t" << Q[y][1] << endl;
    }
    cout << endl;
    
    #ifdef PRINT_DEBUG
    print_array(ni, nj, A, R, Q);
    #endif

    cout << dA.arr << " " << dQ.arr << " " << dR.arr << endl;

    dA.free();
    dR.free();
    dQ.free();
    cudaDeviceReset();

    return 0;
}
