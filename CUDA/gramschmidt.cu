#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <cassert>
using namespace std;

/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"
#include "matrix.h"

#include "host_kernel.h"

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
            A[i][j] = ((DATA_TYPE)(i+1) * (j+1)) / ni;
            Q[i][j] = 0;
        }
    for (i = 0; i < nj; i++)
        for (j = 0; j < nj; j++)
            R[i][j] = 0;
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

/**
 *  Computes the normalization of the k-st column of A using a reduction in shared memory.
 *  The kernel is launched once, with a single block of dimension BLOCK_DIM.
 *
 *  Returns the norm of A^(k) in R[k][k]
 */
__global__ void column_norm(DeviceArr2D A, DeviceArr2D R, int k) {
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

    if (threadIdx.x == 0)
        R[k][k] = sqrt(norm[0]);
}

__global__ void copy_to_q(DeviceArr2D A, DeviceArr2D Q, DeviceArr2D R, int k) {
    //  Q^(k) <- normalized A^(k)
    int SUBCOL = blockIdx.x * blockDim.x;
    int tid = threadIdx.x + SUBCOL;

    if (tid < A.y)
       Q[tid][k] = A[tid][k] / R[k][k];
}

/**
 *  Update R (lower triangular matrix) by multiplying Q^(k) and A_{k+1, y}
 */
__device__ void recompute(DeviceArr2D A, DeviceArr2D Q, DeviceArr2D R, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + k;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Mi porto in shared memory la k-esima colonna di Q, che viene usata per ogni x 
    __shared__ DATA_TYPE qk[blockDim.y];

    if (y < Q.y)
        qk[threadIdx.y] = Q[y][k];

    if (x < R.x and y < A.y) {
        if (blockIdx.y == 0 && threadIdx.y == 0)
            R[k][x] = 0;
    
        R[k][x] += A[y][x] * Q[y][k];

    }

//non lo posso fare qua nonostante le griglie abbiano la stessa dimensione, perché R_kx non è
//ancora completo
        //in base al threadID, faccio gemm tra Q e A e salvo in R[k][j]
//        for (int ly = threadIdx.y; ly < blockDim.y; ly++) {
//            r_kj_partial[threadIdx.x] += Q[ly][threadIdx.x] * A[ly][threadIdx.x];
//        }
//        atomicAdd(&R[k][threadIdx.x], r_kj_partial[threadIdx.x]);
}

__global__ void update_a(DeviceArr2D A, DeviceArr2D R, DeviceArr Q, int k) {

    int x = blockIdx.x * blockDim.x + threadIdx.x + k;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ DATA_TYPE qk[blockDim.y];    //< k-esima colonna di Q
    __shared__ DATA_TYPE r_k[blockDim.y];   //< k-esima sottoriga di R per indici di colonna da k+1 a nj (A.y)

    if (x < R.xi && threadIdx.y == 0)
        r_k[threadIdx.x] = R[k][x];

    if (y < Q.y && threadIdx.y == 0)
        qk[threadIdx.y] = Q[y][k];

    if (x < R.x and y < A.y) {
        A[y][x] -= qk[threadIdx.y] * r_k[threadIdx.x];

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

    /*
    clock_gettime(CLOCK_REALTIME, rt + 0);
    kernel_gramschmidt(ni, nj, A, R, Q);
    clock_gettime(CLOCK_REALTIME, rt + 1);
    wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    printf("gramschmidt (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * ni * nj * nj / (1.0e9 * wt));
    */

    DeviceArr2D dA(ni, nj);
    DeviceArr2D dR(nj, nj);
    DeviceArr2D dQ(ni, nj);

    for(int i=0; i<A.y; i++)
        A[i][0] = 1;
    cudaMemcpy(dA.arr, A.arr, sizeof(DATA_TYPE) * A.x * A.y, cudaMemcpyHostToDevice);

    column_norm<<<1, 32>>>(dA, dR, 0);

    copy_to_q<<<floordiv(A.y, BLOCK_DIM), BLOCK_DIM>>>(dA, dQ, dR, 0);

    cudaMemcpy(Q.arr, dQ.arr, sizeof(DATA_TYPE) * Q.x * Q.y, cudaMemcpyDeviceToHost);
    for(int i=0; i<Q.y; i++)
        cout << Q[i][0] << endl;

    dA.free();
    dR.free();
    dQ.free();
    
    cout << "Freed memory" << endl;

    return 0;
}
