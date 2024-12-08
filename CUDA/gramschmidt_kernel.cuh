#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/**
 *  Computes the normalization of the k-st column of A using a reduction in shared memory.
 *  The kernel is launched once, with a single block of dimension BLOCK_DIM.
 *
 *  Returns the norm of A^(k) in R[k][k]
 */
__global__ void column_norm(DeviceArr2D A, DeviceArr2D R, int k) {
    __shared__ DATA_TYPE norm[BLOCK_DIM];
    norm[threadIdx.x] = 0;

    //IMPROVEMENT: bring A^(k) in shmem

    int SUBCOL_DIM = floordiv(A.y, blockDim.x);

    for (int ly=threadIdx.x; ly < threadIdx.x + SUBCOL_DIM; ly++) {
        if (ly < A.y)
            norm[threadIdx.x] += A[ly][k] * A[ly][k];
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


__global__ void copy_to_q(DeviceArr2D A, DeviceArr2D R, DeviceArr2D Q, int k) {
    //  Q^(k) <- normalized A^(k)
    int SUBCOL = blockIdx.x * blockDim.x;
    int tid = threadIdx.x + SUBCOL;

    if (tid < A.y)
       Q[tid][k] = A[tid][k] / R[k][k];
}

/**
 *  Update R (lower triangular matrix) by multiplying Q^(k) and A_{k+1, y}
 */
__global__ void recompute(DeviceArr2D A, DeviceArr2D R, DeviceArr2D Q, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + k;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Mi porto in shared memory la k-esima colonna di Q, che viene usata per ogni x 
    __shared__ DATA_TYPE qk[BLOCK_DIM];

    if (y < Q.y)
        qk[threadIdx.y] = Q[y][k];

    if (x < R.x and y < A.y) {
        if (blockIdx.y == 0 && threadIdx.y == 0)
            R[k][x] = 0;
    
        R[k][x] += A[y][x] * qk[threadIdx.y];

    }
}

__global__ void column_product_a_q(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, DATA_TYPE *__restrict__ q, int ni, int nj, int k) {
    //porto in memlria 32 valori di a
    __shared__ DATA_TYPE s_q_col_k[BLOCK_DIM];

    int thisCol = blockIdx.x + k;
    int thisRow = blockIdx.y * blockDim.y + threadIdx.y;

    //Porto in memoria condivisa le sezioni di interesse di a e q
    int a_row = blockDim.y* blockIdx.y + threadIdx.y;
    int a_col = k + blockIdx.x * blockDim.x;
    if(a_row < ni){
        //DATO CHE A è letta solo una volta, non serve portarla in memoria condivisa
        s_q_col_k[threadIdx.y] = a[a_row *ni + a_col] * q[a_row *ni + k];
    }
    __syncthreads();

    //Funnel Reduction
    if(a_row < ni){
        for (int b=blockDim.y / 2; b>0; b >>= 1) {  //  funnel pattern of reduction
            if (threadIdx.y < b)
                s_q_col_k[threadIdx.y] += s_q_col_k[threadIdx.y + b];
            __syncthreads();
        }

        if (threadIdx.y == 0)
            atomicAdd(&r[k*ni + a_col],s_q_col_k[0]);
    }
}


__global__ void update_a(DeviceArr2D A, DeviceArr2D R, DeviceArr2D Q, int k) {

    int x = blockIdx.x * blockDim.x + threadIdx.x + k;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ DATA_TYPE qk[BLOCK_DIM];    //< k-esima colonna di Q
    __shared__ DATA_TYPE r_k[BLOCK_DIM];   //< k-esima sottoriga di R per indici di colonna da k+1 a nj (A.y)

    if (x < R.x && threadIdx.y == 0)
        r_k[threadIdx.x] = R[k][x];

    if (y < Q.y && threadIdx.y == 0)
        qk[threadIdx.y] = Q[y][k];

    if (x < R.x and y < A.y)
        A[y][x] -= qk[threadIdx.y] * r_k[threadIdx.x];

}
