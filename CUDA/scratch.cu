

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

