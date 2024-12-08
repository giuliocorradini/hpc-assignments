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

extern "C"
{
#include "utils.h"
}

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

#include <cassert>
#include "gramschmidt_kernel.cuh"

void gold_std(int ni, int nj, Arr2D &A, Arr2D &R, Arr2D &Q) {
    int k=0;

    // Consideriamo la colonna k-esima di A
    DATA_TYPE nrm = 0;

    //  Calcoliamo la norma di A^(k)
    for (int i = 0; i < ni; i++)
        nrm += A[i][k] * A[i][k];

    //  che viene salvata in nel k-esimo elemento diagonale di R
    R[k][k] = sqrt(nrm);

    // la k-esima colonna di Q è la normalizzazione della k-esima colonna di A
    // R[k][k] è una very busy expression
    for (int i = 0; i < ni; i++)
        Q[i][k] = A[i][k] / R[k][k];

    // Per ogni colonna successiva alla k-esima (definita nell'outer loop)
    for (int j = k + 1; j < nj; j++) {
        R[k][j] = 0;

        // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
        for (int i = 0; i < ni; i++)
            R[k][j] += Q[i][k] * A[i][j];
    }
}


__global__ void dot_product_a_q(DATA_TYPE *__restrict__ a, DATA_TYPE *__restrict__ r, DATA_TYPE *__restrict__ q, int ni, int nj, int k) {

    
    //porto in memlria 32 valori di a
    __shared__ DATA_TYPE s_q_col_k[BLOCK_DIM];

    //Porto in memoria condivisa le sezioni di interesse di a e q
    int a_row = blockDim.y* blockIdx.y + threadIdx.y;
    int a_col = blockDim.x* blockIdx.y + threadIdx.x;
    //coordinate del thread, per chiarezza
    int y_thread = threadIdx.y;
    int x_thread = threadIdx.x;
    if(a_row < ni){
        //DATO CHE A è letta solo una volta, non serve portarla in memoria condivisa
        s_q_col_k[threadIdx.y] = a[a_row + a_col] * q[a_row + k];
    }
    __syncthreads();

    //RIDUCZIONE AD IMBUTO
    if(a_row < ni){
        for(int i = 2; i <= blockIdx.y; i*=2){
            if(y_thread % i == 0){
                s_q_col_k[y_thread] += s_q_col_k[y_thread+(i/2)];
            }
           __syncthreads(); 
        }
        //alla fine l'ultimo del blocco thread aggiorna la memoria condivisa
        if(y_thread==0 && x_thread == 0){
            atomicAdd(&r[k*ni+a_col], s_q_col_k[y_thread]);
        }
    }
}


void recompute_first_row() {
    //Arrange
    int DIM = 3;

    Arr2D A(DIM, DIM);
    Arr2D R(DIM, DIM);
    Arr2D Q(DIM, DIM);
    int STOP_AT = min(1, A.x);
    
    Arr2D gsA(DIM, DIM);
    Arr2D gsR(DIM, DIM);
    Arr2D gsQ(DIM, DIM);

    for (int i = 0; i < A.y; i++)
    {
        for (int j = 0; j < A.x; j++)
        {
            A[j][i] = (i + 1) * (j + 1);
            cout << A[j][i] << " ";
            gsA[j][i] = (i + 1) * (j + 1);

            R[i][j] = -1;
        }
        cout << endl;
    }
    //Act
    DeviceArr2D dA(A.x, A.y);
    DeviceArr2D dR(R.x, R.y);
    DeviceArr2D dQ(Q.x, Q.y);

    cudaMemcpy(dA.arr, A.arr, sizeof(DATA_TYPE) * A.x * A.y, cudaMemcpyHostToDevice);
    
    for (int k=0; k<STOP_AT; k++) {
        column_norm<<<1, BLOCK_DIM>>>(dA, dR, k);
        copy_to_q<<<floordiv(A.y, BLOCK_DIM), BLOCK_DIM>>>(dA, dR, dQ, k);

        // Operations on A right edge
        dim3 block(BLOCK_DIM, BLOCK_DIM);
        //dim3 column_grid(floordiv(A.x-k, BLOCK_DIM), floordiv(A.y, BLOCK_DIM));
        dim3 column_grid(1, 1);

        int num_blocks = (DIM + BLOCK_DIM - 1) / BLOCK_DIM;
        dot_product_a_q<<<num_blocks, BLOCK_DIM>>>(dA.arr, dR.arr, dQ.arr, DIM, DIM, k);

        //update_a<<<column_grid, block>>>(dA, dR, dQ, k);
    }
    
    cudaMemcpy(A.arr, dA.arr, sizeof(DATA_TYPE) * A.x * A.y, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q.arr, dQ.arr, sizeof(DATA_TYPE) * Q.x * Q.y, cudaMemcpyDeviceToHost);
    cudaMemcpy(R.arr, dR.arr, sizeof(DATA_TYPE) * R.x * R.y, cudaMemcpyDeviceToHost);

    gold_std(DIM, DIM, gsA, gsR, gsQ);

    //Assert
    cout << R[0][1] << " " << gsR[0][1] << endl;
    //assert(R[0][1] == gsR[0][1]);
    cout << R[0][2] << " " << gsR[0][2] << endl;
    assert(R[0][2] == gsR[0][2]);

}
/**
 * Test recomputation of R row
 */
int main() {
    cout << "Testing recompute" << endl;

    recompute_first_row();

    cout << "All tests passed" << endl;
    return 0;
}