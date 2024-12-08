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

#include "host_kernel.h"
#include "gramschmidt_kernel.cuh"

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
 *  Host function for gramschmidt computation. Kernels are launched from host with VRAM resident data
 *  TODO: stream operations
 */
void cu_gramschmidt(Arr2D &A, Arr2D &R, Arr2D &Q) {
    DeviceArr2D dA(A.x, A.y);
    DeviceArr2D dR(R.x, R.y);
    DeviceArr2D dQ(Q.x, Q.y);

    cudaMemcpy(dA.arr, A.arr, sizeof(DATA_TYPE) * A.x * A.y, cudaMemcpyHostToDevice);
    
    for (int k=0; k<A.x; k++) {
        column_norm<<<1, 32>>>(dA, dR, 2);
        copy_to_q<<<floordiv(A.y, BLOCK_DIM), BLOCK_DIM>>>(dA, dR, dQ, k);

        dim3 dimBlock_a_q(1,BLOCK_DIM);             //sono colonne verticali
        dim3 dimGrid_a_q(A.x-k, (A.y + BLOCK_DIM - 1)/BLOCK_DIM);    //1 x #colonne
        column_product_a_q<<<dimGrid_a_q, dimBlock_a_q>>>(dA.arr, dR.arr, dQ.arr, A.x, A.y, k);

        dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 dimGrid(1, 1);
        update_a<<<dimGrid, dimBlock>>>(dA, dR, dQ, k);
    }
    
    cudaMemcpy(A.arr, dA.arr, sizeof(DATA_TYPE) * A.x * A.y, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q.arr, dQ.arr, sizeof(DATA_TYPE) * Q.x * Q.y, cudaMemcpyDeviceToHost);
    cudaMemcpy(R.arr, dR.arr, sizeof(DATA_TYPE) * R.x * R.y, cudaMemcpyDeviceToHost);
    
    dA.free();
    dR.free();
    dQ.free();
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;

    Arr2D A(ni, nj);
    Arr2D R(nj, nj);
    Arr2D Q(ni, nj);

    Arr2D Agpu(ni, nj);
    Arr2D Rgpu(nj, nj);
    Arr2D Qgpu(ni, nj);

    /* Initialize array(s). */
    init_array(ni, nj, A, R, Q);
    init_array(ni, nj, Agpu, Rgpu, Qgpu);

    struct {
        struct timespec start;
        struct timespec finish;
    } rt;
    double wt;

    #ifdef INSTRUMENT_HOST
    clock_gettime(CLOCK_REALTIME, &rt.start);
    kernel_gramschmidt(ni, nj, A, R, Q);
    clock_gettime(CLOCK_REALTIME, &rt.finish);
    wt = (rt.finish.tv_sec - rt.start.tv_sec) + 1.0e-9 * (rt.finish.tv_nsec - rt.start.tv_nsec);
    printf("gramschmidt (Host) : %9.3f sec\n", wt);
    #endif

    clock_gettime(CLOCK_REALTIME, &rt.start);
    cu_gramschmidt(Agpu, Rgpu, Qgpu);
    clock_gettime(CLOCK_REALTIME, &rt.finish);
    wt = (rt.finish.tv_sec - rt.start.tv_sec) + 1.0e-9 * (rt.finish.tv_nsec - rt.start.tv_nsec);
    printf("gramschmidt (Device) : %9.3f sec\n", wt);

    return 0;
}
