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

void column_copy_unitary() {
    int TARGET_COL = 0;
    Arr2D A(1, 10);
    Arr2D Q(1, 10);
    Arr2D R(TARGET_COL+1, TARGET_COL+1);
    
    Arr2D expectedQ(1, 10);


    for (int y=0; y<A.y; y++) {
        A[y][TARGET_COL] = (y+1);
        expectedQ[y][TARGET_COL] = (y+1);
    }
    R[TARGET_COL][TARGET_COL] = 1;

    DeviceArr2D A_dev(1, 10);
    DeviceArr2D R_dev(1, 1);
    DeviceArr2D Q_dev(1, 10);

    cudaMemcpy(A_dev.arr, A.arr, A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(R_dev.arr, R.arr, R.size(), cudaMemcpyHostToDevice);

    copy_to_q<<<floordiv(A.y, BLOCK_DIM), BLOCK_DIM>>>(A_dev, R_dev, Q_dev, TARGET_COL);

    cudaMemcpy(Q.arr, Q_dev.arr, Q.size(), cudaMemcpyDeviceToHost);

    for (int y=0; y<Q.y; y++)
        assert(Q[y][TARGET_COL] == expectedQ[y][TARGET_COL]);
}

void column_copy_divisor() {
    int TARGET_COL = 0;
    Arr2D A(1, 10);
    Arr2D Q(1, 10);
    Arr2D R(TARGET_COL+1, TARGET_COL+1);
    
    Arr2D expectedQ(1, 10);


    for (int y=0; y<A.y; y++) {
        A[y][TARGET_COL] = (y+1);
        expectedQ[y][TARGET_COL] = (y+1) / 2.0;
    }
    R[TARGET_COL][TARGET_COL] = 2.0;

    DeviceArr2D A_dev(1, 10);
    DeviceArr2D R_dev(1, 1);
    DeviceArr2D Q_dev(1, 10);

    cudaMemcpy(A_dev.arr, A.arr, A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(R_dev.arr, R.arr, R.size(), cudaMemcpyHostToDevice);

    copy_to_q<<<floordiv(A.y, BLOCK_DIM), BLOCK_DIM>>>(A_dev, R_dev, Q_dev, TARGET_COL);

    cudaMemcpy(Q.arr, Q_dev.arr, Q.size(), cudaMemcpyDeviceToHost);

    for (int y=0; y<Q.y; y++)
        //cout << Q[y][TARGET_COL] << " " << expectedQ[y][TARGET_COL] << endl;
        assert(Q[y][TARGET_COL] == expectedQ[y][TARGET_COL]);
}

/**
 * Test normalization of a column
 */
int main() {
    cout << "Testing column copy to Q" << endl;

    column_copy_unitary();
    column_copy_divisor();

    cout << "All tests passed" << endl;
    return 0;
}