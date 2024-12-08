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

void column_norm() {
    Arr2D A(1, 10);
    Arr2D R(1, 1);
    DATA_TYPE norm = 0;
    for (int y=0; y<A.y; y++) {
        A[y][0] = (y+1);
        norm += (y+1) * (y+1);
    }

    norm = sqrt(norm);

    DeviceArr2D A_dev(1, 10);
    DeviceArr2D R_dev(1, 1);

    cudaMemcpy(A_dev.arr, A.arr, A.size(), cudaMemcpyHostToDevice);

    column_norm<<<1, 32>>>(A_dev, R_dev, 0);
    cudaMemcpy(R.arr, R_dev.arr, R.size(), cudaMemcpyDeviceToHost);

    cout << "R: " << R[0][0] << " norm: " << norm << endl;

    assert(R[0][0] == norm);

    A_dev.free();
    R_dev.free();
}

void matrix_norm() {
    Arr2D A(10, 10);
    Arr2D R(10, 10);
    DATA_TYPE norm = 0;
    for (int y=0; y<A.y; y++) {
        A[y][2] = (y+1);
        norm += (y+1) * (y+1);
    }

    norm = sqrt(norm);

    DeviceArr2D A_dev(10, 10);
    DeviceArr2D R_dev(10, 10);

    cudaMemcpy(A_dev.arr, A.arr, A.size(), cudaMemcpyHostToDevice);

    column_norm<<<1, 32>>>(A_dev, R_dev, 2);
    cudaMemcpy(R.arr, R_dev.arr, R.size(), cudaMemcpyDeviceToHost);

    cout << "R: " << R[2][2] << " norm: " << norm << endl;

    assert(R[2][2] == norm);

    A_dev.free();
    R_dev.free();
}

/**
 * Test normalization of a column
 */
int main() {
    cout << "Testing norm" << endl;


    column_norm();
    matrix_norm();

    cout << "All tests passed" << endl;
    return 0;
}