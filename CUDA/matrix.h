#pragma once

#include "gramschmidt.h"

class Arr2D
{
public:
    DATA_TYPE *arr = nullptr;
    const int x, y; //x is the number of columns, y the number of rows 


    virtual DATA_TYPE * operator[] (int i) {
        return arr + i * x;
    }

    virtual int size() {
        return sizeof(DATA_TYPE) * x * y;
    }

    Arr2D(int x, int y): x(x), y(y) {
        arr = new DATA_TYPE[x * y];
    }

    ~Arr2D() {
        delete [] arr;
    }

};

class DeviceArr2D
{
public:
    int x, y;
    DATA_TYPE *arr;

    DeviceArr2D(int x, int y): x(x), y(y) {
        cudaMalloc(&arr, sizeof(DATA_TYPE) * x * y);
    }

    void free() {
        cudaFree(arr);
    }


    __device__ constexpr DATA_TYPE * operator[] (int i) {
        return arr + i * x;
    }
};
