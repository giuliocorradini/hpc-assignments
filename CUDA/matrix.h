#pragma once

#include "gramschmidt.h"

class Arr2D
{
public:
    DATA_TYPE *arr = nullptr;
    const int x, y; //x is the number of columns, y the number of rows 

    Arr2D(int x, int y): x(x), y(y) {}

    virtual DATA_TYPE * operator[] (int i) {
        return arr + i * x;
    }

    virtual int size() {
        return sizeof(DATA_TYPE) * x * y;
    }

    virtual Arr2D & operator= (Arr2D &other) = 0;
    virtual bool isDevice() = 0;
};

class HostArr2D: public Arr2D
{
public:
    HostArr2D(int x, int y): Arr2D(x, y) {
        arr = new DATA_TYPE[x * y];
    }

    ~HostArr2D() {
        delete [] arr;
    }

    virtual bool isDevice() {
        return false;
    }

    virtual Arr2D & operator= (Arr2D &other) {
        if (other.size() > size())
            return *this;

        if (other.isDevice())
            cudaMemcpy(arr, other.arr, other.size(), cudaMemcpyDeviceToHost);
        else
            memcpy(arr, other.arr, other.size());
        
        return *this;
    }
};

class DeviceArr2D: public Arr2D
{
public:
    DeviceArr2D(int x, int y): Arr2D(x, y) {
        cudaMalloc(&arr, sizeof(DATA_TYPE) * x * y);
    }

    void free() {
        //  Not implemented in destructor because causes problems during program termination
        cudaFree(arr);
    }

    virtual bool isDevice() {
        return true;
    }

    virtual Arr2D & operator= (Arr2D &other) {
        if (other.size() > size())
            return *this;

        if (other.isDevice())
            cudaMemcpy(arr, other.arr, other.size(), cudaMemcpyDeviceToDevice);
        else
            cudaMemcpy(arr, other.arr, other.size(), cudaMemcpyHostToDevice);
        
        return *this;
    }
};
