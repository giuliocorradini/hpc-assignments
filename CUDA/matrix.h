#pragma once

#include "gramschmidt.h"

struct Arr2D
{
    DATA_TYPE *arr;
    const int x, y; //x is the number of columns, y the number of rows 

public:
    Arr2D(int x, int y): x(x), y(y) {
        arr = new DATA_TYPE[x * y];
    }

    ~Arr2D() {
        delete [] arr;
    }

    DATA_TYPE * operator[] (int i) {
        return arr + i * x;
    }
};
