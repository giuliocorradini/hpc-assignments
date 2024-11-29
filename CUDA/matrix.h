#pragma once

#include "gramschmidt.h"

struct Arr2D
{
    DATA_TYPE *arr;
    const int n, m; //size of the tab in 2 dimensions

public:
    Arr2D(int n, int m): n(n), m(m) {
        arr = new DATA_TYPE[n * m];
    }

    ~Arr2D() {
        delete [] arr;
    }

    DATA_TYPE * operator[] (int i) {
        return arr + i * m;
    }
};
