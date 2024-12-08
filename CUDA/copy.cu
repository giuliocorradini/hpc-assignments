#include <cassert>
#include <iostream>
using namespace std;

using DATA_TYPE = float;

__global__ void copy_from_device(DATA_TYPE *dst) {
    if (threadIdx.x == 0)
        *dst = 42;
}

void test_copy_from_device() {
    DATA_TYPE src;
    DATA_TYPE *dst;

    cudaMalloc(&dst, sizeof(DATA_TYPE));

    copy_from_device<<<1, 32>>>(dst);

    cudaMemcpy(&src, dst, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    assert(src == 42 && "dst not copied");
}

int main() {
    test_copy_from_device();

    return 0;
}
