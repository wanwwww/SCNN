#include <stdio.h>
#include <cuda.h> // 或者 #include <cuda_runtime.h>

int main() {
    printf("CUDA Version: %d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    return 0;
}
