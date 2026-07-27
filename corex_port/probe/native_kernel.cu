// native_kernel.cu — confirm the ivcore11 GPU actually executes a kernel when
// compiled through the CoreX clang native path (clang -x cuda --cuda-gpu-arch=ivcore11).
// This isolates the blocker: the device computes fine; only NV *PTX text* ingestion
// (cuModuleLoadData) is rejected. CUDA.jl cannot use this path because its Julia->GPU
// compiler emits NV PTX, not ivcore11 device images.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add1(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    const int n = 256;
    float h[n];
    for (int i = 0; i < n; ++i) h[i] = (float)i;
    float *d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    add1<<<(n + 63) / 64, 64>>>(d, n);
    cudaError_t err = cudaDeviceSynchronize();
    cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("sync=%s h[0]=%.1f h[255]=%.1f (expect 1.0 / 256.0)\n",
           cudaGetErrorString(err), h[0], h[255]);
    cudaFree(d);
    return 0;
}
