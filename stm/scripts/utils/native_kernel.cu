#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void scale_entropy(const float *input, float *output, const float scale, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

extern "C" void launch_scale_entropy(const float *input, float *output, float scale, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    scale_entropy<<<blocks, threads, 0, stream>>>(input, output, scale, n);
}

// Minimal sanity test when compiled standalone.
int main() {
    const int n = 4;
    float h_in[n] = {1.f, 2.f, 3.f, 4.f};
    float h_out[n] = {};
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    launch_scale_entropy(d_in, d_out, 0.5f, n, 0);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    std::printf("native-kernel-ok %.2f\\n", h_out[0]);
    return 0;
}
