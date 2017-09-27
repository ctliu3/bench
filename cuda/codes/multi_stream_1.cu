#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

const int N = 1 << 20;

// test in 980Ti, cuda 8.0

// nvcc -default-stream per-thread multi_stream_1.cu -o concurrency.exe
// GPU time: 27.289588

// nvcc multi_stream_1.cu -o no-concurrency.exe
// output: GPU time: 3.638738

__global__ void kernel(float *x, int n) {
  int tid = threadIdx.x + blockDim.x * gridDim.x;
  for (int i = tid; i < n; i+= blockDim.x * gridDim.x) { // why?
    for (int j = 0; j < 100; ++j) {
      x[i] = sqrt(pow(3.14159, i)) + j;
    }
  }
}

int main() {
  const int num_streams = 8;
  cudaStream_t streams[num_streams];
  float* data[num_streams];

  double start = seconds();
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&data[i], N * sizeof(float));
    kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
    kernel<<<1, 1>>>(0, 0); // use the defualt steram
  }
  // if you want to make the 8 streams run in concurrency, you should add
  // `-default-stream per-thread` in the compile option

  cudaDeviceSynchronize();
  printf("GPU time: %lf\n", seconds() - start);

  return 0;
}
