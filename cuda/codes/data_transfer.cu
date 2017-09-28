#include "common.h"

// test on cuda 8.0
// For large data, DtoH is a little faster then HtoD (~8%), but not much.
// Less mount of data makes more difference, HtoD faster

// Data size: 10485760 * sizeof(4) ~= 40M
// Time(%)      Time     Calls       Avg       Min       Max  Name
// 52.10%  1.4179ms         1  1.4179ms  1.4179ms  1.4179ms  [CUDA memcpy HtoD]
// 47.90%  1.3037ms         1  1.3037ms  1.3037ms  1.3037ms  [CUDA memcpy DtoH]

// Data size: 1024 * sizeof(4)
// Time(%)      Time     Calls       Avg       Min       Max  Name
// 70.06%  3.5200us         1  3.5200us  3.5200us  3.5200us  [CUDA memcpy DtoH]
// 29.94%  1.5040us         1  1.5040us  1.5040us  1.5040us  [CUDA memcpy HtoD]


int main() {
  const unsigned int N = 224*224*3;
  const unsigned int bytes = N * sizeof(int);

  int* h_a = (int*)malloc(bytes);
  int* d_a;
  cudaMalloc((int**)&d_a, bytes);

  memset(h_a, 0, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

  return 0;
}
