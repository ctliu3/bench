#include "common.h"
#include <cstdio>
#include <string>

// test on cuda 8.0
// test this before optimize your code

// ==129644== NVPROF is profiling process 129644, command: ./pinned_data.exe
// 
// Device: GeForce GTX 980 Ti
// Transfer size (MB): 40
// 
// Pageable transfers
//   Host to Device bandwidth (GB/s): 2.953402
//   Device to Host bandwidth (GB/s): 1.446348
// 
// Pinned transfers
//   Host to Device bandwidth (GB/s): 3.030097
//   Device to Host bandwidth (GB/s): 3.264647
// ==129644== Profiling application: ./pinned_data.exe
// ==129644== Profiling result:
// Time(%)      Time     Calls       Avg       Min       Max  Name
//  59.45%  40.688ms         2  20.344ms  12.818ms  27.870ms  [CUDA memcpy DtoH]
//  40.55%  27.752ms         2  13.876ms  13.799ms  13.954ms  [CUDA memcpy HtoD]
// 
// ==129644== API calls:
// Time(%)      Time     Calls       Avg       Min       Max  Name
//  82.22%  430.74ms         2  215.37ms  15.549ms  415.19ms  cudaMallocHost
//  13.52%  70.816ms         4  17.704ms  12.845ms  28.991ms  cudaMemcpy
//   3.33%  17.442ms         2  8.7208ms  8.6235ms  8.8181ms  cudaFreeHost
//   0.46%  2.4302ms       364  6.6760us     182ns  272.45us  cuDeviceGetAttribute
//   0.12%  639.93us         1  639.93us  639.93us  639.93us  cudaGetDeviceProperties
//   0.09%  489.13us         4  122.28us  4.9200us  466.51us  cudaEventSynchronize
//   0.07%  361.78us         4  90.445us  82.880us  106.72us  cuDeviceTotalMem
//   0.06%  333.84us         1  333.84us  333.84us  333.84us  cudaMalloc
//   0.05%  284.57us         1  284.57us  284.57us  284.57us  cudaFree
//   0.05%  237.17us         4  59.291us  56.406us  66.453us  cuDeviceGetName
//   0.01%  57.469us         8  7.1830us  3.9490us  11.889us  cudaEventRecord
//   0.00%  18.352us         4  4.5880us     909ns  9.0990us  cudaEventDestroy
//   0.00%  17.406us         4  4.3510us  1.1440us  9.3890us  cudaEventCreate
//   0.00%  11.592us         4  2.8980us  2.4350us  3.5330us  cudaEventElapsedTime
//   0.00%  3.8370us        12     319ns     184ns     682ns  cuDeviceGet
//   0.00%  2.4640us         3     821ns     294ns  1.6560us  cuDeviceGetCount

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
      cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileCopies(float *h_a,
                   float *h_b,
                   float *d,
                   unsigned int n,
                   const char *desc) {
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

int main() {
  unsigned int n = 10 * 1024 * 1024;
  const unsigned int bytes = n * sizeof(float);

  // host
  float* h_aPageable, *h_bPageable;
  float* h_aPinned, *h_bPinned;

  // device
  float* d_a;

  h_aPageable = (float*)malloc(bytes);
  h_bPageable = (float*)malloc(bytes);
  // Always check whether it is successful
  checkCuda(cudaMallocHost((void**)&h_aPinned, bytes));
  checkCuda(cudaMallocHost((void**)&h_bPinned, bytes));
  checkCuda(cudaMalloc((void**)&d_a, bytes));

  for (int i = 0; i < n; ++i) {
    h_aPageable[i] = i;
  }

  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_aPageable, 0, bytes);
  memset(h_aPinned, 0, bytes);

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  std::string s_pageable = "Pageable";
  std::string s_pinned = "Pinned";
  profileCopies(h_aPageable, h_bPageable, d_a, n, s_pageable.c_str());
  profileCopies(h_aPinned, h_bPinned, d_a, n, s_pinned.c_str());

  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}
