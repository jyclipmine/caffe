#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#include <string>

using namespace caffe; // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  Blob<float> blob(3000, 1, 1, 1);
  clock_t start, finish;
  // run loop
  while (true) {
    LOG(INFO) << "-------------------------------------------";
    start = clock();
    blob.mutable_cpu_data();
    finish = clock();
    LOG(INFO) << "GPU to CPU: " << 1000 * (finish - start) / CLOCKS_PER_SEC
        << " ms";
    
    start = clock();
    blob.mutable_gpu_data();
    finish = clock();
    LOG(INFO) << "CPU to GPU: " << 1000 * (finish - start) / CLOCKS_PER_SEC
        << " ms";
  }
  return 0;
}
