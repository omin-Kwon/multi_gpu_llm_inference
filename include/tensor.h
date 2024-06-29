#pragma once

#include <vector>

using std::vector;

/* [Tensor Structure] for CPU */
struct TensorCPU {
  size_t ndim = 0;
  size_t shape[4];
  float *buf = nullptr;

  TensorCPU(const vector<size_t> &shape_);
  TensorCPU(const vector<size_t> &shape_, float *buf_);
  ~TensorCPU();

  size_t num_elem();
};


/* [Tensor Structure] for GPU*/
struct TensorGPU {
  size_t ndim = 0;
  size_t shape[4];
  float *buf = nullptr;
  cudaStream_t stream_param;

  TensorGPU(const vector<size_t> &shape_);
  TensorGPU(const vector<size_t> &shape_, float *buf_);
  ~TensorGPU();

  size_t num_elem();
};



typedef TensorCPU ParameterCPU;

typedef TensorGPU ParameterGPU;
typedef TensorGPU ActivationGPU;
typedef TensorGPU Parameter;
typedef TensorGPU Activation;
typedef TensorGPU Tensor;




