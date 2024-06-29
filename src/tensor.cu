#include "model.h"
#include <iostream>

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */

/* [TensorCPU Constructor] */
TensorCPU::TensorCPU(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  //cudaHostAlloc is used to allocate memory on host
  cudaHostAlloc(&buf, N_ * sizeof(float), cudaHostAllocDefault);
}



TensorCPU::TensorCPU(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  //cuda memory allocation
  cudaHostAlloc(&buf, N_ * sizeof(float), cudaHostAllocDefault);
  //memory copy from host to host
  memcpy(buf, buf_, N_ * sizeof(float));
}

TensorCPU::~TensorCPU() {
  //cuda memory free
  if(buf != nullptr) cudaFreeHost(buf);
}

size_t TensorCPU::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

//--------------------------------------------------------------------------------------------//
/* [TensorGPU Constructor] */
TensorGPU::TensorGPU(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  //buf is a address of pointer of GPU memory
  CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(float)));
}



TensorGPU::TensorGPU(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  //make stream
  CHECK_CUDA(cudaStreamCreate(&stream_param));
  //cuda memory allocation
  CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(float)));
  //memory copy from host to device
  CHECK_CUDA(cudaMemcpyAsync(buf, buf_, N_ * sizeof(float), cudaMemcpyHostToDevice, stream_param));
}

TensorGPU::~TensorGPU() {
  //cuda memory free
  CHECK_CUDA(cudaFree(buf));
  //stream_destroy
  //todo
}

size_t TensorGPU::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}