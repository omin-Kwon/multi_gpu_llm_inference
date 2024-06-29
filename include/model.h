#pragma once

#include <cstdio>
#include <vector>

#include "tensor.h"

using std::vector;

static size_t tokens_per_prompt = 16;

/* Transformer model configuration */
#define NUM_VOCAB 50257
#define MAX_SEQ_LEN 1024
#define HIDDEN_DIM 768
#define NUM_HEAD 12
#define NUM_LAYER 12

/* Model parameter offsets */
#define OFFSET1 (3 * HIDDEN_DIM)               // 2304
#define OFFSET2 HIDDEN_DIM * (3 * HIDDEN_DIM)  // 768*2304
#define OFFSET3 HIDDEN_DIM                     // 768
#define OFFSET4 HIDDEN_DIM *HIDDEN_DIM         // 768*768
#define OFFSET5 4 * HIDDEN_DIM                 // 3072
#define OFFSET6 HIDDEN_DIM * (4 * HIDDEN_DIM)  // 768*3072
#define OFFSET7 MAX_SEQ_LEN *HIDDEN_DIM        // 1024*768
#define OFFSET8 NUM_VOCAB *HIDDEN_DIM          // 50257*768
#define NGPU 4 
#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)



void alloc_and_set_parameters(float *param);
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token);
void free_parameters();