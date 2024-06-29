#include "layer.h"
#include "model.h"
#include <iostream>


/* Token + Positional Embedding
 * @param [in1]  in: [s]
 * @param [in2] wte: [NUM_VOCAB, H]
 * @param [in3] wpe: [MAX_SEQ_LEN, H]
 * @param [out] out: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void token_pos_embedding(vector<int> in, Tensor *wte, Tensor *wpe,
                         Tensor *out) {
  size_t s = in.size();
  size_t H = wte->shape[1];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[i * H + j] = wte->buf[in[i] * H + j] + wpe->buf[i * H + j];
    }
  }
}


//kernel : token_pos_embedding_gpu(gpu_input[i], wte, wpe, embd_a[i]);
__global__ void token_pos_embedding_kernel(int *in_prompt, int* in_generated, float *wte, float *wpe, float *out, size_t input_seq_len, size_t output_seq_len){
  //blockIdx.x : batch index
  //blockDim.x : seq_len
  //threadIdx.x : token index
  size_t H = HIDDEN_DIM;
  size_t s = blockDim.x;
  int my_token;
  if(threadIdx.x >= input_seq_len){
    my_token = in_generated[blockIdx.x * output_seq_len + threadIdx.x - input_seq_len];
  }
  else{
    my_token = in_prompt[blockIdx.x * input_seq_len + threadIdx.x];
  }
  //out : [batch_size, input_seq_len, H]
  for(size_t i = 0; i < H; i++){
    out[blockIdx.x * s * H + threadIdx.x * H + i] = wte[my_token * H + i] + wpe[threadIdx.x * H + i];
  }
}

void token_pos_embedding_gpu(int* in_prompt, int* in_generated, Tensor *wte, Tensor *wpe, Tensor *out, size_t batch_size, size_t input_seq_len, size_t output_seq_len, size_t t,  cudaStream_t stream){
  //Tensor *in pointer resides in CPU memory and it points to GPU memory
  //set the grid and block size
  //invoke the kernel
  dim3 gridDim(batch_size, 1, 1);
  dim3 blockDim((input_seq_len+t), 1, 1);
  token_pos_embedding_kernel<<<gridDim, blockDim, 0, stream>>>(in_prompt, in_generated ,wte->buf, wpe->buf, out->buf, input_seq_len, output_seq_len);
}



/* GELU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void gelu(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    float x = inout->buf[i];
    inout->buf[i] =
        0.5 * x *
        (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
  }
}

__global__ void gelu_kernel(float *inout, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = inout[idx];
    inout[idx] = 0.5 * x * (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
  }
}


void gelu_gpu(Tensor *inout, size_t N, cudaStream_t stream) {
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory
  //set the grid and block size
  size_t N_per_block = 64;
  size_t num_blocks = (N + N_per_block - 1) / N_per_block;
  //invoke the kernel
  gelu_kernel<<<num_blocks, N_per_block, 0, stream>>>(inout->buf, N);
}




/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void softmax(Tensor *inout) {
  size_t s = inout->shape[0];
  size_t H = inout->shape[1];

  for (size_t i = 0; i < s; i++) {
    float max_val = inout->buf[i * H];
    for (size_t j = 0; j < H; j++) {
      if (inout->buf[i * H + j] > max_val) { max_val = inout->buf[i * H + j]; }
    }

    float sum = 0;
    for (size_t j = 0; j < H; j++) {
      inout->buf[i * H + j] = exp(inout->buf[i * H + j] - max_val);
      sum += inout->buf[i * H + j];
    }

    for (size_t j = 0; j < H; j++) { inout->buf[i * H + j] /= sum; }
  }
}


__global__ void softmax_kernel(float *inout, size_t group_size, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_mem[256*24];
    if (idx < group_size * M) {
        float* row = inout + idx * N;

        //Step1 : load the row into shared memory
        for(int i = 0; i < N; i++){
            shared_mem[threadIdx.x * N + i] = row[i];
        }
        __syncthreads();
        //Step2 : Find the maximum value in the row
        float max_val = -INFINITY;
        for(int i = 0; i < N; i++){
            if(shared_mem[threadIdx.x * N + i] > max_val){
                max_val = shared_mem[threadIdx.x * N + i];
            }
        }
        //Step3 : Calculate the exponentials and sum them
        float sum = 0.0f;
        for(int i = 0; i < N; i++){
            shared_mem[threadIdx.x * N + i] = expf(shared_mem[threadIdx.x * N + i] - max_val);
            sum += shared_mem[threadIdx.x * N + i];
        }
        //Step4 : Normalize the values
        for(int i = 0; i < N; i++){
            shared_mem[threadIdx.x * N + i] /= sum;
        }
        //Step5 : store the result back to global memory
        for(int i = 0; i < N; i++){
            row[i] = shared_mem[threadIdx.x * N + i];
        }
    }
}



void softmax_gpu(Tensor *inout, size_t group_size, size_t M, size_t N, cudaStream_t stream) {
    // Tensor *inout pointer resides in CPU memory and it points to GPU memory
    // Set the grid and block size
    //(num_head*batch_size, seq_len, seq_len);
    size_t block_size = 256;
    size_t grid_size = (group_size * M + block_size - 1) / block_size;

    // Calculate the required shared memory size
    //size_t shared_memory_size = block_size * sizeof(float);

    // Invoke the kernel
    softmax_kernel<<<grid_size, block_size, 0, stream>>>(inout->buf, group_size, M, N);
}



/* Layer Normalization
 * @param [in1 & out] inout: [s, H]
 * @param [in2]       gamma: [H]
 * @param [in3]        beta: [H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta) {
  size_t s = inout->shape[0];
  size_t H = inout->shape[1];

  float eps = 1e-5;
  for (size_t i = 0; i < s; i++) {
    float mean = 0;
    float var = 0;

    for (size_t j = 0; j < H; j++) {
      mean += inout->buf[i * H + j];
      var += inout->buf[i * H + j] * inout->buf[i * H + j];
    }

    mean /= H;
    var = var / H - mean * mean;

    for (size_t j = 0; j < H; j++) {
      inout->buf[i * H + j] = (inout->buf[i * H + j] - mean) *
                                  (1.0 / sqrt(var + eps)) * gamma->buf[j] +
                              beta->buf[j];
    }
  }
}

__global__ void layer_norm_kernel(float *inout, float *gamma, float *beta, size_t N, size_t H) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float mean = 0;
    float var = 0;
    float temp = 0; 
    for (size_t i = 0; i < H; i++) {
      temp = inout[idx*H + i];
      mean += temp;
      var += temp*temp;
    }
    mean /= H;
    var = var / H - mean * mean;

    for (size_t i = 0; i < H; i++) {
      inout[idx*H + i] = (inout[idx*H + i] - mean) * (1.0 / sqrt(var + 1e-5)) * gamma[i] + beta[i];
    }
  }
}




void layer_norm_gpu(Tensor *inout, Tensor *gamma, Tensor *beta, size_t batch_size, size_t seq_len, cudaStream_t stream) {
  //it should be same process as layer_norm in cpu
  //but it has to be parallelized with cuda kernel
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory
  //set the grid and block size
  size_t H = inout->shape[2];
  size_t N = batch_size * seq_len; //batch 128, seq_len : (8-16)
  size_t num_blocks = (N + 63) / 64;
  //invoke the kernel
  layer_norm_kernel<<<num_blocks, 64, 0, stream>>>(inout->buf, gamma->buf, beta->buf, N, H);
}




/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [K, N]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = in->shape[0];
  size_t K = in->shape[1];
  size_t N = w->shape[1];

#pragma omp parallel for
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out->buf[i * N + j] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[i * N + j] += in->buf[i * K + k] * w->buf[k * N + j];
      }
      out->buf[i * N + j] += b->buf[j];
    }
  }
}


// __global__ void linear_kernel(float *in, float *w, float *b, float *out, size_t M, size_t K, size_t N) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < M){
//     for (size_t j = 0; j < N; j++) {
//       out[idx * N + j] = 0;
//       for (size_t k = 0; k < K; k++) {
//         out[idx * N + j] += in[idx * K + k] * w[k * N + j];
//       }
//       out[idx * N + j] += b[j];
//     }
//   }
// }

// /* Linear GPU
//   * @param [in1]  in: [b, s, H]
//   * @param [in2]   w: [H, H_]
//   * @param [in3]   b: [H_]
//   * @param [out] out: [b, s, H_]
// */
// void linear_gpu(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, size_t seq_len ,cudaStream_t stream) {
//   //Tensor *in pointer resides in CPU memory and it points to GPU memory
//   size_t M = batch_size * seq_len; //in->shpae[1] is input_seq_len + output_seq_len = total_seq_len
//   size_t K = in->shape[2];
//   size_t N = w->shape[1];
//   //set the grid and block size
//   //Each SM cacluates the 2D tile of output matrix
//   size_t N_per_block = 64;
//   size_t num_blocks = (M + N_per_block - 1) / N_per_block;
//   //kernel invocation
//   linear_kernel<<<num_blocks, N_per_block, 0, stream>>>(in->buf, w->buf, b->buf, out->buf, M, K, N);
// }


__global__ void linear_kernel(float *in, float *w, float *b, float *out, size_t M, size_t K, size_t N) {
    int tile_size = 16;
    int el = 4;
    int globalCol = (blockDim.x * blockIdx.x + threadIdx.x) * el;
    int globalRow = (blockDim.y * blockIdx.y + threadIdx.y) * el;
    int localCol = threadIdx.x;
    int localRow = threadIdx.y;
    int loadRow = globalRow - el * localRow;
    int loadCol = globalCol - el * localCol;
    int r = (localRow / el) * tile_size;
    int c = (localRow % el) * tile_size + localCol;


    __shared__ float localA[64][64];
    __shared__ float localB[64][64];

    float privateA[4][4];
    float privateB[4][4];
    float sum[4][4] = {0.0f};

    for (int k = 0; k < K; k += 64) {
        for(int i =0; i < 16; i++){
          localA[r +i][c] = in[(loadRow + r + i)*K + (k+c)];
          localB[r + i][c] = w[(k + r + i )*N+ (loadCol+c)];
        }
        __syncthreads();

        for (int l = 0; l < 64; l += 4) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    privateA[i][j] = localA[localRow * 4 + i][l + j];
                    privateB[i][j] = localB[l + i][localCol * 4 + j];
                }
            }
            for (int i = 0; i < 4; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 4; j++) {
                        sum[i][j] += privateA[i][m] * privateB[m][j];
                    }
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (globalRow + i < M && globalCol + j < N) {
                out[(globalRow + i) * N + (globalCol + j)] = sum[i][j] + b[globalCol + j];
            }
        }
    }
}

void linear_gpu(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, size_t seq_len ,cudaStream_t stream) {
    //in : [b, s, H]
    //out : [b, s, 3*H]
    
    size_t M = batch_size * seq_len; // 512*(16-23)
    size_t K = in->shape[2]; //768 <- 64 * 12
    size_t N = w->shape[1]; //768*3 <- 64*12*3

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 63) / 64, (M + 63) / 64);

    linear_kernel<<<gridDim, blockDim, 0, stream>>>(in->buf, w->buf, b->buf, out->buf, M, K, N);
}



__global__ void linear_split_qkv_head_fusion_kernel(float *in, float *w, float *b, float *q_out, float *k_out, float *v_out, size_t M, size_t K, size_t N, size_t n_head, size_t batch_size, size_t seq_len){
    
    int tile_size = 16;
    int el = 4;
    int globalCol = (blockDim.x * blockIdx.x + threadIdx.x) * el;
    int globalRow = (blockDim.y * blockIdx.y + threadIdx.y) * el;
    int localCol = threadIdx.x;
    int localRow = threadIdx.y;
    int loadRow = globalRow - el * localRow;
    int loadCol = globalCol - el * localCol;
    int r = (localRow / el) * tile_size;
    int c = (localRow % el) * tile_size + localCol;


    __shared__ float localA[64][64];
    __shared__ float localB[64][64];

    float privateA[4][4];
    float privateB[4][4];
    float sum[4][4] = {0.0f};

    for (int k = 0; k < K; k += 64) {
        for(int i =0; i < 16; i++){
          localA[r +i][c] = in[(loadRow + r + i)*K + (k+c)];
          localB[r + i][c] = w[(k + r + i )*N+ (loadCol+c)];
        }
        __syncthreads();

        for (int l = 0; l < 64; l += 4) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    privateA[i][j] = localA[localRow * 4 + i][l + j];
                    privateB[i][j] = localB[l + i][localCol * 4 + j];
                }
            }
            for (int i = 0; i < 4; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 4; j++) {
                        sum[i][j] += privateA[i][m] * privateB[m][j];
                    }
                }
            }
        }

        __syncthreads();
    }


    int idx = (globalCol)/64;

    int q_k_v = idx / 12;
    int head_idx = idx % 12;
    int e_globalCol = globalCol % 64;

    if(q_k_v == 0){
      for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
          q_out[head_idx * M * (64) + (globalRow + i) * (64) + (e_globalCol + j)] = sum[i][j] + b[globalCol + j];
        }
      }
    }
    else if(q_k_v == 1){
      for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
          k_out[head_idx * M * (64) + (globalRow + i) * (64) + (e_globalCol + j)] = sum[i][j] + b[globalCol + j];
        }
      }

    }
    else{
      for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
          v_out[head_idx * M * (64) + (globalRow + i) * (64) + (e_globalCol + j)] = sum[i][j] + b[globalCol + j];
        }
      }
    }
}



void linear_split_qkv_head_fusion(Tensor *in, Tensor *w, Tensor *b, Tensor *q_out, Tensor *k_out, Tensor *v_out, size_t n_head, size_t batch_size, size_t seq_len ,cudaStream_t stream){
    //in : [b, s, H]
    //out : q: [n_h, b_s, H/n_h], k: [n_h, b_s, H/n_h], v: [n_h, b_s, H/n_h]
    
    size_t M = batch_size * seq_len; // 512*(16-23)
    size_t K = in->shape[2]; //768 <- 64 * 12
    size_t N = w->shape[1]; //768*3 <- 64*12*3

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 63) / 64, (M + 63) / 64);
    linear_split_qkv_head_fusion_kernel<<<gridDim, blockDim, 0, stream>>>(in->buf, w->buf, b->buf, q_out->buf, k_out->buf, v_out->buf, M, K, N, n_head, batch_size, seq_len);
}



/* Matmul
 * @param [in1]  in1: [M, K]
 * @param [in2]  in2: [K, N]
 * @param [out]  out: [M, N]
 */
void matmul(Tensor *in1, Tensor *in2, Tensor *out) {
  size_t M = in1->shape[0];
  size_t K = in1->shape[1];
  size_t N = in2->shape[1];

#pragma omp parallel for
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out->buf[i * N + j] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[i * N + j] += in1->buf[i * K + k] * in2->buf[k * N + j];
      }
    }
  }
}


// //Optimized Version
// __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
//     int el = 4;

//     int globalCol = (blockDim.x * blockIdx.x + threadIdx.x) * el;
//     int globalRow = (blockDim.y * blockIdx.y + threadIdx.y) * el;

//     int localCol = threadIdx.x;
//     int localRow = threadIdx.y;

//     int loadRow = globalRow - 4 * localRow;
//     int loadCol = globalCol - 4 * localCol;

//     int r = (localRow / 4) * 16;
//     int c = (localRow % 4) * 16 + localCol;

//     if (globalCol >= N || globalRow >= M) return;

//     __shared__ float localA[64][64];
//     __shared__ float localB[64][64];

//     float privateA[4][4];
//     float privateB[4][4];
//     float sum[4][4] = {0.0f};

//     // K 차원을 따라 반복. 타일 크기는 64
//     for (int k = 0; k < K; k += 64) {
//         // 공유 메모리에 로드
//         for (int i = 0; i < 16; i++) {
//             localA[r + i][c] = A[(loadRow + r + i) * K + (k + c)];
//             localB[r + i][c] = B[(k + r + i) * N + (loadCol + c)];
//         }

//         __syncthreads();

//         // K 차원 내부에서 16번 반복
//         for (int l = 0; l < 64; l += 4) {
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 4; j++) {
//                     privateA[i][j] = localA[(localRow * 4 + i)][(l + j)];
//                     privateB[i][j] = localB[(l + i)][(localCol * 4 + j)];
//                 }
//             }

//             for (int i = 0; i < 4; i++) {
//                 for (int m = 0; m < 4; m++) {
//                     for (int j = 0; j < 4; j++) {
//                         sum[i][j] += privateA[i][m] * privateB[m][j];
//                     }
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             if ((globalRow + i) < M && (globalCol + j) < N) {
//                 C[(globalRow + i) * N + (globalCol + j)] = sum[i][j];
//             }
//         }
//     }
// }

// void matmul_gpu(Tensor *in1, Tensor *in2, Tensor *out, size_t group_size, size_t M, size_t K, size_t N, cudaStream_t stream) {
//     dim3 blockDim(16, 16);
//     dim3 gridDim((N + 63) / 64, (M + 63) / 64, group_size);

//     matmul_kernel<<<gridDim, blockDim, 0, stream>>>(in1->buf, in2->buf, out->buf, M, N, K);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//     }
// }

//------------------
//Minimal Version

__global__ void matmul_kernel(float *in1, float *in2, float *out, size_t M, size_t K, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = blockIdx.z;

    if (row < M && col < N) {
        float value = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            value += in1[idx * M * K + row * K + k] * in2[idx * K * N + k * N + col];
        }
        out[idx * M * N + row * N + col] = value;
    }
}

void matmul_gpu(Tensor *in1, Tensor *in2, Tensor *out, size_t group_size, size_t M, size_t K, size_t N, cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y, group_size);

    matmul_kernel<<<grid_size, block_size, 0, stream>>>(in1->buf, in2->buf, out->buf, M, K, N);

    // CUDA 오류 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}


// ////Minimul
// __global__ void final_matmul_kernel(float* in1, float* in2, float* out, size_t M, size_t K, size_t N) {
//   size_t globalCol = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t globalRow = blockIdx.y * blockDim.y + threadIdx.y;
//   if(globalRow < M && globalCol < N){
//     float sum = 0;
//     for(size_t i = 0; i < K; i++){
//       sum += in1[globalRow * K + i] * in2[i * N + globalCol];
//     }
//     out[globalRow * N + globalCol] = sum;
//   }
// }



// void final_matmul(Tensor *in1, Tensor *in2, Tensor *out, size_t M, size_t K, size_t N, cudaStream_t stream) {
//   //Tensor *in1 pointer resides in CPU memory and it points to GPU memory
//   //allocate device memory
//   //in1 : [batch_size * seq_len, hidden_dim], in2 : [hidden_dim, NUM_VOCAB]
//   //out : [batch_size * seq_len, NUM_VOCAB]
//   dim3 gridDim((N+15)/16, (M+15)/16);
//   dim3 blockDim(16, 16);
//   //invoke the kernel
//   final_matmul_kernel<<<gridDim, blockDim, 0, stream>>>(in1->buf, in2->buf, out->buf, M, K, N);
// }

//Optim1
static __global__ void optimized_matmul_kernel(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    int el = 4;

    int globalCol = (blockDim.x * blockIdx.x + threadIdx.x) * el;
    int globalRow = (blockDim.y * blockIdx.y + threadIdx.y) * el;

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    int loadRow = globalRow - 4 * localRow;
    int loadCol = globalCol - 4 * localCol;

    int r = (localRow / 4) * 16;
    int c = (localRow % 4) * 16 + localCol;

    // if (globalCol >= N || globalRow >= M) return;

    __shared__ float localA[64][64];
    __shared__ float localB[64][64];

    float privateA[4][4];
    float privateB[4][4];
    float sum[4][4] = {0.0f};

    // K 차원을 따라 반복합니다. 타일 크기는 64입니다.
    for (int k = 0; k < K; k += 64) {
        // 공유 메모리에 로드합니다.
        for (int i = 0; i < 16; i++) {
            if ((loadRow + r + i) < M && (k + c) < K) {
                localA[r + i][c] = A[(loadRow + r + i) * K + (k + c)];
            } else {
                localA[r + i][c] = 0.0f;
            }

            if ((k + r + i) < K && (loadCol + c) < N) {
                localB[r + i][c] = B[(k + r + i) * N + (loadCol + c)];
            } else {
                localB[r + i][c] = 0.0f;
            }
        }

        // 스레드 동기화
        __syncthreads();

        // K 차원 내에서 16번 반복합니다.
        for (int l = 0; l < 64; l += 4) {
            // 작업 항목 8x8로 개인 메모리에 로드합니다.
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    privateA[i][j] = localA[(localRow * 4 + i)][(l + j)];
                    privateB[i][j] = localB[(l + i)][(localCol * 4 + j)];
                }
            }

            // privateA x privateB를 계산하고 sum에 저장합니다.
            for (int i = 0; i < 4; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 4; j++) {
                        sum[i][j] += privateA[i][m] * privateB[m][j];
                    }
                }
            }
        }

        // 스레드 동기화
        __syncthreads();
    }

    // 결과를 C에 저장합니다.
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if ((globalRow + i) < M && (globalCol + j) < N) {
                C[(globalRow + i) * N + (globalCol + j)] = sum[i][j];
            }
        }
    }
}

void final_matmul(Tensor *in1, Tensor *in2, Tensor *out, size_t M, size_t K, size_t N, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 63) / 64, (M + 63) / 64);

    optimized_matmul_kernel<<<gridDim, blockDim, 0, stream>>>(in1->buf, in2->buf, out->buf, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}





/* Transpose
 * @param [in1]  in: [M, N]
 * @param [out] out: [N, M]
 */
void transpose(Tensor *in, Tensor *out) {
  size_t M = in->shape[0];
  size_t N = in->shape[1];

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) { out->buf[j * M + i] = in->buf[i * N + j]; }
  }
}

// __global__ void transpose_kernel(float *in, float *out, size_t group_size, size_t M, size_t N) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if(idx < group_size){
//     for (size_t i = 0; i < M; i++) {
//       for (size_t j = 0; j < N; j++) { out[idx * N * M + j * M + i] = in[idx * M * N + i * N + j]; }
//     }
//   }
// }



// void transpose_gpu(Tensor *in, Tensor *out, size_t group_size ,size_t M, size_t N, cudaStream_t stream) {
//   // in : [group_size, M, N]->out : [group_size, N, M]
//   int grid_size = group_size/32;
//   int block_size = 32;
//   transpose_kernel<<<grid_size, block_size, 0, stream>>>(in->buf, out->buf, group_size, M, N);
// }


__global__ void transposeKernel(float* in, float* out, size_t group_size, size_t M, size_t N) {
    // 그룹 인덱스를 계산
    size_t group = blockIdx.z;

    // 그룹 내 행과 열 인덱스를 계산
    size_t x = blockIdx.x * blockDim.x + threadIdx.x; // 행 인덱스
    size_t y = blockIdx.y * blockDim.y + threadIdx.y; // 열 인덱스

    // 그룹의 시작 포인터 계산
    float* group_in = in + group * M * N;
    float* group_out = out + group * M * N;

    // 범위 내의 인덱스인지 확인
    if (x < N && y < M) {
        group_out[x * M + y] = group_in[y * N + x];
    }
}

void transpose_gpu(Tensor *in, Tensor *out, size_t group_size ,size_t M, size_t N, cudaStream_t stream){
    // 블록과 그리드 크기 정의
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y, group_size);

    // 병렬로 커널 실행
    transposeKernel<<<gridSize, blockSize, 0, stream>>>(in->buf, out->buf, group_size, M, N);
}




//Minimal Version
__global__ void transpose_no_group_kernel(float *in, float *out, size_t M, size_t N) {
    size_t globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    size_t globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalRow < M && globalCol < N) {
        out[globalCol * M + globalRow] = in[globalRow * N + globalCol];
    }
}


void transpose_no_group_gpu(Tensor *in, Tensor *out, size_t M, size_t N, cudaStream_t stream) {
    // in : [M, N] -> out : [N, M]
    dim3 block_size(32, 32);  // Each block transposes a 32x32 tile
    dim3 grid_size((N + 31) / 32, (M + 31) / 32);  // Each block transposes a part of the matrix

    transpose_no_group_kernel<<<grid_size, block_size, 0, stream>>>(in->buf, out->buf, M, N);
}


/* Scaling
 * @param [in1 & out] inout: [N]
 * @param [in2]       scale: [1]
 * 'N' is the number of elements in the tensor.
 */
void scaling(Tensor *inout, float scale) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) { inout->buf[i] *= scale; }
}

//scaling kernel
__global__ void scaling_kernel(float *inout, float scale, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { inout[idx] *= scale; }
}


void scaling_gpu(Tensor *inout, float scale, cudaStream_t stream) {
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory
  //kernel invocation
  size_t N = inout->num_elem();
  scaling_kernel<<<(N + 255) / 256, 256, 0, stream>>>(inout->buf, scale, N);
}



/* Generate mask
 * @param [in & out] inout: [s, s]
 * 's' is the number of tokens in the prompt.
 */
void generate_mask(Tensor *inout) {
  size_t s = inout->shape[0];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < s; j++) {
      if (i >= j) {
        inout->buf[i * s + j] = 0;
      } else {
        inout->buf[i * s + j] = -1e10;
      }
    }
  }
}

//generate_mask_kernel
__global__ void generate_mask_kernel(float *inout, size_t s) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < s * s) {
    size_t i = idx / s;
    size_t j = idx % s;
    if (i >= j) {
      inout[idx] = 0;
    } else {
      inout[idx] = -1e10;
    }
  }
}


void generate_mask_gpu(Tensor *inout, cudaStream_t stream) {
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory.
  //inout->buf resides in GPU memory
  //kernel invocation
  size_t seq_len = inout->shape[0];
  //max seq_len is 8 + 16 = 24
  //parallelize with 32 threads
  size_t n_blocks = (seq_len * seq_len + 31) / 32;
  generate_mask_kernel<<<n_blocks, 32, 0, stream>>>(inout->buf, seq_len);
}



/* Copy
 * @param [in1]  in: [N]
 * @param [out] out: [N]
 * 'N' is the number of elements in the tensor.
 */
void copy(Tensor *in, Tensor *out) {
  size_t N = in->num_elem();

  for (size_t i = 0; i < N; i++) { out->buf[i] = in->buf[i]; }
}

void copy_gpu(Tensor *in, Tensor *out, cudaStream_t stream) {
  //Tensor *in pointer resides in CPU memory and it points to GPU memory
  size_t N = in->num_elem();
  //std::cout<<"N: "<<N<<std::endl;
  cudaMemcpyAsync(out->buf, in->buf, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

/* Add
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
void add(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) { inout->buf[i] += x->buf[i]; }
}

__global__ void add_kernel(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { inout[idx] += x[idx]; }
}


void add_gpu(Tensor *inout, Tensor *x, size_t group_size, size_t seq_len, size_t hidden_dim ,cudaStream_t stream) {
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory
  //cuda memcopy device to device
  //inout : [group_size, seq_len, seq_len], x : [group_size, seq_len, seq_len]
  size_t N = group_size * seq_len * hidden_dim;
  add_kernel<<<(N + 255) / 256, 256, 0, stream>>>(inout->buf, x->buf, N);
}


__global__ void add_copy_kernel(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { 
    int temp = inout[idx];
    int temp2 = x[idx];
    inout[idx] = temp + temp2;
    x[idx] = temp + temp2;
  }
}

void add_copy_gpu(Tensor *inout, Tensor *x, size_t group_size, size_t seq_len, size_t hidden_dim ,cudaStream_t stream) {
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory
  //cuda memcopy device to device
  size_t N = group_size * seq_len * hidden_dim;
  add_copy_kernel<<<(N + 255) / 256, 256, 0, stream>>>(inout->buf, x->buf, N);
}




__global__ void add_mask_kernel(float *inout, float *x, size_t seq_len, size_t total_seq_len, size_t group_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<group_size){
    for(int i = 0; i < seq_len; i++){
      for(int j = 0; j < seq_len; j++){
        inout[idx*seq_len*seq_len + i*seq_len + j] += x[i*total_seq_len + j];
      }
    }
  }
}


void add_mask(Tensor *inout, Tensor *x, size_t group_size, size_t seq_len, size_t total_seq_len, cudaStream_t stream) {
  //Tensor *inout pointer resides in CPU memory and it points to GPU memory
  //cuda memcopy device to device
  //inout : [group_size, seq_len, seq_len], x : [total_seq_len, total_seq_len]
  add_mask_kernel<<<group_size/32, 32, 0, stream>>>(inout->buf, x->buf, seq_len, total_seq_len, group_size);
}



/* Add using CUDA GPU
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
void add_cuda(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  float *d_inout;
  float *d_x;

  cudaMalloc(&d_inout, N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  add_kernel<<<(N + 255) / 256, 256>>>(d_inout, d_x, N);

  cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost);
}

/* Split into QKV
 * @param [in1]  in: [s, H]
 * @param [out] out: [3, s, H/3]
 */
void split_qkv(Tensor *in, Tensor *out) {
  size_t s = in->shape[0];
  size_t H = in->shape[1];

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < s; j++) {
      for (size_t k = 0; k < H / 3; k++) {
        out->buf[i * s * (H / 3) + j * (H / 3) + k] =
            in->buf[i * (H / 3) + j * 3 * (H / 3) + k];
      }
    }
  }
}



// void split_qkv_head_gpu(Tensor *in, Tensor *q_out, Tensor *k_out, Tensor *v_out, size_t n_head, size_t batched_seq, size_t H, cudaStream_t stream) {
//   //Tensor *in pointer resides in CPU memory and it points to GPU memory
//   //cuda memcopy device to device
//   //in : [b, s, 3*H] -> q_out : [n_head, b, s, H/n_head], k_out : [n_head, b, s, H/n_head], v_out : [n_head, b, s, H/n_head]
//   size_t H_ = H/n_head;
//   size_t t_s = batched_seq;
//   for(int i = 0; i < n_head; i++){
//     for(int j = 0; j < t_s; j++){
//       cudaMemcpyAsync(q_out->buf + i*t_s*H_ + j*H_, in->buf + j*3*H + 0 + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//       cudaMemcpyAsync(k_out->buf + i*t_s*H_ + j*H_, in->buf + j*3*H + H + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//       cudaMemcpyAsync(v_out->buf + i*t_s*H_ + j*H_, in->buf + j*3*H + 2*H + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//     }
//   }
// }

// CUDA 커널 함수
__global__ void split_qkv_kernel(const float* __restrict__ in, float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out, size_t n_head, size_t batched_seq, size_t H, size_t H_) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = n_head * batched_seq * H_;

    if (idx < total_elements) {
        size_t head = idx / (batched_seq * H_);
        size_t remainder = idx - head * (batched_seq * H_);
        size_t seq = remainder / H_;
        size_t offset = remainder - seq * H_;

        size_t in_index_base = seq * 3 * H + head * H_ + offset;
        size_t out_index = head * batched_seq * H_ + seq * H_ + offset;

        q_out[out_index] = in[in_index_base];
        k_out[out_index] = in[in_index_base + H];
        v_out[out_index] = in[in_index_base + 2 * H];
    }
}

// 메인 함수
void split_qkv_head_gpu(Tensor *in, Tensor *q_out, Tensor *k_out, Tensor *v_out, size_t n_head, size_t batched_seq, size_t H, cudaStream_t stream) {
    size_t H_ = H / n_head;
    size_t total_elements = n_head * batched_seq * H_;

    // 블록과 그리드 크기 설정
    size_t threads_per_block = 256;
    size_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // CUDA 커널 호출
    split_qkv_kernel<<<num_blocks, threads_per_block, 0, stream>>>(in->buf, q_out->buf, k_out->buf, v_out->buf, n_head, batched_seq, H, H_);

    // 커널 실행 오류 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}




/* Split into heads
 * @param [in1]  in: [3, s, H]
 * @param [out] out: [3, n_head, s, H/n_head]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 * 'n_head' is the number of heads.
 */
void split_head(Tensor *in, size_t n_head, Tensor *out) {
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < n_head; j++) {
      for (size_t k = 0; k < s; k++) {
        for (size_t l = 0; l < H / n_head; l++) {
          out->buf[i * n_head * s * H / n_head + j * s * H / n_head +
                   k * H / n_head + l] =
              in->buf[i * s * H + k * H + j * H / n_head + l];
        }
      }
    }
  }
}

// void split_head_gpu(Tensor *in, size_t n_head, Tensor *out, size_t batched_seq, size_t H, cudaStream_t stream) {
//   //Tensor *in pointer resides in CPU memory and it points to GPU memory
//   //cuda memcopy device to device
//   // in : [3, b, s, H] -> out : [3, n_head, b, s, H/n_head]
//   size_t H_ = H/n_head;
//   size_t t_s = batched_seq;
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < n_head; j++){
//       for(int k = 0; k < t_s; k++){
//         cudaMemcpyAsync(out->buf + i*n_head*t_s*H_ + j*t_s*H_ + k*H_, in->buf + i*t_s*H + k*H + j*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//       }
//     }
//   }
// }




/* Extract Q, K, V from QKV head
 * @param [in1]       in: [3, n_head, s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]        q: [s, H_]
 * @param [out]        k: [s, H_]
 * @param [out]        v: [s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q,
                 Tensor *k, Tensor *v) {
  size_t s = in->shape[2];
  size_t H_ = in->shape[3];  // = HIDDEN_DIM/NUM_HEAD

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H_; j++) {
      q->buf[i * H_ + j] =
          in->buf[0 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
      k->buf[i * H_ + j] =
          in->buf[1 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
      v->buf[i * H_ + j] =
          in->buf[2 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
    }
  }
}

void extract_qkv_gpu(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v, size_t s, size_t H_, cudaStream_t stream) {
  //Tensor *in pointer resides in CPU memory and it points to GPU memory
  //cuda memcopy device to device
  //in : [3, n_head, b, s, H_] -> q : [b, s, H_], k : [b, s, H_], v : [b, s, H_]
  for(int i = 0; i < s; i++){
    cudaMemcpyAsync(q->buf + i*H_, in->buf + 0*n_head*s*H_ + head_idx*s*H_ + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(k->buf + i*H_, in->buf + 1*n_head*s*H_ + head_idx*s*H_ + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v->buf + i*H_, in->buf + 2*n_head*s*H_ + head_idx*s*H_ + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }
}







/* Merge each heads
 * @param [in1]       in: [s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]      out: [n_head, s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out) {
  size_t s = in->shape[0];
  size_t H_ = in->shape[1];  // = HIDDEN_DIM/NUM_HEAD

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H_; j++) {
      out->buf[head_idx * s * H_ + i * H_ + j] = in->buf[i * H_ + j];
    }
  }
}

void merge_head_gpu(Tensor *in, size_t head_idx, size_t n_head, Tensor *out, size_t s, size_t H_, cudaStream_t stream) {
  //Tensor *in pointer resides in CPU memory and it points to GPU memory
  //cuda memcopy device to device
  //in : [s, H_] -> out : [n_head, s, H_]
  for(int i = 0; i < s; i++){
    cudaMemcpyAsync(out->buf + head_idx*s*H_ + i*H_, in->buf + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }
}



/* Concatenate each heads
 * @param [in1]     in: [n_head, s, H_]
 * @param [out]    out: [s, H_*n_head]
 * 'n_head' is the number of heads.
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 */
void concat_head(Tensor *in, Tensor *out) {
  size_t n_head = in->shape[0];
  size_t s = in->shape[1];
  size_t H_ = in->shape[2];  // = HIDDEN_DIM/NUM_HEAD

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < n_head; j++) {
      for (size_t k = 0; k < H_; k++) {
        out->buf[i * n_head * H_ + j * H_ + k] =
            in->buf[j * s * H_ + i * H_ + k];
      }
    }
  }
}

// void concat_head_gpu(Tensor *in, Tensor *out, size_t n_head, size_t batch_size, size_t seq_len, size_t hidden_dim ,cudaStream_t stream) {
//   //Tensor *in pointer resides in CPU memory and it points to GPU memory
//   //cuda memcopy device to device
//   //in : [n_head, batch_size, seq_len, H/n_head] -> out : [batch_size, seq_len, H]
//   size_t H_ = hidden_dim/n_head;
//   size_t t_s = batch_size*seq_len;
//   //std::cout<<"concat_head_gpu: "<<" H_ "<<H_<<" t_s "<<t_s<<std::endl;
//   for(int i = 0; i < t_s; i++){
//     for(int j = 0; j < n_head; j++){
//       cudaMemcpyAsync(out->buf + i*hidden_dim + j*H_, in->buf + j*t_s*H_ + i*H_, H_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
//     }
//   }
// }

__global__ void concat_head_kernel(float *in, float *out, size_t n_head, size_t batch_size, size_t seq_len, size_t H_) {
    size_t hidden_dim = n_head * H_;
    size_t t_s = batch_size * seq_len;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < t_s * hidden_dim) {
        size_t i = idx / hidden_dim;  // batch_size * seq_len dimension
        size_t j = (idx - i * hidden_dim) / H_;  // n_head dimension
        size_t k = idx - i * hidden_dim - j * H_;  // H_ dimension

        out[i * hidden_dim + j * H_ + k] = in[j * t_s * H_ + i * H_ + k];
    }
}

void concat_head_gpu(Tensor *in, Tensor *out, size_t n_head, size_t batch_size, size_t seq_len, size_t hidden_dim, cudaStream_t stream) {
    //in : [n_head, batch_size, seq_len, H/n_head] -> out : [batch_size, seq_len, H]
    size_t H_ = hidden_dim / n_head;
    size_t t_s = batch_size * seq_len;
    size_t total_size = t_s * hidden_dim;

    // Calculate grid and block dimensions
    size_t block_size = 256;
    size_t grid_size = (total_size + block_size - 1) / block_size;

    // Kernel invocation
    concat_head_kernel<<<grid_size, block_size, 0, stream>>>(in->buf, out->buf, n_head, batch_size, seq_len, H_);
}

/* Greedy Max Sampling
 * @param  [in1]  in: [s, V]
 * @return [ret] out: [1]
 * 's' is the number of tokens in the prompt.
 * 'V' is the number of vocabulary.
 */
int top1_sampling(Tensor *in) {
  size_t s = in->shape[0];
  size_t V = in->shape[1];

  int out = 0;
  float max = -INFINITY;
  for (size_t i = 0; i < V; i++) {
    if (in->buf[(s - 1) * V + i] > max) {
      max = in->buf[(s - 1) * V + i];
      out = i;
    }
  }

  return out;
}



__global__ void top1_sampling_gpu_kernel(float *in, size_t batch_size, size_t seq_idx, int* gpu_output, int output_len, size_t t){
  
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < batch_size){
    float max = -INFINITY;
    int new_id = 0;
    for(int i = 0; i < NUM_VOCAB; i++){
      if(in[idx*seq_idx*NUM_VOCAB + (seq_idx-1)*NUM_VOCAB + i] > max){
        max = in[idx*seq_idx*NUM_VOCAB + (seq_idx-1)*NUM_VOCAB + i];
        new_id = i;
      }
    }
    gpu_output[idx*output_len + (t)] = new_id;
  }
}


void top1_sampling_gpu(Tensor *in, size_t batch_size, size_t input_seq_len, size_t t, int* gpu_output, int output_len, cudaStream_t stream) {
  //Tensor *in pointer resides in CPU memory and it points to GPU memory
  //cuda memcopy device to host
  //in : [batch_size, seq_len, NUM_VOCAB] -> out : [batch_size]
  size_t seq_idx = input_seq_len + t;
  int grid_size = (batch_size + 31)/32;
  int block_size = 32;
  top1_sampling_gpu_kernel<<<grid_size, block_size, 0, stream>>>(in->buf, batch_size, seq_idx, gpu_output, output_len, t);
  //add the next_token_id end of the out
}


// __global__ void top1_sampling_gpu_kernel(float *in, size_t batch_size, size_t seq_idx, int* gpu_output, int output_len, size_t t) {
//     extern __shared__ float shared_mem[];
//     int* shared_idx = (int*)&shared_mem[blockDim.x];

//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t tid = threadIdx.x;

//     if (idx < batch_size) {
//         // Initialize shared memory
//         shared_mem[tid] = -INFINITY;
//         shared_idx[tid] = -1;

//         for (int i = tid; i < NUM_VOCAB; i += blockDim.x) {
//             float val = in[idx * NUM_VOCAB + seq_idx * NUM_VOCAB + i];
//             if (val > shared_mem[tid]) {
//                 shared_mem[tid] = val;
//                 shared_idx[tid] = i;
//             }
//         }

//         __syncthreads();

//         // Perform parallel reduction to find the max value and index
//         for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//             if (tid < stride) {
//                 if (shared_mem[tid] < shared_mem[tid + stride]) {
//                     shared_mem[tid] = shared_mem[tid + stride];
//                     shared_idx[tid] = shared_idx[tid + stride];
//                 }
//             }
//             __syncthreads();
//         }

//         if (tid == 0) {
//             gpu_output[idx * output_len + t] = shared_idx[0];
//         }
//     }
// }

// void top1_sampling_gpu(Tensor *in, size_t batch_size, size_t input_seq_len, size_t t, int* gpu_output, int output_len, cudaStream_t stream) {
//     // Tensor *in pointer resides in CPU memory and it points to GPU memory
//     // in : [batch_size, seq_len, NUM_VOCAB] -> out : [batch_size]
//     size_t seq_idx = input_seq_len + t;
//     int block_size = 256;
//     int grid_size = (batch_size + block_size - 1) / block_size;
//     size_t shared_memory_size = block_size * sizeof(float) + block_size * sizeof(int);
//     top1_sampling_gpu_kernel<<<grid_size, block_size, shared_memory_size, stream>>>(in->buf, batch_size, seq_idx, gpu_output, output_len, t);
// }
