#pragma once

#include <vector>

#include "tensor.h"

using std::vector;

#define MATH_PI 3.1415926535f

/* Elementwise operations */
void gelu(Tensor *inout);
void add(Tensor *inout, Tensor *x);
void add_cuda(Tensor *inout, Tensor *x);
void scaling(Tensor *inout, float scale);


void gelu_gpu(Tensor *inout, size_t N, cudaStream_t stream);
void add_gpu(Tensor *inout, Tensor *x, size_t group_size, size_t seq_len, size_t hidden_dim ,cudaStream_t stream);
void scaling_gpu(Tensor *inout, float scale, cudaStream_t stream);
void add_copy_gpu(Tensor *inout, Tensor *x, size_t group_size, size_t seq_len, size_t hidden_dim ,cudaStream_t stream);
void add_mask(Tensor *inout, Tensor *x, size_t group_size, size_t seq_len, size_t total_seq_len, cudaStream_t stream);





/* Matmul operations */
void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void matmul(Tensor *in1, Tensor *in2, Tensor *out);

void linear_gpu(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, size_t seq_len ,cudaStream_t stream);
void matmul_gpu(Tensor *in1, Tensor *in2, Tensor *out, size_t group_size ,size_t M, size_t K, size_t N, cudaStream_t stream);
void final_matmul(Tensor *in1, Tensor *in2, Tensor *out, size_t M, size_t K, size_t N, cudaStream_t stream);
void linear_split_qkv_head_fusion(Tensor *in, Tensor *w, Tensor *b, Tensor *q_out, Tensor *k_out, Tensor *v_out, size_t n_head, size_t batch_size, size_t seq_len ,cudaStream_t stream);


/* Data movement operations */
void copy(Tensor *in, Tensor *out);
void transpose(Tensor *in, Tensor *out);
void split_qkv(Tensor *in, Tensor *out);
void split_head(Tensor *in, size_t n_head, Tensor *out);
void concat_head(Tensor *in, Tensor *out);
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q,
                 Tensor *k, Tensor *v);
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out);
void token_pos_embedding(vector<int> in, Parameter *wte, Parameter *wpe,
                         Tensor *out);

//for gpu version
void copy_gpu(Tensor *in, Tensor *out, cudaStream_t stream);
void transpose_gpu(Tensor *in, Tensor *out, size_t group_size ,size_t M, size_t N, cudaStream_t stream);
void transpose_no_group_gpu(Tensor *in, Tensor *out, size_t M, size_t N, cudaStream_t stream);
void split_qkv_head_gpu(Tensor *in, Tensor *q_out, Tensor *k_out, Tensor *v_out, size_t n_head, size_t batched_seq, size_t H, cudaStream_t stream);
//void split_head_gpu(Tensor *in, size_t n_head, Tensor *out, size_t s, size_t H, cudaStream_t stream);
void concat_head_gpu(Tensor *in, Tensor *out, size_t n_head, size_t batch_size, size_t seq_len, size_t hidden_dim ,cudaStream_t stream);
void token_pos_embedding_gpu(int* in_prompt, int* in_generated, Tensor *wte, Tensor *wpe, Tensor *out, size_t batch_size, size_t input_seq_len, size_t output_seq_len, size_t t,  cudaStream_t stream);

/* Other operations */
void softmax(Tensor *inout);
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta);
void generate_mask(Tensor *inout);
int top1_sampling(Tensor *in);

void softmax_gpu(Tensor *inout, size_t group_size, size_t M, size_t N, cudaStream_t stream);
void layer_norm_gpu(Tensor *inout, Tensor *gamma, Tensor *beta, size_t batch_size, size_t seq_len, cudaStream_t stream);
void generate_mask_gpu(Tensor *inout, cudaStream_t stream);
void top1_sampling_gpu(Tensor *in, size_t batch_size, size_t input_seq_len, size_t t, int* gpu_output, int output_len, cudaStream_t stream);