#include <mpi.h>

#include <cmath>
#include <cstdio>

#include "layer.h"
#include "model.h"
#include <iostream>

static int ngpu = 4;
cudaStream_t streams[NGPU];
static int *gpu_input[NGPU], *gpu_output[NGPU];


ParameterCPU *attn_b[NUM_LAYER], *attn_w[NUM_LAYER];
ParameterCPU *proj_b[NUM_LAYER], *proj_w[NUM_LAYER];
ParameterCPU *ln_1_b[NUM_LAYER], *ln_1_g[NUM_LAYER];
ParameterCPU *ln_2_b[NUM_LAYER], *ln_2_g[NUM_LAYER];
ParameterCPU *mlp1_b[NUM_LAYER], *mlp1_w[NUM_LAYER];
ParameterCPU *mlp2_b[NUM_LAYER], *mlp2_w[NUM_LAYER];
ParameterCPU *ln_f_b, *ln_f_g;
ParameterCPU *wpe, *wte;

// Parameter *attn_b_gpu[NGPU], *attn_w_gpu[NGPU];
// Parameter *proj_b_gpu[NGPU], *proj_w_gpu[NGPU];
// Parameter *ln_1_b_gpu[NGPU], *ln_1_g_gpu[NGPU];
// Parameter *ln_2_b_gpu[NGPU], *ln_2_g_gpu[NGPU];
// Parameter *mlp1_b_gpu[NGPU], *mlp1_w_gpu[NGPU];
// Parameter *mlp2_b_gpu[NGPU], *mlp2_w_gpu[NGPU];
// Parameter *ln_f_b_gpu[NGPU], *ln_f_g_gpu[NGPU];
// Parameter *wpe_gpu[NGPU], *wte_gpu[NGPU];



// It has four GPUs, so we need to allocate the memory for the parameters for each GPU
// Also, in each GPU, we need to allocate the memory for the parameters for each layer
// So, Let's allocate the paramter for double pointer. attn_b_gpu[i] means the double pointer for the parameters for the i-th GPU
//attn_b_gpu[i] is the array of the pointers for the parameters and the length of the array is NUM_LAYER  
ParameterGPU **attn_b_gpu[NGPU], **attn_w_gpu[NGPU];
ParameterGPU **proj_b_gpu[NGPU], **proj_w_gpu[NGPU];
ParameterGPU **ln_1_b_gpu[NGPU], **ln_1_g_gpu[NGPU];
ParameterGPU **ln_2_b_gpu[NGPU], **ln_2_g_gpu[NGPU];
ParameterGPU **mlp1_b_gpu[NGPU], **mlp1_w_gpu[NGPU];
ParameterGPU **mlp2_b_gpu[NGPU], **mlp2_w_gpu[NGPU];
ParameterGPU *ln_f_b_gpu[NGPU], *ln_f_g_gpu[NGPU];
ParameterGPU *wpe_gpu[NGPU], *wte_gpu[NGPU];





// declare the pointer for the GPU memory
ActivationGPU *embd_a[NGPU], *ffn_proj_a[NGPU];
ActivationGPU *mha_qkv_proj_a[NGPU], *mha_out_a[NGPU], *mha_split_head_a[NGPU],
    *mha_mask_a[NGPU], *mha_q_a[NGPU], *mha_k_a[NGPU], *mha_v_a[NGPU],
    *mha_attn_out_a[NGPU], *mha_concat_head_a[NGPU];
ActivationGPU *attn_score_a[NGPU], *k_transposed_a[NGPU];
ActivationGPU *wte_transposed_a[NGPU], *residual_a[NGPU], *logit_a[NGPU];
ActivationGPU *transformer_block_a[NGPU];
int *next_token_id[NGPU];




void alloc_and_set_parameters(float *param) {
  size_t pos = 0;
  int order[] = {
      0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9,
  };
  for (int i = 0; i < NUM_LAYER; i++) {
    attn_b[order[i]] = new ParameterCPU({3 * HIDDEN_DIM}, param + pos);
    pos += OFFSET1;
    attn_w[order[i]] = new ParameterCPU({HIDDEN_DIM, 3 * HIDDEN_DIM}, param + pos);
    pos += OFFSET2;
    proj_b[order[i]] = new ParameterCPU({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    proj_w[order[i]] = new ParameterCPU({HIDDEN_DIM, HIDDEN_DIM}, param + pos);
    pos += OFFSET4;
    ln_1_b[order[i]] = new ParameterCPU({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_1_g[order[i]] = new ParameterCPU({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_2_b[order[i]] = new ParameterCPU({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_2_g[order[i]] = new ParameterCPU({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    mlp1_b[order[i]] = new ParameterCPU({4 * HIDDEN_DIM}, param + pos);
    pos += OFFSET5;
    mlp1_w[order[i]] = new ParameterCPU({HIDDEN_DIM, 4 * HIDDEN_DIM}, param + pos);
    pos += OFFSET6;
    mlp2_b[order[i]] = new ParameterCPU({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    mlp2_w[order[i]] = new ParameterCPU({4 * HIDDEN_DIM, HIDDEN_DIM}, param + pos);
    pos += OFFSET6;
  }
  ln_f_b = new ParameterCPU({HIDDEN_DIM}, param + pos);
  pos += OFFSET3;
  ln_f_g = new ParameterCPU({HIDDEN_DIM}, param + pos);
  pos += OFFSET3;
  wpe = new ParameterCPU({MAX_SEQ_LEN, HIDDEN_DIM}, param + pos);
  pos += OFFSET7;
  wte = new ParameterCPU({NUM_VOCAB, HIDDEN_DIM}, param + pos);
  pos += OFFSET8;
}

void free_parameters() {
  for (int i = 0; i < NUM_LAYER; i++) {
    delete attn_b[i];
    delete attn_w[i];
    delete proj_b[i];
    delete proj_w[i];
    delete ln_1_b[i];
    delete ln_1_g[i];
    delete ln_2_b[i];
    delete ln_2_g[i];
    delete mlp1_b[i];
    delete mlp1_w[i];
    delete mlp2_b[i];
    delete mlp2_w[i];
  }
  delete ln_f_b;
  delete ln_f_g;  
  delete wpe;
  delete wte;
}


void alloc_activations_gpu(size_t batch_size, size_t seq_len){
  for(int gpu_id = 0; gpu_id < ngpu; gpu_id++){
    CHECK_CUDA(cudaSetDevice(gpu_id));
    embd_a[gpu_id] = new Activation({batch_size, seq_len, HIDDEN_DIM});
    ffn_proj_a[gpu_id] = new Activation({batch_size, seq_len, 4 * HIDDEN_DIM});
    mha_qkv_proj_a[gpu_id] = new Activation({batch_size, seq_len, 3 * HIDDEN_DIM});
    mha_out_a[gpu_id] = new Activation({batch_size, seq_len, HIDDEN_DIM});
    mha_mask_a[gpu_id] = new Activation({seq_len, seq_len});
    
    mha_q_a[gpu_id] = new Activation({NUM_HEAD, batch_size, seq_len, HIDDEN_DIM / NUM_HEAD});
    mha_k_a[gpu_id] = new Activation({NUM_HEAD, batch_size, seq_len, HIDDEN_DIM / NUM_HEAD});
    mha_v_a[gpu_id] = new Activation({NUM_HEAD, batch_size, seq_len, HIDDEN_DIM / NUM_HEAD});
    mha_attn_out_a[gpu_id] = new Activation({NUM_HEAD, batch_size, seq_len, HIDDEN_DIM / NUM_HEAD});
    mha_concat_head_a[gpu_id] = new Activation({batch_size, seq_len, HIDDEN_DIM});
    
    attn_score_a[gpu_id] = new Activation({NUM_HEAD, batch_size, seq_len, seq_len});
    k_transposed_a[gpu_id] = new Activation({NUM_HEAD, batch_size, HIDDEN_DIM / NUM_HEAD, seq_len});
    
    wte_transposed_a[gpu_id] = new Activation({HIDDEN_DIM, NUM_VOCAB});
    residual_a[gpu_id] = new Activation({batch_size, seq_len, HIDDEN_DIM});
    logit_a[gpu_id] = new Activation({batch_size, seq_len, NUM_VOCAB});
    transformer_block_a[gpu_id] = new Activation({batch_size, seq_len, HIDDEN_DIM});
    //for next token id
    CHECK_CUDA(cudaMalloc(&next_token_id[gpu_id], batch_size * sizeof(int)));
  }
  //generate the positional mask
  for(int i = 0; i < ngpu; i++){
    CHECK_CUDA(cudaSetDevice(i));
    generate_mask_gpu(mha_mask_a[i], streams[i]);
    transpose_no_group_gpu(wte_gpu[i], wte_transposed_a[i], NUM_VOCAB, HIDDEN_DIM, streams[i]);
  }
}


void alloc_parameters_gpu(){
  //we have to allocate the memory for the parameters for each layer
  for(int gpu_id= 0; gpu_id <ngpu; gpu_id++){
      cudaSetDevice(gpu_id);
      //attn_b_gpu, attn_w_gpu, proj_b_gpu, proj_w_gpu, ln_1_b_gpu, ln_1_g_gpu, ln_2_b_gpu, ln_2_g_gpu, mlp1_b_gpu, mlp1_w_gpu, mlp2_b_gpu, mlp2_w_gpu
      //attn_b_gpu[i] is the array of the pointers to the Parameter struct
      //allocate the memory for the parameters for each layer
      attn_b_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      attn_w_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      proj_b_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      proj_w_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      ln_1_b_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      ln_1_g_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      ln_2_b_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      ln_2_g_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      mlp1_b_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      mlp1_w_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      mlp2_b_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      mlp2_w_gpu[gpu_id] = new ParameterGPU*[NUM_LAYER];
      for(int j = 0 ; j < NUM_LAYER; j++){
        attn_b_gpu[gpu_id][j] = new ParameterGPU({3 * HIDDEN_DIM},attn_b[j]->buf);
        attn_w_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM, 3 * HIDDEN_DIM}, attn_w[j]->buf);
        proj_b_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM}, proj_b[j]->buf);
        proj_w_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM, HIDDEN_DIM}, proj_w[j]->buf);
        ln_1_b_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM}, ln_1_b[j]->buf);
        ln_1_g_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM}, ln_1_g[j]->buf);
        ln_2_b_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM}, ln_2_b[j]->buf);
        ln_2_g_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM}, ln_2_g[j]->buf);
        mlp1_b_gpu[gpu_id][j] = new ParameterGPU({4 * HIDDEN_DIM}, mlp1_b[j]->buf);
        mlp1_w_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM, 4 * HIDDEN_DIM}, mlp1_w[j]->buf);
        mlp2_b_gpu[gpu_id][j] = new ParameterGPU({HIDDEN_DIM}, mlp2_b[j]->buf);
        mlp2_w_gpu[gpu_id][j] = new ParameterGPU({4 * HIDDEN_DIM, HIDDEN_DIM}, mlp2_w[j]->buf);
      }
      //allocate the memory for the parameters for the final layer
      ln_f_b_gpu[gpu_id] = new ParameterGPU({HIDDEN_DIM}, ln_f_b->buf);
      ln_f_g_gpu[gpu_id] = new ParameterGPU({HIDDEN_DIM}, ln_f_g->buf);
      wpe_gpu[gpu_id] = new ParameterGPU({MAX_SEQ_LEN, HIDDEN_DIM}, wpe->buf);
      wte_gpu[gpu_id] = new ParameterGPU({NUM_VOCAB, HIDDEN_DIM}, wte->buf);
  }
}


void free_activations() {
  //so we need to delete the memory for each pointer
  for(int i = 0; i < ngpu; i++){
    CHECK_CUDA(cudaSetDevice(i));
    delete embd_a[i];
    delete ffn_proj_a[i];
    delete mha_qkv_proj_a[i];
    delete mha_out_a[i];
    delete mha_mask_a[i];
    delete mha_q_a[i];
    delete mha_k_a[i];
    delete mha_v_a[i];
    delete mha_attn_out_a[i];
    delete mha_concat_head_a[i];
    delete attn_score_a[i];
    delete k_transposed_a[i];
    delete wte_transposed_a[i];
    delete residual_a[i];
    delete logit_a[i];
    delete transformer_block_a[i];
  }
}

/* (Position-wise) Feed-Forward Network
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in3] mlp1_b: [4*HIDDEN_DIM]
 * @param [in4] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [in5] mlp2_b: [HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
// void ffn(Activation *in, Parameter *mlp1_w, Parameter *mlp1_b,
//          Parameter *mlp2_w, Parameter *mlp2_b, Activation *out) {
//   /* Projection Up:
//     [seq_len, HIDDEN_DIM] -> [seq_len, 4*HIDDEN_DIM] */
//   linear(in, mlp1_w, mlp1_b, ffn_proj_a);

//   /* GELU */
//   gelu(ffn_proj_a);

//   /* Projection Down:
//     [seq_len, 4*HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
//   linear(ffn_proj_a, mlp2_w, mlp2_b, out);
// }



void ffn_gpu(Activation *in, Parameter *mlp1_w, Parameter *mlp1_b,
         Parameter *mlp2_w, Parameter *mlp2_b, Activation *out, size_t batch_size, size_t seq_len, size_t hidden_dim, cudaStream_t stream, int gpu_id){
  //The same process as the CPU version, but we need to add the stream parameter
  //It should be executed in the gpu, it shouldn't be communicated with the CPU
  //So we need to add the stream parameter and invoke the multiple sequential kernels in the GPU
  //Projection Up Kernel
  linear_gpu(in, mlp1_w, mlp1_b, ffn_proj_a[gpu_id], batch_size, seq_len, stream);
  //GELU Kernel
  gelu_gpu(ffn_proj_a[gpu_id], batch_size* seq_len * 4 * hidden_dim, stream);
  //Projection Down Kernel
  linear_gpu(ffn_proj_a[gpu_id], mlp2_w, mlp2_b, out, batch_size, seq_len, stream);
}



/* Attention
 * @param [in1]    q: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in2]    k: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in3]    v: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in4] mask: [seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [out]  out: [seq_len, HIDDEN_DIM/NUM_HEAD]
 */
// void attention(Activation *q, Activation *k, Activation *v, Activation *mask,
//                Activation *out) {
//   /* Get Attention score by q @ k */
//   transpose(k, k_transposed_a);
//   matmul(q, k_transposed_a, attn_score_a);

//   /* Scaling */
//   scaling(attn_score_a, (1.0 / sqrt(k->shape[1])));

//   /* Masking */
//   add(attn_score_a, mask);

//   /* Softmax */
//   softmax(attn_score_a);

//   /* Attention score @ v */
//   matmul(attn_score_a, v, out);
// }

void attention_gpu(Activation *q, Activation *k, Activation *v, Activation *mask,
               Activation *out, size_t batch_size, size_t seq_len, size_t total_seq_len , cudaStream_t stream, int gpu_id) {
  //The same process as the CPU version, but we need to add the stream parameter
  //It should be executed in the gpu, it shouldn't be communicated with the CPU
  //So we need to add the stream parameter and invoke the multiple sequential kernels in the GPU
  //Get Attention score by q @ k Kernel
  transpose_gpu(k, k_transposed_a[gpu_id], NUM_HEAD*batch_size, seq_len, HIDDEN_DIM/NUM_HEAD, stream);
  matmul_gpu(q, k_transposed_a[gpu_id], attn_score_a[gpu_id], NUM_HEAD*batch_size, seq_len, HIDDEN_DIM/NUM_HEAD, seq_len ,stream);
  //Scaling Kernel
  scaling_gpu(attn_score_a[gpu_id], (1.0 / sqrt(k->shape[3])), stream);
  //Masking Kernel
  add_mask(attn_score_a[gpu_id], mask, NUM_HEAD*batch_size, seq_len, total_seq_len ,stream);
  //Softmax Kernel
  softmax_gpu(attn_score_a[gpu_id], NUM_HEAD*batch_size, seq_len, seq_len, stream);
  //Attention score @ v Kernel
  matmul_gpu(attn_score_a[gpu_id], v, out, NUM_HEAD*batch_size, seq_len, seq_len ,HIDDEN_DIM / NUM_HEAD, stream);
}



/* (Masked) Multi-Head Self Attention
 * @param [in1]     in: [seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [seq_len, HIDDEN_DIM]
 */
// void mha(Activation *in, Parameter *attn_b, Parameter *attn_w,
//          Parameter *proj_b, Parameter *proj_w, Activation *out) {
//   /* QKV projection:
//     [seq_len, HIDDEN_DIM] ->
//     [seq_len, 3*HIDDEN_DIM] */
//   linear(in, attn_w, attn_b, mha_qkv_proj_a);

//   /* Split into Q, K, V:
//     [seq_len, 3*HIDDEN_DIM] ->
//     [3, seq_len, HIDDEN_DIM] */
//   split_qkv(mha_qkv_proj_a, mha_split_qkv_a);

//   /* Split into multiple heads:
//     [3, seq_len, HIDDEN_DIM] ->
//     [3, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
//   split_head(mha_split_qkv_a, NUM_HEAD, mha_split_head_a);

//   /* Generate mask to hide future inputs */
//   generate_mask(mha_mask_a);

//   /* Perform Attention over each head:
//     [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] ->
//     [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
//   for (size_t idx = 0; idx < NUM_HEAD; idx++) {
//     /* Extract Q, K, V from qkv_head */
//     extract_qkv(mha_split_head_a, idx, NUM_HEAD, mha_q_a, mha_k_a, mha_v_a);

//     /* Attention */
//     attention(mha_q_a, mha_k_a, mha_v_a, mha_mask_a, mha_attn_out_a);

//     /* Merge each head's attn output
//       [seq_len, HIDDEN_DIM/NUM_HEAD] ->
//       [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
//     merge_head(mha_attn_out_a, idx, NUM_HEAD, mha_merge_head_a);
//   }

//   /* Concat each heads:
//     [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] ->
//     [seq_len, HIDDEN_DIM] */
//   concat_head(mha_merge_head_a, mha_concat_head_a);

//   /* OUT projection:
//     [seq_len, HIDDEN_DIM] -> [seq_len, HIDDEN_DIM] */
//   linear(mha_concat_head_a, proj_w, proj_b, out);
// }


void mha_gpu(Activation *in, Parameter *attn_b, Parameter *attn_w,
         Parameter *proj_b, Parameter *proj_w, Activation *out, size_t batch_size, size_t seq_len, size_t total_seq_len, cudaStream_t stream, int gpu_id) {
  //The same process as the CPU version, but we need to add the stream parameter
  //It should be executed in the gpu, it shouldn't be communicated with the CPU
  //So we need to add the stream parameter and invoke the multiple sequential kernels in the GPU
  //QKV projection Kernel
  linear_gpu(in, attn_w, attn_b, mha_qkv_proj_a[gpu_id], batch_size, seq_len ,stream);
  //Generate mask to hide future inputs Kernel
  //generate_mask_gpu(mha_mask_a[gpu_id], stream);
  split_qkv_head_gpu(mha_qkv_proj_a[gpu_id], mha_q_a[gpu_id], mha_k_a[gpu_id], mha_v_a[gpu_id], NUM_HEAD, batch_size*seq_len, HIDDEN_DIM, stream);
  attention_gpu(mha_q_a[gpu_id], mha_k_a[gpu_id], mha_v_a[gpu_id], mha_mask_a[gpu_id], mha_attn_out_a[gpu_id], batch_size, seq_len, total_seq_len, stream, gpu_id);


  //Perform Attention over each head Kernel
  // for (size_t idx = 0; idx < NUM_HEAD; idx++) {
  //   //Extract Q, K, V from qkv_head Kernel
  //   extract_qkv_gpu(mha_split_head_a[gpu_id], idx, NUM_HEAD, mha_q_a[gpu_id], mha_k_a[gpu_id], mha_v_a[gpu_id], batch_size*seq_len, hidden_dim, stream);
  //   //Attention Kernel
  //   attention_gpu(mha_q_a[gpu_id], mha_k_a[gpu_id], mha_v_a[gpu_id], mha_mask_a[gpu_id], mha_attn_out_a[gpu_id], batch_size, seq_len, hidden_dim, stream,gpu_id);
  //   //Merge each head's attn output Kernel
  //   merge_head_gpu(mha_attn_out_a[gpu_id], idx, NUM_HEAD, mha_merge_head_a[gpu_id], batch_size*seq_len, hidden_dim, stream);
  // }
  //Concat each heads Kernel
  concat_head_gpu(mha_attn_out_a[gpu_id], mha_concat_head_a[gpu_id], NUM_HEAD ,batch_size, seq_len, HIDDEN_DIM, stream);
  //OUT projection Kernel
  linear_gpu(mha_concat_head_a[gpu_id], proj_w, proj_b, out, batch_size, seq_len, stream);
}











/* Transformer Block
 * @param [in1]      in: [seq_len, HIDDEN_DIM]
 * @param [in2]  attn_b: [3*HIDDEN_DIM]
 * @param [in3]  attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4]  proj_b: [HIDDEN_DIM]
 * @param [in5]  proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [in6]  ln_1_b: [HIDDEN_DIM]
 * @param [in7]  ln_1_g: [HIDDEN_DIM]
 * @param [in8]  ln_2_b: [HIDDEN_DIM]
 * @param [in9]  ln_2_g: [HIDDEN_DIM]
 * @param [in10] mlp1_b: [4*HIDDEN_DIM]
 * @param [in11] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in12] mlp2_b: [HIDDEN_DIM]
 * @param [in13] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]     out: [seq_len, HIDDEN_DIM]
 */
// void transformer_block(Activation *in, Parameter *attn_b, Parameter *attn_w,
//                        Parameter *proj_b, Parameter *proj_w, Parameter *ln_1_b,
//                        Parameter *ln_1_g, Parameter *ln_2_b, Parameter *ln_2_g,
//                        Parameter *mlp1_b, Parameter *mlp1_w, Parameter *mlp2_b,
//                        Parameter *mlp2_w, Activation *out) {
//   /* Copy Residual */
//   copy(in, residual_a);

//   /* Layer Normalization */
//   layer_norm(in, ln_1_g, ln_1_b);

//   /* Masked Multi-Head Self-Attention */
//   mha(in, attn_b, attn_w, proj_b, proj_w, mha_out_a);

//   /* Add Residual */
//   add(mha_out_a, residual_a);

//   /* Copy Residual */
//   copy(mha_out_a, residual_a);

//   /* Layer Normalization */
//   layer_norm(mha_out_a, ln_2_g, ln_2_b);

//   /* Position-wise Feed-Forward Network */
//   ffn(mha_out_a, mlp1_w, mlp1_b, mlp2_w, mlp2_b, out);

//   /* Add Residual */
//   add(out, residual_a);
// }


void transformer_block_gpu(ActivationGPU *in, ParameterGPU *attn_b, ParameterGPU *attn_w,
                       ParameterGPU *proj_b, ParameterGPU *proj_w, ParameterGPU *ln_1_b,
                       ParameterGPU *ln_1_g, ParameterGPU *ln_2_b, ParameterGPU *ln_2_g,
                       ParameterGPU *mlp1_b, ParameterGPU *mlp1_w, ParameterGPU *mlp2_b,
                       ParameterGPU *mlp2_w, ActivationGPU *out, size_t batch_size, size_t seq_len, size_t total_seq_len, cudaStream_t stream, int gpu_id) {
  //The same process as the CPU version, but we need to add the stream parameter
  //It should be executed in the gpu, it shouldn't be communicated with the CPU
  //So we need to add the stream parameter and invoke the multiple sequential kernels in the GPU
  //Copy Residual Kernel
  //Do copy with extra stream.
  copy_gpu(in, residual_a[gpu_id], stream);
  //Layer Normalization Kernel
  layer_norm_gpu(in, ln_1_g, ln_1_b, batch_size, seq_len, stream);
  //Masked Multi-Head Self-Attention Kernel
  mha_gpu(in, attn_b, attn_w, proj_b, proj_w, mha_out_a[gpu_id], batch_size, seq_len, total_seq_len, stream, gpu_id);
  add_copy_gpu(mha_out_a[gpu_id], residual_a[gpu_id], batch_size, seq_len, HIDDEN_DIM, stream);
  //Layer Normalization Kernel
  layer_norm_gpu(mha_out_a[gpu_id], ln_2_g, ln_2_b, batch_size, seq_len, stream);
  //Position-wise Feed-Forward Network Kernel
  ffn_gpu(mha_out_a[gpu_id], mlp1_w, mlp1_b, mlp2_w, mlp2_b, out, batch_size, seq_len, HIDDEN_DIM, stream, gpu_id);
  //Add Residual Kernel
  add_gpu(out, residual_a[gpu_id], batch_size, seq_len, HIDDEN_DIM, stream);
}


/* [Model Computation: Token Generation] */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  //Distruibute the prompt to each rank
  // n_token is the number of tokens to generate for each prompt
  size_t prompts_per_rank = n_prompt / mpi_size;
  // allocate the memory for the local input
  int *local_input = new int[prompts_per_rank * tokens_per_prompt];
  // allocate the memory for the local output
  int *local_output = new int[prompts_per_rank * n_token];
  // distribute the input to each rank
  MPI_Scatter(input, prompts_per_rank*tokens_per_prompt, MPI_INT, local_input,
              prompts_per_rank * tokens_per_prompt, MPI_INT, 0, MPI_COMM_WORLD);


  //stream create, malloc, and copy for each GPU

  //prompts_per_gpu is the number of prompts to process for each GPU = batch_size
  int prompts_per_gpu = prompts_per_rank / ngpu;
  int batch_size = prompts_per_gpu;
  int input_seq_len = tokens_per_prompt;
  int output_seq_len = n_token;
  int total_seq_len = input_seq_len + output_seq_len;

  // create the streams for each GPU and allocate the memory for the input and output for each GPU
  for(int i = 0; i < ngpu; i++){
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    //CHECK_CUDA(cudaStreamCreate(&streams_extra[i]));
    CHECK_CUDA(cudaMalloc(&gpu_input[i], batch_size * input_seq_len * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&gpu_output[i], batch_size * output_seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpyAsync(gpu_input[i], local_input + i * batch_size * input_seq_len, batch_size * input_seq_len * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
  }

  //allocate parameters
  alloc_parameters_gpu();
  //allocate the activation
  alloc_activations_gpu(batch_size, total_seq_len);

  // Generate tokens iteratively
  for(int t = 0; t < output_seq_len; t++){
     // Token + Positional Embedding for Input Prompt
    for(int i =0; i <ngpu; i++){
      CHECK_CUDA(cudaSetDevice(i));
      token_pos_embedding_gpu(gpu_input[i], gpu_output[i], wte_gpu[i], wpe_gpu[i], embd_a[i], batch_size, input_seq_len, output_seq_len, t, streams[i]);
    }
    //Forward path of Transformer blocks
    for(int i = 0; i < ngpu; i++){
      CHECK_CUDA(cudaSetDevice(i));
      for (size_t l = 0; l < NUM_LAYER; l++) {
        transformer_block_gpu(embd_a[i], attn_b_gpu[i][l], attn_w_gpu[i][l], proj_b_gpu[i][l], proj_w_gpu[i][l],
                              ln_1_b_gpu[i][l], ln_1_g_gpu[i][l], ln_2_b_gpu[i][l], ln_2_g_gpu[i][l],
                              mlp1_b_gpu[i][l], mlp1_w_gpu[i][l], mlp2_b_gpu[i][l], mlp2_w_gpu[i][l],
                              transformer_block_a[i], batch_size, input_seq_len + t, total_seq_len ,streams[i], i);
        /* Copy output to embd_a for next block */
        copy_gpu(transformer_block_a[i], embd_a[i], streams[i]);
      }
    }
    //Final Layer Normalization
    for(int i = 0; i < ngpu; i++){
      CHECK_CUDA(cudaSetDevice(i));
      layer_norm_gpu(embd_a[i], ln_f_g_gpu[i], ln_f_b_gpu[i], batch_size, input_seq_len + t, streams[i]);
      final_matmul(embd_a[i], wte_transposed_a[i], logit_a[i], batch_size*(input_seq_len + t),  HIDDEN_DIM, NUM_VOCAB, streams[i]);
    }

    // Greedy sampling (only last timestep is considered)
    for(int i = 0; i < ngpu; i++){
      CHECK_CUDA(cudaSetDevice(i));
      top1_sampling_gpu(logit_a[i], batch_size, input_seq_len, t, gpu_output[i], output_seq_len, streams[i]);
    }
  }

  // Copy the output from each GPU
  for(int i = 0; i < ngpu; i++){
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(local_output + i * batch_size * output_seq_len, gpu_output[i], batch_size * output_seq_len * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
  }
  // Gather the output from each rank
  MPI_Gather(local_output, prompts_per_rank * n_token, MPI_INT, output,
              prompts_per_rank * n_token, MPI_INT, 0, MPI_COMM_WORLD);
}
