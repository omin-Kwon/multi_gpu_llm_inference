/* Last updated: 24.06.10 21:00 */
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model.h"

static size_t num_prompts = 1;
static size_t num_generate_token = 8;
static bool run_validation = false;

static char input_fname[100] = "./data/input.bin";
static char param_fname[100] = "/shpc24/project_model_parameters.bin";
static char answer_fname[100] = "./data/answer.bin";
static char output_fname[100] = "./data/output.bin";

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_help() {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stdout,
            " Usage: ./main [-i 'pth'] [-p 'pth'] [-o 'pth'] [-a 'pth']"
            " [-t 'tokens'] [-n 'prompts'] [-v] [-h]\n");
    fprintf(stdout, " Options:\n");
    fprintf(stdout, "  -i: Input binary path (default: ./data/input.bin)\n");
    fprintf(stdout,
            "  -p: Model parameter path (default: "
            "/shpc24/project_model_parameters.bin)\n");
    fprintf(stdout, "  -o: Output binary path (default: ./data/output.bin)\n");
    fprintf(stdout, "  -a: Answer binary path (default: ./data/answer.bin)\n");
    fprintf(stdout, "  -n: Number of input prompts (default: 1)\n");
    fprintf(stdout, "  -t: Number of tokens to generate (default: 8)\n");
    fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
    fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
  }
}

void parse_args(int argc, char **argv) {
  int args;
  while ((args = getopt(argc, argv, "i:o:a:p:n:t:vswh")) != -1) {
    switch (args) {
      case 'i': strcpy(input_fname, optarg); break;
      case 'o': strcpy(output_fname, optarg); break;
      case 'a': strcpy(answer_fname, optarg); break;
      case 'p': strcpy(param_fname, optarg); break;
      case 'n': num_prompts = atoi(optarg); break;
      case 't': num_generate_token = atoi(optarg); break;
      case 'v': run_validation = true; break;
      case 'h':
        print_help();
        exit(0);
        break;
      default:
        print_help();
        exit(0);
        break;
    }
  }
  
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    fprintf(stdout, "\n=============================================\n");
    fprintf(stdout, " Model: GPT-2 125M\n");
    fprintf(stdout, "---------------------------------------------\n");
    fprintf(stdout, " Validation: %s\n", run_validation ? "ON" : "OFF");
    fprintf(stdout, " Number of Prompts: %ld\n", num_prompts);
    fprintf(stdout, " Number of Tokens to generate: %ld\n", num_generate_token);
    fprintf(stdout, " Input binary path: %s\n", input_fname);
    fprintf(stdout, " Model parameter path: %s\n", param_fname);
    fprintf(stdout, " Answer binary path: %s\n", answer_fname);
    fprintf(stdout, " Output binary path: %s\n", output_fname);
    fprintf(stdout, "=============================================\n\n");
  }
}

int validate(int *output, int *answer, int size_) {
  int mismatch_idx = -1;
  int tolerance = (int) (size_ * 0.0005);  // Error tolerance percentage

  for (int i = 0; i < size_; i++) {
    /* Check if the output and answer are the same */
    if (output[i] != answer[i] || std::isnan(output[i])) {
      tolerance--;

      /* Record the first mismatched token number*/
      if (mismatch_idx == -1) mismatch_idx = i;

      /* Break if tolerance is reached */
      if (tolerance < 0) { return mismatch_idx; }
    }
  }

  return -1;
}

void *read_binary(const char *fname, size_t *size) {
  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    fprintf(stdout, "[ERROR] Cannot open file \'%s\'\n", fname);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_t size_ = ftell(f);
  rewind(f);

  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    fprintf(stdout, "[ERROR] Cannot read file \'%s\'\n", fname);
    exit(-1);
  }
  fclose(f);

  if (size != NULL) *size = (size_t)(size_ / 4);  // 4 bytes per float or int

  return buf;
}

void write_binary(int *output, const char *filename, int size_) {
  FILE *f = (FILE *) fopen(filename, "w");
  fwrite(output, sizeof(int), size_, f);
  fclose(f);
}

int main(int argc, char **argv) {
  int mpi_rank, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  parse_args(argc, argv);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZATION                                                 //
  ////////////////////////////////////////////////////////////////////

  int *input = nullptr, *output = nullptr;
  float *param = nullptr;
  size_t param_size = 0;

  /* Initialize input, output and parameters */
  if (mpi_rank == 0) fprintf(stdout, "Initializing input and parameters...");
  if (mpi_rank == 0) {
    /* Load input (size: num_prompts x tokens_per_prompt) from file  */
    size_t input_size;
    input = (int *) read_binary(input_fname, &input_size);

    if (input_size % tokens_per_prompt != 0) {
      fprintf(stdout, "Invalid input size\n");
      exit(1);
    }

    /* Allocate output (size: num_prompts x num_generate_token) */
    output = (int *) malloc(num_prompts * num_generate_token * sizeof(int));
  }
  param = (float *) read_binary(param_fname, &param_size);
  alloc_and_set_parameters(param);

  if (mpi_rank == 0) fprintf(stdout, "Done\n");
  //8 + num_generate_token 8
  assert(tokens_per_prompt + num_generate_token <= MAX_SEQ_LEN);

  ////////////////////////////////////////////////////////////////////
  // MODEL COMPUTATION                                              //
  ////////////////////////////////////////////////////////////////////

  double st = 0.0, et = 0.0;
  for (size_t i = 0; i < 4; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    fprintf(stdout, "Generating tokens...");
    fflush(stdout);
    st = get_time();
  }

  /* Call the main computation (optimization target) of the program. */
  generate_tokens(input, output, num_prompts, num_generate_token);
  
  for (size_t i = 0; i < 4; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    et = get_time();

    /* Print the result */
    fprintf(stdout, "Done!\n");
    fprintf(stdout, "Elapsed time: %lf (sec)\n", et - st);
    fprintf(stdout, "Throughput: %lf (tokens/sec)\n",
            num_prompts * num_generate_token / (et - st));
  }

  ////////////////////////////////////////////////////////////////////
  // FINALIZATION                                                   //
  ////////////////////////////////////////////////////////////////////

  /* Finalize parameters */
  if (mpi_rank == 0) fprintf(stdout, "Finalizing...");
  free_parameters();
  if (mpi_rank == 0) fprintf(stdout, "Done\n");

  if (mpi_rank == 0) {
    /* Save output */
    fprintf(stdout, "Saving output to %s...", output_fname);
    write_binary(output, output_fname, num_prompts * num_generate_token);
    fprintf(stdout, "Done\n");

    /* Validation */
    if (run_validation) {
      fprintf(stdout, "Validation...");

      int *answer = (int *) read_binary(answer_fname, NULL);
      int ret = validate(output, answer, num_prompts * num_generate_token);
      if (ret == -1) {
        fprintf(stdout, "PASS\n");
      } else {
        fprintf(stdout,
                "FAIL\nFirst mismatch "
                "at prompt[#%ld], token_ID[#%ld] (output[%d]=%d <-> "
                "answer[%d]=%d)\n",
                ret / num_generate_token, ret % num_generate_token, ret,
                output[ret], ret, answer[ret]);
      }
    }
  }

  /* MPI Finalization */
  MPI_Finalize();

  return 0;
}
