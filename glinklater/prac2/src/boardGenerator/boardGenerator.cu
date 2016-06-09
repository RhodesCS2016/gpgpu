#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define MAXVAL 1   // values must be between 1 and MAXVAL - so there will be duplicates

typedef unsigned char ubyte;

void split_int (ubyte* buffer, uint randVal) {
  for (int i = 0; i < sizeof(uint); i++) {
    buffer[i] = ((1 << i) & randVal) >> i;
  }
}

int main(int argc, char **argv) {
  int board_size = 64;

  opterr = 0;
  int c;

  while ((c = getopt (argc, argv, (const char*)"s:")) != -1) {
    switch (c) {
      case 's':
        board_size = atoi(optarg);
        break;
      case '?':
        break;
      default:
        break;
    }
  }

  char* seed = (char*)malloc(8);
  scanf("%s", seed);
  size_t random_seed = *(size_t*)seed;
  free(seed);

  size_t data_length = (size_t)(board_size*board_size);
  size_t adjusted_data_length = ((data_length / sizeof(uint)) + 1);

  fprintf(stderr, "generating %d x %d board\n", board_size, board_size);
  fprintf(stderr, "\n");

  curandGenerator_t gen;

  ubyte *result_data = (ubyte*)malloc(data_length);
  uint *h_data = (uint*)malloc(adjusted_data_length * sizeof(uint));
  uint *d_data;  // memory for result
  checkCudaErrors( cudaMalloc((void**)&d_data, adjusted_data_length * sizeof(uint)) );

  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, random_seed) );
  checkCudaErrors( curandGenerate(gen, d_data, adjusted_data_length) );

  checkCudaErrors( cudaMemcpy(h_data, d_data, adjusted_data_length * sizeof(uint), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaFree(d_data) );

  fprintf(stderr, "seed: %zu\n", random_seed);
  fprintf(stderr, "data_length: %zu\n", data_length);
  fprintf(stderr, "adjusted_data_length: %zu\n", adjusted_data_length * sizeof(uint));

  ubyte *buffer = (ubyte*)malloc(sizeof(uint));

  for (int i = 0; i < adjusted_data_length; i++) {
    split_int(buffer, h_data[i]);
    for (int j = 0; j < sizeof(uint); j++) {
      result_data[i*sizeof(uint) + j] = buffer[j];
    }
  }

  int counter = 0;
  while (counter < data_length) {
    printf("%u", result_data[counter++]);
    if (counter % board_size == 0)
      printf("\n");
    else
      printf(",");
  }

  free(h_data);
  free(result_data);
  // printf("\n");
}
