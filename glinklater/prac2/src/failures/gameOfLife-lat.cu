
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <utility>
#include <algorithm>
#include <cuda_runtime.h>
#include <queue>
#include "helper_cuda.h"
#include "mylib.h"

typedef std::queue<cudaStream_t*> StreamQueue;

__global__ void simpleLifeKernel(const ubyte *lifeData, uint worldWidth, uint worldSize, ubyte *resultLifeData) {
  for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x) {
    uint x = cellId % worldWidth;
    uint yAbs = cellId - x;
    uint xLeft = (x + worldWidth - 1) % worldWidth;
    uint xRight = (x + 1) % worldWidth;
    uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
    uint yAbsDown = (yAbs + worldWidth) % worldSize;

    uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
      + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
      + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

    resultLifeData[x + yAbs] =
      aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
  }
}

// populate a queue with specified number of non-blocking streams.
void setupCUDAStreams(StreamQueue *stream_q, int stream_count) {
  for (int i = 0; i < stream_count; i++) {
    cudaStream_t tmp;
    cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking);
    stream_q->push(&tmp);
  }
}

// get the next stream from the queue (round-robin)
cudaStream_t *getNextCUDAStream(StreamQueue *stream_q) {
  cudaStream_t *next = stream_q->front();
  stream_q->pop();
  stream_q->push(next);
  return next;
}

void runSimpleLifeKernel(
  ubyte *d_lifeData,
  ubyte *d_lifeDataBuffer,
  size_t worldWidth,
  size_t worldHeight,
  size_t iterationsCount,
  ushort threadsCount,
  StreamQueue *stream_q)
{
  assert((worldWidth * worldHeight) % threadsCount == 0);
  size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
  ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

  for (size_t i = 0; i < iterationsCount; ++i) {
    simpleLifeKernel<<<blocksCount, threadsCount, 0, *getNextCUDAStream(stream_q)>>>(d_lifeData, worldWidth, worldWidth*worldHeight, d_lifeDataBuffer);
    std::swap(d_lifeData, d_lifeDataBuffer);
  }
}

world *gameWorld;
board *gameBoard;
FILE *out_file;

int main (int argc, char **argv)
{
  int iterations = 10000;
  ushort threadsCount = 64;
  char *in_filename = NULL;
  char *out_filename = NULL;
  int stream_count = 1;
  size_t board_size = 48;

  float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  gameBoard = (board*)malloc(sizeof(board));

  opterr = 0;
  int c;

  while ((c = getopt (argc, argv, (const char*)"o:f:i:t:s:x:")) != -1) {
    switch (c) {
      case 'i':
        iterations = atoi(optarg);
        break;
      case 'f':
        in_filename = optarg;
        break;
      case 'o':
        out_filename = optarg;
        break;
      case 't':
        threadsCount = atoi(optarg);
        break;
      case 's':
        board_size = atoi(optarg);
        break;
      case 'x':
        stream_count = atoi(optarg);
        break;
      case '?':
        break;
      default:
        break;
    }
  }

  printf("iterations: %d\n", iterations);
  printf("in_file: %s\n", in_filename);
  printf("out_file: %s\n", out_filename);
  printf("threadsCount: %u\n", threadsCount);
  printf("\n");

  if (!in_filename) {
    printf("Please specify a board file\n");
    exit(1);
  }

  initWorld(board_size, board_size, gameWorld);
  initBoard(fopen(in_filename, "r"), gameBoard, gameWorld);
  if (out_filename) out_file = fopen(out_filename, "w+");

  ubyte *d_data;
  ubyte *d_resultData;
  StreamQueue stream_q;
  setupCUDAStreams(&stream_q, stream_count);

  checkCudaErrors(cudaMalloc((ubyte**)&d_data, gameWorld->dataLength));
  checkCudaErrors(cudaMemset(d_data, 0, gameWorld->dataLength));
  checkCudaErrors(cudaMemcpy(d_data, gameBoard->data, gameWorld->dataLength, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((ubyte**)&d_resultData, gameWorld->dataLength));
  checkCudaErrors(cudaMemset(d_resultData, 0, gameWorld->dataLength));

  cudaEventRecord(start);  // start timing
  runSimpleLifeKernel(d_data, d_resultData, board_size, board_size, iterations, threadsCount, &stream_q);
  cudaEventRecord(stop);  // stop timing
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);  //

  printf("Elapsed Time for %d iterations: %f", iterations, milli);

  checkCudaErrors(cudaMemcpy(gameBoard->data, d_data, gameWorld->dataLength, cudaMemcpyDeviceToHost));
  printBoard(gameBoard->data, gameWorld);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(gameBoard->data);
  free(gameBoard->resultData);
  if (out_filename) fclose(out_file);
  printf("\n");
  checkCudaErrors(cudaDeviceReset());
}
