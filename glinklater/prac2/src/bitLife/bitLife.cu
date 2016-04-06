
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <utility>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "../mylib.h"

/*
 * Helper Kernels
 */

/// CUDA kernel that encodes byte-per-cell data to bit-per-cell data.
/// Needs to be invoked for each byte in encoded data (cells / 8).

/*
 * bitLifeEncodeKernel
 * Encode the life data of 8 cells into a single byte
 */
__global__ void bitLifeEncodeKernel(
  const ubyte* lifeData,
  size_t encWorldSize,
  ubyte* resultEncodedLifeData
) {
 	for (size_t outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
 			outputBucketId < encWorldSize;
 			outputBucketId += blockDim.x * gridDim.x) {

 		size_t cellId = outputBucketId << 3;

 		ubyte result = lifeData[cellId] << 7 | lifeData[cellId + 1] << 6
      | lifeData[cellId + 2] << 5	| lifeData[cellId + 3] << 4
      | lifeData[cellId + 4] << 3 | lifeData[cellId + 5] << 2
 			| lifeData[cellId + 6] << 1 | lifeData[cellId + 7];

 		resultEncodedLifeData[outputBucketId] = result;
 	}
}

/// Runs a kernel that encodes byte-per-cell data to bit-per-cell data.
void runBitLifeEncodeKernel(const ubyte* d_lifeData, world *gameWorld, ubyte* d_encodedLife) {

 	assert(gameWorld->worldWidth % 8 == 0);
 	size_t worldEncDataWidth = gameWorld->worldWidth / 8;
 	size_t encWorldSize = worldEncDataWidth * gameWorld->worldHeight;

 	ushort threadsCount = 256;
 	assert(encWorldSize % threadsCount == 0);
 	size_t reqBlocksCount = encWorldSize / threadsCount;
 	ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

 	bitLifeEncodeKernel<<<blocksCount, threadsCount>>>(d_lifeData, encWorldSize, d_encodedLife);
 	checkCudaErrors(cudaDeviceSynchronize());
}

/// CUDA kernel that decodes data from bit-per-cell to byte-per-cell format.
/// Needs to be invoked for each byte in encoded data (cells / 8).

/*
 * bitLifeDecodeKernel
 * Decode the life data of 8 cells contained in a single byte into a eight
 * separate bytes.
 */
__global__ void bitLifeDecodeKernel(
    const ubyte* encodedLifeData,
    uint encWorldSize,
    ubyte* resultDecodedlifeData
  ) {

 	for (uint outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
 			outputBucketId < encWorldSize;
 			outputBucketId += blockDim.x * gridDim.x) {

 		uint cellId = outputBucketId << 3;
 		ubyte dataBucket = encodedLifeData[outputBucketId];

 		resultDecodedlifeData[cellId] = dataBucket >> 7;
 		resultDecodedlifeData[cellId + 1] = (dataBucket >> 6) & 0x01;
 		resultDecodedlifeData[cellId + 2] = (dataBucket >> 5) & 0x01;
 		resultDecodedlifeData[cellId + 3] = (dataBucket >> 4) & 0x01;
 		resultDecodedlifeData[cellId + 4] = (dataBucket >> 3) & 0x01;
 		resultDecodedlifeData[cellId + 5] = (dataBucket >> 2) & 0x01;
 		resultDecodedlifeData[cellId + 6] = (dataBucket >> 1) & 0x01;
 		resultDecodedlifeData[cellId + 7] = dataBucket & 0x01;
 	}
}


/// Runs a kernel that decodes data from bit-per-cell to byte-per-cell format.
void runBitLifeDecodeKernel(const ubyte* d_encodedLife, world *gameWorld, ubyte* d_lifeData) {

 	assert(gameWorld->worldWidth % 8 == 0);
 	uint worldEncDataWidth = gameWorld->worldWidth / 8;
 	uint encWorldSize = worldEncDataWidth * gameWorld->worldHeight;

 	ushort threadsCount = 256;
 	assert(encWorldSize % threadsCount == 0);
 	uint reqBlocksCount = encWorldSize / threadsCount;
 	ushort blocksCount = ushort(std::min(32768u, reqBlocksCount));

 	// decode life data back to byte per cell format
 	bitLifeDecodeKernel<<<blocksCount, threadsCount>>>(d_encodedLife, encWorldSize, d_lifeData);
 	checkCudaErrors(cudaDeviceSynchronize());
}

/*
 * bitLife Kernel
 * Compute array and bit offsets required to access all cells in the Moore
 * Neighbourhood and determine the result state of the cell under evaluation.
 *
 * This kernel is executed once per iteration by as many threads and blocks as
 * required to complete the iteration.
 *
 * The number of bytes worth of cell data that each thread processes is
 * variable.
 */
__global__ void bitLifeKernel(
    const ubyte* lifeData,
    uint worldDataWidth,
    uint worldHeight,
    uint bytesPerThread,
    ubyte* resultLifeData) {

  uint worldSize = (worldDataWidth * worldHeight);

  for (uint cellId = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x)
      * bytesPerThread;
      cellId < worldSize;
      cellId += blockDim.x * gridDim.x * bytesPerThread) {

    // Calculate data offsets
    // Start at block x - 1.
    uint x = (cellId + worldDataWidth - 1) % worldDataWidth;
    uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
    uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
    uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

    // Initialize data with previous byte and current byte.
    uint data0 = (uint)lifeData[x + yAbsUp] << 16;
    uint data1 = (uint)lifeData[x + yAbs] << 16;
    uint data2 = (uint)lifeData[x + yAbsDown] << 16;

    x = (x + 1) % worldDataWidth;
    data0 |= (uint)lifeData[x + yAbsUp] << 8;
    data1 |= (uint)lifeData[x + yAbs] << 8;
    data2 |= (uint)lifeData[x + yAbsDown] << 8;

    for (uint i = 0; i < bytesPerThread; ++i) {
      // get the bit coordinate of the cell under evaluation.
      uint oldX = x;  // old x is referring to current center cell
      x = (x + 1) % worldDataWidth;

      // extract state of the cell under evaluation.
      data0 |= (uint)lifeData[x + yAbsUp];
      data1 |= (uint)lifeData[x + yAbs];
      data2 |= (uint)lifeData[x + yAbsDown];

      // evaluate cell iteratively.
      uint result = 0;
      for (uint j = 0; j < 8; ++j) {
        uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000)
          + (data2 & 0x14000);
        aliveCells >>= 14;
        aliveCells = (aliveCells & 0x3) + (aliveCells >> 2)
          + ((data0 >> 15) & 0x1u) + ((data2 >> 15) & 0x1u);

        result = result << 1
          | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u))
          ? 1u
          : 0u);

        data0 <<= 1;
        data1 <<= 1;
        data2 <<= 1;
      }

      // write result
      resultLifeData[oldX + yAbs] = result;
    }
  }
}

/// Runs a kernel that evaluates given world of bit-per-cell density using algorithm specified by parameters.
bool runBitLifeKernel(
    ubyte *&d_encodedLifeData,
    ubyte *&d_encodedlifeDataBuffer,
		world *gameWorld,
    size_t iterationsCount,
    ushort threadsCount,
    uint bytesPerThread
  ) {

	// World has to fit into 8 bits of every byte exactly.
	if (gameWorld->worldWidth % 8 != 0) {
    fprintf(stderr, "World has to fit into 8 bits of every byte exactly.\n");
		return false;
	}

	size_t worldEncDataWidth = gameWorld->worldWidth / 8;
	if (worldEncDataWidth % bytesPerThread != 0) {
    fprintf(stderr, "bytesPerThread must align with world size.\n");
		return false;
	}

	size_t encWorldSize = worldEncDataWidth * gameWorld->worldHeight;
	if (encWorldSize > std::numeric_limits<uint>::max()) {
    fprintf(stderr, "World is too big to fit into a uint\n");
		return false;
	}

	if ((encWorldSize / bytesPerThread) % threadsCount != 0) {
    fprintf(stderr, "Number of threads must align with world size and bytesPerThread.\n");
		return false;
	}

	size_t reqBlocksCount = (encWorldSize / bytesPerThread) / threadsCount;
	ushort blocksCount = ushort(std::min(size_t(32768), reqBlocksCount));

  // exec kernel
  for (size_t i = 0; i < iterationsCount; ++i) {
    bitLifeKernel<<<blocksCount, threadsCount>>>(
      d_encodedLifeData,
      uint(worldEncDataWidth),
      uint(gameWorld->worldHeight),
      bytesPerThread,
      d_encodedlifeDataBuffer
    );
    std::swap(d_encodedLifeData, d_encodedlifeDataBuffer);
  }

	checkCudaErrors(cudaDeviceSynchronize());
  fprintf(stderr, "bitLife Kernel executed successfully.\n");
	return true;
}

bool fullBitLifeKernel (board *gameBoard, size_t iterationsCount, ushort threadsCount, uint bytesPerThread, float *milli) {
  world *gameWorld = gameBoard->_world;
  ubyte *d_encodedData;
  ubyte *d_encodedDataBuffer;
  ubyte *d_data;
  uint worldEncDataWidth = gameWorld->worldWidth / 8;
 	uint encWorldSize = worldEncDataWidth * gameWorld->worldHeight;

  checkCudaErrors(cudaMalloc((ubyte**)&d_data, gameWorld->dataLength));
  checkCudaErrors(cudaMemset(d_data, 0, gameWorld->dataLength));
  checkCudaErrors(cudaMemcpy(d_data, gameBoard->data, gameWorld->dataLength, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((ubyte**)&d_encodedData, encWorldSize));
  checkCudaErrors(cudaMemset(d_encodedData, 0, encWorldSize));

  runBitLifeEncodeKernel(d_data, gameWorld, d_encodedData);

  checkCudaErrors(cudaFree(d_data));

  checkCudaErrors(cudaMalloc((ubyte**)&d_encodedDataBuffer, encWorldSize));
  checkCudaErrors(cudaMemset(d_encodedDataBuffer, 0, encWorldSize));

  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start);  // start timing

  bool ret = runBitLifeKernel(d_encodedData, d_encodedDataBuffer, gameWorld, iterationsCount, threadsCount, bytesPerThread);

  cudaEventRecord(stop);  // stop timing
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(milli, start, stop);

  checkCudaErrors(cudaFree(d_encodedDataBuffer));

  checkCudaErrors(cudaMalloc((ubyte**)&d_data, gameWorld->dataLength));
  checkCudaErrors(cudaMemset(d_data, 0, gameWorld->dataLength));

  runBitLifeDecodeKernel(d_encodedData, gameWorld, d_data);

  checkCudaErrors(cudaFree(d_encodedData));

  checkCudaErrors(cudaMemcpy(gameBoard->data, d_data, gameWorld->dataLength, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_data));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return ret;
}

/*
 * Main
 */

board *gameBoard;
world *gameWorld;
FILE *out_file;

int main (int argc, char **argv)
{
  int iterations = 10000;
  ushort threadsCount = 64;
  char *in_filename = NULL;
  char *out_filename = NULL;
  size_t board_size = 48;
  size_t bytesPerThread = 8;

  float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  gameBoard = (board*)malloc(sizeof(board));
  gameWorld = (world*)malloc(sizeof(world));

  opterr = 0;
  int c;

  while ((c = getopt (argc, argv, (const char*)"o:f:i:t:s:b:")) != -1) {
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
      case 'b':
        bytesPerThread = (size_t)atoi(optarg);
        break;
      case 's':
        board_size = atoi(optarg);
        break;
      case '?':
        break;
      default:
        break;
    }
  }

  // printf("iterations: %d\n", iterations);
  // printf("in_file: %s\n", in_filename);
  // printf("out_file: %s\n", out_filename);
  // printf("threadsCount: %u\n", threadsCount);
  // printf("\n");

  if (!in_filename) {
    printf("Please specify a board file\n");
    exit(1);
  }

  initWorld(board_size, board_size, gameWorld);
  initBoard(fopen(in_filename, "r"), gameBoard, gameWorld);
  if (out_filename) out_file = fopen(out_filename, "w+");

  fullBitLifeKernel(gameBoard, iterations, threadsCount, bytesPerThread, &milli);

  reportTime(iterations, board_size, threadsCount, milli);

  // checkCudaErrors(cudaMemcpy(gameBoard->data, d_data, BOARD_BYTES, cudaMemcpyDeviceToHost));
  // printBoard(gameBoard->data, gameWorld);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(gameBoard->data);
  free(gameBoard->resultData);
  if (out_filename) fclose(out_file);
  // printf("\n");
  checkCudaErrors(cudaDeviceReset());
}
