#include <cuda_runtime.h>
#include <stdio.h>


/* Naive kernel for transposing a rectangular host array. */

const int TILE_DIM = 32;

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


void initialData(float *in,  const int size)
{  // initialise matrix
    for (int i = 0; i < size; i++)
   	 {   in[i] = (float)(rand() & 0xFF) / 10.0f;    }
    return;
}

void printData(float *in,  const int size)
{  // print matrix
    for (int i = 0; i < size; i++)
   	 {   printf("%3.0f ", in[i]);     }
    printf("\n");
    return;
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols)
{  // check that transposed matrix is correct
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int index = i*cols + j;
            if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
                match = 0;
                printf("different on (%d, %d) (offset=%d) element in transposed matrix: host %f gpu %f\n", i, j,  index, hostRef[index], gpuRef[index]);
                break;
            }
        }
        if (!match) break;
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nrows, const int ncols)
{  // transpose using CPU
    for (int iy = 0; iy < ncols; ++iy)
    {
     for (int ix = 0; ix < nrows; ++ix)
    	 {   out[ix * ncols + iy] = in[iy * nrows + ix];     }
    }
}

__global__ void justcopy(float *out, float *in, const int nrows, const int ncols)
{  // routine to copy data from one matrix to another -- no transposition done
    // get matrix coordinate (ix,iy)
     unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // copy data as is with boundary test
    if (ix < nrows && iy < ncols)
  	  { out[ix * ncols + iy] = in[ix * ncols + iy];  }
}

__global__ void naivetranspose(float *out, float *in, const int nrows, const int ncols)
{  // naive routine to transpose a matrix -- no optimisations considered
    // get matrix coordinate (ix,iy)
     unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
     unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose with boundary test
    if (ix < nrows && iy < ncols)
  	  { out[ix * ncols + iy] = in[iy * nrows + ix]; }
}

__global__ void lessnaivetranspose(float *out, float *in, const int nrows, const int ncols)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.x + threadIdx.y;
  unsigned int width = gridDim.x * blockDim.x;

  for (int j=0; j<blockDim.x; j+=blockDim.y) {
    out[x*width + (y+j)] = in[(y+j)*width + x];
  }
}

__global__ void opttranspose(float *out, float *in, const int nrows, const int ncols)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.x + threadIdx.y;
  int width = gridDim.x * blockDim.x;

  for (int j = 0; j < blockDim.x; j += blockDim.y)
     tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * blockDim.x + threadIdx.x;  // transpose block offset
  y = blockIdx.x * blockDim.x + threadIdx.y;

  for (int j = 0; j < blockDim.x; j += blockDim.y)
     out[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initialise CUDA timing
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    bool iprint = 0;

    // set up array size 1024
    int nrows = 1 << 10;
    int ncols = 1 << 10;

    int blockx = 16;
    int blocky = 16;

    // interpret command line arguments if present
    if (argc > 1) iprint = atoi(argv[1]);

    if (argc > 2) blockx  = atoi(argv[2]);

    if (argc > 3) blocky  = atoi(argv[3]);

    if (argc > 4) nrows  = atoi(argv[4]);

    if (argc > 5) ncols  = atoi(argv[5]);

    printf(" with matrix nrows %d ncols %d\n", nrows, ncols);
    size_t ncells = nrows * ncols;
    size_t nBytes = ncells * sizeof(float);

    // execution configuration
    dim3 block (blockx, blocky);
    dim3 grid  ((nrows + block.x - 1) / block.x, (ncols + block.y - 1) / block.y);
    dim3 grid_opt (nrows/block.x, ncols/block.x);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nrows * ncols);

    //  transpose at host side
    transposeHost(hostRef, h_A, nrows, ncols);

    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // execute justcopy kernel
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    cudaEventRecord(start); // start timing
    justcopy<<<grid, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nrows * ncols);

    float ibnd = 2 * ncells * sizeof(float) /  	(1024.0 * 1024.0 * 1024.0) / (milli/1000);  // convert bytes and millisec to GB/sec
    // ibnd = 2 * ncells * sizeof(float) / 1e9 / milli/1000;
    printf("justcopy kernel elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> effective bandwidth %f GB/s\n", milli, grid.x, grid.y, block.x, block.y, ibnd);

    // execute naive transpose kernel
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    cudaEventRecord(start); // start timing
    // naivetranspose<<<grid, block>>>(d_C, d_A, nrows, ncols);
    // lessnaivetranspose<<<grid_opt, block>>>(d_C, d_A, nrows, ncols);
    opttranspose<<<grid_opt, block>>>(d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    ibnd = 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / (milli/1000);
    printf("optimised transpose elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> effective bandwidth %f GB/s\n", milli, grid_opt.x, grid_opt.y, block.x, block.y, ibnd);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
