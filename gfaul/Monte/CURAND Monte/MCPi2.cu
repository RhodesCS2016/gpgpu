//To Karin: This is my curand solution for monte carlo, it can't run much more than 500,000 threads without crashing, please see the comment at the top of the MCPi.cu file, 
//which is my (much more powerful) thrust solution to the same problem

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define SEED 35791246

__global__ void MC(unsigned int seed, int * z, int n)
{

	int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	curandState state;
	curand_init (seed, i, 1, &state);
	
	if (i < n)
	{	
		float x = curand_uniform(&state);
		float y = curand_uniform(&state);
		double result=x*x+y*y;

		z[i] = 0;
		if (result < 1.0)
			z[i] = 1;
	}
}


int main (int argc, char*argv[] )
{
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int niter=200000;
	int z;
	double pi;

	int block_x = 32;
	int block_y = 32;
      
	dim3 block(block_x,block_y);
	 
	dim3 grid(32,16);
	
	unsigned int *h_z = (unsigned int *)malloc(sizeof(unsigned int)*niter);     
	
	int *d_z;  
	checkCudaErrors(cudaMalloc((void**)&d_z,sizeof(int)*niter));
	
	cudaEventRecord(start); // start timing
	
	MC<<<grid,block>>>(SEED, d_z, niter);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

	printf("Kernel execution time (ms): %f \n",milli);
	
	checkCudaErrors(cudaDeviceSynchronize());
	
	checkCudaErrors( cudaMemcpy(h_z, d_z, sizeof(float)*niter, cudaMemcpyDeviceToHost) );
	
	int cnt = 0;
	for (int q = 0; q<niter; q++)
		if(h_z[q]==1)
			cnt++;
	
	thrust::device_vector<int> d_vec(d_z, d_z+niter);
	z = thrust::reduce(d_vec.begin(), d_vec.end());
	
	pi = (double)z/niter*4;
	printf("Pi = %f with %d iterations, the kernel cannot do more than 500000 threads without crashing\n",pi,niter);
	
	checkCudaErrors( cudaFree(d_z) );
}