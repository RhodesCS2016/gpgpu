#include "cuda_runtime.h"
// To Karin: I implemented this solution using the thrust libruary for RNG in the kernel, i was able to run 250,000,000 without crashing the graphics card, it seems to be a far more powerful 
//librurary than CURAND, i have implemented a CURAND solution as well, its in the file MCPi2.cu 

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
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#define SEED 35791246

__global__ void MC_thrust (int * z, int n)
{
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	if (i<n)
	{
		thrust::default_random_engine rng;
		
		rng.discard(i);
		
		thrust::uniform_real_distribution<float> rNum(0,1);
		
		float x = rNum(rng);
		float y = rNum(rng);
		
		float point = x*x+y*y;
		z[i] = 0;
		//printf("%f\n",point);
		if (point<=1)
			z[i] = 1;
	}
	
}

int main (int argc, char*argv[] )
{
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int niter=250000000;
	int z;
	double pi;

	int block_x = 32;
	int block_y = 32;
      
	dim3 block(block_x,block_y);
	 
	dim3 grid(1024,256);
	
	//unsigned int *h_z = (unsigned int *)malloc(sizeof(unsigned int)*niter);     
	
	int *d_z;  
	checkCudaErrors(cudaMalloc((void**)&d_z,sizeof(int)*niter));
	
	//########################################################Monte Carlo using thrust##########################################################
	cudaEventRecord(start); // start timing
	
	MC_thrust<<<grid,block>>>(d_z, niter);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

	printf("Kernel execution time (ms): %f \n",milli);
	
	checkCudaErrors(cudaDeviceSynchronize());
	
	//checkCudaErrors( cudaMemcpy(h_z, d_z, sizeof(float)*niter, cudaMemcpyDeviceToHost) );
		
	thrust::device_vector<int> d_vec(d_z, d_z+niter);
	z = thrust::reduce(d_vec.begin(), d_vec.end());
	
	pi = (double)z/niter*4;
	printf("Pi = %f with %d iterations\n",pi, niter);
	
	checkCudaErrors( cudaFree(d_z) );
}