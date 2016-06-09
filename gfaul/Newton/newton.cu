//Karin: No matter what i do, the program still crashes when it reaches the end of the main function, from my debugging I think it has something to do with the RNG of thrust, but im not entirely sure. 
//The code works and executes perfectly but like I said, cannot fathom the crash at the end :(

////////////////////////////////////////////////////////////////////////
// GPU version of Newton's method to calculate square roots
// Inefficient code to calculate square root of n random numbers using CURAND
////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
#define MAXVAL 100000   // values must be between 1 and MAXVAL - so there will be duplicates

__constant__ int iterations_G;			
__constant__ int n_G;	

__global__ void squareRoot_2D_2D ( int n, unsigned int *x, float *y, int iter )
// Applies Newton's method to compute the square root 
// of the appropriate int in x and places result in y
{
   int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   float inc;
   int c = (abs((int) x[i]) % MAXVAL) + 1; // change random number generated to a value between 1 and MAXVAL
   float r = c;
   if (i<n)   // to prevent excess processors from accessible out-of-bounds data
   { for(int j=0; j<iter; j++)   // iterations of Newton's method - based on parameter iter
        { // actual Newton's method
          inc = r + r;
          inc = (r*r - c)/inc;
          r = r - inc;
        }
	   x[i] = c;  // copy actual number used into x
       y[i] = r;  // write in result - i.e. square root of x value
   }
}

__global__ void squareRoot ( int n, unsigned int *x, float *y, int iter )
// Applies Newton's method to compute the square root 
// of the appropriate int in x and places result in y
{
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float inc;
   int c = (abs((int) x[i]) % MAXVAL) + 1; // change random number generated to a value between 1 and MAXVAL
   float r = c;
   if (i<n)   // to prevent excess processors from accessible out-of-bounds data
   { for(int j=0; j<iter; j++)   // iterations of Newton's method - based on parameter iter
        { // actual Newton's method
          inc = r + r;
          inc = (r*r - c)/inc;
          r = r - inc;
        }
	   x[i] = c;  // copy actual number used into x
       y[i] = r;  // write in result - i.e. square root of x value
   }
}


int main (int argc, char*argv[] )
{     int iterations = 10; // number of iterations for Newton's method
      int n = 1000000; //  dimension of data
	  int block_x = 8;
	  int block_y = 4;
      //int block = 1024; // block size
	  dim3 block(block_x,block_y);
	  //int grid = n / block + 1;  // round up number of blocks
	  dim3 grid(31250,1);

 // initialise card

	findCudaDevice(argc, (const char**) argv);   // from helper_cuda.h 
 
  // initialise CUDA timing

	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on host and device
		unsigned int *h_x = (unsigned int *)malloc(sizeof(unsigned int)*n);     
		float *h_y = (float *)malloc(sizeof(float)*n);     
	//	unsigned int *d_x;  // memory for initial numbers
	//	checkCudaErrors(cudaMalloc((void**)&d_x,sizeof(unsigned int)*n));
		float *d_y;  // memory for result
		checkCudaErrors(cudaMalloc((void**)&d_y,sizeof(float)*n));
    
	// random number generation
		cudaEventRecord(start);  // start timing

	//	curandGenerator_t gen;
	//	checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	//	checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
	//	checkCudaErrors( curandGenerate(gen, d_x, n) );
 
		//random number using thrust
		thrust::host_vector<unsigned int> h_vec (n);
		thrust::generate(h_vec.begin(), h_vec.end(), rand);
		
		thrust::device_vector<unsigned int> d_vec = h_vec;

		//d_x = thrust::raw_pointer_cast(&d_vec[0]);
 
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milli, start, stop);  // time random generation

		printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, n/(0.001*milli));
	
	
	
	// execute kernel and time it
	//	checkCudaErrors(cudaMemcpyToSymbol("iterations_G", &iterations, sizeof(int))); 	
	//	checkCudaErrors(cudaMemcpyToSymbol("n_G", &n, sizeof(int)));			
		
		cudaEventRecord(start); // start timing
		
		squareRoot_2D_2D<<<grid,block>>>(n ,thrust::raw_pointer_cast(&d_vec[0]), d_y, iterations);
		//squareRoot<<<grid,block>>>(n, d_x, d_y, iterations);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

		printf("Square root kernel execution time (ms): %f \n",milli);

		checkCudaErrors(cudaDeviceSynchronize());  // flush print queues

		// copy back results
		checkCudaErrors( cudaMemcpy(h_y, d_y, sizeof(float)*n, cudaMemcpyDeviceToHost) );	
		checkCudaErrors( cudaMemcpy(h_x, thrust::raw_pointer_cast(&d_vec[0]), sizeof(unsigned int)*n, cudaMemcpyDeviceToHost) );	  

      /*for (int k = 0; k < n; k++) // Really just for testing to check how close results are to actual square roots
      {
         printf("x = %u", h_x[k]);
         printf("  sqrt(x) = %f", h_y[k]);
         float z = h_y[k]*h_y[k];
         printf("  diff = %f\n", z - h_x[k]);
      } */

	 // Tidy up library

	//	checkCudaErrors( curandDestroyGenerator(gen) );

	// Release memory and exit cleanly
		printf("\nI am here!!\n");
		free(h_x);
		free(h_y);
	//	checkCudaErrors( cudaFree(d_vec) );
		checkCudaErrors( cudaFree(d_y) );

	// CUDA exit 
		printf("I am here!!\n");
		cudaDeviceReset();
		printf("I am here!!\n");
		
		return 0;
}

