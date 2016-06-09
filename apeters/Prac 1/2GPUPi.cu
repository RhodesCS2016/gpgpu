////////////////////////////////////////////////////////////
// GPU Monte Carlo Approximation of Pi.
// The process assumes a circle of radius 1 with centre at 1,1
// Therefore the square has a length of 2.
// AUTHOR: Antonio Peters
////////////////////////////////////////////////////////////
//HEADERS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>
///////////////////////////////////////////////////////////
//Thrust
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <cstdio>
///////////////////////////////////////////////////////////
//SOME CONSTANT DEFINITIONS
using namespace std;
#define ITTER 	40	//number of iterations
#define LOOP 	1				//number of loops per thread
#define LPR 	(ITTER + LOOP -1)/LOOP		//Number of itterations needed for a given loop size
#define n 		2*ITTER //array length for the random numbers
#define block 	16 //block dimension
#define grid 	LPR/block +1 //grid dimension with overflow
///////////////////////////////////////////////////////////
//RADIUS CALCULATOR
///////////////////////////////////////////////////////////
__global__ void MCPi (float *z, int *C)
//Use the monte carlo method to find the radius and store the values in a double block sized array.
{
	int i = blockIdx.x*blockDim.x + threadIdx.x; //get global thread ID
	if (i < LPR) // Check that it is within the bounds, if not its completely useless
	{
		int count = 0; //initialise counter
		for( int j = i; j < ITTER; j = j + LPR)
		{
			float x = z[j]; // get x value
			printf("%f\n", x);
			float y = z[j+ITTER]; // get y value
			printf("%f\n", y);
			
			if (x*x + y*y <= 1) count++; //add to the counter if length is less than 1
		}
		
		C[i] = count;
	}
}
//////////////////////////////////////////////////////////
//MAIN PROGRAM (a lot of this was taken from newton.cu)
//////////////////////////////////////////////////////////
int main(int argc, char const *argv[])
{	
	//////////////////////////////////////////////////////
	// initialise card
	findCudaDevice(argc, (const char**) argv);
	//////////////////////////////////////////////////////
	// initialise CUDA timing
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//////////////////////////////////////////////////////
	// allocate memory on device
	//float *d_z;  // memory for initial numbers
	//checkCudaErrors(cudaMalloc((void**)&d_z,sizeof(float)*n)); //essentially a matrix of ITTER*2 size
	///////////////////////////////////////////////////////
	// random number generation
	cudaEventRecord(start);  // start timing

	thrust::device_vector<float> d_rand(n,0);
	thrust::generate(d_rand.begin(), d_rand.end(),rand);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // time random generation

	printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n", milli, n/(0.001*milli));
	////////////////////////////////////////////////////////
	// execute kernel and time it
	cudaEventRecord(start); // start timing

	float * d_z = thrust::raw_pointer_cast(d_rand.data());

	printf("Step 1\n");

	int *d_d;  // memory for block sums array
	checkCudaErrors(cudaMalloc((void**)&d_d,sizeof(int)*LPR)); //so that each block has its own array element to update
    
    MCPi<<<grid,block>>>(d_z, d_d);
	///////////////
	unsigned int *h_d = (unsigned int *) malloc(sizeof(unsigned int *)*LPR); 
	checkCudaErrors( cudaMemcpy(h_d, d_d, sizeof(int)*LPR, cudaMemcpyDeviceToHost) );
	for(int j = 0; j < LPR; j++) {
        printf("%d ", h_d[j]);
    }
    printf("\n");
    ////////////////
	printf("Step 2\n");

	thrust::device_vector<int> d_vec (d_d, d_d + LPR);
	
	printf("Step 3\n");

	float sum = (float) thrust::reduce(d_vec.begin(), d_vec.end(), (int) 0, thrust::plus<int>());

	printf("Step 4\n");

	float pi = 4*(sum/ITTER);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

	checkCudaErrors(cudaDeviceSynchronize());  // flush print queues
	///////////////////////////////////////////////////////////
	//print results
	printf("Monte Carlo Pi kernel execution time (ms): %f \n Pi Approximation: %f \n",milli, pi);
	////////////////////////////////////////////////////////////
	// Release memory and exit cleanly
	checkCudaErrors( cudaFree(d_z) );
	////////////////////////////////////////////////////////////
	// CUDA exit 
	cudaDeviceReset();
	////////////////////////////////////////////////////////////
}