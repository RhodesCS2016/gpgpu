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
__global__ void MCPi (float *z, unsigned int *C)
//Use the monte carlo method to find the radius and store the values in a double block sized array.
{
	int i = blockIdx.x*blockDim.x + threadIdx.x; //get global thread ID
	if (i < LPR) // Check that it is within the bounds, if not its completely useless
	{
		int count = 0; //initialise counter
		for( int j = i; j < ITTER; j = j + LPR)
		{
			float x = z[j]; // get x value
			float y = z[j+ITTER]; // get y value

			if (x*x + y*y <= 1) count++; //add to the counter if length is less than 1
		}
		
		C[i] = count;
	}
}
//////////////////////////////////////////////////////////
//PI FINDER
//////////////////////////////////////////////////////////
__global__ void PiFind (unsigned int *C, unsigned int *pi, int len)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x; //get array ID
	printf("%d\n", len);
	if(i < len/2){
		
		C[i] = C[i] + C[i + gridDim.x/2];
		printf("%d %d\n", i, C[i]);
		if(i ==0 && len%2 == 1) C[i] = C[i] + C[len-1];

		int k = i%blockDim.x; //threadIdx.x;

		int x = blockDim.x/2; //get block size for summing 
		while(x >= 1 && k < x)
		{
			if(i+x < len/2)
			{
				C[i] += C[i+x];
				__threadfence_block();
				printf("%d %d\n", i, C[i]);
			}
			x = x/2;
		} //sum all counts in a block

		if( k == 0 ) pi[blockIdx.x] = C[i]; //add block sum to total sum array
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
	float *d_z;  // memory for initial numbers
	checkCudaErrors(cudaMalloc((void**)&d_z,sizeof(float)*n)); //essentially a matrix of ITTER*2 size
	unsigned int *d_d;  // memory for block sums array
	checkCudaErrors(cudaMalloc((void**)&d_d,sizeof(unsigned int)*LPR)); //so that each block has its own array element to update
    ///////////////////////////////////////////////////////
	// random number generation
	cudaEventRecord(start);  // start timing

	curandGenerator_t gen;
	checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
	checkCudaErrors( curandGenerateUniform(gen, d_z, n) );

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // time random generation

	printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n", milli, n/(0.001*milli));
	////////////////////////////////////////////////////////
	// execute kernel and time it
	cudaEventRecord(start); // start timing

	MCPi<<<grid,block>>>(d_z, d_d);
	///////////////
	unsigned int *h_d = (unsigned int *) malloc(sizeof(unsigned int *)*LPR); 
	checkCudaErrors( cudaMemcpy(h_d, d_d, sizeof(unsigned int)*LPR, cudaMemcpyDeviceToHost) );
	for(int j = 0; j < LPR; j++) {
        printf("%d ", h_d[j]);
    }
    printf("\n");
    ////////////////

    int arrsize = (LPR/2 + block - 1)/block; 
    printf("%d \n", arrsize);

    while (arrsize > 1)
	{
		unsigned int *d_pi;  // memory for reduced pi sum
		checkCudaErrors(cudaMalloc((void**)&d_pi,sizeof(unsigned int)*arrsize));

		int len = sizeof(d_d)/sizeof(d_d[0]);
		PiFind<<<arrsize,block>>>(d_d, d_pi, len);
		////////////////////////////
		unsigned int *h_pi = (unsigned int *) malloc(sizeof(unsigned int *)*arrsize); 
		checkCudaErrors( cudaMemcpy(h_pi, d_pi, sizeof(unsigned int)*arrsize, cudaMemcpyDeviceToHost) );	
		for(int j = 0; j <= arrsize; j++) {
	    	printf("%d ", h_pi[j]);
		}
		printf("\n");
		//////////////////////////////
		
		checkCudaErrors( cudaFree(d_d) );
		unsigned int *d_d;  // memory for block sums array
		checkCudaErrors(cudaMalloc((void**)&d_d,sizeof(unsigned int)*arrsize));
		checkCudaErrors( cudaFree(d_pi));
		arrsize = (LPR/2 + block - 1)/block;
		printf("%d \n", arrsize);
	}

	unsigned int *d_pi;  // memory for reduced pi sum
	checkCudaErrors(cudaMalloc((void**)&d_pi,sizeof(unsigned int)));

	int len = sizeof(d_d)/sizeof(d_d[0]);
	PiFind<<<arrsize,block>>>(d_d, d_pi,len);
	////////////////////////////
	unsigned int *h_print = (unsigned int *) malloc(sizeof(unsigned int *)); 
	checkCudaErrors( cudaMemcpy(h_print, d_pi, sizeof(unsigned int), cudaMemcpyDeviceToHost) );	
	printf("%d \n", h_print[0]);
	//////////////////////////////
	
	checkCudaErrors( cudaFree(d_d) );
	unsigned int *d_last;  // memory for block sums array
	checkCudaErrors(cudaMalloc((void**)&d_last,sizeof(unsigned int)));
	checkCudaErrors( cudaFree(d_pi));

	unsigned int *h_pi = (unsigned int *) malloc(sizeof(unsigned int *)); 
	checkCudaErrors( cudaMemcpy(h_pi, d_last, sizeof(unsigned int), cudaMemcpyDeviceToHost) );	
	float c = (float) h_pi[0];
	checkCudaErrors( cudaFree(d_last) );
	free(h_pi);

	float pi = (c/ITTER)*4;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution

	checkCudaErrors(cudaDeviceSynchronize());  // flush print queues
	///////////////////////////////////////////////////////////
	//print results
	printf("Monte Carlo Pi kernel execution time (ms): %f \n Pi Approximation: %f \n",milli, pi);
	/////////////////////////////////////////////////////////////
	// Tidy up library
	checkCudaErrors( curandDestroyGenerator(gen) );
	////////////////////////////////////////////////////////////
	// Release memory and exit cleanly
	checkCudaErrors( cudaFree(d_z) );
	////////////////////////////////////////////////////////////
	// CUDA exit 
	cudaDeviceReset();
	////////////////////////////////////////////////////////////
}