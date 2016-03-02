////////////////////////////////////////////////////////////////////////
// GPU version of Newton's method to calculate square roots
// Inefficient code to calculate square root of n random numbers using CURAND
////////////////////////////////////////////////////////////////////////

#include "device_launch_parameters.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;
#define MAXVAL 100000   // values must be between 1 and MAXVAL - so there will be duplicates
#define SEED 76543210

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


int main (int argc, char **argv)
{
   int dimx = 1024; // default block size if no params.
   int dimy = 1; 
   int iterations = 10; // number of iterations for Newton's method
   int n = 1000000; //  dimension of data
   int use_thrust = 1;
   int curand_func = 1; // 1 = xorwow, 2 = philox4, 3 = quasi_sobol32

   // int block = 32; // block size

   // initialise card

   // findCudaDevice(argc, (const char**) argv);   // from helper_cuda.h 

   // get args

   // if (argc > 2) 
   // {
   //    dimx = atoi(argv[1]);
   //    dimy = atoi(argv[2]);
   //    if (argc > 3)
   //       iterations = atoi(argv[3]);
   // }

   opterr = 0;
   int c;

   while ((c = getopt (argc, argv, (const char*)"x:y:i:n:t123")) != -1) {
      switch (c) {
         case '1':
            curand_func = 1;
            break;
         case '2':
            curand_func = 2;
            break;
         case '3':
            curand_func = 3;
            break;
         case 'x':
            dimx = atoi(optarg);
            break;
         case 'y':
            dimy = atoi(optarg);
            break;
         case 'i':
            iterations = atoi(optarg);
            break;
         case 'n':
            n = atoi(optarg);
         case 't':
            use_thrust = 1;
         case '?':
            break;
         default:
            break;
         }
      }

   char* rng; // name of random method
   switch (curand_func)
   {
      case 2:
         rng = "philox4";
         break;
      case 3:
         rng = "quasi_sobol32";
         break;
      default:
         rng = "xorwow";
         break;
   }

   int gridn = n / (dimx * dimy) + 1;  // round up number of blocks

   // printf("%d; %d; %d; %d\n", dimx, dimy, iterations, n);

   dim3 block(dimx, dimy);
   dim3 grid(gridn, 1);
 
   // initialise CUDA timing

   float rand_milli;
   float sqrt_milli;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // allocate memory on host and device
   unsigned int *h_x = (unsigned int *)malloc(sizeof(unsigned int)*n);     
   float *h_y = (float *)malloc(sizeof(float)*n);     
   unsigned int *d_x;  // memory for initial numbers
   checkCudaErrors(cudaMalloc((void**)&d_x,sizeof(unsigned int)*n));
   float *d_y;  // memory for result
   checkCudaErrors(cudaMalloc((void**)&d_y,sizeof(float)*n));
   
   // random number generation
   cudaEventRecord(start);  // start timing

   curandGenerator_t gen;

   if (curand_func == 2) {
      checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10) );
      checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, SEED) );
      checkCudaErrors( curandGenerate(gen, d_x, n) );
   } else if (curand_func == 3) {
      checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32) );
      //checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, SEED) );
      checkCudaErrors( curandGenerate(gen, d_x, n) );
   } else {
      checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW) );
      checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, SEED) );
      checkCudaErrors( curandGenerate(gen, d_x, n) );
   }
 
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&rand_milli, start, stop);  // time random generation

   // printf("CURAND %s RNG  execution time (ms): %f,  samples/sec: %e \n",
   //    rng, rand_milli, n/(0.001*rand_milli));

   // execute kernel and time it

   cudaEventRecord(start); // start timing

   squareRoot<<<grid,block>>>(n, d_x, d_y, iterations);

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&sqrt_milli, start, stop);  // stop timing actual kernel execution

   // printf("Square root kernel execution time (ms): %f \n",sqrt_milli);

   printf("%s,%f,%d,%e,%d,%d,%d,%d,%f,%d\n", rng, rand_milli, n, n/(0.001*rand_milli), block.x, block.y, grid.x, grid.y, sqrt_milli, iterations);

   checkCudaErrors(cudaDeviceSynchronize());  // flush print queues

   // copy back results
   checkCudaErrors( cudaMemcpy(h_y, d_y, sizeof(float)*n, cudaMemcpyDeviceToHost) );   
   checkCudaErrors( cudaMemcpy(h_x, d_x, sizeof(unsigned int)*n, cudaMemcpyDeviceToHost) );    

   // for (int k = 0; k < n; k++) // Really just for testing to check how close results are to actual square roots
   // {
   //    printf("x = %u", h_x[k]);
   //    printf("  sqrt(x) = %f", h_y[k]);
   //    float z = h_y[k]*h_y[k];
   //    printf("  diff = %f\n", z - h_x[k]);
   // }
   // Tidy up library

   checkCudaErrors( curandDestroyGenerator(gen) );

   // Release memory and exit cleanly

   free(h_x);
   free(h_y);
   checkCudaErrors( cudaFree(d_x) );
   checkCudaErrors( cudaFree(d_y) );

   // CUDA exit 

   cudaDeviceReset();
}

