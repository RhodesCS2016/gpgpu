//	Naive CUDA C Implementation of Mandelbrot set

#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>

#define ITTER 10000
#define ER2 4

/* screen ( integer) coordinate */
#define iXmax 4096
#define iYmax 4096
/* world ( double) coordinate = parameter plane*/
#define CxMin -2.5
#define CxMax 1.5
#define CyMin -2.0
#define CyMax 2.0
/* rescale for image*/
#define PixelWidth (CxMax-CxMin)/iXmax
#define PixelHeight (CyMax-CyMin)/iYmax

__global__ void mandelbrot(int n, unsigned int *grid)
//Determines the mandelbrot set over a given plane
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * iXmax + ix;

    if (ix < iXmax && iy < iXmax)
    {
    	double Cy=CyMin + iy*PixelHeight;
        if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
        double Cx=CxMin + ix*PixelWidth;
        
        double Zx=0.0;
        double Zy=0.0;
    	double Zx2=Zx*Zx;
        double Zy2=Zy*Zy;
    	int i;

        for (i=0;i<ITTER && ((Zx2+Zy2)<ER2);i++)
        {
            Zy=2*Zx*Zy + Cy;
            Zx=Zx2-Zy2 +Cx;
            Zx2=Zx*Zx;
            Zy2=Zy*Zy;
        };
        if (i==ITTER)
        {
			grid[idx] = 1;
        }
        else
        {
        	grid[idx] = 0;
        }
    }
}

int main(int argc, char*argv[] )
{
	// create new file,give it a name and open it in binary mode
    FILE * fp;
    char *filename="mancuda.ppm";
    fp= fopen(filename,"wb");
    // write ASCII header to the file
    fprintf(fp,"P6\n #\n %d\n %d\n 255\n",iXmax,iYmax);
    
    // initialise card
	findCudaDevice(argc, (const char**) argv);
  	// initialise CUDA timing
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// allocate memory on host and device
	int n 	= iXmax*iYmax;
	unsigned int *h_x 	= (unsigned int *)malloc(sizeof(unsigned int)*n);     
	unsigned int *d_x;  
	checkCudaErrors(cudaMalloc((void**)&d_x,sizeof(unsigned int)*n));

	// execute kernel and time it
	cudaEventRecord(start); // start timing

	// determine grid and block sizes
    dim3 block(32,32);
    dim3 grid((iXmax + block.x - 1) / block.x, (iYmax + block.y - 1) / block.y);

    mandelbrot<<<grid,block>>>(n, d_x);

    checkCudaErrors(cudaDeviceSynchronize());  // flush print queues

    // copy back results
    checkCudaErrors( cudaMemcpy(h_x, d_x, sizeof(float)*n, cudaMemcpyDeviceToHost) );   

	cudaEventRecord(stop); //stop timing
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);  // stop timing actual kernel execution
	//
	printf("Mandelbrot kernel execution time (ms): %f \n",milli);

	// write results to file
	int i;
	static unsigned char color[3];
	for (i=0;i<n;i++)
	{
		if (h_x[i]==1)
        { /*  interior of Mandelbrot set = black */
            color[0]=0;
            color[1]=0;
            color[2]=0;                           
        }
        else 
        { /* exterior of Mandelbrot set = white */
            color[0]=255; /* Red*/
            color[1]=255;  /* Green */ 
            color[2]=255;/* Blue */
        };
        /*write color to the file*/
        fwrite(color,1,3,fp);
	}
    
	fclose(fp);
    return 0;
}