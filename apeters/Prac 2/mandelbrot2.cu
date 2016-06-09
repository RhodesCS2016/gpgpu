//	Optimised CUDA C Implementation of Mandelbrot set

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

/* number fo streams*/
#define nstreams 2

/* loop counter*/
#define LOOP 2
#define iXL iXmax/LOOP

__global__ void mandelbrot(int j, unsigned int *grid)
//Determines the mandelbrot set over a given plane
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y + gridDim.y * blockDim.y * j;
    unsigned int iy2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ( ix < iXL && iy < iYmax)
    {
        while (ix < iXmax)
        {
            unsigned int idx = iy2 * iXmax + ix;

        	double Cy=CyMin + iy*PixelHeight;
            if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
            double Cx=CxMin + ix*PixelWidth;
            
            double Zx   = 0.0;
            double Zy   = 0.0;
        	double Zx2  = Zx * Zx;
            double Zy2  = Zy * Zy;
        	int i;

            for (i=0;i<ITTER && ((Zx2+Zy2)<ER2);i++)
            {
                Zy  = 2 * Zx * Zy + Cy;
                Zx  = Zx2 - Zy2 + Cx;
                Zx2 = Zx * Zx;
                Zy2 = Zy * Zy;
            };

            if (i==ITTER)
            {
    			grid[idx] = 1;
            }
            else
            {
            	grid[idx] = 0;
            }

            ix += iXL;
        }
    }
}

int main(int argc, char*argv[] )
{
	// create new file,give it a name and open it in binary mode
    FILE * fp;
    char *filename="mancuda2.ppm";
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

    // create streams
    cudaStream_t stream[nstreams];
    int bytesPerStream = (sizeof(unsigned int)*n)/nstreams;

    // determine grid and block sizes
    dim3 block(32,32);
    dim3 grid(((iXmax + block.x - 1) / block.x)/LOOP, ((iYmax + block.y - 1) / block.y)/nstreams);

    int j;
    for (j=0; j< nstreams; j++){ cudaStreamCreate(&stream[j]); }
    for (j=0; j< nstreams; j++)
    {
        int offset = (n/nstreams)*j;
	    mandelbrot<<<grid,block,0,stream[j]>>>(j, &d_x[offset]);
        cudaMemcpyAsync(&h_x[offset], &d_x[offset], bytesPerStream, cudaMemcpyDeviceToHost, stream[j]);
    }
    for (j=0; j< nstreams; j++)
    {
        cudaStreamSynchronize(stream[j]);  
        cudaStreamDestroy(stream[j]);  // destroy stream
    }

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