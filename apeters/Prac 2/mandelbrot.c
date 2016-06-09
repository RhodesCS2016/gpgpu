/*  Code taken from:
    https://rosettacode.org/wiki/Mandelbrot_set#C
    Accessed on: 17/03/2016

    Some minor changes have been made to make it more comprable to the CUDA counterparts

 c program:
 --------------------------------
  1. draws Mandelbrot set for Fc(z)=z*z +c
  using Mandelbrot algorithm ( boolean escape time )
 -------------------------------         
 2. technique of creating ppm file is  based on the code of Claudio Rocchini
 http://en.wikipedia.org/wiki/Image:Color_complex_plot.jpg
 create 24 bit color graphic file ,  portable pixmap file = PPM 
 see http://en.wikipedia.org/wiki/Portable_pixmap
 to see the file use external application ( graphic viewer)
  */
 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <math.h>

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
 /* */
 #define PixelWidth (CxMax-CxMin)/iXmax
 #define PixelHeight (CyMax-CyMin)/iYmax

 int main()
 {
    /*create new file,give it a name and open it in binary mode  */
    FILE * fp;
    char *filename="manc.ppm";
    fp= fopen(filename,"wb"); /* b -  binary mode */
    /*write ASCII header to the file*/
    fprintf(fp,"P6\n #\n %d\n %d\n 255\n",iXmax,iYmax);

    /* screen ( integer) coordinate */
    int iX,iY;
    /* world ( double) coordinate = parameter plane*/
    double Cx,Cy;
    /* Z=Zx+Zy*i  ;   Z0 = 0 */
    double Zx, Zy;
    double Zx2, Zy2; /* Zx2=Zx*Zx;  Zy2=Zy*Zy  */
    /*  */
    int Iteration;

    /*Create array to store values*/
    int n   = iXmax*iYmax;
    unsigned int *h_x   = {(unsigned int *)malloc(sizeof(unsigned int)*n)};
    
    /*initiate timer*/
    time_t start,end;
    start=clock();

    /*begin process*/
    for(iY=0;iY<iYmax;iY++)
    {
        Cy=CyMin + iY*PixelHeight;
        if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
        for(iX=0;iX<iXmax;iX++)
        {         
            Cx=CxMin + iX*PixelWidth;
            /* initial value of orbit = critical point Z= 0 */
            Zx=0.0;
            Zy=0.0;
            Zx2=Zx*Zx;
            Zy2=Zy*Zy;
            /* */
            unsigned int idx = iY * iXmax + iX;
            for (Iteration=0;Iteration<ITTER && ((Zx2+Zy2)<ER2);Iteration++)
            {
                Zy=2*Zx*Zy + Cy;
                Zx=Zx2-Zy2 +Cx;
                Zx2=Zx*Zx;
                Zy2=Zy*Zy;
            };
            if (Iteration==ITTER)
            { /*  interior of Mandelbrot set = black */
                h_x[idx] = 1;                       
            }
            else 
            { /* exterior of Mandelbrot set = white */
                h_x[idx] =0;
            };
        }
    }
    /*end timer*/
    end = clock() - start;
    float msec = end * 1000 / CLOCKS_PER_SEC;
    printf("Mandelbrot host execution time (ms): %f \n", msec);
    

    // write results to file
    static unsigned char color[3];
    int i;
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