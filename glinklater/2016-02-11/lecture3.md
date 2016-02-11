# Lecture 3

## TODO

* Programming model

## Program Structure

1. Allocate Memory
2. Copy data from CPU memory to GPU memory
3. Invoke the CUDA kernel to perform program specific computation
4. Copy results back from GPU memory to CPU memory
5. Destroy GPU memory. (MEMORY LEAKS!!!)

***

* Host: CPU with its memory (host memory)
* Devce: GPU with its memory (6 different types of memory on the GPU Card)

* Kernel is the key component -> the code that runs on each core of the GPU.
* CUDA manages scheduling kernels on GPU threads.
* From the host you can define how the kernel should be mapped to the device.
* Once a kernel has been launched, control returns to the CPU.
  * WE NEED SEMAPHORES!!!

## Programming model

* Programming Model is an abstraction of architectures that acts as a bridge between the application and its implementation on available hardware.
* Model is typically embedded ina  programming language or environment.
* CUDA provides the following API
  * a way to organise threads (warps) on GPU
  * a way to access memory on GPU
  * both are hierarchical.

This allows us to create programs in a general way that is not bound to a specific architecture and leaves the low level optimisation to the designers of the CUDA language -> "Separations of concerns".

## Managing Memory

CUDA runtime provides functions:
* cudaMalloc - allocate device memory (similar to C malloc)
* cudaMemcpy - transfer between host and device. (memcpy on host)
* cudaMemset - initialise memory. (memset on host)
* cudaFree - destroy memory allocation on device. (free on host)

## cudaMemcpy

see API docs

Returns ```cudaError_t types```

  e.g. ```cudaSuccess``` or ```cudaErrorMemoryAllocation```

Use ```cudaGetErrorString()``` to convert to string

^ NOTE!!! NBNBNBNB

## Example: Summing 2 Arrays

```c
// host
void sumarrays (float * a, float * b, float * c, const int n)
{
  for (int i=0; i<n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main (void) {
  float * h_a, * h_b, * h_c;
  h_a = (float * ) malloc (nElem*sizeof(float));
  h_b = (float * ) malloc (nElem*sizeof(float));
  h_c = (float * ) malloc (nElem*sizeof(float));

  sumarrays (h_a, h_b, h_c, nElem);

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
```

```c
// host + device
__global__ void sumarrays (float * a, float * b, float * c, const int n)
{
  int i = threadIdx.x;
  if (i < N) c[i] = a[i] + b[i]
}

int main (void) {
  float * h_a, * h_b, * h_c;
  float * d_a, * d_b, * d_c;
  // allocate on host
  h_a = (float ** ) malloc (nElem*sizeof(float));
  h_b = (float ** ) malloc (nElem*sizeof(float));
  h_c = (float ** ) malloc (nElem*sizeof(float));
  // initialise on host

  cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

  d_a = (float ** ) cudaMalloc (nElem*sizeof(float));
  d_b = (float ** ) cudaMalloc (nElem*sizeof(float));
  d_c = (float ** ) cudaMalloc (nElem*sizeof(float));

  // call kernel
  sumarrays <<<1, nElem>>> (d_a, d_b, d_c, nElem);

  cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a)
  cudaFree(d_b)
  cudaFree(d_c)

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
```
