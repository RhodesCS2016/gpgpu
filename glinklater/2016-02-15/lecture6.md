# Lecture 6

## Results for "HandsOn" Task

| 2D Block Size | 2D Grid Size | Kernel Execution Time (ms) | 1D Grid Size | Kernel Execution time (ms) -> 1 data item per thread | Kernel execution time (ms) -> 16 data items per thread |
|:-|:-|:-|:-|:-|:-|
| 32x32 | | | | | |
| 32x16 | | | | | |
| 16x32 | | | | | |
| 16x16 | | | | | |

## Execution Model Summary

Suppose we have 1000 blocks with 128 threads each -> How is this executed?
* 8-12 blocks probably running on each SMM
* Each block has 4 warps (128/32) -> 32-48 warps
* At each clock cycle, SMM warp scheduler does the following:
  * Decides which warps not waiting for something
  * Chooses from warps not waiting for something
    * data coming from device memory (memory latency)
    * completion of earlier instructions (pipeline delay)
* Programmer doesn't worry about this detail - but

## CUDA Syntax

We can define our own functions:
```c
__device__ float myfunction(int i) { }
```
The above cannot be called on the host

Functions for both the Host and Device can be defined as follows:
```c
__host__ __device__ float myfunction2() { }
```

Call device functions from within the kernel:
```c
__global__ kernel () {
  myfunction2 <<<1, 1>>> ();
}
```

## Error Management

All errors are handled by the host

Most CUDA API calls return cudaError_t
-> Enumeration type with cudaSuccess = 0

```char * cudaErrorGetString(cudaError_t err)``` returns a readable string.

```c
void CHECK (cudaError_t err) {
  if (err)
  {
    printf("Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaErrorGetString(err));
    exit(1);
  }
}

// ... other

CHECK(cudaFree(my_ptr));
```

Most Cuda calls are asynchronous -> difficult to determine where error arose.

Also kernel launches have no return value

```cudaError_t cudaGetLastError()```
* returns error code for last runetime function called (including kernel launches)
* once called, error state cleared.

For kernel calls, must first call ```cudaDeviceSynchronize();```

Then, ```e = cudaGetLastError();```

## Device Management

* Application can query and select GPUs
  * ```cudaGetDeviceCount(int *count)```
  * ```cudaSetDevice(int device)```
  * ```cudaGetDevice(int *device)```
  * ```cudaGetDeviceProperties(cudaDeviceProp *prop, int device)```
* Multiple host threads can share a device
* A single host thread can manage multiple devices.

## Global memory

* Visible to all threads + CPU
* Located on the device
* Persists between kernel calls in the same Application
* Allocation and deallocation handled by the programmer (on the host)

## Per-thread memory

Variables declared in kernel are allocated per thread
* Only accessible by thread
* Has lifetime of thread

Compiler controls where these variables are stored in physical memory
* Registers (on-chip) -> fastest form of memory on SM, but limited.
* Local memory (mostly off-chip)
