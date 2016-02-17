# Lecture 6

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

## Local memory

* Register space spills into local memory
* Amount per thread 512KB
* Contains variables eligible for being registers but cannot fit.
  * Arrays too large for registers
  * Local arrays with dynamic indexes
* Physically located in L1/L2 (2MB) Caches and device memory.

## Constant memory

* Special area of device memory
* 64 KB
* Read only from kernel
  * Cache (10KB per SM)
* Constants are declared at file scope
* Constant values from host code
* Intended to be broadcast to all threads in a warp - NB for performance.

## Texture memory

* Read only
* Texture cache - 24KB per SM
* Allocate and manage global memory
* Qualify kernel pointer argument as ```const __restrict__```
* Optimised for 2D spacial locality -> good performance for threads in a warp accessing 2D data
* L1 shares this space on Maxwell arch.

## Variable Qualifiers

| Qualifier | Var Decl. | Memory | Scope | Lifetime |
|:-|:-|:-|:-|:-|:-|:-|
| | Atomic variables (not arrays) | register | thread | thread |
| | Arrays | local | thread | thread |
| \__shared__ | float* | shared | block | block |
| \__device__ | float* | global | global | program |
| \__constant__ | float* | constant | global | program |
\* Can be scalar or array

## Memory Constraints

* 64K registers and 48KB shared Memory per SMM
* In order for a second block to run on the same SMM, each block must at most use 32K registers and 24 KB shared Memory
* To allow a third block each block must use at most 21.3K registers and 16KB shared memory... and so on
* Tradeoff between memory use and parallelism **<- NB**
