# Lecture 10

## TODO

* Recap memory Hierarchy
* Memory Managment
* Aligned and coalesced access.

## Recap on Memory Hierarchy

* Registers -> private to a thread
* Shared memory -> per-block shared memory
* Local memory -> resides in same space as global memory, cached in L1 (per SM) and L2 (per device) caches. private to a thread.
* Constant memory -> resides in device memory but cached; constant variable declared globally outside of the kernel (from host); RO to kernels and broadcast read.
* Texture memory -> resides in device memory but cached; RO and 2D optimized.
* Global Memory -> resides in device memory; accessible to all SMs.

Arranged top down: fastest -> slowest and smallest -> largest.

## Caches

* L1 -> one per SM
* L2 -> shared by all SMs
* RO constant -> one per SM
* RO texture -> one per SM
* Only memory load operations are cached -> not store operations.
  * Writes do not affect our caches at all.

## Memory Transfer Host <--> Device

* Allocation on device -> ```cudaMalloc()```
* Copy to/from device -> ```cudaMemcpy()```
  * This is a synchronous command, both host and device stop what they are doing for this.
* Initialise memory on device -> ```cudaMemset()```
* Deallocation on device -> ```cudaFree()```


* What happens when you are doing a memcpy and the page gets dropped?
* Solution: copy to pinned memory then copy to device(default)
  * pinned memory is non-pagable memory
* Else allocate pinned memory on host.
* This happens automatically.

## Pinned memory on the host

* Use ```cudaMallocHost()``` and ```cudaFreeHost()```
  * Allocates bytes of host memory that are page-locked.
  * Thus can be read/written with a much higher bandwidth than pagable memory.
  * But more expensive to allocate/deallocate.
  * Generally beneficial for > 10MB data transfers.
  * CAVEAT: allocating excessive amounts of pinned memory can degrade host system performance (reduce the amount of pageable memory)
    * We can deal with this by using ```cudaFreeHost()``` ASAP to give us back our pages.

## Zero-copy memory

* Pinned host memory that can be mapped in advance to device address space.
* Both host and device can access this memory
* Uses:
  * Leveraging host memory when insufficient device memory
  * Acoiding explicit transfer between host and device.
  * Improving PCIe transfer rates.
* ```cudaHostAlloc()``` with additional 3rd parameter:
  * ```cudaHostAllocDefault``` -> same as ```cudaMallocHost```
  * ```cudaHostAllocMapped``` -> returns host memory that is mapped to device.
  * **NOTE** this is an enum.
  * **NOTE** this is horrendously expensive...
  * Could be used when you have multiple devices and you don't want to copy multiple times.
  * This is also a point where race conditions are a definite possibility. Make sure to synchronise device first.

## Device access to zero-copy memory

* Obtain pointer for device to access host memory
```c
cudaHostGetDevicePointer(void **d_ptr, void **h_ptr, unsigned int flag);
```
  * Returns a device pointer that can be referenced on device to access mapped, pinned host memory.
  * Flag is currently unused; set to zero.
  * Remember every access by device to this memory must pass over the PCIe bus -> slow compared to global device memory.
  * **SYNCHRONISE THIS WHEN GETTING RESULTS FROM DEVICE ON HOST**

## Unified Virtual Addressing

* Takes zero-copy for memory one-step further.
* On UVA enables devices to share a single virtual address space.
* Thus, eliminates the need to get explicit device pointer (pointing to mapped, zero-copy memory on host) -> use host pointer directly.
* Why is this useful? Maybe allows multiple devices to access other device's memory?

## Unified Memory

* Provides unified memory supposer > CUDA 6.0
* Creates a pool of managed memory -> where allocation is accessible on both CPU and GPU through the same pointer.
  * Like C# is to C++ and C
* Underlying system automatically migrates data in unified memory space between device and host.
* Data movement is transparent to application -> hence simplified coding.
* Managed memory == allocations managed by system
* Unmanaged memory == allocations managed by application (explicit code).
* Host can access managed memory.

* Static allocation (only in file and global-scopes)
```c
__device__ __managed__ int y;
```

* Dynamic allocation:
```c
cudaMallocManaged(void **ptr, size_t size, unsigned int flags = 0);
```

* Program using managed memory is functionally unchanged from unmanaged counterpart.
* Advantages:
  * automatic data migration and avoids duplicate pointers.
* **NOTE**: This is a managed memory system, it still consumes resources on both device and host.

## Memory Access Patterns

* Most GPU applications tend to be limited by memory bandwidth
* Since most device data access begins in global memory, optimizing bandwidth is important.
* Memory operations are issued per warp
* Each thread provides a single address and the warp presents a single memory access request containing the 32 addresses.
* Depending on distributing of addresses in warp, memory accesses are classified into different patterns.
  1. Aligned / Coalesced Access
    * All accesses to global memory go through L2 cache; some also go through L1 cache
    * If both are used, memory access serviced by a 128-byte transaction.
    * if only L2 used, 32-byte transaction used
    * L1 cache line is 128 bytes, mapping to a 128-byte aligned segment in memory
    * If each thread in warp requests 4-byte value -> this results in 128 bytes per request.
    * Aligned memory access occurs when first address of memory transaction is an even multiple of cache granularity.
    * Misaligned access will cause wasted bandwidth. (multiple memory reads required.)
    * Coalesced memory access occurs when all threads in warp access contiguous chunk of memory.
    * Aligned and coalesced access means a warp accesses a contiguous chunk of memory starting at an aligned address.
      * *This is the ideal situation.*
