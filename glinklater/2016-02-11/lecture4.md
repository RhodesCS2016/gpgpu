# Lecture 4

## Leftovers

Note: branching logic is inefficient on GPU, unless we split the then and else logic onto different warps; we really don't mind doing a little bit of excess work.

cudaMemcpy needs the correct direction, host and device do not have mutual unique memory addresses.

## Threads

CUDA threads are different to data-parallel tasks
* Each thread performs the same operations on a subset of a data structure.
* Threads execute independently

CUDA threads are very light-weight (not like CPU threads)
* Little creation overhead
* Near-Instant context switching. **<- STRENGTH OF GPU**

## Thread Hierarchy

* GPU can handle thousands of threads
* CUDA programming model allows a kernel launch to specify more threads than the GPU can execute concurrently.
* Two level Hierarchy
  * Blocks can be assembled in a 2d structure
  * Each block can contain an 2d array of threads
  * for example: a 2x3 array of blocks with 3x5 threads each results in a total of 90 threads.
  * result is we can work in up to three dimensions.
* All threads issued by a kernel are collectively called a grid.

* All threads in a grid share the same global memory
* Threads in a block can cooperate with eachother
  * Block-local synchronization
  * Block-level shared memory
* Threads in different blocks cannot cooperate.

## Definitions

* Thread
  * smallest unit
  * exists within a thread block
  * executes an instance of the kernel
  * has a thread ID within its block, program counter, registers and per-thread private memory
* Block
  * set of concurrently executing threads that can cooperate among themselves through barrier synchronization and shared memory
  * has a block id within the grid
* Grid
  * array of blocks that execute the same kernel
  * read inputs and write inputs to global memory
  * synchronize between dependent kernel calls

## Memory

* Thread has per thread local memory
* Block has block shared memory
* Kernel has per-device global memory

## CUDA Thread Indexing

* Each thread hsa two unique coordinates:
  * ```blockIdx```
  * ```threadIdx```

These are built in variables of type ```uint3```
  * Struct with three fields: x, y and z
  * e.g. ```blockIdx.y``` or ```threadIdx.z```

UINT3 ->
does memory work with ```ulong``` addresses and if not then is the architecture limited to 8GB of memory?

## Thread Layout

* Built-in variables
  * blockDim (block size in number of threads)
  * gridDim (grid size in number of blocks)
* Dimensions specified as uint3
* Grid usually organized as 2d array of blocks
* Block usually organised as 3d array of threads
* Setting grid and block dimensions is done via dim3 types:
```c
dim3 example (6); // This is perfectly legitimate.
dim3 grid (2,3);
dim3 block (3,5);
// Note we cannot define 3d grids and blocks inline.
kernelx <<<grid, block>>>(some, args);
```
  * These are used inside host code.

Example index:
```c
d_a[blockIdx.x * blockDim.x + threadIdx.x];
```

NB: See CUDA Grid image in lecture notes!
