# Lecture 5

## Leftovers

```blockDim``` and ```gridDim``` are structs of type ```dim3``` and needs to be accessed as follows:

```c
/**
 *          GRID
 *           Y
 *   ______________________
 *   |      |      |      |
 *   | 0,0  | 0,1  | 0,2  |
 * X |______|______|______|
 *   |      |      |      |
 *   | 1,0  | 1,1  | 1,2  |
 *   |______|______|______|
 *
 */

gridDim.x => 2
       .y => 3
       .z => 0
```

Remember to round up grid size to the next multiple of block size:

```
Data Size = 6
Block Size = 4
Grid Size = 6/4 = 2
```

1024 Threads in a block is maximum.
Limit exists because of a lack of registers (hardware limitation).

## Streaming Multiprocessor

* exists as a scalable array of SMs
* Each SM has:
  * CUDA cores (dynamically allocated)
  * Shared memory / L1 Cache
  * Register file
  * Load/store units
  * Special function units (ALU/FPU)
  * Warp scheduler (in hardware because hardware FTW).
    * Do CPUs have hardware schedulers? if not, maybe they should?
* Threads in blocks execute concurrently on the assigned SM

## Warp Execution

* Multiple blocks can be assigned to the same SM
* Each block is executed when the resources available
* SM partitions the thread blocks assigned to it into groups of 32 Threads (warp)
* Schedules warps on available resources
* All threads in a warp execute the same instructions at the same time, BUT
* threads can follow different paths (although this decreases parallelism)
* Thus, SIMT rather than SIMD.

## Memory access

* 6 types
  * registers
  * shared block Memory
  * constant Cache
  * texture Cache
  * global Cache
  * *L1 Cache*
  * *L2 Cache*


* L1 Cache
  * Spillover for registers
  * typically sits on chip in shared memor spillovery or texture Cache


* L2 Cache
  * Typically sits in global memory
  * slower than L1 but faster than typical global memory

## Resources

See diagram in lecture slides.

* Resources at ANY ONE POINT IN TIME (any overrages get queued):
  * Limites within SMM: [1] 32 concurrent blocks, [2] 1024 threads/block, [3] 2048 threads total
  * 1 block of 2048 threads -> forbidden by [2]
  * 2 blocks of 1024 threads -> feasible on one SMM
  * 4 blocks of 512 threads -> feasible on one SMM
  * 4 blocks of 1024 threads -> forbidden by [3], feasible on two SMM
  * 8 blocks of 256 threads -> feasible on same SMM
  * 256 blocks of 8 threads -> forbidden by [1], feasible with 8 SMM, not feasible with our hardware (stops at 5 SMM).
