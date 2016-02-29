# Lecture 16

## Synchronisation

* Key mechanism in a parallel language
* Two approaches: barriers and memory fences
* Barrier: all threads wait until all other calling threads reach barrier point.
* Memory fence: all threads stall until all modifications to memory are visible to all other calling threads.
* Both of these are available for both grid and block level.

## Weakly ordered memory model.

* Means memory accesses are not necessarily executed in the order in which they appear in the program.
  * e.g. the order threads write to different memories is not necessarily the same in which they are executed.

## Explicit barrier

* Use ```syncthreads();```
* Be careful to ensure all threads reach the same point.
```c
if (threadId % 2 == 0)
{
  // ...
  __syncthreads();
} else {
  // ...
  __syncthreads();
}
```
  * If all threads don't reach same point -> undefined behaviour -> race condition?
  * **NOTE** -> *DON'T PUT ```__syncthreads();``` IN BRANCHING EXECUTION PATHS. See Below.*
  ```c
  if (threadId % 2 == 0)
  {
    // ...    
  } else {
    // ...
  }
  __syncthreads();
  ```

## Memory Fence
* Ensure that all memory write before the fence is visible to other threads after the fence.
* To set a memory fence within a thread block: ```void __threadfence_block();```
  * Does not perform any thread synchronization, so it is not necessary for all threads in a block to actually execute this instruction.
* Memory fence at grid level: ``` void __threadfence();```
  * Stalls the calling thread until all of its writes to global memory are visible to all threads in the same grid.
* Less "draconian" than synchronozing every single thread in the block.

## Warp shuffle instruction

* Warp shuffle is an optimization.
* Warp -> implicityly synchronize group of threads (```warpsize == 32```)
* WarpID (```warpid```)
  * Identifier of warp in block (```threadIdx.x / 32```)
* LaneID (```laneid```)
  * Coordinate of the thread in a warp: ```threadIdx.x % 32```
  * exists as a register -> we can get hold of this value.
* Instruction to exchange data in a warp
* Threads can "read" other threads' registers.
* No shared memory is needed.
* Available starting CC3.0

* Variants:
  * idx -> info moves in an any-to-any fashion.
  * up -> info moves to thread with higher idx.
  * down -> info moves to thread with lower idx.
  * bfly -> butterfly (XOR) exchange with mask.

## Intrinsic \__shfl instruction

```c
int __shfl (int var, int srcLane, int width=warpsize);
```

* Returns value of var passed to ```shfl``` by the thread lane.

## Calling \__shfl

* Can change warpsize to any power-of-two
* Means we can subdivide a warp into sections, but then shuffleID not the same as laneid
  ```c
  shuffleID = threadIdx.x % width
  int y = __shfl(x, 3, 16); // I am still very confused about this -_-'
  ```
  * Threads 0 to 15 recieve value of x from thread 3.
  * Threads 16 to 31 recieve value from thread 19.
  * When lane index is same for all threads -> instruction performs a warp broadcast operation.

## Identifying threads relative to calling thread

```c
int __shfl_up (int var, unsigned int delta, int width=warpsize);
```

* Source lane is calculated by subtracting delta from the callers lane index.

## Shfl down

* Similar to shfl up
* difference is that you add delta to lane index.

## Shfl XOR (butterfly)

```c
int __shfl_xor (int var, int lanemask, int width=warpsize);
```

* Calculates source lane by performing bit-wise XOR of the caller's lane index with lanemask.
  * Crosses between pairs of threads.
