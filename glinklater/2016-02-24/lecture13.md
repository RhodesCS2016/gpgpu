# Lecture 13

## Reducing with unrolling

* Using cyclic partitioning of data -> each thread works on data from more than one block
* Process two blocks at a time by having each thread sum its data in block i

## Dynamic Parallelism

* So far, all kernels invoked from host -> so GPU workload completely under control of CPU
* Ability to dynamically add parallelism to GPU at arbitrary points in kernel -> dynamic parallelism
* Use a more hierarchical approach where parallelism can be exposed in multiple levels in GPU kernel -> via recursive algorithm
* Delay decision of how many blocks and grids to create until runtime.

## Nested Execution

* Uses concept of two types of kernel execution
  * parent -> kernel that calls extra kernels, can't terminate until all children are terminated.
  * child -> invoked by parent

* Grid launches in device thread are visible across a thread block.
  * if thread 0 launches a kernel then grid is visible across parent block.
* Parent and child grids share global and constant memory, but have their own local and shared memory.
* Two points in execution of child-grid where memory is consistent between parent and child: at start of child-grid and when all children have completed.

## Nested Hello World

```c
__global nestedHW(int const iSize, int iDepth)
{
  tid = threadIdx.x;
  printf("Recurse %d: Hello World from thread %d, block %d\n",
    iDepth,
    tid,
    blockIdx.x
  );
  if (iSize == 1) return;

  int nthreads = iSize >> 1;
  if (tid == 0 && nthreads > 0)
  {
    nestedHW<<<1,nthreads>>> (nthreads, ++iDepth);
    printf("Nested depth %d\n", iDepth);
  }
}
```
