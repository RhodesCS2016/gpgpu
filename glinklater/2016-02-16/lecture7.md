# Lecture 7

## What we know about warps

* blocks divided into threads
* threads grouped into warps
* 32 threads per warps
* all threads in a warp executed in SIMT fashion
  * All threads execute same instruction on different data

* Irrespective of logical block partitioning (1D, 2D, 3D) -> hardware view of a thread block is 1D
* So, each thread has a unique ID in 3 Dimensions i.e. ->

```c
ID = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
```

## Warps per block

* number = ceil (ThreadsPerBlock / warpSize)
* Warps never split across block
* if threadsPerBlock is not a multiple of 32, some threads in last warp will be inactive.
* Inactive threads still consume SM resources (registers, local memory etc.)
* So make blockDim a multiple of 32.

## CPU Branching

* CPUs use fairly complex branching prediction logic to predict which branch (e.g. if statement) control flow is likely to take.
* If correct, only small performance penalty for branching.
* If not, the CPU stalls as it needs to flush instruction pipeline.
* GPU does not have any of this logic, it simply processes each instruction as it comes.

## GPU Branching

* No branch prediction, all threads in a warp must execute identical instructions.
* If threads in a warp diverge (e.g. when processing an if statement) -> the warp executes each branch path serially.
* Threads that do not need to execute current path are disabled.
* However, execution cost is sum of both branches
* Potentially large loss of performance
* So, avoid different execution paths in the same warp if possible.

## Avoiding Divergence

```c
if (id % 2 == 0) a = 100.0f;
else b = 100f;
c[id] = a + b;
```

* Thread approach - even numbered threads set a, odd numbers set b
* Warp approach - even numbered warps set a, odd numbered warps set b

```c
if ((id/warpSize) % 2 == 0) // Do as above
```

## How Nvidia GPUs handle Divergence

* Predicated instructions that are carried out only if a logical flag is true
```assembly
p: a = b + c
```

* In the previous example all threads compute the logical predicate and two predicated instructions:
```assembly
p = (d % 2 == 0);
p: a = 100.0f;
!p: b = 100.0f;
```

* No need to worry about illegal memory accesses or divide by zero errors.

## Handling Divergence

If the branches are big, nvcc compiler inserts code to check if all threads in a warp take the same branch (warp voting) and then branches accordingly.
```
p = ...
if (any(p))
{
  p: ...
  p: ...
}
if (any(!p))
{
  !p: ...
  !p: ...
}
```
