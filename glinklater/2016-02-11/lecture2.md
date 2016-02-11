# Lecture 2

## TODO

* Parallel concepts
* Nvidia GPUs
* Heterogeneous parallel paradigm
* Hello World in CUDA

## Parallel Computing

Task Parallelism
* Tasks or functions can be operated independently and largely in parallel.
* Distributed across cores

Data Parallelism
* Data items can be operated on at the same time
* Distributed across cores
* CUDA is well suited to this type of parallelism.

## Data Dependency

* Data Dependence occurs whena program statement refers to the data of a preceding statement
* Data Dependence limits parallelism.
  * "Embarassingly Parallel Problem"
  * Use semaphore to solve this and break problem into phases.

## Data partitioning

Block partitioning
* consecutive data is chunked and each chunk is given to a single thread.

Cyclic partitioning
* Fewer items are chunked
* each chunk is allocated to a separate thread
* each thread operates on several chunks
  * Similar to a work queue.
  * advantage when some parts of the data compute quicker than others.  

## Flynn's Taxonomy

* S/M: single/multiple
* I/D: instruction/data -> task/data
* i.e. SISD -> single instruction, single data

***

* SISD: traditional computer
* SIMD: vector processors -> allows us to think sequentially but still get some speedup
* MISD: unknown -> each core operates on same data via separate instruction Streams
* MIMD: multiple cores operate on multiple data. each core has own instruction set. Modern Computers are an example of these.
* SIMT: single instruction multiple threads -> allows efficient use of processor: while one thread is waiting for slow IO to finish another can use the processor

## Metrics

* Latency
* Bandwidth
* Throughput
* Occupancy

## Heterogeneous Computing

* Homogenous computing uses one or more processor of the same architecture. (i.e. multi-processor)
* Heterogeneous involves multiple architecture -> i.e. CPU + GPU

## Nvidia Cards

* Board Family
  * Tegra -> Embedded
  * Geforce -> consumer / gaming
  * Quadro -> visualiztion
  * Tesla -> data-center parallel computing (general purpose + stream processing)

* Microarcitecture Classes
  * Tesla (don't confuse with above) -> released Nov. 2006 (CC 1.0)
  * Fermi -> released 2010 (CC 2.0, 2.1)
  * Kepler -> released 2012 (CC 3.0 to 3.7)
  * Maxwell -> released 2014 (gen. 1 CC 5.0, gen. 2 CC 5.2 and 5.3)
  * Pascal -> soon...
  * Volta -> not so soon...

## Heterogeneous architecture

Parallel -> GPU
serial -> CPU

## Heterogeneous app

* 2 parts
  * Host Code: runs on CPU
  * Device (CUDA) Code: runs on GPU
* Typically initialized by CPU
* CPU manages environment, code and data for device before loading compute-intensive tasks on device.
* Device acts as a hardware accelerator.

## Heterogeneous Computing paradigm

* GPU and CPU are complimentary
* GPU cannot replace CPU

[diagram]

as parallelism and data size incr(CUDA)ease, enter gpu paradigm

as parallelism and data size decreases, enter cpu paradigm

## CUDA -> Compute Unified Device Architecture

* CUDA is "not a language" -> its an API
* Accessible through APIs, compiler directives, CUDA-accelerated libs and extensions to standard languages (C, C++ and Python)
  * Take a look at "THRUST" (CUDA-accelerated lib)
* just to confuse us more -> its also a general purpose parallel computing platform as well.

*Definition -> CUDA is a platform that exposes an API.*

* CUDA Driver API -> Low level, more flexible and more control
* CUDA Runtime API -> Easier to use, more high level.

## Components

You need three things
* Driver
* Toolkit
  * nvcc CUDA compiler
  * Profiling and Debugging
  * Libs
* SDK
  * demo examples + error-checking utilities.
  * no docs... FUCK!

## Tools

* NVIDIA NSight IDE
* CUDA-GDB
* Visual and command line profiler (nvprof)
* CUDA-MEMCHECK memory analyser
* GPU device managment tools

#### Go look at the prac.

## Compilation

* nvcc compiler separates CUDA C program into host and device code
* Host code (standard C) compiled with gcc
* Device code (CUDA C), containing data parallel functions called kernels, is compiled further with nvcc
* Linking stage: CUDA runtime libraries addition

## Hello World in CUDA

```c
// file helloC.cu

#include <stdio.h>

int main(void)
{
  printf("Hello World");
  return 0;
}
```

```c
// file helloG.cu

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void)
{
  printf("Hello World from GPU!");
}

int main(void)
{
  printf("Hello World from CPU!");
  // <<<[BLOCK], [THREADS]>>>
  helloFromGPU <<<1, 10>>> ();
  cudaDeviceReset ();
  return 0;
}
```
