# Welcome to GPGPUs

## CPU

#### Von Neumann Processor

The standard fetch-execute instruction cycle that most (if not all) modern processors follow - at least conceptually.

#### Pipelining

Multiple instructions are overlapped across many processors

#### Superscalar processors

**Intel**

14 Stage Pipeline;
3 ALU;
1 FPU (addition);
1 FPU (multiplication);
2 load/store;
FP division is slow;

You can technically get 3 integer and 2 floating-point results per cycle.

#### Technical Challenges
* Challenges
  * Compiler needs to extract best performance by reordering instructions to be most efficient.
  * out-of-order CPU execution to avoid delays like waiting for IO
  * branch prediction to minimise delays (loops, if-then-else, any conditional)
* All of the above results in a limit to the number of pipelines that can be used.
* Optimisation through executing code before it is needed (prediction)
* 95% of Intel processors devoted to control and data.

#### Memory Hierarchy

Register (1 cycle) -> L1 Cache (4 cycles) -> L2 Cache (20 cycles) -> DDR (120 cycles).

We don't really have any control of this, all we can do is improve performance by exploiting data locality.

## GPU

* Simplified logic
* Usually used for floating point computation
* no out-of-order execution, really bad at branch statements
* Very high graphics memory bandwidth but low memory capacity (190 GB/s -> 8 GB memory)
* gtx 750i
  * 640 Cores
  * 5 SMX (streaming cores)
    * each core has 4 blocks with 32 SIMD (single instruction, multiple data cores)
    * each block operates in lockstep (single clock for all cores in block), all 32 cores have to be doing the same thing at the same time.

#### CUDA

C programming with a few extensions and some C++ features (i.e. templates)

Code runs on CPU, CUDA runs on GPU.

## TODO

* Parallel programming concepts
* Basic CUDA
* CUDA execution
* CUDA memory model
* Libs
* Streams and concurrency
* Optimisation
