# Lecture 8

## Branch Efficiency

* Ratio of non-divergent branches to total branches
```
branch_efficiency = 100 * (numBranches - numDivergentBranches / numBranches)
```
* Efficiency is 100% if no branches are divergent

* Warp divergence should be the first thing you look for in a new app.
* Worst case close factor 32x in performance if one thread needs an expensive branch, while rest do nothing.

## Resource Partitioning

* Local execution context of warp includes: registers, program counters, shared memory
* Remains on-chip for lifetime of warp -> no cost context switching
* \# of resident thread blocks and warps on SM depends on amount of shared resources.
* Block is active when resources allocated
* Active warps can be classified as:
  * Selected -> actively executing
  * Stalled -> not ready for execution
  * Eligible -> ready for execution but not currently executing.

## Hiding Latency

* Latency is the time it takes from the start of the instruction to the end.
* Thread level parallelism used to maximise utilisation of cores.
* Linked to np. of resident warps.
* Instruction latency is number of clock cycles between instruction being issued and completed.

## Instruction Latency

* Two Types:
  * Arithmetic instructions: 10-20 cycles
  * Memory instructions: 400-800 cycles

* Little's law can be applied to calculate how many active warps needed to hide latency:
  * ```# of warps required = latency x throughput```

## Full arithmetic utilisation

* Consider executing 32-bit floating point multiply-add
  * ```(a + b x c)```
* Instruction latency = 20 cycles
* Throughput (operations/cycle)

  = 128 (cores/SMM)
  = 192 (cores/SMX)

* Parallel coperations = 2560 cycles (Maxwell) or 3840 cycles (Kepler)
* So need 2560/32 = 80 warps (Maxwell) or 120 warps (Kepler)

## Full memory utilisation

| GPU | Instruction lat. | Bandwidth (GB/sec) | Mem Freq. | Bandwidth (B/cycle) | Parallelism (KB) |
|:-:|:-:|:-:|:-:|:-:|
| Kepler (Tesla K20) | 800 | 250 | 2.6 | 96 | 77 |
| Maxwell (GTX 750 Ti) | 800 | 86.4 | 5.4 | 16 | 13 |

* Parallelism is for entire device - because global Memory
* Suppose each thread moves one float (4B) - ```need 13K / 4 = 3250``` threads (```3250/32 = 102 warps```) to hide latency on Maxwell.

## Occupancy

* Ratio of active warps to maximum number of warps per SM
  * ```occupancy = active warps / max warps```
* CUDA GPU Occupancy Calculator
  * ```./CUDA/v7.x/tools/<calc>```

## Synchronisation

* Barrier synchronisation common in most parallel languages.
* Two levels in CUDA:
  * System-level: wait for all work on both host and device to complete ```cudaDeviceSynchronize();```
  * Block-level: wait for al threads in a block to reach the same point in execution ```__syncthreads();``` **<- NB BLOCK LEVEL**
* Important to ensure all threads can reach the Barrier -> otherwise deadlock.

* All memory changes by threads will be visible to all threads after sync.
* Race conditions, or hazards, are unordered accesses by multiple threads in different warps to the same memory
  * Read-after-write
  * Write-after-write
  * Write-after-read
* No thread synchronization between threads in different blocks -> need to use multiple kernels.

## Scalability

* Implies that providing additional parallel resources yield speedup relative to the added amount
* Desirable for any parallel application
* Serial code is inherently not scalable.
* Parallel code is potentially scalable, but how much depends on the algorithm.
* Scalability vs. Efficiency?
* CUDA programs very scalable because thread blocks can be distributed across number of SMs and executed in any order.
