# Lecture 15

## TODO

* Shared Memory (SM)
* SM Banks

## Non-coalesced memory access

* Sometimes not possible to do away with Non-coalesced global memory accesses
* SM can assist: low-latency (20-30x less than global memory) on-chip memory that offers higher bandwidth than global memory (64 KB)
* Useful as:
  * Intra-block communication channel
  * Program managed cache for global memory data
  * Scratch pad memory for transforming data to improve global memory access patterns.

## Shared memory

* Fixed amount per thread block (64K)
* Same lifetime as thread block to which allocated.
* SM accesses issued per warp -> ideally in one transaction.
* Worst case: 32 unique transactions
* if multiple threads access same word in SM -> one thread fetches the word, and broadcast to other threads in the warp via multicast.
* Critical resource -> the more SM used by a kernel, the fewer possible concurrently active thread blocks.

## Allocation of shared memory

* Statically:
```c
__shared__ float tile[size_y][size_x];
```

* Dynamically -> if size not known at compile time can declare an unsized 1D array in kernel
```c
extern __shared__ int tile[];
// Give size with kernel launch as third parameter
kernel <<<grid, block, size of dynamic SM>>> (...);
```

## Shared memory banks

* To achieve high bandwidth, SM divided into 32 equally sized modules, called banks, which can be accessed simultaneously.
* Aim is to have each thread in warp access a separate bank -> then all accesses can be done in a single transaction.
* Ideal case -> SM load or store accesses only one memory location per bank; can be serviced by one transaction
* Otherwise need multiple transactions -> lower bandwidth utilization.

## Bank Conflict

* When multiple addresses in SM request fall into same memory bank -> bank Conflict
* When conflict occur, request is replayed
* Hardware splits request with a bank conflict into as many separate conflict-free transactions as necessary (decreasing BU)
* Parallel access: multiple addresses across multiple banks
* Serial access: multiple accesses in same bank

## Access patterns

* Optimal parallel pattern
  * Each thread accesses continguous banks
* Irregular parallel pattern
  * Each thread accesses a different bank but not contiguous
  * This is still good.
* Irregular access pattern
  * conflict-free if threads access same address within a bank
  * bank conflict access if accessing different addreses in same bank.

## Address modes

* SM access width defines which SM addresses are in which bank.
* Bank width varies depending on CC
  * Kepler: either 8-byte or 4-byte
  * Maxwell: only 4-byte
* Successive 32-bit (64-bit) words map to successive banks.
* Each bank has a bandwidth of 32-bits / cycle.
* Mapping from SM addresses to bank index: ```idx = addr / 4 bytes % 32 banks```

## Bank Allocation

See slides...

## Examples

* No bank conflicts
  * Access in (row,bank_idx) 0,1; 1,2; 2,3
* 3 conflicts
  * Accesses in 0,1; 1,1; 2,1;

## Memory Padding

* It is a good idea to pad your memory allocations so that each thread keeps looking at the same bank.

See lecture slides
