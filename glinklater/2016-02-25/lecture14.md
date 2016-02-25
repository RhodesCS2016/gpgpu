# Lecture 14

## TODO

* Global Memory reads/writes
* More memory access patters
* Structs of Arrays and Arrays of Structs (SoA vs. AoS)

## Global Memory reads

* All data pipelined through one of the following paths depending on source of data:
  * L1/L2 Cache
  * Constant Cache
  * Read-only Cache
* On maxwell, L1 cache disabled for global reads by default, in which case path goes first goes to L2 Cache, then to device memory.
* If L1 cache enabled (and architecture allowing), global read first hit attempt is L1, then L2, then device.
* Use compiler flag ```-Xptxas -dlcm=ca``` to opt-in to using L1 cache on global loads.

## Memory access patterns

1. Cached vs. uncached -> Load is cached if L1 cache enabled.
2. Aligned vs. unaligned -> Load is aligned if first address is a multiple of 32 or 128 depending on if using L1 cache or not.
3. Coalesced vs. uncoalesced -> Load is coalesced if warp access a contiguous chunk of data.

## Cached Load Patterns

Cached load operations serviced at granularity of L1 cache line (128 bytes)

### Bus Utilization Metric (BU)

```
BU = X bytes requested / Y bytes loaded
```

1. Aligned and coalesced (ideal case) -> 100% Utilization

2. Aligned but not consecutive. -> can be 100% Utilization but only if each thread accesses one data item and all data items are used.

3. Unaligned but consecutive -> 50% Utilization when memory doesn't fall on a cache line. (unfortunate...)

4. Same address for all 32 threads -> 3.125% Utilization, definitely don't use L1 cache; should be using constant cache. (constant cache does one access for all threads in warp (broadcast) as opposed to 32 consecutive lookups on L2 cache)

5. 32 random addresses across memory -> worst case... 0.02% Utilization. don't do this. if you really need to do this then use L2 cache (disable L1).

## Uncached load patterns

* Performed at granularity of memory segments (32-bytes)
* More fine-grained loads -> can lead to better bus utilization.


1. Aligned and coalesced (ideal case) -> 100% Utilization

2. Aligned but not consecutive -> also 100% Utilization

3. Not aligned but consecutive -> 80% Utilization

4. Same address for all 32 threads -> 12.5% Utilization (better than 3.125%)

5. 32 random addresses (improved worst case) -> 3.9% Utilization (better than 0.02% Utilization)

## Optimising Memory Access

* Fixing uncoalesced reads: not always possible as depends on algorithm.
* Fixing misaligned reads:
  * Offset data structures
  * Pad data structures to keep within 128-byte multiples
  * Use uncached reads
  * Direct global reads via read-only cache rather than L1 cache.
* ```Global memory efficiency = requested global memory throughput / required global memory throughput```
  * If low global memory efficiency then try putting things into shared memory.

## Read-only cache

* Previously reserved for texture memory loads
* \>CC3.5 can support global loads as alternative to L1
* Granularity of loads through RO cache is 32-bytes
* Use function ```__ldg() { out[idx] = in[idx] -> out[idx] = __ldg(in, idx) }```

## Global Memory writes

* Not directed through L1 cache
* Only passes through L2 cache
* Performed at 32-byte granularity, but can be one, two or four segments at a time.
* So if two addresses fall within the same 128-byte region but not with the same 64-byte region, one 4-segment transaction will be issued.
* That is, one 4-segment transaction is better than 4x 1-segment transaction

## Global Write Patterns

1. Access is aligned and consecutive -> 1x 4-segment transaction

2. Access is aligned but addresses scattered along 192-byte range -> 2x 1-segment transaction and 1x 2-segment transaction.

3. All Accesses in consecutive 64-byte range and aligned -> 1x 2-segment transaction

## Array of Structures vs. Structures of Arrays

* Consider storing x- and y-coordinates
* Array of structures:
```c
struct innerStruct {float x; float y;};
struct innerStruct myPoints[10];
/**
 * AoS
 * [x, y, x, y, x, y, x, y]
 *  t0;   t1;   t2;   t3;
 */
```

* Structure of Arrays:
```c
struct innerStruct {float x[10]; float y[10]};
/**
 * SoA
 * [x, x, x, x, y, y, y, y]
 *  t0;t1;t2;t3;t0;t1;t2;t3;
 */
```

* Using SoA approach makes full use of GPU memory bandwidth as provides coalesced access.

## Memory performance tuning

* Maximised device memory bandwidth utilization:
  * Aligned and coalesced access reduces wasted bandwidth
  * Sufficient concurrent memory operations to hide memory latency.

* Maximised concurrent memory accesses
  * Increasing the number of indep. memory operations performed in each kernel.
  * Experimenting with execution configuration of the kernel to expose parallelism to each SM.

## What bandwidth can a kernel achieve

* Theoretical bandwidth = absolute maximum bandwidth achieveable with the hardware.
  * As per hardware specs.

* Effective bandwidth = the measured bandwidth that a kernel actually achieves.
  * ```Effective bandwidth (GB/s) = bytes read + bytes written / time elapsed x 10^-9```
