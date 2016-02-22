# Lecture 11

## Question

How can you parallelise calculating the sum of a large list of numbers?

* Split the list and add up sections of the list serially.
* Split list in half, add relative idx in both halves together and store in-place in first half.
* Irrespective of how you decide; it will involve partial sums.
  * CUDA only has barriers; no other semaphores.

## Parallel Reduction

1. Partition List into smaller chunks
2. Have thread calculate the partial sum for each chunk
3. Add the partial results from each chunk into final sum.


* Performing a commutative and associative op across a vector is referred to as reduction.
* Different implementation possibilities:
  * Neighbouring pair
    * Two next each other and does sum.
  * Interleaved pair
    * Sum ```i``` and ```i + stride``` where stride is ideally half the block size (in threads).
    * Does not incur extra memory latency because both solutions require 2x memory fetches.


* Naieve Solution
  * Divergence between each pair of threads.
  * Even threads work, odd do not.


* Better Solution
  * Divide threads better so that a whole warp in a block of 64 threads is not working and can be reallocated elsewhere.
  * First Half of thread block executes first reduction step.
  * Then first quarter executes next reduction step and so on.
  * Optimises thread allocation.


* Even Better Solution
  * Use interleaved pairs
  * More efficient than previous versions because of global memory load and store patterns.
    * Optimises Memory Reads.
  * Has some divergence as in previous version.
  * Coalesced Read for first and second operands.

## Unrolling Loops

* Technique to optimise loop execution by reducing frequency of branches and loop iteration operations.
* Rewrite body of loop multiple times -> number of copies = loop unrolling factor.
* Best used when number of iterations known.

```c
for (int i=0; i<100; i++) a[i] = b[i] + c[i];

// Better for latency hiding
// Fewer comparisons and more efficient use of CPU pipeline.
for (int i=0; i<100; i+=2)
{
  a[i] = b[i] + c[i];
  a[i+1] = b[i+1] + c[i+1];
}
```

## Reducing with Unrolling.
