# Lecture 19

## Events

### Creation/Destruction

* Declare event: ```cudaEvent_t myevent;```
* Create event: ```cudaEventCreate(&myevent);```
* Destroy event: ```cudaEventDestroy(&myevent);```

### Recording Events

* Events mark a point in stream execution.
* Can be used to chunk if executing stream operations have reached a given point.
* Think of them as operations added to the stream whose only action when popped ...

### Waiting on Events

* Blocks the calling host thread: ```cudaEventSynchronize(myevent);```
* Analagous to cudaStreamSynchronize();
* But allows host to wait for a specified intermediate point and not complete stream execution.

### Measuring elapsed time with Events

* Use ```cudaEventElapsedTime(&time, startevent, stopevent);``` with events.
  ```c
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // launch kernel
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ```

## Host view of concurrency

* CUDA operations can be classified as:
  * Memory related ops; many are synchronous

## Non-null Streams

* Non-blocking on host
* But can be blocking (```cudaStreamCreate```) or Non-blocking on device.
* operations within a non-null stream can be blocked by operations in the NULL stream. NULL stream == blocking stream;
* Operations in a non-blocking stream won't block on ops in NULL stream.
  ```c
  Kernel1<<<1,1,0,stream1>>>();

  // this is blocking the other Streams
  Kernel2<<<1,1>>>();

  Kernel3<<<1,1,0,stream2>>>();
  ```

## Creating non-blocking Streams

* Use ```cudaStreamCreateWithFlags(cudaStream_t *mystream, unsigned int flag)```
* Where ```flag = cudaStreamNonBlocking```.
* if we used cudaStreamNonBlocking stream in previous example, none of the kernels would block.

## More on synchronization.

* Implicit synchronization
  * cudaMemcpy -> host blocks till data transfer complete.
  * Memory related operations on device typically wait for all previous operations on device to end:
    * Device Memory allocation
    * Device memset
    * Memcpy between addresses on same Device
    * Modification to L1/shared memory configuration
* Explicit synchronization
  * ```cudaDeviceSynchronize```
  * ```cudaStreamSynchronize```
  * ```cudaEventSynchronize```
  * ```cudaStreamWaitEvent```

## Atomic Operations

* Performs a mathematical operation, but does so in a single uninterruptable operation without any interference from other threads.
* Prevents the problem where an application needs threads to update a counter in shared memory
```c
__shared__ int counter;
// ...
if (/* ... */) counter++;
```
* Problem if two or more threads try to dio this at the same time.

* With atomic instructions, the read/add/write becomes a single operation, happening one after the other.

* Several different operations are supported, mostly only for integers.
  * Addition ```atomicAdd```
  * Min/Max ```atomicMin, atomicMax```
  * Inc/Dec
  * Unconditional swap (int and 32-bit float) ```atomicExch```
  * Subtraction
  * Compare-and-swap

## Atomic Compare-and-swap

* ```int atomicCAS (int *addr, int compare, int value)```
* if compare equals old value at address then val is stored instead.
* In either case, routine returns the value of old
* Seems a bizarre routine at first sight but can be very useful in atomic locks.
* Can also be used to implement 64-bit floating point atomic addition.

## Global Atomic locks

```c
// global variable: 0 = unlocked; 1 = locked
__device__ int lock=0;
__global__ void kernel()
{
  if (threadIdx.x == 0)
  {
    // set lock
    do {
      // do things
    } while (atomicCAS(&lock, 0, 1));

    // free lock.
    lock = 0;
  }
}
```

## Safe global atomic lock

```c
__device__ int lock=0;
__global__ void kernel ()
{
  if (threadIdx.x == 0) {
    do {} while (atomicCAS(&lock,0,1));
    __threadfence(); // wait for writes to finish -> introduces safety.
    lock = 0;
  }
}
```
