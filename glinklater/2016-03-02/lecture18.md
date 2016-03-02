# Lecture 18

## CUDA Stream

* Refers to a sequence of asynchronous CUDA operations that execute on a
device in order defined by the host code
* Stream encapsulates these operations, maintains their ordering, permits operations to be added to the end of the queue, and allows querying of the status of the stream.
* Execution of operation in stream is asynchronous with respect to host.
* Operations in different streams have no ordering restrictions.

## Implementing Grid-level concurrency

* Use multiple streams to launch multiple simultaneous kernels -> grid-level concurrency
* Asynchronous nature of operations in stream, allows their execution to be overlapped with other operations in host-device system.
* Can hide latency of certain operations
* From software point of view, streams operate concurrently -> however, hardware may need to serialize then (e.g. PCIe bus contention, limited SMs, etc.)

### Battery is dying...
