# Lecture 12

## Libraries

* Advantages of using these libraries:
  * Already parallelised
  * Already implemented
  * Already debugged
  * Already optimised.
* Disadvantages
  * wat?
* NVIDIA provides a handful of libraries and there are also a number of third party libraries.

## Nvidia Libraries

* CUFFT -> Fast Fourier transforms
* CUBLAS -> linear algebra
* CUSPARSE -> sparse base linear algebra
* Libm -> math.h
* CURAND -> random numbers
* NPP -> Image and signal processing.
* Thrust -> c++ stl equivalent (GUI stuff)

See the docs.

* Open Source
  * MAGMA -> linear algebra
  * CULA Tools -> linear algebra
  * CUSP -> sparse base linear algebra
  * CUDPP -> ?

## CUDA Libm (math.h equivalent)

* C99 plus extras
* Basic ops: x*y etc.
* Trig
* etc.

## CURAND

* XORWOW pseudo-random generator
* Sobol' quasi-random number generators
* Host API for generating random numbers in bulk.
* Inline implementation allows use GPU inside functions/kernels


* Features
  * Host API: call kernel on host, generates numbers on GPU, consume on host or on GPU.
  * GPU API: generate and consume during kernel execution.

**Steps**

1. Create a generator
2. Set a seed.
3. Generate the data from a distribution.
  * ```GenerateUniform/UniformDouble```: Uniform
  * ```GenerateNormal/NormalDouble```: Gaussian
  * ```GenerateLogNormal/LogNormalDouble```: Log-Normal
  * etc.
4. Destroy the generator

See the docs.

## NPP

*Graeme's Best Friend*

## Thrust

A template library for CUDA -> mimics the C++ stl

Containers
* Manage memory on host and device.
* Help avoid common errors.


Iterators
* Know where data lives
* Define ranges

Algorithms
* Sorts
* Reductions

**This could be EXTREMELY useful**

```c
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib.h>

int main(void)
{
  // Generate 32M random numbers on host
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device
  // the details are hidden from us here.
  // possibly unified memory.
  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer back to host_vector
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  // do stuff.

  return 0;
}
```

Algorithms
* Elementwise
  * forEach
  * reduction
  * zip
  * transforms
* sort
* Function stuff?

**Get this ASAP**
