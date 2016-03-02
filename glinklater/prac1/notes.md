# Newton

*Changes*

* Varying block size
* Varying algorithm
* Varying value of 'n'

*Ranking done by*

* Square Root Algorithm
* CURAND Time
* Samples per second

It is clear from the results that, across the board, the philox4 algorithm is the best for generating up to 10M random numbers as it provides roughly a 5ms performance boost compared to the xorwow and the quasi_sobol32 algorithms. This is likely due to the fact that philox4 generates 4 numbers at a time.

The execution of the sqrt calculation seems to favour larger data sets as the operations involving a data set of 100M numbers completed faster than the operations involving 1M and 10M. I do not know what would cause this.

Also the sqrt algorithm favours block sizes that appear with threads that are in multiples of 32; in particular the 32x32 block size seems to be favoured. This is likely due to a combination of coalesced and aligned memory accesses due to careful indexing of threads.

When inspecting samples per second; it is apparant that the higher values of ```n``` seem to provide the best results in terms of random number sampling. Once again I do not know how to explain this.
