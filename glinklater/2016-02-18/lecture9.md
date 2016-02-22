# Lecture 9

## Covering 4 cycles of latency

```c
/**
 * Executing warps
 *      1        2        3        4        5
 * ----------------------------------------------
 * | Warp 1 | Warp 2 | Warp 3 | Warp 4 | Warp 1 |
 * ----------------------------------------------
 *                    Cycles ->
 *
 * 1st Warp executes and then needs to wait for 4 cycles.
 *
 * To hide the latency, the next warp is executed and then waits
 * for its latency
 *
 * We need 4 warps in order to keep 100% occupancy while there is a warp
 * waiting for some operation.
 *
 * Warp 1 continues executing after the latency has been reached.
 */
```

* Thus ```threads needed = 4 x number of cores``` to hide all latency.
* If latency is 20 cycles, threads needed = ```20 x number of cores.```
