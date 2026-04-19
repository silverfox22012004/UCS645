# Lab 5 - MPI Programming

## How to compile and run

```bash
make          # builds the perfect number finder
make run      # runs with default 2 processes
make run NP=4 # runs with 4 processes

# for daxpy separately
mpicc -Wall -O2 -o mpi_daxpy mpi_daxpy.c
mpirun --oversubscribe -np 4 ./mpi_daxpy
```

---

## Q1: DAXPY Loop

Parallelized the DAXPY operation (X = A*X + Y) for vectors of size 2^16 using MPI_Scatter and MPI_Gather.

| Procs | Time (s) | Speedup | Efficiency | Comm % |
|-------|----------|---------|------------|--------|
| 1 | 0.000070 | 1.00 | 100.00% | - |
| 2 | 0.001211 | 0.09 | 4.28% | 97.36% |
| 4 | 0.000798 | 0.06 | 1.52% | 98.28% |
| 8 | 0.001861 | 0.04 | 0.47% | 98.95% |

So basically adding more processes made it slower lol. The vector is too small (65k elements) and the computation takes microseconds, but MPI_Scatter/Gather takes way longer than the actual math. Comm overhead is 97%+ which means almost all time is spent just moving data around. Would need much bigger vectors to actually see any benefit here.

---

## Q2: Broadcast Race

Compared my own broadcast (loop-based, root sends to each rank one by one) vs the built-in MPI_Bcast. Used an 80MB array.

| Procs | My Bcast (s) | MPI_Bcast (s) | How much faster |
|-------|-------------|--------------|-----------------|
| 1 | 0.000001 | 0.000000 | 3.50x |
| 2 | 0.070164 | 0.019214 | 3.65x |
| 4 | 0.245403 | 0.044397 | 5.53x |
| 8 | 0.691922 | 0.048921 | 14.14x |
| 16 | 3.465557 | 0.187397 | 18.49x |

My version scales linearly since rank 0 has to send to every process one after another. MPI_Bcast uses a tree structure so it distributes the load - once a process receives data it can forward it to others. At 16 processes the built-in version is ~18x faster. Lesson learned: don't try to reimplement collective ops with send/recv loops.

---

## Q3: Dot Product (500M elements)

Dot product of two 500-million element vectors. Each process generates its chunk locally (no need to broadcast the whole array), then we use MPI_Reduce to sum up partial results.

| Procs | Time (s) | Speedup | Efficiency |
|-------|----------|---------|------------|
| 1 | 68.42 | 1.00 | 100.00% |
| 2 | 71.68 | 0.95 | 47.70% |
| 4 | 31.59 | 2.17 | 54.14% |
| 8 | 22.17 | 3.09 | 38.58% |

This one actually shows real speedup since the vectors are huge. 3x speedup at 8 procs is decent. Efficiency drops because of Amdahl's law - the sequential parts (broadcasting the multiplier, the reduce at the end) become a bigger fraction as we add more processes. With 2 procs it was actually slower than serial, probably because the overhead of setting everything up wasn't worth it for just splitting in half.

---

## Q4: Master-Slave Prime Search

Finding primes up to 100,000. Master sends one number at a time to workers, workers check if its prime and send back the result, then get the next number. Uses MPI_ANY_SOURCE so the master just gives work to whoever finishes first.

| Config | Time (s) | Speedup | Efficiency |
|--------|----------|---------|------------|
| 2 (1M+1S) | 0.0927 | 1.00 | 100.00% |
| 4 (1M+3S) | 0.0410 | 2.26 | 75.33% |
| 8 (1M+7S) | 0.0333 | 2.78 | 39.71% |

Dynamic scheduling works well here because checking different numbers takes different amounts of time. But sending one number per message is too fine-grained - at 8 processes the master is basically drowning in tiny messages. Efficiency drops to 40%. Sending batches of numbers per request would help a lot.

---

## Q5: Perfect Number Search

Same master-worker setup but looking for perfect numbers up to 10,000 (numbers where the sum of divisors equals the number itself, like 6 = 1+2+3).

| Procs | Time (s) | Speedup | Efficiency | Comm % |
|-------|----------|---------|------------|--------|
| 1 | 0.0149 | 1.00 | 100.00% | 92.60% |
| 2 | 0.0169 | 0.88 | 44.00% | 93.48% |
| 4 | 0.0216 | 0.69 | 17.25% | 94.76% |
| 8 | 0.0145 | 1.03 | 12.88% | 92.69% |

Adding processes actually made it slower here (negative scaling from 1 to 4 procs). The problem is just too small - checking if a number up to 10000 is perfect takes almost no time, but each check requires a full round-trip message to the master. Over 90% of the time is spent on communication. This is a case where sequential is simply better.

---

## Files

- `master_slave.c` - main program with master-worker pattern for perfect number search
- `divisor_check.c/h` - function to test if a number is perfect
- `primality_test.c/h` - function to test if a number is prime
- `mpi_daxpy.c` - parallel DAXPY implementation with benchmarking
- `broadcast_util.c/h` - manual loop-based broadcast for comparison
- `vec_dotprod.c/h` - local dot product computation
- `makefile` - build rules
