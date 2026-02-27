// Exercise 1: Ring Communication (Improved for Real Scaling)
// Compile: mpicxx -O2 -o ex1_ring ex1_ring_comm.cpp
// Run:     mpirun -np 4 ./ex1_ring

#include <mpi.h>
#include <stdio.h>

void print_separator(int width = 60) {
    for (int i = 0; i < width; i++) printf("-");
    printf("\n");
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;

    // ── Synchronize Before Timing ───────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // ── Heavy Computation (dominates communication) ─────────────
    long long workload = 200000000;   // increase if system is very fast
    long long local_sum = 0;

    for (long long i = 0; i < workload; i++) {
        local_sum += (i % 7) * (i % 5);
    }

    // ── Ring Communication ──────────────────────────────────────
    int value;

    if (rank == 0) {
        value = 100 + (local_sum % 1000);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += (local_sum % 1000);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    // ── End Timing ──────────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double local_time = end - start;

    // ── Gather Times ────────────────────────────────────────────
    double all_times[size];
    MPI_Gather(&local_time, 1, MPI_DOUBLE,
               all_times, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // ── Only Rank 0 Prints ──────────────────────────────────────
    if (rank == 0) {

        // Tp = maximum process time
        double Tp = all_times[0];
        for (int i = 1; i < size; i++) {
            if (all_times[i] > Tp)
                Tp = all_times[i];
        }

        print_separator();
        printf("   EXERCISE 1: RING COMMUNICATION   \n");
        print_separator();
        printf(" Processes (p): %d\n", size);
        print_separator();

        printf("\nExecution Times (per process)\n");
        for (int i = 0; i < size; i++) {
            printf("  Process %d : %.6f s\n", i, all_times[i]);
        }

        print_separator();
        printf("  PERFORMANCE METRICS\n");
        print_separator();
        printf("  %-22s : %.6f s\n", "Execution Time (Tp)", Tp);
        printf("  %-22s : Use np=1 run\n", "Serial Time (T1)");
        printf("  %-22s : T1 / Tp\n", "Speedup (Sp)");
        printf("  %-22s : (Sp/p) * 100\n", "Efficiency (Ep)");
        print_separator();

        printf("\nNOTE:\n");
        printf("1. Run with -np 1 first and record Tp (this is T1).\n");
        printf("2. Run with -np 2,4,8.\n");
        printf("3. Compute: Sp = T1 / Tp,  Ep = Sp/p * 100\n");
        print_separator();
    }

    MPI_Finalize();
    return 0;
}