// Exercise 3: Global Min & Max with Performance Metrics
// Compile: mpicxx -O2 -o ex3_minmax ex3_minmax.cpp
// Run:     mpirun -np 4 ./ex3_minmax

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void line(int n = 70) {
    for (int i = 0; i < n; i++) printf("-");
    printf("\n");
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int NUM_COUNT = 10;
    int numbers[NUM_COUNT];

    srand(time(NULL) + rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Generate random numbers (0–1000)
    for (int i = 0; i < NUM_COUNT; i++)
        numbers[i] = rand() % 1001;

    // Local min & max
    int local_max = numbers[0];
    int local_min = numbers[0];

    for (int i = 1; i < NUM_COUNT; i++) {
        if (numbers[i] > local_max) local_max = numbers[i];
        if (numbers[i] < local_min) local_min = numbers[i];
    }

    // Structure for MAXLOC/MINLOC
    struct {
        int value;
        int rank;
    } local_maxloc, global_maxloc,
      local_minloc, global_minloc;

    local_maxloc.value = local_max;
    local_maxloc.rank  = rank;

    local_minloc.value = local_min;
    local_minloc.rank  = rank;

    // Global max & min with rank
    MPI_Reduce(&local_maxloc, &global_maxloc,
               1, MPI_2INT, MPI_MAXLOC,
               0, MPI_COMM_WORLD);

    MPI_Reduce(&local_minloc, &global_minloc,
               1, MPI_2INT, MPI_MINLOC,
               0, MPI_COMM_WORLD);

    // Also plain reduction
    int global_max, global_min;

    MPI_Reduce(&local_max, &global_max,
               1, MPI_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    MPI_Reduce(&local_min, &global_min,
               1, MPI_INT, MPI_MIN,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double local_time = end - start;

    // Get maximum execution time (Tp)
    double Tp = 0;
    MPI_Reduce(&local_time, &Tp, 1,
               MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {

        static double T1 = 0;
        if (size == 1)
            T1 = Tp;

        double Sp = 0, Ep = 0, Comm = 0;

        if (T1 > 0) {
            Sp = T1 / Tp;
            Ep = (Sp / size) * 100.0;

            double ideal = T1 / size;
            Comm = ((Tp - ideal) / Tp) * 100.0;
        }

        line();
        printf("   EXERCISE 3: GLOBAL MINIMUM & MAXIMUM\n");
        line();
        printf("Processes (p)        : %d\n", size);
        printf("Numbers per process  : %d\n", NUM_COUNT);
        printf("Total numbers        : %d\n", NUM_COUNT * size);
        line();

        printf("Global Maximum       : %d\n", global_max);
        printf("Found by Process     : %d\n",
               global_maxloc.rank);

        printf("\nGlobal Minimum       : %d\n", global_min);
        printf("Found by Process     : %d\n",
               global_minloc.rank);
        line();

        printf("Performance Metrics\n");
        line();
        printf("Execution Time (Tp)  : %.8f s\n", Tp);

        if (size == 1)
            printf("Serial Time (T1)     : %.8f s (record this)\n", Tp);
        else
            printf("Serial Time (T1)     : Use np=1 value\n");

        printf("Speedup (Sp)         : %.4f\n", Sp);
        printf("Efficiency (Ep)      : %.2f%%\n", Ep);
        printf("Communication %%       : %.2f%%\n", Comm);
        line();
    }

    MPI_Finalize();
    return 0;
}