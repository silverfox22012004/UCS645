// FINAL VERSION: Parallel Array Sum with Performance Metrics
// Compile: mpicxx -O2 -o ex2_array_sum ex2_array_sum.cpp
// Run:     mpirun -np 4 ./ex2_array_sum

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void line(int n = 70) {
    for(int i = 0; i < n; i++) printf("-");
    printf("\n");
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ARRAY_SIZE = 100;

    int *full_array = NULL;

    // Handle uneven division
    int base = ARRAY_SIZE / size;
    int remainder = ARRAY_SIZE % size;

    int local_size = base + (rank < remainder ? 1 : 0);
    int *local_array = (int*)malloc(local_size * sizeof(int));

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {

        full_array = (int*)malloc(ARRAY_SIZE * sizeof(int));

        for (int i = 0; i < ARRAY_SIZE; i++)
            full_array[i] = i + 1;

        sendcounts = (int*)malloc(size * sizeof(int));
        displs     = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Distribute data
    MPI_Scatterv(full_array, sendcounts, displs,
                 MPI_INT,
                 local_array, local_size,
                 MPI_INT,
                 0, MPI_COMM_WORLD);

    // Local computation
    int local_sum = 0;
    for (int i = 0; i < local_size; i++)
        local_sum += local_array[i];

    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum,
               1, MPI_INT, MPI_SUM,
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
            T1 = Tp;   // Record serial time

        double Sp = 0, Ep = 0, Comm = 0;

        if (T1 > 0) {
            Sp = T1 / Tp;
            Ep = (Sp / size) * 100.0;

            double ideal = T1 / size;
            Comm = ((Tp - ideal) / Tp) * 100.0;
        }

        line();
        printf("   EXERCISE 2: PARALLEL ARRAY SUM\n");
        line();
        printf("Processes (p)      : %d\n", size);
        printf("Array Size (n)     : %d\n", ARRAY_SIZE);
        printf("Global Sum         : %d\n", global_sum);
        printf("Expected Sum       : 5050\n");
        printf("Average            : %.2f\n",
               (double)global_sum / ARRAY_SIZE);
        printf("Verification       : %s\n",
               global_sum == 5050 ? "CORRECT ✓" : "INCORRECT ✗");
        line();

        printf("Performance Metrics\n");
        line();
        printf("Execution Time Tp  : %.8f s\n", Tp);

        if (size == 1)
            printf("Serial Time T1     : %.8f s (record this)\n", Tp);
        else
            printf("Serial Time T1     : Use value from np=1 run\n");

        printf("Speedup (Sp)       : %.4f\n", Sp);
        printf("Efficiency (Ep)    : %.2f%%\n", Ep);
        printf("Communication %%     : %.2f%%\n", Comm);
        line();

        printf("\nRESULT TABLE ENTRY\n");
        printf("%-8s %-12.8f %-10.4f %-12.2f %-10.2f\n",
               "p=" , Tp, Sp, Ep, Comm);
        line();
    }

    MPI_Finalize();
    return 0;
}