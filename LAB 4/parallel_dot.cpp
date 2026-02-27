// Exercise 4: Parallel Dot Product with Performance Metrics
// Compile: mpicxx -O2 -o ex4_dot ex4_dot.cpp
// Run:     mpirun -np 4 ./ex4_dot

#include <mpi.h>
#include <stdio.h>

void line(int n = 70) {
    for(int i = 0; i < n; i++) printf("-");
    printf("\n");
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 8;

    if (N % size != 0) {
        if(rank == 0)
            printf("Vector size must be divisible by number of processes.\n");
        MPI_Finalize();
        return 0;
    }

    int chunk = N / size;

    int A[N], B[N];
    int local_A[chunk], local_B[chunk];

    // Initialize vectors in Process 0
    if (rank == 0) {
        int tempA[N] = {1,2,3,4,5,6,7,8};
        int tempB[N] = {8,7,6,5,4,3,2,1};

        for(int i = 0; i < N; i++) {
            A[i] = tempA[i];
            B[i] = tempB[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Scatter vectors
    MPI_Scatter(A, chunk, MPI_INT,
                local_A, chunk, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B, chunk, MPI_INT,
                local_B, chunk, MPI_INT,
                0, MPI_COMM_WORLD);

    // Compute partial dot product
    int local_dot = 0;
    for(int i = 0; i < chunk; i++)
        local_dot += local_A[i] * local_B[i];

    // Reduce to global result
    int global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot,
               1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    double local_time = end - start;

    // Get max execution time
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
        printf("   EXERCISE 4: PARALLEL DOT PRODUCT\n");
        line();
        printf("Vector Size (n)      : %d\n", N);
        printf("Processes (p)        : %d\n", size);
        line();

        printf("Dot Product Result   : %d\n", global_dot);
        printf("Expected Result      : 120\n");
        printf("Verification         : %s\n",
               global_dot == 120 ? "CORRECT ✓" : "INCORRECT ✗");
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