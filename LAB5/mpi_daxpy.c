#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VEC_LEN 65536  /* 2^16 elements */

int main(int argc, char** argv) {
    int my_rank, num_procs;
    double alpha = 2.5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (VEC_LEN % num_procs != 0) {
        if (my_rank == 0)
            printf("Error: vector length (%d) must be evenly divisible "
                   "by process count (%d)\n", VEC_LEN, num_procs);
        MPI_Finalize();
        return 1;
    }

    int chunk = VEC_LEN / num_procs;

    double *global_x = NULL, *global_y = NULL, *serial_x = NULL;
    double *part_x = (double*)malloc(chunk * sizeof(double));
    double *part_y = (double*)malloc(chunk * sizeof(double));

    double baseline_time = 0.0;

    if (my_rank == 0) {
        global_x = (double*)malloc(VEC_LEN * sizeof(double));
        global_y = (double*)malloc(VEC_LEN * sizeof(double));
        serial_x = (double*)malloc(VEC_LEN * sizeof(double));

        for (int i = 0; i < VEC_LEN; i++) {
            global_x[i] = 1.0;
            global_y[i] = 2.0;
            serial_x[i] = 1.0;
        }

        /* Run the serial version for comparison */
        double ts = MPI_Wtime();
        for (int i = 0; i < VEC_LEN; i++) {
            serial_x[i] = alpha * serial_x[i] + global_y[i];
        }
        baseline_time = MPI_Wtime() - ts;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double wall_begin, wall_finish, c_start;
    double local_comm = 0.0;

    /* ---- Parallel DAXPY ---- */
    wall_begin = MPI_Wtime();

    /* Distribute chunks */
    c_start = MPI_Wtime();
    MPI_Scatter(global_x, chunk, MPI_DOUBLE,
                part_x,   chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(global_y, chunk, MPI_DOUBLE,
                part_y,   chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_comm += (MPI_Wtime() - c_start);

    /* Local DAXPY computation */
    for (int i = 0; i < chunk; i++) {
        part_x[i] = alpha * part_x[i] + part_y[i];
    }

    /* Collect results */
    c_start = MPI_Wtime();
    MPI_Gather(part_x,   chunk, MPI_DOUBLE,
               global_x, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_comm += (MPI_Wtime() - c_start);

    wall_finish = MPI_Wtime();
    double local_elapsed = wall_finish - wall_begin;

    /* Aggregate the worst-case timings across all ranks */
    double peak_elapsed, peak_comm;
    MPI_Reduce(&local_elapsed, &peak_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm,    &peak_comm,    1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double spdup = baseline_time / peak_elapsed;
        double eff   = (spdup / num_procs) * 100.0;
        double cpct  = (peak_comm / peak_elapsed) * 100.0;

        printf("\n+------------- MPI DAXPY Benchmark -------------+\n");
        printf("| Vector length  : 2^16 (%d)\n", VEC_LEN);
        printf("| Processes      : %d\n", num_procs);
        printf("+------------------------------------------------+\n");
        printf("| Serial time    : %f s\n", baseline_time);
        printf("| Parallel time  : %f s\n", peak_elapsed);
        printf("| Comm time      : %f s\n", peak_comm);
        printf("+------------------------------------------------+\n");
        printf("| Speedup        : %.2fx\n", spdup);
        printf("| Efficiency     : %.2f%%\n", eff);
        printf("| Comm overhead  : %.2f%%\n", cpct);
        printf("+------------------------------------------------+\n");

        free(global_x);
        free(global_y);
        free(serial_x);
    }

    free(part_x);
    free(part_y);
    MPI_Finalize();
    return 0;
}
