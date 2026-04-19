#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "divisor_check.h"

#define UPPER_BOUND 10000   /* highest number to examine */
#define MSG_WORK   1
#define MSG_HALT   2

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs < 2) {
        if (my_rank == 0)
            printf("Need at least 2 processes (1 master + 1 worker).\n");
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        /* ========== MASTER PROCESS ========== */
        int next_candidate = 2;
        int running_workers = num_procs - 1;
        int reply;
        MPI_Status st;

        /* Timing bookkeeping */
        double wall_start = MPI_Wtime();
        double comm_accum = 0.0;
        double t0, t1;
        int tested_count = 0;

        printf("Scanning for perfect numbers in range [2, %d] ...\n", UPPER_BOUND);

        while (running_workers > 0) {
            /* Wait for any worker to report back */
            t0 = MPI_Wtime();
            MPI_Recv(&reply, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &st);
            t1 = MPI_Wtime();
            comm_accum += (t1 - t0);

            int sender = st.MPI_SOURCE;

            /* Positive reply means a perfect number was found */
            if (reply > 0) {
                printf("  >> Perfect number detected: %d\n", reply);
            }
            if (reply != 0) {
                tested_count++;
            }

            /* Dispatch next task or shut down the worker */
            t0 = MPI_Wtime();
            if (next_candidate <= UPPER_BOUND) {
                MPI_Send(&next_candidate, 1, MPI_INT, sender,
                         MSG_WORK, MPI_COMM_WORLD);
                next_candidate++;
            } else {
                int halt = 0;
                MPI_Send(&halt, 1, MPI_INT, sender,
                         MSG_HALT, MPI_COMM_WORLD);
                running_workers--;
            }
            t1 = MPI_Wtime();
            comm_accum += (t1 - t0);
        }

        /* Report timing breakdown */
        double wall_end  = MPI_Wtime();
        double wall_total = wall_end - wall_start;
        double comp_time  = wall_total - comm_accum;

        printf("\n--- Performance Summary ---\n");
        printf("Wall-clock time     : %.4f s\n", wall_total);
        printf("Communication time  : %.4f s\n", comm_accum);
        printf("Computation time    : %.4f s\n", comp_time);
        printf("Comm overhead       : %.2f%%\n",
               (comm_accum / wall_total) * 100.0);
        printf("Candidates checked  : %d\n", tested_count);
        printf("Avg time / candidate: %.6f s\n",
               wall_total / tested_count);

    } else {
        /* ========== WORKER PROCESS ========== */
        int outgoing = 0;   /* first message is just a "ready" signal */
        int candidate;
        MPI_Status st;

        double w_start   = MPI_Wtime();
        double w_comm    = 0.0;
        double w_comp    = 0.0;
        double t0, t1;
        int    processed = 0;

        for (;;) {
            /* Send result (or initial ready signal) to master */
            t0 = MPI_Wtime();
            MPI_Send(&outgoing, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            w_comm += (t1 - t0);

            /* Receive next task */
            t0 = MPI_Wtime();
            MPI_Recv(&candidate, 1, MPI_INT, 0, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &st);
            t1 = MPI_Wtime();
            w_comm += (t1 - t0);

            if (st.MPI_TAG == MSG_HALT) break;

            /* Perform the divisor check */
            t0 = MPI_Wtime();
            outgoing = check_perfect_number(candidate);
            t1 = MPI_Wtime();
            w_comp += (t1 - t0);
            processed++;
        }

        double w_end   = MPI_Wtime();
        double w_total = w_end - w_start;

        printf("Worker %d  |  total %.4fs  comm %.4fs (%.1f%%)  "
               "comp %.4fs (%.1f%%)  items %d\n",
               my_rank, w_total,
               w_comm, (w_comm / w_total) * 100.0,
               w_comp, (w_comp / w_total) * 100.0,
               processed);
    }

    MPI_Finalize();
    return 0;
}
