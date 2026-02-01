#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE (1 << 16)   // 65536 elements
#define REPEAT 100       // number of timing iterations

int main() {

    double *A = malloc(sizeof(double) * SIZE);
    double *B = malloc(sizeof(double) * SIZE);
    double alpha = 2.5;

    if (!A || !B) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize vectors
    for (int i = 0; i < SIZE; i++) {
        A[i] = 0.5 * i;
        B[i] = 0.3 * (SIZE - i);
    }

    printf("\n=========== Parallel DAXPY Benchmark ===========\n");
    printf("Formula: A[i] = alpha*A[i] + B[i]\n");
    printf("Elements: %d\n", SIZE);
    printf("-------------------------------------------------\n");
    printf("Threads | Avg Time (ms) | Speedup\n");
    printf("-------------------------------------------------\n");

    double base_time = 0.0;
    double max_gain = 0.0;
    int best_t = 1;

    for (int t = 1; t <= 16; t++) {

        omp_set_num_threads(t);

        // Warm-up pass
        #pragma omp parallel for
        for (int i = 0; i < SIZE; i++)
            A[i] = alpha * A[i] + B[i];

        // Reset A
        for (int i = 0; i < SIZE; i++)
            A[i] = 0.5 * i;

        double total = 0.0;

        for (int r = 0; r < REPEAT; r++) {

            double start = omp_get_wtime();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < SIZE; i++)
                A[i] = alpha * A[i] + B[i];

            double end = omp_get_wtime();
            total += (end - start);

            // Reinitialize A for next run
            for (int i = 0; i < SIZE; i++)
                A[i] = 0.5 * i;
        }

        double avg_ms = (total / REPEAT) * 1000.0;

        if (t == 1)
            base_time = avg_ms;

        double sp = base_time / avg_ms;

        printf("  %2d    |   %10.6f  |  %6.3f\n", t, avg_ms, sp);

        if (sp > max_gain) {
            max_gain = sp;
            best_t = t;
        }
    }

    printf("-------------------------------------------------\n");
    printf("Best speedup %.3fx observed with %d threads\n", max_gain, best_t);
    printf("=================================================\n");

    free(A);
    free(B);

    return 0;
}