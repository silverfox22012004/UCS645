#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define DIM 1000
#define TEST_RUNS 5
#define MAX_T 32

double *A, *B, *C, *REF;

// Convert 2D → 1D index
static inline int pos(int r, int c) {
    return r * DIM + c;
}

// Allocate square matrix
double* alloc_sq_matrix() {
    double *ptr = calloc(DIM * DIM, sizeof(double));
    if (!ptr) {
        printf("Allocation failure!\n");
        exit(1);
    }
    return ptr;
}

// Fill matrices with predictable pattern
void fill_inputs() {
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            A[pos(i,j)] = i + j;
            B[pos(i,j)] = i - j;
        }
    }
}

// Reset output matrix
void zero_out(double *M) {
    for (int i = 0; i < DIM * DIM; i++) M[i] = 0.0;
}

// Baseline multiplication (no parallelism)
double run_serial() {
    double t0 = omp_get_wtime();

    for (int i = 0; i < DIM; i++)
        for (int k = 0; k < DIM; k++) {
            double tmp = A[pos(i,k)];
            for (int j = 0; j < DIM; j++)
                C[pos(i,j)] += tmp * B[pos(k,j)];
        }

    return omp_get_wtime() - t0;
}

// Strategy A: Parallel rows
double run_parallel_rows(int t) {
    omp_set_num_threads(t);
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < DIM; i++)
        for (int k = 0; k < DIM; k++) {
            double tmp = A[pos(i,k)];
            for (int j = 0; j < DIM; j++)
                C[pos(i,j)] += tmp * B[pos(k,j)];
        }

    return omp_get_wtime() - t0;
}

// Strategy B: 2D loop collapse
double run_parallel_grid(int t) {
    omp_set_num_threads(t);
    double t0 = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            for (int k = 0; k < DIM; k++)
                C[pos(i,j)] += A[pos(i,k)] * B[pos(k,j)];

    return omp_get_wtime() - t0;
}

// Quick correctness check
int check(double *X, double *Y) {
    int step = DIM / 8;
    for (int i = 0; i < DIM; i += step)
        for (int j = 0; j < DIM; j += step)
            if (fabs(X[pos(i,j)] - Y[pos(i,j)]) > 1e-6)
                return 0;
    return 1;
}

void divider() {
    printf("------------------------------------------------------------\n");
}

int main() {

    A = alloc_sq_matrix();
    B = alloc_sq_matrix();
    C = alloc_sq_matrix();
    REF = alloc_sq_matrix();

    fill_inputs();

    printf("\nMatrix Size: %d x %d\n", DIM, DIM);
    printf("Max Threads Available: %d\n", omp_get_max_threads());

    divider();
    printf("Running sequential baseline...\n");

    double base = 0;
    for (int i = 0; i < TEST_RUNS; i++) {
        zero_out(C);
        base += run_serial();
    }
    base /= TEST_RUNS;
    for (int i = 0; i < DIM * DIM; i++) REF[i] = C[i];

    printf("Baseline Time: %.3f s\n", base);
    divider();

    printf("\n=== Strategy A: Row Parallelism ===\n");
    printf("Threads | Time | Speedup | Eff(%%)\n");

    double bestA = 0; int bestAt = 1;

    for (int t = 1; t <= MAX_T; t++) {
        double sum = 0;
        for (int r = 0; r < TEST_RUNS; r++) {
            zero_out(C);
            sum += run_parallel_rows(t);
        }
        double avg = sum / TEST_RUNS;
        double sp = base / avg;
        double eff = (sp / t) * 100.0;

        if (sp > bestA) { bestA = sp; bestAt = t; }
        if (t == 2 || t == MAX_T) printf("Check: %s\n", check(REF,C) ? "OK" : "FAIL");

        printf("%4d   | %.3f | %.2fx | %.1f\n", t, avg, sp, eff);
    }

    divider();
    printf("\n=== Strategy B: 2D Parallel Grid ===\n");

    double bestB = 0; int bestBt = 1;

    for (int t = 1; t <= MAX_T; t++) {
        double sum = 0;
        for (int r = 0; r < TEST_RUNS; r++) {
            zero_out(C);
            sum += run_parallel_grid(t);
        }
        double avg = sum / TEST_RUNS;
        double sp = base / avg;

        if (sp > bestB) { bestB = sp; bestBt = t; }

        printf("%4d   | %.3f | %.2fx\n", t, avg, sp);
    }

    divider();
    printf("Best Row Strategy: %.2fx @ %d threads\n", bestA, bestAt);
    printf("Best Grid Strategy: %.2fx @ %d threads\n", bestB, bestBt);

    free(A); free(B); free(C); free(REF);
    return 0;
}