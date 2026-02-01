#include <stdio.h>
#include <omp.h>
#include <math.h>

#define SLICES 100000000L
#define RUNS 3
#define THREAD_CAP 32

// Compute π without parallelism
double pi_serial(long n, double *out_pi) {
    double h = 1.0 / (double)n;
    double sum = 0.0;

    double t_start = omp_get_wtime();

    for (long i = 0; i < n; i++) {
        double x = h * (i + 0.5);
        sum += 4.0 / (1.0 + x * x);
    }

    *out_pi = h * sum;
    return omp_get_wtime() - t_start;
}

// Compute π using OpenMP reduction (more efficient than critical)
double pi_parallel(long n, int tcount, double *out_pi) {
    double h = 1.0 / (double)n;
    double sum = 0.0;

    omp_set_num_threads(tcount);
    double t_start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < n; i++) {
        double x = h * (i + 0.5);
        sum += 4.0 / (1.0 + x * x);
    }

    *out_pi = h * sum;
    return omp_get_wtime() - t_start;
}

void line() {
    printf("------------------------------------------------------------\n");
}

int main() {

    printf("\nNUMERICAL PI ESTIMATION (INTEGRATION METHOD)\n");
    line();
    printf("Intervals: %ld\n", SLICES);
    printf("Reference π: %.15f\n", M_PI);
    printf("Max HW Threads: %d\n", omp_get_max_threads());
    line();

    // --- Serial baseline ---
    double base_time = 0.0, pi_val = 0.0;

    for (int i = 0; i < RUNS; i++)
        base_time += pi_serial(SLICES, &pi_val);

    base_time /= RUNS;

    printf("\nSerial Avg Time: %.4f sec\n", base_time);
    printf("Computed π: %.15f | Error: %.3e\n\n", pi_val, fabs(pi_val - M_PI));

    // --- Parallel tests ---
    printf("Threads | Avg Time | Speedup | Eff(%%) | π Estimate\n");
    printf("--------|----------|---------|---------|-----------------\n");

    double best_gain = 0.0;
    int best_threads = 1;

    for (int t = 1; t <= THREAD_CAP; t++) {

        double t_sum = 0.0, pi_out = 0.0;

        for (int r = 0; r < RUNS; r++)
            t_sum += pi_parallel(SLICES, t, &pi_out);

        double avg = t_sum / RUNS;
        double sp = base_time / avg;
        double efficiency = (sp / t) * 100.0;

        if (sp > best_gain) {
            best_gain = sp;
            best_threads = t;
        }

        printf(" %3d    | %.4f  | %.2fx  | %.1f  | %.15f\n",
               t, avg, sp, efficiency, pi_out);
    }

    line();
    printf("Best configuration: %d threads (%.2fx speedup)\n", best_threads, best_gain);
    line();

    return 0;
}