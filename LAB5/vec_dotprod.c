#include "vec_dotprod.h"
#include <stdlib.h>

double local_dot_product(long long chunk_size, double scale_factor) {
    double *vec_a = (double*)malloc(chunk_size * sizeof(double));
    double *vec_b = (double*)malloc(chunk_size * sizeof(double));
    double partial_sum = 0.0;

    if (vec_a == NULL || vec_b == NULL) return -1.0;

    /* Generate the two vectors locally to avoid global allocation */
    for (long long idx = 0; idx < chunk_size; idx++) {
        vec_a[idx] = 1.0;
        vec_b[idx] = 2.0 * scale_factor;
    }

    /* Accumulate the element-wise product */
    for (long long idx = 0; idx < chunk_size; idx++) {
        partial_sum += vec_a[idx] * vec_b[idx];
    }

    free(vec_a);
    free(vec_b);
    return partial_sum;
}
