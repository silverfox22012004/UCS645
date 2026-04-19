#ifndef VEC_DOTPROD_H
#define VEC_DOTPROD_H

#include <mpi.h>

/*
 * Computes the dot product of two locally-generated vectors.
 * Vector A is filled with 1.0, vector B with (2.0 * scale_factor).
 * Returns the partial dot product for the local chunk.
 */
double local_dot_product(long long chunk_size, double scale_factor);

#endif
