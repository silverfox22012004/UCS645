#ifndef DIVISOR_CHECK_H
#define DIVISOR_CHECK_H

#include <mpi.h>

/*
 * Verifies whether a given integer is a perfect number by
 * summing its proper divisors. Returns the number itself
 * when it is perfect, otherwise returns its negation.
 */
int check_perfect_number(int val);

#endif
