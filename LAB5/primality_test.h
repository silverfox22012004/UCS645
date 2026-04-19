#ifndef PRIMALITY_TEST_H
#define PRIMALITY_TEST_H

#include <mpi.h>

/*
 * Determines whether the provided integer is a prime.
 * If it is prime the function returns the value itself;
 * otherwise it returns the negated value.
 */
int evaluate_prime(int num);

#endif
