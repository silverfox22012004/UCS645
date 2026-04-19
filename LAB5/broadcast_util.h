#ifndef BROADCAST_UTIL_H
#define BROADCAST_UTIL_H

#include <mpi.h>

/*
 * A manual broadcast implementation using point-to-point
 * MPI_Send / MPI_Recv calls inside a simple loop.
 */
void manual_broadcast(double* buf, int count, int root, MPI_Comm comm);

#endif
