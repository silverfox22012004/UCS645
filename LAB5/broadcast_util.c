#include "broadcast_util.h"

void manual_broadcast(double* buf, int count, int root, MPI_Comm comm) {
    int my_rank, num_procs;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_procs);

    if (my_rank == root) {
        /* Root process distributes the buffer to every other rank */
        for (int dest = 0; dest < num_procs; dest++) {
            if (dest != root) {
                MPI_Send(buf, count, MPI_DOUBLE, dest, 0, comm);
            }
        }
    } else {
        /* Non-root ranks receive the buffer from root */
        MPI_Recv(buf, count, MPI_DOUBLE, root, 0, comm, MPI_STATUS_IGNORE);
    }
}
