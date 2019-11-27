"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionality in a worker-slave manner.
"""

from mpi4py import MPI


class parallel_context():
    """
    This function initializes the MPI subsystem.

    Parameters:
    comm:
        MPI communicator to use. If None is given, MPI_COMM_WORLD is used,
        i.e., all available MPI resources and the global communicator is used.
        When combining Elephant with other libraries or applications that also
        run in parallel, you can create multiple communicators to separate the
        individual processes.
        Default: None
    slave_ranks:
        List of ranks to be used as slaves within the communicator. If None,
        all N-1 ranks of the current communicator, hosting N ranks, will be
        used.
        Default: None
    """

    def __init__(self, comm=None, slave_ranks=None):
        # Save communicator
        if not comm:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        # Get size of current MPI communicator
        self.comm_size = MPI.COMM_WORLD.Get_size()

        # Save ranks for slaves
        if not slave_ranks:
            self.slave_ranks = range(1, self.comm_size)
        else:
            slave_ranks = set(slave_ranks)
            if (len(slave_ranks > self.comm_size-1) or
                    max(slave_ranks) > self.comm_size or
                    min(slave_ranks) < 1):
                raise ValueError(
                    "Elements of slave_ranks must be >0 and <N, "
                    "where N is the communicator size.")
            self.slave_ranks = slave_ranks

        self.status = MPI.Status()
        self.rank_name = MPI.Get_processor_name()


def main():
    pc = parallel_context()

    print("%s, %i" % (pc.rank_name, pc.comm_size))


if __name__ == "__main__":
    main()
    