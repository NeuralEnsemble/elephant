"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionalities, e.g., to perform analysis in sliding windows.
"""

import sys
import quantities as pq
import elephant
from mpi4py import MPI


# TODO: Rename as stampede?

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
    worker_ranks:
        List of ranks to be used as workers within the communicator. If None,
        all N-1 ranks of the current communicator, hosting N ranks, will be
        used.
        Default: None
    """

    def __init__(self, comm=None, worker_ranks=None):
        # Save communicator
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        # Get size of current MPI communicator
        self.comm_size = self.comm.Get_size()

        # Save ranks for slaves
        if worker_ranks is None:
            self.worker_ranks = set(range(1, self.comm_size))
        else:
            worker_ranks = set(worker_ranks)
            if (len(worker_ranks) > self.comm_size-1 or
                    max(worker_ranks) > self.comm_size or
                    min(worker_ranks) < 1):
                raise ValueError(
                    "Elements of worker_ranks must be >0 and <N, "
                    "where N is the communicator size.")
            self.worker_ranks = worker_ranks

        # Save status and name of the rank
        self.status = MPI.Status()
        self.rank_name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()
        #self.attributes = self.comm.Get_attr()

        # If this is the master node or any node not in the current
        # communicator, continue with main program. Otherwise start a worker.
        if self.rank in self.worker_ranks:
            self.__run_worker()

    def terminate(self):
        """
        This function initializes the MPI subsystem.

        Parameters:
        comm:
            MPI communicator to use. If None is given, MPI_COMM_WORLD is used,
            i.e., all available MPI resources and the global communicator is
            used. When combining Elephant with other libraries or applications
            that also run in parallel, you can create multiple communicators to
            separate the individual processes.
            Default: None
        worker_ranks:
            List of ranks to be used as slaves within the communicator. If 
            None, all N-1 ranks of the current communicator, hosting N ranks,
            will be used.
            Default: None
        """
        pass

    def __run_worker(self):
        while True:
            req = self.comm.irecv(source=0, tag=11)
            data = req.wait()
            print(data)
        sys.exit(0)


def main():
    pc = parallel_context()
    print("%s, %i" % (pc.rank_name, pc.comm_size))

    s = [elephant.spike_train_generation.homogeneous_poisson_process(
        10*pq.Hz, t_start=0*pq.s, t_stop=20*pq.s)
        for _ in range(pc.comm_size-1)]

    for i in range(pc.comm_size-1):
        req = pc.comm.isend(s[i], i+1, tag=11)
        req.wait()

if __name__ == "__main__":
    main()
