"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionalities, e.g., to perform analysis in sliding windows.
"""

import sys
import quantities as pq
import elephant
from mpi4py import MPI


# MPI message tags
MPI_SEND_HANDLER = 1
MPI_SEND_INPUT = 2
MPI_SEND_OUTPUT = 3
MPI_WORKER_DONE = 4


# TODO: Rename as stampede?

class ParallelContext():
    """
    This function initializes the MPI subsystem.

    Parameters:
    comm: MPI.Communicator or None
        MPI communicator to use. If None is given, MPI_COMM_WORLD is used,
        i.e., all available MPI resources and the global communicator is used.
        When combining Elephant with other libraries or applications that also
        run in parallel, you can create multiple communicators to separate the
        individual processes.
        Default: None
    worker_ranks: list or None
        List of ranks to be used as workers within the communicator. If None,
        all ranks 1..N-1 of the communicator `comm`, hosting N ranks, will be
        used, and rank 0 is considered the master node.
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
        
        # Save number of workers
        self.num_workers = len(self.worker_ranks)
        
        # Save status and name of the rank
        self.status = MPI.Status()
        self.rank_name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()
        # self.attributes = self.comm.Get_attr()

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
        # TODO: Shut down workers
        pass

    def __run_worker(self):
        while True:
            # Await a handler function
            req = self.comm.irecv(source=0, tag=MPI_SEND_HANDLER)
            handler = req.wait()

            # Await data
            req = self.comm.irecv(source=0, tag=MPI_SEND_INPUT)
            data = req.wait()

            # Execute handler
            handler.worker(data)

            # Report back that we are done
#             req = self.parallel_context.comm.isend(
#                 0, 0, tag=MPI_WORKER_DONE)
#             req.wait()

        sys.exit(0)


class JobQ:
    def __init__(self, parallel_context):
        self.parallel_context = parallel_context

    def add_spiketrain_list_job(self, spiketrain_list, handler):
        self.spiketrain_list = spiketrain_list
        self.handler = handler
        
    def execute(self):
#         # Send handler to all processes
#         for i in range(self.parallel_context.comm_size-1):
#             req = self.parallel_context.comm.isend(
#                 self.handler, i+1, tag=MPI_SEND_HANDLER)
#             req.wait()

        # Save status of each worker
        worker_busy = [False for _ in range(self.parallel_context.num_workers)]

        # Send all spike trains
        while len(self.spiketrain_list) > 0:
            if False in worker_busy:
                available_worker = worker_busy.index(False)

                next = self.spiketrain_list.pop()

                # Send handler
                req = self.parallel_context.comm.isend(
                    self.handler, available_worker+1, tag=MPI_SEND_HANDLER)
                req.wait()

                # Send data
                req = self.parallel_context.comm.isend(
                    next, available_worker+1, tag=MPI_SEND_INPUT)
                req.wait()

                worker_busy[available_worker] = True

#             # Completing worker?
#             req = self.comm.irecv(source=0, tag=MPI_WORKER_DONE)


class JobQHandlers():
    def __init__(self):
        pass

    def handler(self):
        pass


class JobQSpikeTrainListHandler(JobQHandlers):
    def worker(self, spiketrain):
        result = elephant.statistics.lv(spiketrain)
        print(result)
        # return result


def main():
    # Initialize context
    pc = ParallelContext()
    print("%s, %i" % (pc.rank_name, pc.comm_size))

    # Create a list of spike trains, one per worker
    s = [elephant.spike_train_generation.homogeneous_poisson_process(
        10*pq.Hz, t_start=0*pq.s, t_stop=20*pq.s)
        for _ in range(pc.comm_size-1)]

    # Create a new queue operating on the current context
    handler = JobQSpikeTrainListHandler()

    new_q = JobQ(pc)
    new_q.add_spiketrain_list_job(s, handler)
    new_q.execute()
    
    # Send one spike train to each worker, tag=11
#     for i in range(pc.comm_size-1):
#         req = pc.comm.isend(s[i], i+1, tag=11)
#         req.wait()


if __name__ == "__main__":
    main()
