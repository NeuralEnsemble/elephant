"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionalities, e.g., to perform analysis in sliding windows.
"""

import sys
import time
import quantities as pq
import elephant
from mpi4py import MPI


# MPI message tags
MPI_SEND_HANDLER = 1
MPI_SEND_INPUT = 2
MPI_SEND_OUTPUT = 3
MPI_WORKER_DONE = 4
MPI_TERM_WORKER = 5


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
            self.worker_ranks = range(1, self.comm_size)
        else:
            worker_ranks = set(worker_ranks)
            if (len(worker_ranks) > self.comm_size-1 or
                    max(worker_ranks) > self.comm_size or
                    min(worker_ranks) < 1):
                raise ValueError(
                    "Elements of worker_ranks must be >0 and <N, "
                    "where N is the communicator size.")
            self.worker_ranks = list(worker_ranks).sort()
        
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
        # Shut down workers
        for worker in self.worker_ranks:
            req = self.comm.isend(
                0, worker, tag=MPI_TERM_WORKER)
            req.wait()

    def __run_worker(self):
        keep_working = True
        
        while keep_working:
            if self.comm.iprobe(source=0, tag=MPI_SEND_HANDLER):
                # Await a handler function
                req = self.comm.irecv(source=0, tag=MPI_SEND_HANDLER)
                handler = req.wait()

                # Await data
                req = self.comm.irecv(source=0, tag=MPI_SEND_INPUT)
                data = req.wait()

                # Execute handler
                handler.worker(data)

                # Report back that we are done
                req = self.comm.isend(
                    True, 0, tag=MPI_WORKER_DONE)
                req.wait()
            elif self.comm.iprobe(source=0, tag=MPI_TERM_WORKER):
                keep_working = False

        sys.exit(0)


class JobQueue:
    def __init__(self, parallel_context):
        self.parallel_context = parallel_context

    def add_spiketrain_list_job(self, spiketrain_list, handler):
        self.spiketrain_list = spiketrain_list
        self.handler = handler

    def execute(self):
        # Save status of each worker
        worker_busy = [False for _ in range(self.parallel_context.num_workers)]

        # Send all spike trains
        while len(self.spiketrain_list) > 0 or True not in worker_busy:
            if False in worker_busy:
                idle_worker_index = worker_busy.index(False)
                idle_worker = self.parallel_context.worker_ranks[
                    idle_worker_index]

                next = self.spiketrain_list.pop()

                # Send handler
                req = self.parallel_context.comm.isend(
                    self.handler, idle_worker, tag=MPI_SEND_HANDLER)
                req.wait()

                # Send data
                req = self.parallel_context.comm.isend(
                    next, idle_worker, tag=MPI_SEND_INPUT)
                req.wait()

                worker_busy[idle_worker_index] = True
                # TODO: Remove
                # print("%i started" % (idle_worker))

            # Any completing worker?
            for worker_index, worker in enumerate(
                    self.parallel_context.worker_ranks):
                if self.parallel_context.comm.iprobe(
                        source=worker, tag=MPI_WORKER_DONE):
                    req = self.parallel_context.comm.irecv(
                        source=worker, tag=MPI_WORKER_DONE)
                    _ = req.wait()
                    worker_busy[worker_index] = False
                    # TODO: Remove
                    # print("%i completed" % (worker))

class JobQueueHandlers():
    def __init__(self):
        pass

    def handler(self):
        pass


class JobQueueSpikeTrainListHandler(JobQueueHandlers):
    def worker(self, spiketrain):
        # Do something complicated
        for _ in range(1000):
            result = elephant.statistics.lv(spiketrain)
        # print(result)
        # return result


def main():
    # Initialize context
    pc = ParallelContext()
    # pc = ParallelContext(worker_ranks=[3,4])
    print("%s, %i" % (pc.rank_name, pc.comm_size))

    # Create a list of spike trains
    spiketrain_list = [
        elephant.spike_train_generation.homogeneous_poisson_process(
            10*pq.Hz, t_start=0*pq.s, t_stop=20*pq.s)
        for _ in range(100)]

    # Create a new queue operating on the current context
    handler = JobQueueSpikeTrainListHandler()

    ta = time.time()
    for s in spiketrain_list:
        # Do something complicated
        for _ in range(1000):
            result = elephant.statistics.lv(s)
        # print(result)
    ta = time.time()-ta

    tb = time.time()
    new_q = JobQueue(pc)
    new_q.add_spiketrain_list_job(spiketrain_list, handler)
    new_q.execute()
    tb = time.time()-tb

    # Send one spike train to each worker
    pc.terminate()

    print("Standard: %f s, Parallel: %f s" % (ta, tb))


if __name__ == "__main__":
    main()
