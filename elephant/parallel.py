"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionalities, e.g., to perform analysis in sliding windows.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
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

class ParallelContext_MPI():
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
    worker_ranks: list of int or None
        List of ranks to be used as workers within the communicator. If None,
        all ranks 1..N-1 of the communicator `comm`, hosting N ranks, will be
        used, and rank 0 is considered the master node. If a specific list of
        ranks is given, only one of the remaining ranks may continue to act
        as the master and use the ParallelContext_MPI; all other ranks must be
        used in a different manner.
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
            self.worker_ranks = list(worker_ranks)
            self.worker_ranks.sort()

        # Save number of workers
        self.num_workers = len(self.worker_ranks)

        # Save status and name of the rank
        self.status = MPI.Status()
        self.rank_name = MPI.Get_processor_name()
        self.rank = self.comm.Get_rank()
        # TODO: Remove this if not required
        # self.attributes = self.comm.Get_attr()

        # If this is the master node or any node not in the current
        # communicator, continue with main program. Otherwise start a worker.
        if self.rank in self.worker_ranks:
            self.__run_worker()

    def terminate(self):
        """
        This function terminates the MPI subsystem.

        This function will send a terminate signal to all workers. It will
        return once all workers acknowledge the signal, so that the master can
        safely exit, knowing that all workers are done computing their thing.
        """
        # Shut down workers
        for worker in self.worker_ranks:
            req = self.comm.isend(
                0, worker, tag=MPI_TERM_WORKER)
            req.wait()

    def __run_worker(self):
        """
        The main worker loop that distributes jobs among the nodes.

        This is the function run on each worker upon initialization. It will
        continue until it receives an MPI_TERM_WORKER message. While running,
        the function waits for the master to instruct the worker to perform a
        function with a specific arguments. After execution, it reports back to
        the master that the worker is done and ready to execute another
        function.
        """
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
                result = handler.worker(data)

                # Report back that we are done
                req = self.comm.isend(
                    True, 0, tag=MPI_WORKER_DONE)
                req.wait()

                # Send return value
                req = self.comm.isend(
                    result, 0, tag=MPI_SEND_OUTPUT)
                req.wait()
            elif self.comm.iprobe(source=0, tag=MPI_TERM_WORKER):
                keep_working = False

        # The worker will exit, since it has no further defined function
        sys.exit(0)

    def add_spiketrain_list_job(self, spiketrain_list, handler):
        self.spiketrain_list = spiketrain_list
        self.handler = handler

    def execute(self):
        # Save status of each worker
        worker_busy = [False for _ in range(self.num_workers)]

        # Save job ID currently executed by each worker
        worker_job_id = [-1 for _ in range(self.num_workers)]

        # Job ID counter
        job_id = 0

        # Save results for each job
        results = {}

        # Send all spike trains
        while job_id < len(self.spiketrain_list) or True in worker_busy:
            # Is there a free worker and work left to do?
            if job_id < len(self.spiketrain_list) and False in worker_busy:
                idle_worker_index = worker_busy.index(False)
                idle_worker = self.worker_ranks[
                    idle_worker_index]

                next_spiketrain = self.spiketrain_list[job_id]

                # Send handler
                req = self.comm.isend(
                    self.handler, idle_worker, tag=MPI_SEND_HANDLER)
                req.wait()

                # Send data
                req = self.comm.isend(
                    next_spiketrain, idle_worker, tag=MPI_SEND_INPUT)
                req.wait()

                worker_busy[idle_worker_index] = True
                worker_job_id[idle_worker_index] = job_id
                job_id += 1

            # Any completing worker?
            for worker_index, worker in enumerate(
                    self.worker_ranks):
                if self.comm.iprobe(
                        source=worker, tag=MPI_WORKER_DONE):
                    req = self.comm.irecv(
                        source=worker, tag=MPI_WORKER_DONE)
                    _ = req.wait()

                    # Get output
                    req = self.comm.irecv(
                        source=worker, tag=MPI_SEND_OUTPUT)
                    result = req.wait()

                    # Save results and mark the worker as idle
                    results[worker_job_id[worker_index]] = result
                    worker_busy[worker_index] = False

        # Return results dictionary
        return results


class JobQueueHandlers():
    def __init__(self):
        pass

    def worker(self):
        pass


class JobQueueSpikeTrainListHandler(JobQueueHandlers):
    def worker(self, spiketrain):
        # Do something complicated
        for _ in range(1000):
            result = elephant.statistics.lv(spiketrain)
        return result


def main():
    # Initialize context (take everything, default)
    # pc = ParallelContext_MPI()

    # Initialize context (use only ranks 3 and 4 as slave, 0 as master)
    pc = ParallelContext_MPI(worker_ranks=[3, 4])
    if pc.rank != 0:
        sys.exit(0)

    print("Master: %s, rank %i; Communicator size: %i" % (
        pc.rank_name, pc.rank, pc.comm_size))

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
    ta = time.time()-ta

    tb = time.time()
    pc.add_spiketrain_list_job(spiketrain_list, handler)
    results = pc.execute()
    tb = time.time()-tb

    # Send one spike train to each worker
    pc.terminate()

    print("Execution times:\nStandard: %f s, Parallel: %f s" % (ta, tb))

    # These results should match
    print("Standard result: %f" % result)
    print("Parallel result: %f" % results[99])
    assert(result == results[99])


if __name__ == "__main__":
    main()
