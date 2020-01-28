"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionalities, e.g., to perform analysis in sliding windows.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import time
import functools
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
    This function provides a fall-back parallel context that executes jobs
    sequentially.
    """

    def __init__(self):
        self.name = "Fall-back parallel context implementing serial processing"
        pass

    def terminate(self):
        """
        This function terminates the serial parallel context.
        """
        pass

    def add_list_job(self, spiketrain_list, handler):
        self.spiketrain_list = spiketrain_list
        self.handler = handler

    def execute(self):
        """
        Executes the current queue of jobs sequentially, as if jobs were called
        in a for-loop, and returns all results when done.

        Returns:
        --------
        results : dict
            A dictionary containing results of all submitted jobs. The keys are
            set to the job ID, and integer number counting the submitted jobs,
            starting at 0.
        """
        # Job ID counter
        job_id = 0

        # Save results for each job
        results = {}

        # Send all spike trains
        while job_id < len(self.spiketrain_list):
            next_spiketrain = self.spiketrain_list[job_id]
            results[job_id] = self.handler.worker(next_spiketrain)
            job_id += 1

        # Return results dictionary
        return results


class ParallelContext_MPI(ParallelContext):
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
        self.name = "Parallel context using openMPI"

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

    def execute(self):
        """
        Executes the current queue of jobs on the MPI workers, and returns all
        results when done.

        Returns:
        --------
        results : dict
            A dictionary containing results of all submitted jobs. The keys are
            set to the job ID, and integer number counting the submitted jobs,
            starting at 0.
        """
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
                    results[worker_job_id[worker_index]] = result+1
                    worker_busy[worker_index] = False

        # Return results dictionary
        return results


class GlobalParallelContext():
    def __init__(self):
        self.global_parallel_context = ParallelContext()

    def get_current_context(self):
        return self.global_parallel_context


global_pc = GlobalParallelContext()
global_pc.global_parallel_context = ParallelContext_MPI()


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


class JobQueueExpandHandler(JobQueueHandlers):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        if len(args) > 1:
            self.args = args[1:]
        else:
            self.args = []
        self.kwargs = kwargs

    def worker(self, single_input):
        return self.func(single_input, *self.args, **self.kwargs)


def parallel_context_embarassing_list(func):
    '''
    This is a decorator that transforms the first argument from a list into
    a series of function calls for each element
    '''
    #@functools.wraps(func)
    def embarassing_list_expand(*args, **kwargs):
        # If the first input argument is a list, then feed it into the parallel
        # context
        global global_pc
        if type(args[0]) is list:
            handler = JobQueueExpandHandler(func, *args, **kwargs)
            print(global_pc.get_current_context().name)
            global_pc.get_current_context().add_list_job(args[0], handler)
            results_mpi = global_pc.get_current_context().execute()
            return results_mpi
        else:
            return func(*args, **kwargs)
    return embarassing_list_expand


def main():
    # Initialize serial context (take everything, default)
    pc_serial = ParallelContext()

    # Initialize MPI context (take everything, default)
    pc_mpi = ParallelContext_MPI()

    # Initialize MPI context (use only ranks 3 and 4 as slave, 0 as master)
    # pc_mpi = ParallelContext_MPI(worker_ranks=[3, 4])
    # if pc_mpi.rank != 0:
    #    sys.exit(0)

    print("MPI Context:\nMaster: %s, rank %i; Communicator size: %i" % (
        pc_mpi.rank_name, pc_mpi.rank, pc_mpi.comm_size))

    # =========================================================================
    # Decorator test
    # =========================================================================

    # Create a list of spike trains
    spiketrain_list = [
        elephant.spike_train_generation.homogeneous_poisson_process(
            10*pq.Hz, t_start=0*pq.s, t_stop=20*pq.s)
        for _ in range(10000)]

    global global_pc

    global_pc.global_parallel_context = pc_serial
    td = time.time()
    results_decorate_serial = elephant.statistics.lv(spiketrain_list)
    td = time.time()-td

    global_pc.global_parallel_context = pc_mpi
    te = time.time()
    results_decorate_mpi = elephant.statistics.lv(spiketrain_list)
    te = time.time()-te

    print(
        "Decorator execution times:" +
        "\nSerial: %f s, Parallel: %f s" % (td, te))
    print("Serial result: %f" % results_decorate_serial[105])
    print("Parallel result: %f" % results_decorate_mpi[105])
    assert(
        results_decorate_serial[105] == elephant.statistics.lv(
            spiketrain_list[105]))
    assert(
        results_decorate_mpi[105] == elephant.statistics.lv(
            spiketrain_list[105]))

    # =========================================================================
    # User defined worker
    # =========================================================================
    
    # Create a list of spike trains
    spiketrain_list = [
        elephant.spike_train_generation.homogeneous_poisson_process(
            10*pq.Hz, t_start=0*pq.s, t_stop=20*pq.s)
        for _ in range(100)]

    # Create a new queue operating on the current context
    handler = JobQueueSpikeTrainListHandler()

    # Test 1: Standard
    results_standard = {}
    ta = time.time()
    for s in spiketrain_list:
        # Do something complicated
        for i in range(1000):
            results_standard[i] = elephant.statistics.lv(s)
    ta = time.time()-ta

    # Test 2: Serial Handler
    tb = time.time()
    pc_serial.add_list_job(spiketrain_list, handler)
    results_serial = pc_serial.execute()
    tb = time.time()-tb

    # Test 3: MPI Handler
    tc = time.time()
    pc_mpi.add_list_job(spiketrain_list, handler)
    results_mpi = pc_mpi.execute()
    tc = time.time()-tc

    print(
        "Execution times:" +
        "\nStandard: %f s, Serial: %f s, Parallel: %f s" % (ta, tb, tc))

    # These results should match
    print("Standard result: %f" % results_standard[99])
    print("Serial result: %f" % results_serial[99])
    print("Parallel result: %f" % results_mpi[99])
    assert(results_standard[99] == results_serial[99])
    assert(results_standard[99] == results_mpi[99])

    # Terminate MPI
    pc_mpi.terminate()


if __name__ == "__main__":
    main()
