"""
This package contains an implementation of embarassingly parallel processing
of Elephant functionalities, e.g., to perform analysis in sliding windows.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import time
from functools import wraps
import multiprocessing
import numpy as np
import quantities as pq
import elephant
from mpi4py import MPI


# MPI message tags
MPI_SEND_HANDLER = 1
MPI_SEND_INPUT = 2
MPI_SEND_INPUT_TYPE = 3
MPI_SEND_OUTPUT = 4
MPI_SEND_OUTPUT_TYPE = 5
MPI_WORKER_DONE = 6
MPI_TERM_WORKER = 7


class ParallelContext(object):
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

    def add_list_job(self, arg_list, handler):
        self.arg_list = arg_list
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

        # Send all arguments
        for job_id, arg in enumerate(self.arg_list):
            results[job_id] = self.handler.worker(arg)

        # Return results dictionary
        return results


class ParallelContext_Multithread(ParallelContext):
    """
    This function initializes a parallel context based on threads.

    Parameters:
    n_workers: int
        Number of ranks to be used as workers. If `None` is set, all cores of
        the current system are used.
        Default: None
    """

    def __init__(self, n_workers=None):
        self.name = "Parallel context using threads"

        n_max_workers = int(multiprocessing.cpu_count() - 1)

        # Save ranks for slaves
        if n_workers is None:
            n_workers = n_max_workers

        if n_workers < 1 or n_workers > n_max_workers:
            raise ValueError(
                "Too few available cores, cannot initialize multithreading.")

        # Save number of workers
        self.n_workers = n_workers

    def terminate(self):
        """
        This function terminates the threading subsystem.
        """
        pass

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
        # Create a pool
        pool = multiprocessing.Pool(processes=self.n_workers)

        # Launch jobs
        jobs = []
        for arg in self.arg_list:
            jobs.append(pool.apply_async(
                func=self.handler.worker, args=[arg]))
        pool.close()

        # Wait for jobs to finish
        results = {}
        for job_id, job in enumerate(jobs):
            results[job_id] = job.get()

        pool.join()
        pool.terminate()

        return results


class ParallelContext_MPI(ParallelContext):
    """
    This function initializes a parallel context using the MPI subsystem.

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
        if self.comm_size == 1:
                raise ValueError("Communicator size must be at least 2.")
        if worker_ranks is None:
            worker_ranks = list(range(1, self.comm_size))
        else:
            worker_ranks = list(set(worker_ranks))
        if (len(worker_ranks) > self.comm_size-1 or
                max(worker_ranks) > self.comm_size or
                min(worker_ranks) < 1):
            raise ValueError(
                "Elements of worker_ranks must be >0 and <N, "
                "where N is the communicator size.")
        self.worker_ranks = worker_ranks
        self.worker_ranks.sort()

        # Save number of workers
        self.n_workers = len(self.worker_ranks)

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

                # Get input type: list or single item
                req = self.comm.irecv(
                    source=0, tag=MPI_SEND_INPUT_TYPE)
                result_type = req.wait()

                if result_type == 0:
                    # Get input as single value
                    req = self.comm.irecv(
                        source=0, tag=MPI_SEND_INPUT)
                    data = req.wait()
                    # TODO: Remove print
                    # print("Receiving: " + str(type(data)))
                else:
                    # Get input as a list
                    data = []
                    import copy
                    for _ in range(result_type):
                        req = self.comm.irecv(
                            source=0, tag=MPI_SEND_INPUT)
                        data.append(copy.deepcopy(req.wait()))
                        # TODO: Remove print
                        # print("Receiving list of: " + str(type(data[-1])))

                # Execute handler
                result = handler.worker(data)

                # Report back that we are done
                req = self.comm.isend(
                    True, 0, tag=MPI_WORKER_DONE)
                req.wait()

                # Report back output type: list or single item
                if type(result) is list:
                    result_type = len(result)
                    if result_type == 0:
                        result = None
                else:
                    result_type = 0
                req = self.comm.isend(
                    result_type, 0, tag=MPI_SEND_OUTPUT_TYPE)
                req.wait()

                # Send return value
                if result_type == 0:
                    req = self.comm.isend(
                        result, 0, tag=MPI_SEND_OUTPUT)
                    req.wait()
                else:
                    for list_item in result:
                        req = self.comm.isend(
                            list_item, 0, tag=MPI_SEND_OUTPUT)
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
        worker_busy = [False for _ in range(self.n_workers)]

        # Save job ID currently executed by each worker
        worker_job_id = [-1 for _ in range(self.n_workers)]

        # Job ID counter
        job_id = 0

        # Save results for each job
        results = {}

        # Send all jobs
        while job_id < len(self.arg_list) or True in worker_busy:
            # Is there a free worker and work left to do?
            if job_id < len(self.arg_list) and False in worker_busy:
                idle_worker_index = worker_busy.index(False)
                idle_worker = self.worker_ranks[
                    idle_worker_index]

                # TODO: Remove print
                # print("Sending %i to %i" % (job_id, idle_worker))

                next_arg = self.arg_list[job_id]

                # Send handler
                req = self.comm.isend(
                    self.handler, idle_worker, tag=MPI_SEND_HANDLER)
                req.wait()

                # Report on input type: list or single item
                if type(next_arg) is list:
                    # List of values
                    arg_type = len(next_arg)
                    if arg_type == 0:
                        next_arg = None
                else:
                    # Single value
                    arg_type = 0
                req = self.comm.isend(
                    arg_type, idle_worker, tag=MPI_SEND_INPUT_TYPE)
                req.wait()

                # Send input value
                if arg_type == 0:
                    # Single value
                    # TODO: Remove print
                    # print("Sending: " + str(type(next_arg)))
                    req = self.comm.isend(
                        next_arg, idle_worker, tag=MPI_SEND_INPUT)
                    req.wait()
                else:
                    # List of values
                    for list_item in next_arg:
                        # TODO: Remove print
                        # print("Sending: " + str(type(list_item)))
                        req = self.comm.isend(
                            list_item, idle_worker, tag=MPI_SEND_INPUT)
                        req.wait()

                # Mark worker as busy and record its job ID
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

                    # Get output type: list or single item
                    req = self.comm.irecv(
                        source=worker, tag=MPI_SEND_OUTPUT_TYPE)
                    arg_type = req.wait()

                    if arg_type == 0:
                        # Get output
                        req = self.comm.irecv(
                            source=worker, tag=MPI_SEND_OUTPUT)
                        result = req.wait()
                    else:
                        result = []
                        for _ in range(arg_type):
                            req = self.comm.irecv(
                                source=worker, tag=MPI_SEND_OUTPUT)
                            result.append(req.wait())

                    # Save results and mark the worker as idle
                    results[worker_job_id[worker_index]] = result
                    worker_busy[worker_index] = False

        # Return results dictionary
        return results


class GlobalParallelContext():
    """
    This class defines a global parallel context to be used, e.g., by
    decorators or other parallelized functions.
    """

    def __init__(self):
        self.global_parallel_context = ParallelContext()

    def get_current_context(self):
        return self.global_parallel_context


# This defines the global parallel context
global_pc = GlobalParallelContext()


class JobQueueHandlers(object):
    """
    This is a base class for all job queue handlers.
    """
    def __init__(self):
        pass

    def worker(self):
        pass


class JobQueueListExpandHandler(JobQueueHandlers):
    """
    Jobs that call an Elephant with all elements of a list as the first of the
    parameters.

    Parameters:
    -----------
    func : function
        This is the underlying function to be executed.
    args : list
        Arguments to pass to func, with the exception of
        the first argument (which is later given by iteration of the list).
    kwargs : dict
        Keyword arguments to pass to func.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def worker(self, single_input):
        return self.func(single_input, *self.args, **self.kwargs)


class ParallelContextEmbarassingList():
    '''
    This is a decorator that transforms the first argument from a list into
    a series of function calls for each element
    '''
    def __init__(self):
        pass

    def __call__(self, func):
        @wraps(func)
        def embarassing_list_expand(*args, **kwargs):
            # If the first input argument is a list, then feed it into the
            # parallel context
            global global_pc
            if type(args[0]) is list:
                # TODO: Handle case where len(args)==1
                handler = JobQueueListExpandHandler(func, *args[1:], **kwargs)
                # TODO: Remove this print
                # print(global_pc.get_current_context().name)
                global_pc.get_current_context().add_list_job(args[0], handler)
                results_parallel = global_pc.get_current_context().execute()
                return results_parallel
            else:
                return func(*args, **kwargs)
        return embarassing_list_expand


@ParallelContextEmbarassingList()
def spike_train_generation(rate, n, t_start, t_stop):
    '''
    Returns a list of `n` spiketrains corresponding to the rates given in
    `rate_list`.
    '''
    return [elephant.spike_train_generation.homogeneous_poisson_process(
            rate, t_start=0*pq.s, t_stop=20*pq.s) for _ in range(n)]


def main():
    # Override global parallel context to use multithreading
    # TODO: For now, this does not work, thus stick with serial jobs
    # global_pc.global_parallel_context = ParallelContext_Multithread()

    # Test if script is running with mpirun, e.g.,:
    #     mpirun -n <cores> python elephant/parallel.py
    # If not, don't bother about MPI.
    if MPI.COMM_WORLD.Get_size() < 2:
        test_mpi = False
    else:
        test_mpi = True
    # TODO: For later, check why MPI fails so miserably, don't test for now
    # test_mpi = False

    # Initialize serial context (take everything, default)
    pc_serial = ParallelContext()

    # Initialize MPI context (take everything, default)
    if test_mpi:
        pc_mpi = ParallelContext_MPI()

    # Initialize MPI context (take everything, default)
    pc_mp = ParallelContext_Multithread()

    # Initialize MPI context (use only ranks 3 and 4 as slave, 0 as master)
    # pc_mpi = ParallelContext_MPI(worker_ranks=[3, 4])
    # if pc_mpi.rank != 0:
    #    sys.exit(0)

    print("MP Context:\nWorkers: %i\n" % (pc_mp.n_workers))
    if test_mpi:
        print(
            "MPI Context:\nMaster: %s, rank %i; Communicator size: %i\n"
            "Workers: %i" % (
                pc_mpi.rank_name, pc_mpi.rank, pc_mpi.comm_size,
                pc_mpi.n_workers))

    # =========================================================================
    # Test 1: Spike train generation
    # =========================================================================

    # Create a list of lists of spiketrains Each inner list of spike trains has
    # n_spiketrains entries, each with the same rate. The rate for the i-th
    # inner list is given by rate_list[i-1].
    rate_list = list(np.linspace(10, 20, 20)*pq.Hz)
    n_spiketrains = 1000

    ta = time.time()
    spiketrain_list_standard = [
        spike_train_generation(
            rate, n=n_spiketrains, t_start=0*pq.s, t_stop=20*pq.s)
        for rate in rate_list]
    ta = time.time()-ta
    print("Standard generation done.\n")

    # Create a new queue operating on the current context
    handler_generate = JobQueueListExpandHandler(
        spike_train_generation,
        n=n_spiketrains, t_start=0*pq.s, t_stop=20*pq.s)

    tb = time.time()
    pc_serial.add_list_job(
        arg_list=rate_list, handler=handler_generate)
    spiketrain_list_serial = pc_serial.execute()
    tb = time.time()-tb
    print("Serial generation done.\n")

    tc = time.time()
    if test_mpi:
        pc_mpi.add_list_job(rate_list, handler_generate)
        spiketrain_list_mpi = pc_mpi.execute()
    tc = time.time()-tc
    print("MPI generation done.\n")

    td = time.time()
    pc_mp.add_list_job(
        arg_list=rate_list, handler=handler_generate)
    spiketrain_list_mp = pc_mp.execute()
    td = time.time()-td
    print("MP generation done.\n")

    te = time.time()
    spiketrain_list_decorated = spike_train_generation(
        rate_list, n=n_spiketrains, t_start=0*pq.s, t_stop=20*pq.s)
    te = time.time()-te
    print("Decorator-style generation done.\n")

    print(
        "Generation execution times:" +
        "\nStandard: %f s, Serial: %f s, MPI: %f s, MP: %f s, Dec: %f s" %
        (ta, tb, tc, td, te))

    # =========================================================================
    # Test 2: Calculate a time histogram
    # =========================================================================

    # In the following, we calculate the PSTH for list of n_spiketrains
    # spiketrains.

    # Create a new queue operating on the current context
    handler = JobQueueListExpandHandler(
        elephant.statistics.time_histogram, binsize=50 * pq.ms, output='rate')

    # Test 1: Standard
    results_standard = {}
    ta = time.time()
    for i, s in enumerate(spiketrain_list_standard):
        results_standard[i] = elephant.statistics.time_histogram(
            s, binsize=50 * pq.ms, output='rate')
    ta = time.time()-ta
    print("Standard calculation done.\n")

    # Test 2: Serial Handler
    tb = time.time()
    pc_serial.add_list_job(spiketrain_list_standard, handler)
    results_serial = pc_serial.execute()
    tb = time.time()-tb
    print("Serial calculation done.\n")

    # Test 3: MPI Handler
    if test_mpi:
        tc = time.time()
        pc_mpi.add_list_job(spiketrain_list_standard, handler)
        results_mpi = pc_mpi.execute()
        tc = time.time()-tc
        print("MPI calculation done.\n")
    else:
        tc = 999

    # Test 4: MP Handler
    td = time.time()
    pc_mp.add_list_job(spiketrain_list_standard, handler)
    results_mp = pc_mp.execute()
    td = time.time()-td
    print("MP calculation done.\n")

    print(
        "Calculation execution times:" +
        "\nStandard: %f s, Serial: %f s, MPI: %f s, MP: %f s" %
        (ta, tb, tc, td))

    # These results should match, test last element of results
    cmp1 = len(rate_list)-1
    cmp2 = 5
    print("Standard result: %f" % results_standard[cmp1][cmp2])
    print("Serial result: %f" % results_serial[cmp1][cmp2])
    if test_mpi:
        print("MPI result: %f" % results_mpi[cmp1][cmp2])
    print("MP result: %f" % results_mp[cmp1][cmp2])
    assert(results_standard[cmp1][cmp2] == results_serial[cmp1][cmp2])
    if test_mpi:
        assert(results_standard[cmp1][cmp2] == results_mpi[cmp1][cmp2])
    assert(results_standard[cmp1][cmp2] == results_mp[cmp1][cmp2])

    # Terminate MPI
    if test_mpi:
        pc_mpi.terminate()


if __name__ == "__main__":
    main()
