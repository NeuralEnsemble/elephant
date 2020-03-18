import time
from functools import wraps

from elephant.parallel import SingleProcess, MPIPoolExecutor, \
    ProcessPoolExecutor, MPICommExecutor


def benchmark(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__}: {duration:.3f} sec")
        return res

    return wrapped


@benchmark
def test_single_process(handler, args_list, **kwargs):
    SingleProcess().execute(handler=handler,
                            args_iterate=args_list, **kwargs)


@benchmark
def test_mpi_pool(handler, args_list, **kwargs):
    mpi_pool = MPIPoolExecutor()
    mpi_pool.execute(handler=handler,
                     args_iterate=args_list, **kwargs)
    print(mpi_pool)


@benchmark
def test_mpi_comm_exec(handler, args_list, **kwargs):
    MPICommExecutor().execute(handler=handler, args_iterate=args_list,
                              **kwargs)


@benchmark
def test_process_pool(handler, args_list, **kwargs):
    ProcessPoolExecutor().execute(handler=handler,
                                  args_iterate=args_list, **kwargs)


def test_sleep():
    # test to check the overhead of spawning MPI, OpenMP processes
    sleep_sec = [0.5] * 10
    test_single_process(handler=time.sleep, args_list=sleep_sec)
    test_process_pool(handler=time.sleep, args_list=sleep_sec)
    # test_mpi_pool(handler=time.sleep, args_list=sleep_sec)
    test_mpi_comm_exec(handler=time.sleep, args_list=sleep_sec)


if __name__ == '__main__':
    # mpirun python -m mpi4py.futures parallel/tests.py
    test_sleep()
