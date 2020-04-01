import mpi4py.futures
from mpi4py import MPI

from .parallel import ProcessPoolExecutor


class MPIPoolExecutor(ProcessPoolExecutor):
    """
    The `MPIPoolExecutor` class uses a pool of MPI processes to execute calls
    asynchronously.

    `MPIPoolExecutor` is recommended to use on cluster nodes which support
    MPI-2 protocol.

    Notes
    -----
    `-m mpi4py.futures` command line option is needed to execute Python scripts
    with MPI:

    .. code-block:: sh

       mpiexec -n numprocs python -m mpi4py.futures pyfile [arg] ...

    For more information of how to launch MPI processes in Python refer to
    https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#command-line
    """

    def _create_executor(self):
        return mpi4py.futures.MPIPoolExecutor(self.max_workers)

    def _extra_repr(self):
        extra_old = super(MPIPoolExecutor, self)._extra_repr()
        info = dict(MPI.INFO_ENV)
        return "{old}, {mpi}".format(old=extra_old, mpi=info)


class MPICommExecutor(MPIPoolExecutor):
    """
    Legacy MPI-1 implementation for cluster nodes which do not support MPI-2
    protocol.

    Parameters
    ----------
    comm : MPI.Intracomm or None
        MPI (intra)communicator. If None, set to `MPI.COMM_WORLD`.
        Default: None
    root : int
        Designated master process.
        Default: 0

    Notes
    -----
    `-m mpi4py.futures` command line option is needed to execute Python scripts
    with MPI:

    .. code-block:: sh

       mpiexec -n numprocs python -m mpi4py.futures pyfile [arg] ...

    For more information of how to launch MPI processes in Python refer to
    https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#command-line
    """
    def __init__(self, comm=None, root=0):
        super(MPICommExecutor, self).__init__(max_workers=None)
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.root = root

    def _create_executor(self):
        return mpi4py.futures.MPICommExecutor(comm=self.comm, root=self.root)
