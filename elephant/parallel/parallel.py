"""
The ``elephant.parallel`` package provides classes to parallelize calls to any
user-specified function.

The typical use case is calling a function many times with different
parameters.

.. note::  This parallelization module is independent from Elephant and can be
           easily used in other projects.

Tutorial
--------

:doc:`View tutorial <../tutorials/parallel>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master?filepath=doc/tutorials/parallel.ipynb


Available Executors
-------------------

.. autosummary::
    :toctree: parallel/

    ProcessPoolExecutor
    MPIPoolExecutor
    MPICommExecutor


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import concurrent.futures
from functools import update_wrapper, partial

import mpi4py.futures
from mpi4py import MPI


class SingleProcess(object):
    """
    A fall-back parallel context that executes jobs sequentially.
    """

    def __repr__(self):
        return "{name}({extra})".format(name=self.__class__.__name__,
                                        extra=self._extra_repr())

    def _extra_repr(self):
        return ""

    @staticmethod
    def _update_handler(handler, **kwargs):
        handler_wrapper = partial(handler, **kwargs)
        update_wrapper(handler_wrapper, handler)
        return handler_wrapper

    def execute(self, handler, args_iterate, **kwargs):
        """
        Executes the queue of
        `[handler(arg, **kwargs) for arg in args_iterate]` in a single process
        (no speedup).

        Parameters
        ----------
        handler : callable
            A function to be executed for each argument in `args_iterate`.
        args_iterate : list
            A list of (different) values of the first argument of the `handler`
            function.
        kwargs
            Additional key arguments to `handler`.

        Returns
        -------
        results : list
            The result of applying the `handler` for each `arg` in the
            `args_iterate`. The `i`-th item of the resulted list corresponds to
            `args_iterate[i]` (the order is preserved).
        """
        handler = self._update_handler(handler, **kwargs)
        results = [handler(arg) for arg in args_iterate]
        return results


class ProcessPoolExecutor(SingleProcess):
    """
    The wrapper of python built-in `concurrent.futures.ProcessPoolExecutor`
    class.

    `ProcessPoolExecutor` is recommended to use if you have one physical
    machine (laptop or PC).

    Parameters
    ----------
    max_workers : int or None
        The maximum number of processes that can be used to
        execute the given calls. If None or not given then as many
        worker processes will be created as the machine has processors.
        Default: None
    """
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def _extra_repr(self):
        return "max_workers={0}".format(self.max_workers)

    def _create_executor(self):
        return concurrent.futures.ProcessPoolExecutor(self.max_workers)

    def execute(self, handler, args_iterate, **kwargs):
        """
        Executes the queue of
        `[handler(arg, **kwargs) for arg in args_iterate]` in multiple
        processes within one machine (`ProcessPoolExecutor`) or multiple
        nodes (`MPIPoolExecutor` and `MPICommExecutor`).

        Parameters
        ----------
        handler : callable
            A function to be executed for each argument in `args_iterate`.
        args_iterate : list
            A list of (different) values of the first argument of the `handler`
            function.
        kwargs
            Additional key arguments to `handler`.

        Returns
        -------
        results : list
            The result of applying the `handler` for each `arg` in the
            `args_iterate`. The `i`-th item of the resulted list corresponds to
            `args_iterate[i]` (the order is preserved).
        """
        handler = self._update_handler(handler, **kwargs)

        # if not initialized, MPICommExecutor crashes if run without
        # -m mpi4py.futures mode
        results = []

        with self._create_executor() as executor:
            results = executor.map(handler, args_iterate)
            # print(executor, results)
        results = list(results)  # convert a map to a list

        return results


class MPIPoolExecutor(ProcessPoolExecutor):
    """
    The `MPIPoolExecutor` class uses a pool of MPI processes to execute calls
    asynchronously.

    `MPIPoolExecutor` is recommended to use on cluster nodes which support
    MPI-2 protocol.

    Notes
    -----
    `-m mpi4py.futures` command line option is needed to execute python scripts
    with MPI:

    .. code-block:: sh

       mpiexec -n numprocs python -m mpi4py.futures pyfile [arg] ...

    For more information of how to launch MPI processes in python refer to
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
    `-m mpi4py.futures` command line option is needed to execute python scripts
    with MPI:

    .. code-block:: sh

       mpiexec -n numprocs python -m mpi4py.futures pyfile [arg] ...

    For more information of how to launch MPI processes in python refer to
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
