import concurrent.futures
from functools import update_wrapper, partial


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
            `args_iterate`. The `i`-th item of the resulting list corresponds
            to `args_iterate[i]` (the order is preserved).
        """
        handler = self._update_handler(handler, **kwargs)
        results = [handler(arg) for arg in args_iterate]
        return results


class ProcessPoolExecutor(SingleProcess):
    """
    The wrapper of Python built-in `concurrent.futures.ProcessPoolExecutor`
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
            `args_iterate`. The `i`-th item of the resulting list corresponds
            to `args_iterate[i]` (the order is preserved).
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
