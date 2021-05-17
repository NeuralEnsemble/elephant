"""
The ``elephant.parallel`` package provides classes to parallelize calls to any
user-specified function.

The typical use case is calling a function many times with different
parameters.

.. note::  This parallelization module is independent from Elephant and can be
           used to parallelize mutually independent calls to arbitrary
           functions.

Tutorial
--------

:doc:`View tutorial <../tutorials/parallel>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/parallel.ipynb


Available Executors
-------------------

.. autosummary::
    :toctree: _toctree/parallel/

    ProcessPoolExecutor
    MPIPoolExecutor
    MPICommExecutor


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import warnings

from .parallel import ProcessPoolExecutor, SingleProcess

try:
    from .mpi import MPIPoolExecutor, MPICommExecutor
except ImportError:
    # mpi4py is missing
    warnings.warn("mpi4py package is missing. Please run 'pip install mpi4py' "
                  "in a terminal to activate MPI features.")

__all__ = [
    "ProcessPoolExecutor",
    "SingleProcess",
    "MPIPoolExecutor",
    "MPICommExecutor"
]
