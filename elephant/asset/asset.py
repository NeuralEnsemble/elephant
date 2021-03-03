# -*- coding: utf-8 -*-
"""
ASSET is a statistical method :cite:`asset-Torre16_e1004939` for the detection
of repeating sequences of synchronous spiking events in parallel spike trains.


ASSET analysis class object of finding patterns
-----------------------------------------------

.. autosummary::
    :toctree: _toctree/asset/

    ASSET


Patterns post-exploration
-------------------------

.. autosummary::
    :toctree: _toctree/asset/

    synchronous_events_intersection
    synchronous_events_difference
    synchronous_events_identical
    synchronous_events_no_overlap
    synchronous_events_contained_in
    synchronous_events_contains_all
    synchronous_events_overlap


Tutorial
--------

:doc:`View tutorial <../tutorials/asset>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/asset.ipynb


Examples
--------
In this example we

  * simulate two noisy synfire chains;
  * shuffle the neurons to destroy visual appearance;
  * run ASSET analysis to recover the original neurons arrangement.

1. Simulate two noise synfire chains, shuffle the neurons to destroy the
   pattern visually, and store shuffled activations in neo.SpikeTrains.

   >>> import neo
   >>> import numpy as np
   >>> import quantities as pq
   >>> np.random.seed(10)
   >>> spiketrain = np.linspace(0, 50, num=10)
   >>> np.random.shuffle(spiketrain)
   >>> spiketrains = np.c_[spiketrain, spiketrain + 100]
   >>> spiketrains += np.random.random_sample(spiketrains.shape) * 5
   >>> spiketrains = [neo.SpikeTrain(st, units='ms', t_stop=1 * pq.s)
   ...                for st in spiketrains]

2. Create `ASSET` class object that holds spike trains.

   `ASSET` requires at least one argument - a list of spike trains. If
   `spiketrains_y` is not provided, the same spike trains are used to build an
   intersection matrix with.

   >>> from elephant import asset
   >>> asset_obj = asset.ASSET(spiketrains, bin_size=3*pq.ms)

3. Build the intersection matrix `imat`:

   >>> imat = asset_obj.intersection_matrix()

4. Estimate the probability matrix `pmat`, using the analytical method:

   >>> pmat = asset_obj.probability_matrix_analytical(imat,
   ...                                                kernel_width=50*pq.ms)

5. Compute the joint probability matrix `jmat`, using a suitable filter:

   >>> jmat = asset_obj.joint_probability_matrix(pmat, filter_shape=(5, 1),
   ...                                           n_largest=3)

6. Create the masked version of the intersection matrix, `mmat`, from `pmat`
   and `jmat`:

   >>> mmat = asset_obj.mask_matrices([pmat, jmat], thresholds=.9)

7. Cluster significant elements of imat into diagonal structures:

   >>> cmat = asset_obj.cluster_matrix_entries(mmat, max_distance=11,
   ...                                         min_neighbors=3, stretch=5)

9. Extract sequences of synchronous events:

   >>> sses = asset_obj.extract_synchronous_events(cmat)

The ASSET found the following sequences of synchronous events:

>>> sses
{1: {(36, 2): {5},
  (37, 4): {1},
  (40, 6): {4},
  (41, 7): {8},
  (43, 9): {2},
  (47, 14): {7},
  (48, 15): {0},
  (50, 17): {9}}}

To visualize them, refer to Viziphant documentation and an example plot
:func:`viziphant.asset.plot_synchronous_events`.

"""
from __future__ import division, print_function, unicode_literals

import math
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import neo
import numpy as np
import quantities as pq
import scipy.spatial
import scipy.stats
from sklearn.cluster import dbscan
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from tqdm import trange, tqdm

import elephant.conversion as conv
from elephant import spike_train_surrogates
from elephant.utils import get_cuda_capability_major

try:
    from mpi4py import MPI

    mpi_accelerated = True
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except ImportError:
    mpi_accelerated = False
    size = 1
    rank = 0


__all__ = [
    "ASSET",
    "synchronous_events_intersection",
    "synchronous_events_difference",
    "synchronous_events_identical",
    "synchronous_events_no_overlap",
    "synchronous_events_contained_in",
    "synchronous_events_contains_all",
    "synchronous_events_overlap"
]


# =============================================================================
# Some Utility Functions to be dealt with in some way or another
# =============================================================================


def _signals_same_attribute(signals, attr_name):
    """
    Check whether a list of signals (`neo.AnalogSignal` or `neo.SpikeTrain`)
    have same attribute `attr_name`. If so, return that value. Otherwise,
    raise ValueError.

    Parameters
    ----------
    signals : list
        A list of signals (e.g. `neo.AnalogSignal` or `neo.SpikeTrain`) having
        attribute `attr_name`.

    Returns
    -------
    pq.Quantity
        The value of the common attribute `attr_name` of the list of signals.

    Raises
    ------
    ValueError
        If `signals` is an empty list.

        If `signals` have different `attr_name` attribute values.
    """
    if len(signals) == 0:
        raise ValueError('Empty signals list')
    attribute = getattr(signals[0], attr_name)
    for sig in signals[1:]:
        if getattr(sig, attr_name) != attribute:
            raise ValueError(
                "Signals have different '{}' values".format(attr_name))
    return attribute


def _quantities_almost_equal(x, y):
    """
    Returns True if two quantities are almost equal, i.e., if `x - y` is
    "very close to 0" (not larger than machine precision for floats).

    Parameters
    ----------
    x : pq.Quantity
        First Quantity to compare.
    y : pq.Quantity
        Second Quantity to compare. Must have same unit type as `x`, but not
        necessarily the same shape. Any shapes of `x` and `y` for which `x - y`
        can be calculated are permitted.

    Returns
    -------
    np.ndarray
        Array of `bool`, which is True at any position where `x - y` is almost
        zero.

    Notes
    -----
    Not the same as `numpy.testing.assert_allclose` (which does not work
    with Quantities) and `numpy.testing.assert_almost_equal` (which works only
    with decimals)
    """
    eps = np.finfo(float).eps
    relative_diff = (x - y).magnitude
    return np.all([-eps <= relative_diff, relative_diff <= eps], axis=0)


def _transactions(spiketrains, bin_size, t_start, t_stop, ids=None):
    """
    Transform parallel spike trains into a list of sublists, called
    transactions, each corresponding to a time bin and containing the list
    of spikes in `spiketrains` falling into that bin.

    To compute each transaction, the spike trains are binned (with adjacent
    exclusive binning) and clipped (i.e., spikes from the same train falling
    in the same bin are counted as one event). The list of spike IDs within
    each bin form the corresponding transaction.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or list of tuple
        A list of `neo.SpikeTrain` objects, or list of pairs
        (Train_ID, `neo.SpikeTrain`), where `Train_ID` can be any hashable
        object.
    bin_size : pq.Quantity
        Width of each time bin. Time is binned to determine synchrony.
    t_start : pq.Quantity
        The starting time. Only spikes occurring at times `t >= t_start` are
        considered. The first transaction contains spikes falling into the
        time segment `[t_start, t_start+bin_size]`.
        If None, takes the value of `spiketrain.t_start`, common for all
        input `spiketrains` (raises ValueError if it's not the case).
        Default: None
    t_stop : pq.Quantity
        The ending time. Only spikes occurring at times `t < t_stop` are
        considered.
        If None, takes the value of `spiketrain.t_stop`, common for all
        input `spiketrains` (raises ValueError if it's not the case).
        Default: None
    ids : list of int, optional
        List of spike train IDs.
        If None, the IDs `0` to `N-1` are used, where `N` is the number of
        input spike trains.
        Default: None

    Returns
    -------
    list of list
        A list of transactions, where each transaction corresponds to a time
        bin and represents the list of spike train IDs having a spike in that
        time bin.

    Raises
    ------
    TypeError
        If `spiketrains` is not a list of `neo.SpikeTrain` or a list of tuples
        (id, `neo.SpikeTrain`).
    """

    if all(isinstance(st, neo.SpikeTrain) for st in spiketrains):
        trains = spiketrains
        if ids is None:
            ids = range(len(spiketrains))
    else:
        # (id, SpikeTrain) pairs
        try:
            ids, trains = zip(*spiketrains)
        except TypeError:
            raise TypeError('spiketrains must be either a list of ' +
                            'SpikeTrains or a list of (id, SpikeTrain) pairs')

    # Bin the spike trains and take for each of them the ids of filled bins
    binned = conv.BinnedSpikeTrain(
        trains, bin_size=bin_size, t_start=t_start, t_stop=t_stop)
    filled_bins = binned.spike_indices

    # Compute and return the transaction list
    return [[train_id for train_id, b in zip(ids, filled_bins)
             if bin_id in b] for bin_id in range(binned.n_bins)]


def _analog_signal_step_interp(signal, times):
    """
    Compute the step-wise interpolation of a signal at desired times.

    Given a signal (e.g. a `neo.AnalogSignal`) `s` taking values `s[t0]` and
    `s[t1]` at two consecutive time points `t0` and `t1` (`t0 < t1`), the value
    of the step-wise interpolation at time `t: t0 <= t < t1` is given by
    `s[t] = s[t0]`.

    Parameters
    ----------
    signal : neo.AnalogSignal
        The analog signal, containing the discretization of the function to
        interpolate.
    times : pq.Quantity
        A vector of time points at which the step interpolation is computed.

    Returns
    -------
    pq.Quantity
        Object with same shape of `times` and containing
        the values of the interpolated signal at the time points in `times`.
    """
    dt = signal.sampling_period

    # Compute the ids of the signal times to the left of each time in times
    time_ids = np.floor(
        ((times - signal.t_start) / dt).rescale(
            pq.dimensionless).magnitude).astype('i')

    return (signal.magnitude[time_ids] * signal.units).rescale(signal.units)


# =============================================================================
# HERE ASSET STARTS
# =============================================================================


def _stretched_metric_2d(x, y, stretch, ref_angle, working_memory=None):
    r"""
    Given a list of points on the real plane, identified by their abscissa `x`
    and ordinate `y`, compute a stretched transformation of the Euclidean
    distance among each of them.

    The classical euclidean distance `d` between points `(x1, y1)` and
    `(x2, y2)`, i.e., :math:`\sqrt((x1-x2)^2 + (y1-y2)^2)`, is multiplied by a
    factor

    .. math::

            1 + (stretch - 1.) * \abs(\sin(ref_angle - \theta)),

    where :math:`\theta` is the angle between the points and the 45 degree
    direction (i.e., the line `y = x`).

    The stretching factor thus steadily varies between 1 (if the line
    connecting `(x1, y1)` and `(x2, y2)` has inclination `ref_angle`) and
    `stretch` (if that line has inclination `90 + ref_angle`).

    Parameters
    ----------
    x : (n,) np.ndarray
        Array of abscissas of all points among which to compute the distance.
    y : (n,) np.ndarray
        Array of ordinates of all points among which to compute the distance
        (same shape as `x`).
    stretch : float
        Maximum stretching factor, applied if the line connecting the points
        has inclination `90 + ref_angle`.
    ref_angle : float
        Reference angle in degrees (i.e., the inclination along which the
        stretching factor is 1).

    Returns
    -------
    D : (n,n) np.ndarray
        Square matrix of distances between all pairs of points.

    """
    alpha = np.deg2rad(ref_angle)  # reference angle in radians

    # Create the array of points (one per row) for which to compute the
    # stretched distance
    points = np.column_stack([x, y])

    x_array = np.expand_dims(x, axis=0)
    y_array = np.expand_dims(y, axis=0)

    def calculate_stretch_mat(theta_mat, D_mat):
        # Transform [-pi, pi] back to [-pi/2, pi/2]
        theta_mat[theta_mat < -np.pi / 2] += np.pi
        theta_mat[theta_mat > np.pi / 2] -= np.pi

        # Compute the matrix of stretching factors for each pair of points.
        # Equivalent to:
        #   stretch_mat = 1 + (stretch - 1.) * np.abs(np.sin(alpha - theta))
        _stretch_mat = np.subtract(alpha, theta_mat, out=theta_mat)
        _stretch_mat = np.sin(_stretch_mat, out=_stretch_mat)
        _stretch_mat = np.abs(_stretch_mat, out=_stretch_mat)
        _stretch_mat = np.multiply(stretch - 1, _stretch_mat, out=_stretch_mat)
        _stretch_mat = np.add(1, _stretch_mat, out=_stretch_mat)

        _stretch_mat = np.multiply(D_mat, _stretch_mat, out=_stretch_mat)

        return _stretch_mat

    if working_memory is None:
        # Compute the matrix D[i, j] of euclidean distances among points
        # i and j
        D = pairwise_distances(points)

        # Compute the angular coefficients of the line between each pair of
        # points

        # dX[i,j]: x difference between points i and j
        # dY[i,j]: y difference between points i and j
        dX = x_array.T - x_array
        dY = y_array.T - y_array

        # Compute the matrix Theta of angles between each pair of points
        theta = np.arctan2(dY, dX, dtype=np.float32)

        stretch_mat = calculate_stretch_mat(theta, D)
    else:
        start = 0
        # x and y sizes are the same
        stretch_mat = np.empty((len(x), len(y)), dtype=np.float32)
        for D_chunk in pairwise_distances_chunked(
                points, working_memory=working_memory):
            chunk_size = D_chunk.shape[0]
            dX = x_array[:, start: start + chunk_size].T - x_array
            dY = y_array[:, start: start + chunk_size].T - y_array

            theta_chunk = np.arctan2(
                dY, dX, out=stretch_mat[start: start + chunk_size, :])

            # stretch_mat (theta_chunk) is updated in-place here
            calculate_stretch_mat(theta_chunk, D_chunk)

            start += chunk_size

    # Return the stretched distance matrix
    return stretch_mat


def _interpolate_signals(signals, sampling_times, verbose=False):
    """
    Interpolate signals at given sampling times.
    """
    # Reshape all signals to one-dimensional array object (e.g. AnalogSignal)
    for i, signal in enumerate(signals):
        if signal.ndim == 2:
            signals[i] = signal.flatten()
        elif signal.ndim > 2:
            raise ValueError('elements in fir_rates must have 2 dimensions')

    if verbose:
        print('create time slices of the rates...')

    # Interpolate in the time bins
    interpolated_signal = np.vstack([_analog_signal_step_interp(
        signal, sampling_times).rescale('Hz').magnitude
                                     for signal in signals]) * pq.Hz

    return interpolated_signal


class _GPUBackend:
    """
    Parameters
    ----------
    max_chunk_size: int or None, optional
        Defines the maximum chunk size used in the `_split_axis` function. The
        users typically don't need to set this parameter manually - it's used
        to simulate scenarios when the input matrix is so large that it cannot
        fit into GPU memory. Setting this parameter manually can resolve GPU
        memory errors in case automatic parameters adjustment fails.

    Notes
    -----
    1. PyOpenCL backend takes some time to compile the kernel for the first
       time - the caching will affect your benchmarks unless you run each
       program twice.
    2. Pinned Host Memory.
       Host (CPU) data allocations are pageable by default. The GPU cannot
       access data directly from pageable host memory, so when a data transfer
       from pageable host memory to device memory is invoked, the CUDA driver
       must first allocate a temporary page-locked, or "pinned", host array,
       copy the host data to the pinned array, and then transfer the data from
       the pinned array to device memory, as illustrated at
       https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
       Same for OpenCL. Therefore, Python memory analyzers show increments in
       the used RAM each time an OpenCL/CUDA buffer is created. As with any
       Python objects, PyOpenCL and PyCUDA clean up and free allocated memory
       automatically when garbage collection is executed.
    """
    def __init__(self, max_chunk_size=None):
        self.max_chunk_size = max_chunk_size

    def _choose_backend(self):
        # If CUDA is detected, always use CUDA.
        # If OpenCL is detected, don't use it by default to avoid the system
        # becoming unresponsive until the program terminates.
        use_cuda = int(os.getenv("ELEPHANT_USE_CUDA", '1'))
        use_opencl = int(os.getenv("ELEPHANT_USE_OPENCL", '1'))
        cuda_detected = get_cuda_capability_major() != 0
        if use_cuda and cuda_detected:
            return self.pycuda
        if use_opencl:
            return self.pyopencl
        return self.cpu

    def _split_axis(self, chunk_size, axis_size, min_chunk_size=None):
        chunk_size = min(chunk_size, axis_size)
        if self.max_chunk_size is not None:
            chunk_size = min(chunk_size, self.max_chunk_size)
        if min_chunk_size is not None and chunk_size < min_chunk_size:
            raise ValueError(f"[GPU not enough memory] Impossible to split "
                             f"the array into chunks of size at least "
                             f"{min_chunk_size} to fit into GPU memory")
        n_chunks = math.ceil(axis_size / chunk_size)
        chunk_size = math.ceil(axis_size / n_chunks)  # align in size
        if min_chunk_size is not None:
            chunk_size = max(chunk_size, min_chunk_size)
        split_idx = list(range(0, axis_size, chunk_size))
        last_id = split_idx[-1]
        last_size = axis_size - last_id  # last is the smallest
        split_idx = list(zip(split_idx[:-1], split_idx[1:]))
        if min_chunk_size is not None and last_size < min_chunk_size:
            # Overlap the last chunk with the previous.
            # The overlapped part (intersection) will be computed twice.
            last_id = axis_size - min_chunk_size
        split_idx.append((last_id, axis_size))
        return chunk_size, split_idx


class _JSFUniformOrderStat3D(_GPUBackend):
    def __init__(self, n, d, precision='float', verbose=False,
                 cuda_threads=64, cuda_cwr_loops=32, tolerance=1e-5,
                 max_chunk_size=None):
        super().__init__(max_chunk_size=max_chunk_size)
        if d > n:
            raise ValueError(f"d ({d}) must be less or equal n ({n})")
        self.n = n
        self.d = d
        self.precision = precision
        self.verbose = verbose and rank == 0
        self.cuda_threads = cuda_threads
        self.cuda_cwr_loops = cuda_cwr_loops
        self.map_iterations = self._create_iteration_table()
        bits = 32 if precision == "float" else 64
        self.dtype = np.dtype(f"float{bits}")
        self.tolerance = tolerance

    @property
    def num_iterations(self):
        # map_iterations table is populated with element indices, not counts;
        # therefore, we add 1
        return self.map_iterations[:, -1].sum() + 1

    def _create_iteration_table(self):
        # do not use numpy arrays - they are limited to uint64
        map_iterations = [list(range(self.n))]
        for row_id in range(1, self.d):
            prev_row = map_iterations[row_id - 1]
            curr_row = [0] * (row_id + 1)
            for col_id in range(row_id + 1, self.n):
                cumsum = prev_row[col_id] + curr_row[-1]
                curr_row.append(cumsum)
            map_iterations.append(curr_row)
        # here we can wrap the resulting array in numpy:
        # if at least one item is greater than 2<<63 - 1,
        # the data type will be set to 'object'
        map_iterations = np.vstack(map_iterations)
        return map_iterations

    def _combinations_with_replacement(self):
        # Generate sequences of {a_i} such that
        #   a_0 >= a_1 >= ... >= a_(d-1) and
        #   d-i <= a_i <= n, for each i in [0, d-1].
        #
        # Almost equivalent to
        # list(itertools.combinations_with_replacement(range(n, 0, -1), r=d))
        # [::-1]
        #
        # Example:
        #   _combinations_with_replacement(n=13, d=3) -->
        #   (3, 2, 1), (3, 2, 2), (3, 3, 1), ... , (13, 13, 12), (13, 13, 13).
        #
        # The implementation follows the insertion sort algorithm:
        #   insert a new element a_i from right to left to keep the reverse
        #   sorted order. Now substitute increment operation for insert.
        if self.d > self.n:
            return
        if self.d == 1:
            for matrix_entry in range(1, self.n + 1):
                yield (matrix_entry,)
            return
        sequence_sorted = list(range(self.d, 0, -1))
        input_order = tuple(sequence_sorted)  # fixed
        while sequence_sorted[0] != self.n + 1:
            for last_element in range(1, sequence_sorted[-2] + 1):
                sequence_sorted[-1] = last_element
                yield tuple(sequence_sorted)
            increment_id = self.d - 2
            while increment_id > 0 and sequence_sorted[increment_id - 1] == \
                    sequence_sorted[increment_id]:
                increment_id -= 1
            sequence_sorted[increment_id + 1:] = input_order[increment_id + 1:]
            sequence_sorted[increment_id] += 1

    def cpu(self, log_du):
        log_1 = np.log(1.)
        # Compute the log of the integral's coefficient
        logK = np.sum(np.log(np.arange(1, self.n + 1)))
        # Add to the 3D matrix u a bottom layer equal to 0 and a
        # top layer equal to 1. Then compute the difference du along
        # the first dimension.

        # prepare arrays for usage inside the loop
        di_scratch = np.empty_like(log_du, dtype=np.int32)
        log_du_scratch = np.empty_like(log_du)

        # precompute log(factorial)s
        # pad with a zero to get 0! = 1
        log_factorial = np.hstack((0, np.cumsum(np.log(range(1, self.n + 1)))))

        # compute the probabilities for each unique row of du
        # only loop over the indices and do all du entries at once
        # using matrix algebra
        # initialise probabilities to 0
        P_total = np.zeros(
            log_du.shape[0],
            dtype=np.float32 if self.precision == 'float' else np.float64
        )

        for iter_id, matrix_entries in enumerate(
                tqdm(self._combinations_with_replacement(),
                     total=self.num_iterations,
                     desc="Joint survival function",
                     disable=not self.verbose)):
            # if we are running with MPI
            if mpi_accelerated and iter_id % size != rank:
                continue
            # we only need the differences of the indices:
            di = -np.diff((self.n,) + matrix_entries + (0,))

            # reshape the matrix to be compatible with du
            di_scratch[:, range(len(di))] = di

            # use precomputed factorials
            sum_log_di_factorial = log_factorial[di].sum()

            # Compute for each i,j the contribution to the probability
            # given by this step, and add it to the total probability

            # Use precomputed log
            np.copyto(log_du_scratch, log_du)

            # for each a=0,1,...,A-1 and b=0,1,...,B-1, replace du with 1
            # whenever di_scratch = 0, so that du ** di_scratch = 1 (this
            # avoids nans when both du and di_scratch are 0, and is
            # mathematically correct)
            log_du_scratch[di_scratch == 0] = log_1

            di_log_du = di_scratch * log_du_scratch
            sum_di_log_du = di_log_du.sum(axis=1)
            logP = sum_di_log_du - sum_log_di_factorial

            P_total += np.exp(logP + logK)

        if mpi_accelerated:
            totals = np.zeros_like(P_total)

            # exchange all the results
            mpi_float_type = MPI.FLOAT \
                if self.precision == 'float' else MPI.DOUBLE
            comm.Allreduce(
                [P_total, mpi_float_type],
                [totals, mpi_float_type],
                op=MPI.SUM)

            # We need to return the collected totals instead of the local
            # P_total
            P_total = totals

        return P_total

    def _compile_template(self, template_name, **kwargs):
        from jinja2 import Template
        cu_template_path = Path(__file__).parent / template_name
        cu_template = Template(cu_template_path.read_text())
        asset_cu = cu_template.render(
            precision=self.precision,
            CWR_LOOPS=self.cuda_cwr_loops,
            N=self.n, D=self.d, **kwargs)
        return asset_cu

    def pyopencl(self, log_du, device_id=0):
        import pyopencl as cl
        import pyopencl.array as cl_array

        self._check_input(log_du)

        it_todo = self.num_iterations
        u_length = log_du.shape[0]

        context = cl.create_some_context(interactive=False)
        if self.verbose:
            print("Available OpenCL devices:\n", context.devices)
        device = context.devices[device_id]

        # A queue bounded to the device
        queue = cl.CommandQueue(context)

        max_l_block = device.local_mem_size // (
                self.dtype.itemsize * (self.d + 2))
        n_threads = min(self.cuda_threads, max_l_block,
                        device.max_work_group_size)
        if n_threads > 32:
            # It's more efficient to make the number of threads
            # a multiple of the warp size (32).
            n_threads -= n_threads % 32

        iteration_table_str = ", ".join(f"{val}LU" for val in
                                        self.map_iterations.flatten())
        iteration_table_str = "{%s}" % iteration_table_str

        log_factorial = np.r_[0, np.cumsum(np.log(range(1, self.n + 1)))]
        logK = log_factorial[-1]
        log_factorial_str = ", ".join(f"{val:.10f}" for val in log_factorial)
        log_factorial_str = "{%s}" % log_factorial_str
        atomic_int = 'int' if self.precision == 'float' else 'long'

        # GPU_MAX_HEAP_SIZE OpenCL flag is set to 2 Gb (1 << 31) by default
        mem_avail = min(device.max_mem_alloc_size, device.global_mem_size,
                        1 << 31)
        # 4 * (D + 1) * size + 8 * size == mem_avail
        chunk_size = mem_avail // (4 * log_du.shape[1] + self.dtype.itemsize)
        chunk_size, split_idx = self._split_axis(chunk_size=chunk_size,
                                                 axis_size=u_length)

        P_total = np.empty(u_length, dtype=self.dtype)
        P_total_gpu = cl_array.Array(queue, shape=chunk_size, dtype=self.dtype)

        for i_start, i_end in split_idx:
            log_du_gpu = cl_array.to_device(queue, log_du[i_start: i_end],
                                            async_=True)
            P_total_gpu.fill(0, queue=queue)
            chunk_size = i_end - i_start
            l_block = min(n_threads, chunk_size)
            l_num_blocks = math.ceil(chunk_size / l_block)
            grid_size = math.ceil(it_todo / (n_threads * self.cuda_cwr_loops))
            if grid_size > l_num_blocks:
                # make grid_size divisible by l_num_blocks
                grid_size -= grid_size % l_num_blocks
            else:
                # grid_size must be at least l_num_blocks
                grid_size = l_num_blocks

            if self.verbose:
                print(f"[Joint prob. matrix] it_todo={it_todo}, "
                      f"grid_size={grid_size}, L_BLOCK={l_block}, "
                      f"N_THREADS={n_threads}")

            # OpenCL defines unsigned long as uint64, therefore we're adding
            # the LU suffix, not LLU, which would indicate unsupported uint128
            # data type format.
            asset_cl = self._compile_template(
                template_name="joint_pmat.cl",
                L=f"{chunk_size}LU",
                L_BLOCK=l_block,
                L_NUM_BLOCKS=l_num_blocks,
                ITERATIONS_TODO=f"{it_todo}LU",
                logK=f"{logK:.10f}f",
                iteration_table=iteration_table_str,
                log_factorial=log_factorial_str,
                ATOMIC_UINT=f"unsigned {atomic_int}",
                ASSET_ENABLE_DOUBLE_SUPPORT=int(self.precision == "double")
            )

            program = cl.Program(context, asset_cl).build()

            # synchronize
            cl.enqueue_barrier(queue)

            kernel = program.jsf_uniform_orderstat_3d_kernel
            kernel(queue, (grid_size,), (n_threads,),
                   P_total_gpu.data, log_du_gpu.data, g_times_l=True)

            P_total_gpu[:chunk_size].get(ary=P_total[i_start: i_end])

        return P_total

    def pycuda(self, log_du):
        try:
            # PyCuda should not be in requirements-extra because CPU limited
            # users won't be able to install Elephant.
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray
            import pycuda.driver as drv
            from pycuda.compiler import SourceModule
        except ImportError as err:
            raise ImportError(
                "Install pycuda with 'pip install pycuda'") from err

        self._check_input(log_du)

        it_todo = self.num_iterations
        u_length = log_du.shape[0]

        device = pycuda.autoinit.device

        max_l_block = device.MAX_SHARED_MEMORY_PER_BLOCK // (
                    self.dtype.itemsize * (self.d + 2))
        n_threads = min(self.cuda_threads, max_l_block,
                        device.MAX_THREADS_PER_BLOCK)
        if n_threads > device.WARP_SIZE:
            # It's more efficient to make the number of threads
            # a multiple of the warp size (32).
            n_threads -= n_threads % device.WARP_SIZE

        log_factorial = np.r_[0, np.cumsum(np.log(range(1, self.n + 1)))]
        log_factorial = log_factorial.astype(self.dtype)
        logK = log_factorial[-1]

        free, total = drv.mem_get_info()
        # 4 * (D + 1) * size + 8 * size == mem_avail
        chunk_size = free // (4 * log_du.shape[1] + self.dtype.itemsize)
        chunk_size, split_idx = self._split_axis(chunk_size=chunk_size,
                                                 axis_size=u_length)

        P_total = np.empty(u_length, dtype=self.dtype)
        P_total_gpu = gpuarray.GPUArray(chunk_size, dtype=self.dtype)
        log_du_gpu = drv.mem_alloc(4 * chunk_size * log_du.shape[1])

        for i_start, i_end in split_idx:
            drv.memcpy_htod_async(dest=log_du_gpu, src=log_du[i_start: i_end])
            P_total_gpu.fill(0)
            chunk_size = i_end - i_start
            l_block = min(n_threads, chunk_size)
            l_num_blocks = math.ceil(chunk_size / l_block)
            grid_size = math.ceil(it_todo / (n_threads * self.cuda_cwr_loops))
            grid_size = min(grid_size, device.MAX_GRID_DIM_X)
            if grid_size > l_num_blocks:
                # make grid_size divisible by l_num_blocks
                grid_size -= grid_size % l_num_blocks
            else:
                # grid_size must be at least l_num_blocks
                grid_size = l_num_blocks

            if self.verbose:
                print(f"[Joint prob. matrix] it_todo={it_todo}, "
                      f"grid_size={grid_size}, L_BLOCK={l_block}, "
                      f"N_THREADS={n_threads}")

            asset_cu = self._compile_template(
                template_name="joint_pmat.cu",
                L=f"{chunk_size}LLU",
                L_BLOCK=l_block,
                L_NUM_BLOCKS=l_num_blocks,
                ITERATIONS_TODO=f"{it_todo}LLU",
                logK=f"{logK:.10f}f",
            )

            module = SourceModule(asset_cu)

            iteration_table_gpu, _ = module.get_global("iteration_table")
            iteration_table = self.map_iterations.astype(np.uint64)
            drv.memcpy_htod(iteration_table_gpu, iteration_table)

            log_factorial_gpu, _ = module.get_global("log_factorial")
            drv.memcpy_htod(log_factorial_gpu, log_factorial)

            drv.Context.synchronize()

            kernel = module.get_function("jsf_uniform_orderstat_3d_kernel")
            kernel(P_total_gpu.gpudata, log_du_gpu, grid=(grid_size, 1),
                   block=(n_threads, 1, 1))

            P_total_gpu[:chunk_size].get(ary=P_total[i_start: i_end])

        return P_total

    def _cuda(self, log_du):
        # Compile a self-contained joint_pmat_old.cu file and run it
        # in a terminal. Having this function is useful to debug ASSET CUDA
        # application because it's self-contained and the logic is documented.
        # Don't use this backend when the 'log_du' arrays are huge because
        # of the disk I/O operations.
        # A note to developers: remove this backend in half a year once the
        # pycuda backend proves to be stable.

        self._check_input(log_du)

        asset_cu = self._compile_template(
            template_name="joint_pmat_old.cu",
            L=f"{log_du.shape[0]}LLU",
            N_THREADS=self.cuda_threads,
            ITERATIONS_TODO=f"{self.num_iterations}LLU",
            ASSET_DEBUG=int(self.verbose)
        )
        with tempfile.TemporaryDirectory() as asset_tmp_folder:
            asset_cu_path = os.path.join(asset_tmp_folder, 'asset.cu')
            asset_bin_path = os.path.join(asset_tmp_folder, 'asset.o')
            with open(asset_cu_path, 'w') as f:
                f.write(asset_cu)
            # -O3 optimization flag is for the host code only;
            # by default, GPU device code is optimized with -O3.
            # -w to ignore warnings.
            compile_cmd = ['nvcc', '-w', '-O3', '-o', asset_bin_path,
                           asset_cu_path]
            if self.precision == 'double' and get_cuda_capability_major() >= 6:
                # atomicAdd(double) requires compute capability 6.x
                compile_cmd.extend(['-arch', 'sm_60'])
            compile_status = subprocess.run(
                compile_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if self.verbose:
                print(compile_status.stdout.decode())
                print(compile_status.stderr.decode(), file=sys.stderr)
            compile_status.check_returncode()
            log_du_path = os.path.join(asset_tmp_folder, "log_du.dat")
            P_total_path = os.path.join(asset_tmp_folder, "P_total.dat")
            with open(log_du_path, 'wb') as f:
                log_du.tofile(f)
            run_status = subprocess.run(
                [asset_bin_path, log_du_path, P_total_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if self.verbose:
                print(run_status.stdout.decode())
                print(run_status.stderr.decode(), file=sys.stderr)
            run_status.check_returncode()
            with open(P_total_path, 'rb') as f:
                P_total = np.fromfile(f, dtype=self.dtype)

        return P_total

    def _check_input(self, log_du):
        it_todo = self.num_iterations
        if it_todo > np.iinfo(np.uint64).max:
            raise ValueError(f"it_todo ({it_todo}) is larger than MAX_UINT64."
                             " Only Python backend is supported.")
        # Don't convert log_du to float32 transparently for the user to avoid
        # situations when the user accidentally passes an array with float64.
        # Doing so wastes memory for nothing.
        if log_du.dtype != np.float32:
            raise ValueError("'log_du' must be a float32 array")
        if log_du.shape[1] != self.d + 1:
            raise ValueError(f"log_du.shape[1] ({log_du.shape[1]}) must be "
                             f"equal to D+1 ({self.d + 1})")

    def compute(self, u):
        if u.shape[1] != self.d:
            raise ValueError("Invalid input data shape axis 1: expected {}, "
                             "got {}".format(self.d, u.shape[1]))
        # A faster and memory efficient implementation of
        # du = np.diff(u, prepend=0, append=1, axis=1).astype(np.float32)
        du = np.empty((u.shape[0], u.shape[1] + 1), dtype=np.float32)
        du[:, 0] = u[:, 0]
        np.subtract(u[:, 1:], u[:, :-1], out=du[:, 1:-1])
        np.subtract(1, u[:, -1], out=du[:, -1])

        # precompute logarithms
        # ignore warnings about infinities, see inside the loop:
        # we replace 0 * ln(0) by 1 to get exp(0 * ln(0)) = 0 ** 0 = 1
        # the remaining infinities correctly evaluate to
        # exp(ln(0)) = exp(-inf) = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            log_du = np.log(du, out=du)

        jsf_backend = self._choose_backend()

        P_total = jsf_backend(log_du)

        # Captures non-finite values like NaN, inf
        inside = (P_total > -self.tolerance) & (P_total < 1 + self.tolerance)
        outside_vals = P_total[~inside]
        if len(outside_vals) > 0:
            # A watchdog for unexpected results.
            warnings.warn(f"{len(outside_vals)}/{P_total.shape[0]} values of "
                          "the computed joint prob. matrix lie outside of the "
                          f"valid [0, 1] interval:\n{outside_vals}\nIf you're "
                          "using PyOpenCL backend, make sure you've disabled "
                          "GPU Hangcheck as described here https://"
                          "software.intel.com/content/www/us/en/develop/"
                          "documentation/get-started-with-intel-oneapi-"
                          "base-linux/top/before-you-begin.html\n"
                          "Clipping the output array to 0 and 1.")
            P_total = np.clip(P_total, a_min=0., a_max=1., out=P_total)

        return P_total


class _PMatNeighbors(_GPUBackend):
    """
    Parameters
    ----------
    filter_shape : tuple of int
        A pair of integers representing the kernel shape `(l, w)`.
    n_largest : int
        The number of largest neighbors to collect for each entry in `mat`.
    """

    def __init__(self, filter_shape, n_largest, max_chunk_size=None):
        super().__init__(max_chunk_size=max_chunk_size)
        self.n_largest = n_largest
        self.max_chunk_size = max_chunk_size

        filter_size, filter_width = filter_shape
        if filter_width >= filter_size:
            raise ValueError('filter_shape width must be lower than length')
        if not ((filter_width % 2) and (filter_size % 2)):
            warnings.warn(
                'The kernel is not centered on the datapoint in whose'
                'calculation it is used. Consider using odd values'
                'for both entries of filter_shape.')

        # Construct the kernel
        filt = np.ones((filter_size, filter_size), dtype=bool)
        filt = np.triu(filt, -filter_width)
        filt = np.tril(filt, filter_width)
        if n_largest > len(filt.nonzero()[0]):
            raise ValueError(f"Too small filter shape {filter_shape} to "
                             f"select {n_largest} largest elements.")

        self.filter_kernel = filt

    def _check_input(self, mat):
        symmetric = np.all(np.diagonal(mat) == 0.5)
        # Check consistent arguments
        filter_size = self.filter_kernel.shape[0]
        if (symmetric and mat.shape[0] < 2 * filter_size - 1) \
                or (not symmetric and min(mat.shape) < filter_size):
            raise ValueError(f"'filter_shape' {self.filter_kernel.shape} is "
                             f"too large for the input matrix of shape "
                             f"{mat.shape}")
        if mat.dtype != np.float32:
            raise ValueError("The input matrix dtype must be float32.")

    def pyopencl(self, mat):
        import pyopencl as cl
        import pyopencl.array as cl_array
        from jinja2 import Template

        context = cl.create_some_context(interactive=False)
        device = context.devices[0]
        queue = cl.CommandQueue(context)

        # if the matrix is symmetric the diagonal was set to 0.5
        # when computing the probability matrix
        symmetric = np.all(np.diagonal(mat) == 0.5)
        self._check_input(mat)

        filt_size = self.filter_kernel.shape[0]  # filt is a square matrix
        filt_rows, filt_cols = self.filter_kernel.nonzero()
        filt_rows = "{%s}" % ", ".join(f"{row}U" for row in filt_rows)
        filt_cols = "{%s}" % ", ".join(f"{col}U" for col in filt_cols)

        lmat_padded = np.zeros((mat.shape[0], mat.shape[1], self.n_largest),
                               dtype=np.float32)
        if symmetric:
            mat = mat[filt_size:]
            lmat = lmat_padded[filt_size + filt_size // 2: -filt_size // 2 + 1]
        else:
            lmat = lmat_padded[filt_size // 2: -filt_size // 2 + 1]

        # GPU_MAX_HEAP_SIZE OpenCL flag is set to 2 Gb (1 << 31) by default
        mem_avail = min(device.max_mem_alloc_size, device.global_mem_size,
                        1 << 31)
        # 4 * size * n_cols * n_largest + 4 * (size + filt_size) * n_cols
        chunk_size = (mem_avail // 4 - filt_size * lmat.shape[1]) // (
                lmat.shape[1] * (self.n_largest + 1))
        chunk_size, split_idx = self._split_axis(chunk_size=chunk_size,
                                                 axis_size=lmat.shape[0],
                                                 min_chunk_size=filt_size)

        pmat_cl_path = Path(__file__).parent / "pmat_neighbors.cl"
        pmat_cl_template = Template(pmat_cl_path.read_text())

        lmat_gpu = cl_array.Array(
            queue, shape=(chunk_size, lmat.shape[1], self.n_largest),
            dtype=np.float32
        )

        for i_start, i_end in split_idx:
            mat_gpu = cl_array.to_device(queue,
                                         mat[i_start: i_end + filt_size],
                                         async_=True)
            lmat_gpu.fill(0, queue=queue)
            chunk_size = i_end - i_start
            it_todo = chunk_size * (lmat.shape[1] - filt_size + 1)

            pmat_neighbors_cl = pmat_cl_template.render(
                FILT_SIZE=filt_size,
                N_LARGEST=self.n_largest,
                PMAT_COLS=f"{lmat.shape[1]}LU",
                Y_OFFSET=f"{i_start}LU",
                NONZERO_SIZE=self.filter_kernel.sum(),
                SYMMETRIC=int(symmetric),
                filt_rows=filt_rows,
                filt_cols=filt_cols
            )

            program = cl.Program(context, pmat_neighbors_cl).build()

            # synchronize
            cl.enqueue_barrier(queue)

            kernel = program.pmat_neighbors
            # When the grid size is set to the total number of work items to
            # execute and the local size is set to None, PyOpenCL chooses the
            # number of threads automatically such that the total number of
            # work items exactly matches the desired number of iterations.
            kernel(queue, (it_todo,), None, lmat_gpu.data, mat_gpu.data)

            lmat_gpu[:chunk_size].get(ary=lmat[i_start: i_end])

        return lmat_padded

    def pycuda(self, mat):
        from jinja2 import Template
        try:
            # PyCuda should not be in requirements-extra because CPU limited
            # users won't be able to install Elephant.
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray
            import pycuda.driver as drv
            from pycuda.compiler import SourceModule
        except ImportError as err:
            raise ImportError(
                "Install pycuda with 'pip install pycuda'") from err

        # if the matrix is symmetric the diagonal was set to 0.5
        # when computing the probability matrix
        symmetric = np.all(np.diagonal(mat) == 0.5)
        self._check_input(mat)

        device = pycuda.autoinit.device
        n_threads = device.MAX_THREADS_PER_BLOCK

        filt_size = self.filter_kernel.shape[0]
        filt_rows, filt_cols = self.filter_kernel.nonzero()

        lmat_padded = np.zeros((mat.shape[0], mat.shape[1], self.n_largest),
                               dtype=np.float32)
        if symmetric:
            mat = mat[filt_size:]
            lmat = lmat_padded[filt_size + filt_size // 2: -filt_size // 2 + 1]
        else:
            lmat = lmat_padded[filt_size // 2: -filt_size // 2 + 1]

        free, total = drv.mem_get_info()
        # 4 * size * n_cols * n_largest + 4 * (size + filt_size) * n_cols
        chunk_size = (free // 4 - filt_size * lmat.shape[1]) // (
                lmat.shape[1] * (self.n_largest + 1))
        chunk_size, split_idx = self._split_axis(chunk_size=chunk_size,
                                                 axis_size=lmat.shape[0],
                                                 min_chunk_size=filt_size)

        pmat_cu_path = Path(__file__).parent / "pmat_neighbors.cu"
        pmat_cu_template = Template(pmat_cu_path.read_text())

        lmat_gpu = gpuarray.GPUArray(
            (chunk_size, lmat.shape[1], self.n_largest), dtype=np.float32)

        mat_gpu = drv.mem_alloc(4 * (chunk_size + filt_size) * mat.shape[1])

        for i_start, i_end in split_idx:
            drv.memcpy_htod_async(dest=mat_gpu,
                                  src=mat[i_start: i_end + filt_size])
            lmat_gpu.fill(0)
            chunk_size = i_end - i_start
            it_todo = chunk_size * (lmat.shape[1] - filt_size + 1)

            pmat_neighbors_cu = pmat_cu_template.render(
                FILT_SIZE=filt_size,
                N_LARGEST=self.n_largest,
                PMAT_COLS=f"{lmat.shape[1]}LLU",
                Y_OFFSET=f"{i_start}LLU",
                NONZERO_SIZE=self.filter_kernel.sum(),
                SYMMETRIC=int(symmetric),
                IT_TODO=it_todo,
            )

            module = SourceModule(pmat_neighbors_cu)

            filt_rows_gpu, _ = module.get_global("filt_rows")
            drv.memcpy_htod(filt_rows_gpu, filt_rows.astype(np.uint32))

            filt_cols_gpu, _ = module.get_global("filt_cols")
            drv.memcpy_htod(filt_cols_gpu, filt_cols.astype(np.uint32))

            drv.Context.synchronize()

            grid_size = math.ceil(it_todo / n_threads)
            if grid_size > device.MAX_GRID_DIM_X:
                raise ValueError("Cannot launch a CUDA kernel with "
                                 f"{grid_size} num. of blocks. Adjust the "
                                 "'max_chunk_size' parameter.")

            kernel = module.get_function("pmat_neighbors")
            kernel(lmat_gpu.gpudata, mat_gpu, grid=(grid_size, 1),
                   block=(n_threads, 1, 1))

            lmat_gpu[:chunk_size].get(ary=lmat[i_start: i_end])

        return lmat_padded

    def compute(self, mat):
        """
        Build the 3D matrix `L` of largest neighbors of elements in a 2D matrix
        `mat`.

        For each entry `mat[i, j]`, collects the `n_largest` elements with
        largest values around `mat[i, j]`, say `z_i, i=1,2,...,n_largest`,
        and assigns them to `L[i, j, :]`.
        The zone around `mat[i, j]` where largest neighbors are collected from
        is a rectangular area (kernel) of shape `(l, w) = filter_shape`
        centered around `mat[i, j]` and aligned along the diagonal.

        If `mat` is symmetric, only the triangle below the diagonal is
        considered.

        Parameters
        ----------
        mat : np.ndarray
            A square matrix of real-valued elements.

        Returns
        -------
        lmat : np.ndarray
            A matrix of shape `(l, w, n_largest)` containing along the last
            dimension `lmat[i, j, :]` the largest neighbors of `mat[i, j]`.
        """
        backend = self._choose_backend()
        lmat = backend(mat)
        return lmat

    def cpu(self, mat):
        # if the matrix is symmetric the diagonal was set to 0.5
        # when computing the probability matrix
        symmetric = np.all(np.diagonal(mat) == 0.5)
        self._check_input(mat)

        filter_size = self.filter_kernel.shape[0]

        # Initialize the matrix of d-largest values as a matrix of zeroes
        lmat = np.zeros((mat.shape[0], mat.shape[1], self.n_largest),
                        dtype=np.float32)

        N_bin_y = mat.shape[0]
        N_bin_x = mat.shape[1]
        # if the matrix is symmetric do not use kernel positions intersected
        # by the diagonal
        if symmetric:
            bin_range_y = range(filter_size, N_bin_y - filter_size + 1)
        else:
            bin_range_y = range(N_bin_y - filter_size + 1)
            bin_range_x = range(N_bin_x - filter_size + 1)

        # compute matrix of largest values
        for y in bin_range_y:
            if symmetric:
                # x range depends on y position
                bin_range_x = range(y - filter_size + 1)
            for x in bin_range_x:
                patch = mat[y: y + filter_size, x: x + filter_size]
                mskd = patch[self.filter_kernel]
                largest_vals = np.sort(mskd)[-self.n_largest:]
                lmat[y + (filter_size // 2), x + (filter_size // 2), :] = \
                    largest_vals

        return lmat


def synchronous_events_intersection(sse1, sse2, intersection='linkwise'):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of positions `(iK, jK)` of matrix entries and
    associated synchronous events `SK`, finds the intersection among them.

    The intersection can be performed 'pixelwise' or 'linkwise'.

        * if 'pixelwise', it yields a new SSE which retains only events in
          `sse1` whose pixel position matches a pixel position in `sse2`. This
          operation is not symmetric:
          `intersection(sse1, sse2) != intersection(sse2, sse1)`.
        * if 'linkwise', an additional step is performed where each retained
          synchronous event `SK` in `sse1` is intersected with the
          corresponding event in `sse2`. This yields a symmetric operation:
          `intersection(sse1, sse2) = intersection(sse2, sse1)`.

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Each is a dictionary of pixel positions `(i, j)` as keys and sets `S`
        of synchronous events as values (see above).
    intersection : {'pixelwise', 'linkwise'}, optional
        The type of intersection to perform among the two SSEs (see above).
        Default: 'linkwise'

    Returns
    -------
    sse_new : dict
        A new SSE (same structure as `sse1` and `sse2`) which retains only the
        events of `sse1` associated to keys present both in `sse1` and `sse2`.
        If `intersection = 'linkwise'`, such events are additionally
        intersected with the associated events in `sse2`.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    sse_new = sse1.copy()
    for pixel1 in sse1.keys():
        if pixel1 not in sse2.keys():
            del sse_new[pixel1]

    if intersection == 'linkwise':
        for pixel1, link1 in sse_new.items():
            sse_new[pixel1] = link1.intersection(sse2[pixel1])
            if len(sse_new[pixel1]) == 0:
                del sse_new[pixel1]
    elif intersection == 'pixelwise':
        pass
    else:
        raise ValueError(
            "intersection (=%s) can only be" % intersection +
            " 'pixelwise' or 'linkwise'")

    return sse_new


def synchronous_events_difference(sse1, sse2, difference='linkwise'):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), computes the difference between `sse1` and `sse2`.

    The difference can be performed 'pixelwise' or 'linkwise':

        * if 'pixelwise', it yields a new SSE which contains all (and only) the
          events in `sse1` whose pixel position doesn't match any pixel in
          `sse2`.
        * if 'linkwise', for each pixel `(i, j)` in `sse1` and corresponding
          synchronous event `S1`, if `(i, j)` is a pixel in `sse2`
          corresponding to the event `S2`, it retains the set difference
          `S1 - S2`. If `(i, j)` is not a pixel in `sse2`, it retains the full
          set `S1`.

    Note that in either case the difference is a non-symmetric operation:
    `intersection(sse1, sse2) != intersection(sse2, sse1)`.

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Dictionaries of pixel positions `(i, j)` as keys and sets `S` of
        synchronous events as values (see above).
    difference : {'pixelwise', 'linkwise'}, optional
        The type of difference to perform between `sse1` and `sse2` (see
        above).
        Default: 'linkwise'

    Returns
    -------
    sse_new : dict
        A new SSE (same structure as `sse1` and `sse2`) which retains the
        difference between `sse1` and `sse2`.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    sse_new = sse1.copy()
    for pixel1 in sse1.keys():
        if pixel1 in sse2.keys():
            if difference == 'pixelwise':
                del sse_new[pixel1]
            elif difference == 'linkwise':
                sse_new[pixel1] = sse_new[pixel1].difference(sse2[pixel1])
                if len(sse_new[pixel1]) == 0:
                    del sse_new[pixel1]
            else:
                raise ValueError(
                    "difference (=%s) can only be" % difference +
                    " 'pixelwise' or 'linkwise'")

    return sse_new


def _remove_empty_events(sse):
    """
    Given a sequence of synchronous events (SSE) `sse` consisting of a pool of
    pixel positions and associated synchronous events (see below), returns a
    copy of `sse` where all empty events have been removed.

    `sse` must be provided as a dictionary of type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse : dict
        A dictionary of pixel positions `(i, j)` as keys, and sets `S` of
        synchronous events as values (see above).

    Returns
    -------
    sse_new : dict
        A copy of `sse` where all empty events have been removed.

    """
    sse_new = sse.copy()
    for pixel, link in sse.items():
        if link == set([]):
            del sse_new[pixel]

    return sse_new


def synchronous_events_identical(sse1, sse2):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether `sse1` is strictly contained in `sse2`.

    `sse1` is strictly contained in `sse2` if all its pixels are pixels of
    `sse2`,
    if its associated events are subsets of the corresponding events
    in `sse2`, and if `sse2` contains events, or neuron IDs in some event,
    which do not belong to `sse1` (i.e., `sse1` and `sse2` are not identical).

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Dictionaries of pixel positions `(i, j)` as keys and sets `S` of
        synchronous events as values.

    Returns
    -------
    bool
        True if `sse1` is identical to `sse2`.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    # Remove empty links from sse11 and sse22, if any
    sse11 = _remove_empty_events(sse1)
    sse22 = _remove_empty_events(sse2)

    # Return whether sse11 == sse22
    return sse11 == sse22


def synchronous_events_no_overlap(sse1, sse2):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether `sse1` and `sse2` are disjoint.

    Two SSEs are disjoint if they don't share pixels, or if the events
    associated to common pixels are disjoint.

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Dictionaries of pixel positions `(i, j)` as keys and sets `S` of
        synchronous events as values.

    Returns
    -------
    bool
        True if `sse1` is disjoint from `sse2`.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    # Remove empty links from sse11 and sse22, if any
    sse11 = _remove_empty_events(sse1)
    sse22 = _remove_empty_events(sse2)

    # If both SSEs are empty, return False (we consider them equal)
    if sse11 == {} and sse22 == {}:
        return False

    common_pixels = set(sse11.keys()).intersection(set(sse22.keys()))
    if len(common_pixels) == 0:
        return True
    if all(sse11[p].isdisjoint(sse22[p]) for p in common_pixels):
        return True
    return False


def synchronous_events_contained_in(sse1, sse2):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether `sse1` is strictly contained in `sse2`.

    `sse1` is strictly contained in `sse2` if all its pixels are pixels of
    `sse2`, if its associated events are subsets of the corresponding events
    in `sse2`, and if `sse2` contains non-empty events, or neuron IDs in some
    event, which do not belong to `sse1` (i.e., `sse1` and `sse2` are not
    identical).

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Dictionaries of pixel positions `(i, j)` as keys and sets `S` of
        synchronous events as values.

    Returns
    -------
    bool
        True if `sse1` is a subset of `sse2`.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    # Remove empty links from sse11 and sse22, if any
    sse11 = _remove_empty_events(sse1)
    sse22 = _remove_empty_events(sse2)

    # Return False if sse11 and sse22 are disjoint
    if synchronous_events_identical(sse11, sse22):
        return False

    # Return False if any pixel in sse1 is not contained in sse2, or if any
    # link of sse1 is not a subset of the corresponding link in sse2.
    # Otherwise (if sse1 is a subset of sse2) continue
    for pixel1, link1 in sse11.items():
        if pixel1 not in sse22.keys():
            return False
        if not link1.issubset(sse22[pixel1]):
            return False

    # Check that sse1 is a STRICT subset of sse2, i.e. that sse2 contains at
    # least one pixel or neuron id not present in sse1.
    return not synchronous_events_identical(sse11, sse22)


def synchronous_events_contains_all(sse1, sse2):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether `sse1` strictly contains `sse2`.

    `sse1` strictly contains `sse2` if it contains all pixels of `sse2`, if all
    associated events in `sse1` contain those in `sse2`, and if `sse1`
    additionally contains other pixels / events not contained in `sse2`.

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Dictionaries of pixel positions `(i, j)` as keys and sets `S` of
        synchronous events as values.

    Returns
    -------
    bool
        True if `sse1` strictly contains `sse2`.

    Notes
    -----
    `synchronous_events_contains_all(sse1, sse2)` is identical to
    `synchronous_events_is_subsequence(sse2, sse1)`.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    return synchronous_events_contained_in(sse2, sse1)


def synchronous_events_overlap(sse1, sse2):
    """
    Given two sequences of synchronous events (SSEs) `sse1` and `sse2`, each
    consisting of a pool of pixel positions and associated synchronous events
    (see below), determines whether the two SSEs overlap.

    The SSEs overlap if they are not equal and none of them is a superset of
    the other one but they are also not disjoint.

    Both `sse1` and `sse2` must be provided as dictionaries of the type

    .. centered:: {(i1, j1): S1, (i2, j2): S2, ..., (iK, jK): SK},

    where each `i`, `j` is an integer and each `S` is a set of neuron IDs.

    Parameters
    ----------
    sse1, sse2 : dict
        Dictionaries of pixel positions `(i, j)` as keys and sets `S` of
        synchronous events as values.

    Returns
    -------
    bool
        True if `sse1` and `sse2` overlap.

    See Also
    --------
    ASSET.extract_synchronous_events : extract SSEs from given spike trains

    """
    contained_in = synchronous_events_contained_in(sse1, sse2)
    contains_all = synchronous_events_contains_all(sse1, sse2)
    identical = synchronous_events_identical(sse1, sse2)
    is_disjoint = synchronous_events_no_overlap(sse1, sse2)
    return not (contained_in or contains_all or identical or is_disjoint)


def _signals_t_start_stop(signals, t_start=None, t_stop=None):
    if t_start is None:
        t_start = _signals_same_attribute(signals, 't_start')
    if t_stop is None:
        t_stop = _signals_same_attribute(signals, 't_stop')
    return t_start, t_stop


def _intersection_matrix(spiketrains, spiketrains_y, bin_size, t_start_x,
                         t_start_y, t_stop_x, t_stop_y, normalization=None):
    if spiketrains_y is None:
        spiketrains_y = spiketrains

    # Compute the binned spike train matrices, along both time axes
    spiketrains_binned = conv.BinnedSpikeTrain(
        spiketrains, bin_size=bin_size,
        t_start=t_start_x, t_stop=t_stop_x)
    spiketrains_binned_y = conv.BinnedSpikeTrain(
        spiketrains_y, bin_size=bin_size,
        t_start=t_start_y, t_stop=t_stop_y)

    # Compute imat by matrix multiplication
    bsts_x = spiketrains_binned.sparse_matrix
    bsts_y = spiketrains_binned_y.sparse_matrix

    # Compute the number of spikes in each bin, for both time axes
    # 'A1' property returns self as a flattened ndarray.
    spikes_per_bin_x = bsts_x.sum(axis=0).A1
    spikes_per_bin_y = bsts_y.sum(axis=0).A1

    # Compute the intersection matrix imat
    imat = bsts_x.T.dot(bsts_y).toarray().astype(np.float32)
    for ii in range(bsts_x.shape[1]):
        # Normalize the row
        col_sum = bsts_x[:, ii].sum()
        if normalization is None or col_sum == 0:
            norm_coef = 1.
        elif normalization == 'intersection':
            norm_coef = np.minimum(
                spikes_per_bin_x[ii], spikes_per_bin_y)
        elif normalization == 'mean':
            # geometric mean
            norm_coef = np.sqrt(
                spikes_per_bin_x[ii] * spikes_per_bin_y)
        elif normalization == 'union':
            norm_coef = np.array([(bsts_x[:, ii]
                                   + bsts_y[:, jj]).count_nonzero()
                                  for jj in range(bsts_y.shape[1])])
        else:
            raise ValueError(
                "Invalid parameter 'norm': {}".format(normalization))

        # If normalization required, for each j such that bsts_y[j] is
        # identically 0 the code above sets imat[:, j] to identically nan.
        # Substitute 0s instead.
        imat[ii, :] = np.divide(imat[ii, :], norm_coef,
                                out=np.zeros(imat.shape[1],
                                             dtype=np.float32),
                                where=norm_coef != 0)

    # Return the intersection matrix and the edges of the bins used for the
    # x and y axes, respectively.
    return imat


class ASSET(object):
    """
    Analysis of Sequences of Synchronous EvenTs class.

    Parameters
    ----------
    spiketrains_i, spiketrains_j : list of neo.SpikeTrain
        Input spike trains for the first and second time dimensions,
        respectively, to compute the p-values from.
        If `spiketrains_y` is None, it's set to `spiketrains`.
    bin_size : pq.Quantity, optional
        The width of the time bins used to compute the probability matrix.
    t_start_i, t_start_j : pq.Quantity, optional
        The start time of the binning for the first and second axes,
        respectively.
        If None, the attribute `t_start` of the spike trains is used
        (if the same for all spike trains).
        Default: None
    t_stop_i, t_stop_j : pq.Quantity, optional
        The stop time of the binning for the first and second axes,
        respectively.
        If None, the attribute `t_stop` of the spike trains is used
        (if the same for all spike trains).
        Default: None
    verbose : bool, optional
        If True, print messages and show progress bar.
        Default: True


    Raises
    ------
    ValueError
        If the `t_start` & `t_stop` times are not (one of):
          perfectly aligned;

          fully disjoint.

    """

    def __init__(self, spiketrains_i, spiketrains_j=None, bin_size=3 * pq.ms,
                 t_start_i=None, t_start_j=None, t_stop_i=None, t_stop_j=None,
                 verbose=True):
        self.spiketrains_i = spiketrains_i
        if spiketrains_j is None:
            spiketrains_j = spiketrains_i
        self.spiketrains_j = spiketrains_j
        self.bin_size = bin_size
        self.t_start_i, self.t_stop_i = _signals_t_start_stop(
            spiketrains_i,
            t_start=t_start_i,
            t_stop=t_stop_i)
        self.t_start_j, self.t_stop_j = _signals_t_start_stop(
            spiketrains_j,
            t_start=t_start_j,
            t_stop=t_stop_j)
        self.verbose = verbose and rank == 0

        msg = 'The time intervals for x and y need to be either identical ' \
              'or fully disjoint, but they are:\n' \
              'x: ({}, {}) and y: ({}, {}).'.format(self.t_start_i,
                                                    self.t_stop_i,
                                                    self.t_start_j,
                                                    self.t_stop_j)

        # the starts have to be perfectly aligned for the binning to work
        # the stops can differ without impacting the binning
        if self.t_start_i == self.t_start_j:
            if not _quantities_almost_equal(self.t_stop_i, self.t_stop_j):
                raise ValueError(msg)
        elif (self.t_start_i < self.t_start_j < self.t_stop_i) \
                or (self.t_start_i < self.t_stop_j < self.t_stop_i):
            raise ValueError(msg)

        # Compute the binned spike train matrices, along both time axes
        self.spiketrains_binned_i = conv.BinnedSpikeTrain(
            self.spiketrains_i, bin_size=self.bin_size,
            t_start=self.t_start_i, t_stop=self.t_stop_i)
        self.spiketrains_binned_j = conv.BinnedSpikeTrain(
            self.spiketrains_j, bin_size=self.bin_size,
            t_start=self.t_start_j, t_stop=self.t_stop_j)

    @property
    def x_edges(self):
        """
        A Quantity array of `n+1` edges of the bins used for the horizontal
        axis of the intersection matrix, where `n` is the number of bins that
        time was discretized in.
        """
        return self.spiketrains_binned_i.bin_edges.rescale(self.bin_size.units)

    @property
    def y_edges(self):
        """
        A Quantity array of `n+1` edges of the bins used for the vertical axis
        of the intersection matrix, where `n` is the number of bins that
        time was discretized in.
        """
        return self.spiketrains_binned_j.bin_edges.rescale(self.bin_size.units)

    def is_symmetric(self):
        """
        Returns
        -------
        bool
            Whether the intersection matrix  is symmetric or not.

        See Also
        --------
        ASSET.intersection_matrix

        """
        return _quantities_almost_equal(self.x_edges[0], self.y_edges[0])

    def intersection_matrix(self, normalization=None):
        """
        Generates the intersection matrix from a list of spike trains.

        Given a list of `neo.SpikeTrain`, consider two binned versions of them
        differing for the starting and ending times of the binning:
        `t_start_x`, `t_stop_x`, `t_start_y` and `t_stop_y` respectively (the
        time intervals can be either identical or completely disjoint). Then
        calculate the intersection matrix `M` of the two binned data, where
        `M[i,j]` is the overlap of bin `i` in the first binned data and bin `j`
        in the second binned data (i.e., the number of spike trains spiking at
        both bin `i` and bin `j`).

        The matrix entries can be normalized to values between `0` and `1` via
        different normalizations (see "Parameters" section).

        Parameters
        ----------
        normalization : {'intersection', 'mean', 'union'} or None, optional
            The normalization type to be applied to each entry `M[i,j]` of the
            intersection matrix `M`. Given the sets `s_i` and `s_j` of neuron
            IDs in the bins `i` and `j` respectively, the normalization
            coefficient can be:

                * None: no normalisation (row counts)
                * 'intersection': `len(intersection(s_i, s_j))`
                * 'mean': `sqrt(len(s_1) * len(s_2))`
                * 'union': `len(union(s_i, s_j))`
            Default: None

        Returns
        -------
        imat : (n,n) np.ndarray
            The floating point intersection matrix of a list of spike trains.
            It has the shape `(n, n)`, where `n` is the number of bins that
            time was discretized in.

        """
        imat = _intersection_matrix(self.spiketrains_i, self.spiketrains_j,
                                    self.bin_size,
                                    self.t_start_i, self.t_start_j,
                                    self.t_stop_i, self.t_stop_j,
                                    normalization=normalization)
        return imat

    def probability_matrix_montecarlo(self, n_surrogates, imat=None,
                                      surrogate_method='dither_spikes',
                                      surrogate_dt=None):
        """
        Given a list of parallel spike trains, estimate the cumulative
        probability of each entry in their intersection matrix by a Monte Carlo
        approach using surrogate data.

        Contrarily to the analytical version (see
        :func:`ASSET.probability_matrix_analytical`) the Monte Carlo one does
        not incorporate the assumptions of Poissonianity in the null
        hypothesis.

        The method produces surrogate spike trains (using one of several
        methods at disposal, see "Parameters" section) and calculates their
        intersection matrix `M`. For each entry `(i, j)`, the intersection CDF
        `P[i, j]` is then given by:

        .. centered::  P[i, j] = #(spike_train_surrogates such that
                       M[i, j] < I[i, j]) / #(spike_train_surrogates)

        If `P[i, j]` is large (close to 1), `I[i, j]` is statistically
        significant: the probability to observe an overlap equal to or larger
        than `I[i, j]` under the null hypothesis is `1 - P[i, j]`, very small.

        Parameters
        ----------
        n_surrogates : int
            The number of spike train surrogates to generate for the bootstrap
            procedure.
        imat : (n,n) np.ndarray or None, optional
            The floating point intersection matrix of a list of spike trains.
            It has the shape `(n, n)`, where `n` is the number of bins that
            time was discretized in.
            If None, the output of :func:`ASSET.intersection_matrix` is used.
            Default: None
        surrogate_method : {'dither_spike_train', 'dither_spikes',
                            'jitter_spikes',
                            'randomise_spikes', 'shuffle_isis',
                            'joint_isi_dithering'}, optional
            The method to generate surrogate spike trains. Refer to the
            :func:`spike_train_surrogates.surrogates` documentation for more
            information about each surrogate method. Note that some of these
            methods need `surrogate_dt` parameter, others ignore it.
            Default: 'dither_spike_train'
        surrogate_dt : pq.Quantity, optional
            For surrogate methods shifting spike times randomly around their
            original time ('dither_spike_train', 'dither_spikes') or replacing
            them randomly within a certain window ('jitter_spikes'),
            `surrogate_dt` represents the size of that shift (window). For
            other methods, `surrogate_dt` is ignored.
            If None, it's set to `self.bin_size * 5`.
            Default: None

        Returns
        -------
        pmat : np.ndarray
            The cumulative probability matrix. `pmat[i, j]` represents the
            estimated probability of having an overlap between bins `i` and `j`
            STRICTLY LOWER than the observed overlap, under the null hypothesis
            of independence of the input spike trains.

        Notes
        -----
        We recommend playing with `surrogate_dt` parameter to see how it
        influences the result matrix. For this, refer to the ASSET tutorial.

        See Also
        --------
        ASSET.probability_matrix_analytical : analytical derivation of the
                                              matrix

        """
        if imat is None:
            # Compute the intersection matrix of the original data
            imat = self.intersection_matrix()

        if surrogate_dt is None:
            surrogate_dt = self.bin_size * 5

        symmetric = self.is_symmetric()

        # Generate surrogate spike trains as a list surrs
        # Compute the p-value matrix pmat; pmat[i, j] counts the fraction of
        # surrogate data whose intersection value at (i, j) is lower than or
        # equal to that of the original data
        pmat = np.zeros(imat.shape, dtype=np.int32)

        for surr_id in trange(n_surrogates, desc="pmat_bootstrap",
                              disable=not self.verbose):
            if mpi_accelerated and surr_id % size != rank:
                continue
            surrogates = [spike_train_surrogates.surrogates(
                st, n_surrogates=1,
                method=surrogate_method,
                dt=surrogate_dt,
                decimals=None,
                edges=True)[0]
                          for st in self.spiketrains_i]

            if symmetric:
                surrogates_y = surrogates
            else:
                surrogates_y = [spike_train_surrogates.surrogates(
                    st, n_surrogates=1, method=surrogate_method,
                    dt=surrogate_dt, decimals=None, edges=True)[0]
                                for st in self.spiketrains_j]

            imat_surr = _intersection_matrix(surrogates, surrogates_y,
                                             self.bin_size,
                                             self.t_start_i, self.t_start_j,
                                             self.t_stop_i, self.t_stop_j)

            pmat += (imat_surr <= (imat - 1))

            del imat_surr

        if mpi_accelerated:
            pmat = comm.allreduce(pmat, op=MPI.SUM)

        pmat = pmat * 1. / n_surrogates

        if symmetric:
            np.fill_diagonal(pmat, 0.5)

        return pmat

    def probability_matrix_analytical(self, imat=None,
                                      firing_rates_x='estimate',
                                      firing_rates_y='estimate',
                                      kernel_width=100 * pq.ms):
        r"""
        Given a list of spike trains, approximates the cumulative probability
        of each entry in their intersection matrix.

        The approximation is analytical and works under the assumptions that
        the input spike trains are independent and Poisson. It works as
        follows:

            * Bin each spike train at the specified `bin_size`: this yields a
              binary array of 1s (spike in bin) and 0s (no spike in bin;
              clipping used);
            * If required, estimate the rate profile of each spike train by
              convolving the binned array with a boxcar kernel of user-defined
              length;
            * For each neuron `k` and each pair of bins `i` and `j`, compute
              the probability :math:`p_ijk` that neuron `k` fired in both bins
              `i` and `j`.
            * Approximate the probability distribution of the intersection
              value at `(i, j)` by a Poisson distribution with mean parameter
              :math:`l = \sum_k (p_ijk)`,
              justified by Le Cam's approximation of a sum of independent
              Bernouilli random variables with a Poisson distribution.

        Parameters
        ----------
        imat : (n,n) np.ndarray or None, optional
            The intersection matrix of a list of spike trains.
            It has the shape `(n, n)`, where `n` is the number of bins that
            time was discretized in.
            If None, the output of :func:`ASSET.intersection_matrix` is used.
            Default: None
        firing_rates_x, firing_rates_y : list of neo.AnalogSignal or 'estimate'
            If a list, `firing_rates[i]` is the firing rate of the spike train
            `spiketrains[i]`.
            If 'estimate', firing rates are estimated by simple boxcar kernel
            convolution, with the specified `kernel_width`.
            Default: 'estimate'
        kernel_width : pq.Quantity, optional
            The total width of the kernel used to estimate the rate profiles
            when `firing_rates` is 'estimate'.
            Default: 100 * pq.ms

        Returns
        -------
        pmat : np.ndarray
            The cumulative probability matrix. `pmat[i, j]` represents the
            estimated probability of having an overlap between bins `i` and `j`
            STRICTLY LOWER than the observed overlap, under the null hypothesis
            of independence of the input spike trains.
        """
        if imat is None:
            # Compute the intersection matrix of the original data
            imat = self.intersection_matrix()

        symmetric = self.is_symmetric()

        bsts_x_matrix = self.spiketrains_binned_i.to_bool_array()

        if symmetric:
            bsts_y_matrix = bsts_x_matrix
        else:
            bsts_y_matrix = self.spiketrains_binned_j.to_bool_array()

            # Check that the nr. neurons is identical between the two axes
            if bsts_x_matrix.shape[0] != bsts_y_matrix.shape[0]:
                raise ValueError(
                    'Different number of neurons along the x and y axis!')

        # Define the firing rate profiles
        if firing_rates_x == 'estimate':
            # If rates are to be estimated, create the rate profiles as
            # Quantity objects obtained by boxcar-kernel convolution
            fir_rate_x = self._rate_of_binned_spiketrain(bsts_x_matrix,
                                                         kernel_width)
        elif isinstance(firing_rates_x, list):
            # If rates provided as lists of AnalogSignals, create time slices
            # for both axes, interpolate in the time bins of interest and
            # convert to Quantity
            fir_rate_x = _interpolate_signals(
                firing_rates_x, self.spiketrains_binned_i.bin_edges[:-1],
                self.verbose)
        else:
            raise ValueError(
                'fir_rates_x must be a list or the string "estimate"')

        if symmetric:
            fir_rate_y = fir_rate_x
        elif firing_rates_y == 'estimate':
            fir_rate_y = self._rate_of_binned_spiketrain(bsts_y_matrix,
                                                         kernel_width)
        elif isinstance(firing_rates_y, list):
            # If rates provided as lists of AnalogSignals, create time slices
            # for both axes, interpolate in the time bins of interest and
            # convert to Quantity
            fir_rate_y = _interpolate_signals(
                firing_rates_y, self.spiketrains_binned_j.bin_edges[:-1],
                self.verbose)
        else:
            raise ValueError(
                'fir_rates_y must be a list or the string "estimate"')

        # For each neuron, compute the prob. that that neuron spikes in any bin
        if self.verbose:
            print('compute the prob. that each neuron fires in each pair of '
                  'bins...')

        rate_bins_x = (fir_rate_x * self.bin_size).simplified.magnitude
        spike_probs_x = 1. - np.exp(-rate_bins_x)
        if symmetric:
            spike_probs_y = spike_probs_x
        else:
            rate_bins_y = (fir_rate_y * self.bin_size).simplified.magnitude
            spike_probs_y = 1. - np.exp(-rate_bins_y)

        # Compute the matrix Mu[i, j] of parameters for the Poisson
        # distributions which describe, at each (i, j), the approximated
        # overlap probability. This matrix is just the sum of the probability
        # matrices p_ijk computed for each neuron k:
        # p_ijk is the probability that neuron k spikes in both bins i and j.
        # The sum of outer products is equivalent to a dot product.
        if self.verbose:
            print(
                "compute the probability matrix by Le Cam's approximation...")
        Mu = spike_probs_x.T.dot(spike_probs_y)
        # A straightforward implementation is:
        # pmat_shape = spike_probs_x.shape[1], spike_probs_y.shape[1]
        # Mu = np.zeros(pmat_shape, dtype=np.float64)
        # for probx, proby in zip(spike_probs_x, spike_probs_y):
        #     Mu += np.outer(probx, proby)

        # Compute the probability matrix obtained from imat using the Poisson
        # pdfs
        pmat = scipy.stats.poisson.cdf(imat - 1, Mu)

        if symmetric:
            # Substitute 0.5 to the elements along the main diagonal
            if self.verbose:
                print("substitute 0.5 to elements along the main diagonal...")
            np.fill_diagonal(pmat, 0.5)

        return pmat

    def joint_probability_matrix(self, pmat, filter_shape, n_largest,
                                 min_p_value=1e-5, precision='float',
                                 cuda_threads=64, cuda_cwr_loops=32,
                                 tolerance=1e-5):
        """
        Map a probability matrix `pmat` to a joint probability matrix `jmat`,
        where `jmat[i, j]` is the joint p-value of the largest neighbors of
        `pmat[i, j]`.

        The values of `pmat` are assumed to be uniformly distributed in the
        range [0, 1]. Centered a rectangular kernel of shape
        `filter_shape=(l, w)` around each entry `pmat[i, j]`,
        aligned along the diagonal where `pmat[i, j]` lies into, extracts the
        `n_largest` values falling within the kernel and computes their joint
        p-value `jmat[i, j]`.

        Parameters
        ----------
        pmat : np.ndarray
            A square matrix, the output of
            :func:`ASSET.probability_matrix_montecarlo` or
            :func:`ASSET.probability_matrix_analytical`, of cumulative
            probability values between 0 and 1. The values are assumed
            to be uniformly distributed in the said range.
        filter_shape : tuple of int
            A pair of integers representing the kernel shape `(l, w)`.
        n_largest : int
            The number of the largest neighbors to collect for each entry in
            `jmat`.
        min_p_value : float, optional
            The minimum p-value in range `[0, 1)` for individual entries in
            `pmat`. Each `pmat[i, j]` is set to
            `min(pmat[i, j], 1-p_value_min)` to avoid that a single highly
            significant value in `pmat` (extreme case: `pmat[i, j] = 1`) yields
            joint significance of itself and its neighbors.
            Default: 1e-5
<<<<<<< HEAD:elephant/asset.py
=======
        precision : {'float', 'double'}, optional
            Single or double floating-point precision for the resulting `jmat`
            matrix.
              * `'float'`: 32 bits; the tolerance error is ``≲1e-3``.

              * `'double'`: 64 bits; the tolerance error is ``<1e-5``.
            Double floating-point precision is typically x4 times slower than
            the single floating-point equivalent.
            Default: 'float'
        cuda_threads : int, optional
            [CUDA/OpenCL performance parameter that does not influence the
            result.]
            The number of CUDA/OpenCL threads per block (in X axis) between 1
            and 1024 and is used only if CUDA or OpenCL backend is enabled.
            For performance reasons, it should be a multiple of 32.
            Old GPUs (Tesla K80) perform faster with `cuda_threads` larger
            than 64 while new series (Tesla T4) with capabilities 6.x and more
            work best with 32 threads.
            Default: 64
        cuda_cwr_loops : int, optional
            [CUDA/OpenCL performance parameter that does not influence the
            result.]
            A positive integer that defines the number of fast
            'combinations_with_replacement' loops to run to reduce branch
            divergence. This parameter influences the performance when the
            number of iterations is huge (`>1e8`); in such cases, increase
            the value.
            Default: 32
        tolerance : float, optional
            Tolerance is used to catch unexpected behavior of billions of
            floating point additions, when the number of iterations is huge
            or the data arrays are large. A warning is thrown when the
            resulting joint prob. matrix values are outside of the acceptable
            range ``[-tolerance, 1.0 + tolerance]``.
            Default: 1e-5
>>>>>>> master:elephant/asset/asset.py

        Returns
        -------
        jmat : np.ndarray
            The joint probability matrix associated to `pmat`.

        Notes
        -----
        1. By default, if CUDA is detected, CUDA acceleration is used. CUDA
           backend is **~X1000** faster than the Python implementation.
           To turn off CUDA features, set the environment flag
           ``ELEPHANT_USE_CUDA`` to ``0``. Otherwise
        2. If PyOpenCL is installed and detected, PyOpenCL backend is used.
           PyOpenCL backend is **~X100** faster than the Python implementation.
           To turn off OpenCL features, set the environment flag
           ``ELEPHANT_USE_OPENCL`` to ``0``.

           When using PyOpenCL backend, make sure you've disabled GPU Hangcheck
           as described in the `Intel GPU developers documentation
           <https://software.intel.com/content/www/us/en/develop/
           documentation/get-started-with-intel-oneapi-base-linux/top/
           before-you-begin.html>`_. Do it with caution - using your built-in
           Intel graphics card to perform computations may make the system
           unresponsive until the compute program terminates.

        """
        l, w = filter_shape

        # Find for each P_ij in the probability matrix its neighbors and
        # maximize them by the maximum value 1-p_value_min
        pmat = np.asarray(pmat, dtype=np.float32)
        pmat_neighb_obj = _PMatNeighbors(filter_shape=filter_shape,
                                         n_largest=n_largest)
        pmat_neighb = pmat_neighb_obj.compute(pmat)

        pmat_neighb = np.minimum(pmat_neighb, 1. - min_p_value,
                                 out=pmat_neighb)

        # in order to avoid doing the same calculation multiple times:
        # find all unique sets of values in pmat_neighb
        # and store the corresponding indices
        # flatten the second and third dimension in order to use np.unique
        pmat_neighb = pmat_neighb.reshape(pmat.size, n_largest)
        pmat_neighb, pmat_neighb_indices = np.unique(pmat_neighb, axis=0,
                                                     return_inverse=True)

        # Compute the joint p-value matrix jpvmat
        n = l * (1 + 2 * w) - w * (
                w + 1)  # number of entries covered by kernel
        jsf = _JSFUniformOrderStat3D(n=n, d=pmat_neighb.shape[1],
                                     precision=precision,
                                     verbose=self.verbose,
                                     cuda_threads=cuda_threads,
                                     cuda_cwr_loops=cuda_cwr_loops,
                                     tolerance=tolerance)
        jpvmat = jsf.compute(u=pmat_neighb)

        # restore the original shape using the stored indices
        jpvmat = jpvmat[pmat_neighb_indices].reshape(pmat.shape)

        return 1. - jpvmat

    @staticmethod
    def mask_matrices(matrices, thresholds):
        """
        Given a list of `matrices` and a list of `thresholds`, return a boolean
        matrix `B` ("mask") such that `B[i,j]` is True if each input matrix in
        the list strictly exceeds the corresponding threshold at that position.
        If multiple matrices are passed along with only one threshold the same
        threshold is applied to all matrices.

        Parameters
        ----------
        matrices : list of np.ndarray
            The matrices which are compared to the respective thresholds to
            build the mask. All matrices must have the same shape.
            Typically, it is a list `[pmat, jmat]`, i.e., the (cumulative)
            probability and joint probability matrices.
        thresholds : float or list of float
            The significance thresholds for each matrix in `matrices`.

        Returns
        -------
        mask : np.ndarray
            Boolean mask matrix with the shape of the input matrices.

        Raises
        ------
        ValueError
            If `matrices` or `thresholds` is an empty list.

            If `matrices` and `thresholds` have different lengths.

        See Also
        --------
        ASSET.probability_matrix_montecarlo : for `pmat` generation
        ASSET.probability_matrix_analytical : for `pmat` generation
        ASSET.joint_probability_matrix : for `jmat` generation

        """
        if len(matrices) == 0:
            raise ValueError("Empty list of matrices")
        if isinstance(thresholds, float):
            thresholds = np.full(shape=len(matrices), fill_value=thresholds)
        if len(matrices) != len(thresholds):
            raise ValueError(
                '`matrices` and `thresholds` must have same length')

        mask = np.ones_like(matrices[0], dtype=bool)
        for (mat, thresh) in zip(matrices, thresholds):
            mask &= mat > thresh

        # Replace nans, coming from False * np.inf, with zeros
        mask[np.isnan(mask)] = False

        return mask

    @staticmethod
    def cluster_matrix_entries(mask_matrix, max_distance, min_neighbors,
                               stretch, working_memory=None):
        r"""
        Given a matrix `mask_matrix`, replaces its positive elements with
        integers representing different cluster IDs. Each cluster comprises
        close-by elements.

        In ASSET analysis, `mask_matrix` is a thresholded ("masked") version
        of the intersection matrix `imat`, whose values are those of `imat`
        only if considered statistically significant, and zero otherwise.

        A cluster is built by pooling elements according to their distance,
        via the DBSCAN algorithm (see `sklearn.cluster.DBSCAN` class). Elements
        form a neighbourhood if at least one of them has a distance not larger
        than `max_distance` from the others, and if they are at least
        `min_neighbors`. Overlapping neighborhoods form a cluster:

            * Clusters are assigned integers from `1` to the total number `k`
              of clusters;
            * Unclustered ("isolated") positive elements of `mask_matrix` are
              assigned value `-1`;
            * Non-positive elements are assigned the value `0`.

        The distance between the positions of two positive elements in
        `mask_matrix` is given by a Euclidean metric which is stretched if the
        two positions are not aligned along the 45 degree direction (the main
        diagonal direction), as more, with maximal stretching along the
        anti-diagonal. Specifically, the Euclidean distance between positions
        `(i1, j1)` and `(i2, j2)` is stretched by a factor

        .. math::
                 1 + (\mathtt{stretch} - 1.) *
                 \left|\sin((\pi / 4) - \theta)\right|,

        where :math:`\theta` is the angle between the pixels and the 45 degree
        direction. The stretching factor thus varies between 1 and `stretch`.

        Parameters
        ----------
        mask_matrix : np.ndarray
            The boolean matrix, whose elements with positive values are to be
            clustered. The output of :func:`ASSET.mask_matrices`.
        max_distance : float
            The maximum distance between two elements in `mask_matrix` to be
            a part of the same neighbourhood in the DBSCAN algorithm.
        min_neighbors : int
            The minimum number of elements to form a neighbourhood.
        stretch : float
            The stretching factor of the euclidean metric for elements aligned
            along the 135 degree direction (anti-diagonal). The actual
            stretching increases from 1 to `stretch` as the direction of the
            two elements moves from the 45 to the 135 degree direction.
            `stretch` must be greater than 1.
        working_memory : int or None, optional
            The sought maximum memory in MiB for temporary distance matrix
            chunks. When None (default), no chunking is performed. This
            parameter is passed directly to
            ``sklearn.metrics.pairwise_distances_chunked`` function and it
            has no influence on the outcome matrix. Instead, it control the
            memory VS speed trade-off.
            Default: None

        Returns
        -------
        cluster_mat : np.ndarray
            A matrix with the same shape of `mask_matrix`, each of whose
            elements is either:

                * a positive integer (cluster ID) if the element is part of a
                  cluster;
                * `0` if the corresponding element in `mask_matrix` is
                  non-positive;
                * `-1` if the element does not belong to any cluster.

        See Also
        --------
        sklearn.cluster.DBSCAN

        """
        # Don't do anything if mat is identically zero
        if np.all(mask_matrix == 0):
            return mask_matrix

        # List the significant pixels of mat in a 2-columns array
        xpos_sgnf, ypos_sgnf = np.where(mask_matrix > 0)

        # Compute the matrix D[i, j] of euclidean distances between pixels i
        # and j
        try:
            D = _stretched_metric_2d(
                xpos_sgnf, ypos_sgnf, stretch=stretch, ref_angle=45,
                working_memory=working_memory
            )
        except MemoryError as err:
            raise MemoryError("Set 'working_memory=100' or another value to "
                              "chunk the data") from err

        # Cluster positions of significant pixels via dbscan
        core_samples, config = dbscan(
            D, eps=max_distance, min_samples=min_neighbors,
            metric='precomputed')

        # Construct the clustered matrix, where each element has value
        # * i = 1 to k if it belongs to a cluster i,
        # * 0 if it is not significant,
        # * -1 if it is significant but does not belong to any cluster
        cluster_mat = np.zeros_like(mask_matrix, dtype=np.int32)
        cluster_mat[xpos_sgnf, ypos_sgnf] = \
            config * (config == -1) + (config + 1) * (config >= 0)

        return cluster_mat

    def extract_synchronous_events(self, cmat, ids=None):
        """
        Given a list of spike trains, a bin size, and a clustered
        intersection matrix obtained from those spike trains via ASSET
        analysis, extracts the sequences of synchronous events (SSEs)
        corresponding to clustered elements in the cluster matrix.

        Parameters
        ----------
        cmat : (n,n) np.ndarray
            The cluster matrix, the output of
            :func:`ASSET.cluster_matrix_entries`.
        ids : list, optional
            A list of spike train IDs. If provided, `ids[i]` is the identity
            of `spiketrains[i]`. If None, the IDs `0,1,...,n-1` are used.
            Default: None

        Returns
        -------
        sse_dict : dict
            A dictionary `D` of SSEs, where each SSE is a sub-dictionary `Dk`,
            `k=1,...,K`, where `K` is the max positive integer in `cmat` (i.e.,
            the total number of clusters in `cmat`):

            .. centered:: D = {1: D1, 2: D2, ..., K: DK}

            Each sub-dictionary `Dk` represents the k-th diagonal structure
            (i.e., the k-th cluster) in `cmat`, and is of the form

            .. centered:: Dk = {(i1, j1): S1, (i2, j2): S2, ..., (iL, jL): SL}.

            The keys `(i, j)` represent the positions (time bin IDs) of all
            elements in `cmat` that compose the SSE (i.e., that take value `l`
            and therefore belong to the same cluster), and the values `Sk` are
            sets of neuron IDs representing a repeated synchronous event (i.e.,
            spiking at time bins `i` and `j`).
        """
        nr_worms = cmat.max()  # number of different clusters ("worms") in cmat
        if nr_worms <= 0:
            return {}

        # Compute the transactions associated to the two binnings
        tracts_x = _transactions(
            self.spiketrains_i, bin_size=self.bin_size, t_start=self.t_start_i,
            t_stop=self.t_stop_i,
            ids=ids)

        if self.spiketrains_j is self.spiketrains_i:
            diag_id = 0
            tracts_y = tracts_x
        else:
            if self.is_symmetric():
                diag_id = 0
                tracts_y = tracts_x
            else:
                diag_id = None
                tracts_y = _transactions(
                    self.spiketrains_j, bin_size=self.bin_size,
                    t_start=self.t_start_j, t_stop=self.t_stop_j, ids=ids)

        # Reconstruct each worm, link by link
        sse_dict = {}
        for k in range(1, nr_worms + 1):  # for each worm
            # worm k is a list of links (each link will be 1 sublist)
            worm_k = {}
            pos_worm_k = np.array(
                np.where(cmat == k)).T  # position of all links
            # if no link lies on the reference diagonal
            if all([y - x != diag_id for (x, y) in pos_worm_k]):
                for bin_x, bin_y in pos_worm_k:  # for each link

                    # reconstruct the link
                    link_l = set(tracts_x[bin_x]).intersection(
                        tracts_y[bin_y])

                    # and assign it to its pixel
                    worm_k[(bin_x, bin_y)] = link_l

                sse_dict[k] = worm_k

        return sse_dict

    def _rate_of_binned_spiketrain(self, binned_spiketrains, kernel_width):
        """
        Calculate the rate of binned spiketrains using convolution with
        a boxcar kernel.
        """
        if self.verbose:
            print('compute rates by boxcar-kernel convolution...')

        # Create the boxcar kernel and convolve it with the binned spike trains
        k = int((kernel_width / self.bin_size).simplified.item())
        kernel = np.full(k, fill_value=1. / k)
        rate = np.vstack([np.convolve(bst, kernel, mode='same')
                          for bst in binned_spiketrains])

        # The convolution results in an array decreasing at the borders due
        # to absence of spikes beyond the borders. Replace the first and last
        # (k//2) elements with the (k//2)-th / (n-k//2)-th ones, respectively
        k2 = k // 2
        for i in range(rate.shape[0]):
            rate[i, :k2] = rate[i, k2]
            rate[i, -k2:] = rate[i, -k2 - 1]

        # Multiply the firing rates by the proper unit
        rate = rate * (1. / self.bin_size).rescale('Hz')

        return rate
