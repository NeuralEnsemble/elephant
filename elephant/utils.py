"""
.. autosummary::
    :toctree: _toctree/utils

    is_time_quantity
    get_common_start_stop_times
    check_neo_consistency
    check_same_units
    round_binning_errors
"""

from __future__ import division, print_function, unicode_literals

import ctypes
import warnings
from functools import wraps

import neo
import numpy as np
import quantities as pq


__all__ = [
    "deprecated_alias",
    "is_binary",
    "is_time_quantity",
    "get_common_start_stop_times",
    "check_neo_consistency",
    "check_same_units",
    "round_binning_errors"
]


def is_binary(array):
    """
    Parameters
    ----------
    array : np.ndarray or list

    Returns
    -------
    bool
        Whether the input array is binary or not.

    """
    array = np.asarray(array)
    return ((array == 0) | (array == 1)).all()


def deprecated_alias(**aliases):
    """
    A deprecation decorator constructor.

    Parameters
    ----------
    **aliases
        The key-value pairs of mapping old --> new argument names of a
        function.

    Returns
    -------
    callable
        A decorator for the specific mapping of deprecated argument names.

    Examples
    --------
    In the example below, `my_function(binsize)` signature is marked as
    deprecated (but still usable) and changed to `my_function(bin_size)`.

    >>> @deprecated_alias(binsize='bin_size')
    ... def my_function(bin_size):
    ...     pass

    """
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(func_name, kwargs, aliases):
    for old, new in aliases.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError(f"{func_name} received both '{old}' and "
                                f"'{new}'")
            warnings.warn(f"'{old}' is deprecated; use '{new}'",
                          DeprecationWarning)
            kwargs[new] = kwargs.pop(old)


def is_time_quantity(*quantities, allow_none=False):
    """
    Parameters
    ----------
    *quantities : pq.Quantity
         A scalar or array-like to check for being a Quantity with time units.
    allow_none : bool, optional
        Allow the input to be None or not.
        Default: False

    Returns
    -------
    bool
        Whether the input is a time Quantity (True) or not (False).
        If the input is None and `allow_none` is set to True, returns True.

    """
    for quantity in quantities:
        if allow_none and quantity is None:
            continue
        if not isinstance(quantity, pq.Quantity):
            return False
        if quantity.dimensionality.simplified != pq.s.dimensionality:
            return False
    return True


def get_common_start_stop_times(neo_objects):
    """
    Extracts the common `t_start` and the `t_stop` from the input neo objects.

    If a single neo object is given, its `t_start` and `t_stop` is returned.
    Otherwise, the aligned times are returned: the maximal `t_start` and
    minimal `t_stop` across `neo_objects`.

    Parameters
    ----------
    neo_objects : neo.SpikeTrain or neo.AnalogSignal or list
        A neo object or a list of neo objects that have `t_start` and `t_stop`
        attributes.

    Returns
    -------
    t_start, t_stop : pq.Quantity
        Shared start and stop times.

    Raises
    ------
    AttributeError
        If the input neo objects do not have `t_start` and `t_stop` attributes.
    ValueError
        If there is no shared interval ``[t_start, t_stop]`` across the input
        neo objects.
    """
    if hasattr(neo_objects, 't_start') and hasattr(neo_objects, 't_stop'):
        return neo_objects.t_start, neo_objects.t_stop
    try:
        t_start = max(elem.t_start for elem in neo_objects)
        t_stop = min(elem.t_stop for elem in neo_objects)
    except AttributeError:
        raise AttributeError("Input neo objects must have 't_start' and "
                             "'t_stop' attributes")
    if t_stop < t_start:
        raise ValueError(f"t_stop ({t_stop}) is smaller than t_start "
                         f"({t_start})")
    return t_start, t_stop


def check_neo_consistency(neo_objects, object_type, t_start=None,
                          t_stop=None, tolerance=1e-8):
    """
    Checks that all input neo objects share the same units, t_start, and
    t_stop.

    Parameters
    ----------
    neo_objects : list of neo.SpikeTrain or neo.AnalogSignal
        A list of neo spike trains or analog signals.
    object_type : type
        The common type.
    t_start, t_stop : pq.Quantity or None, optional
        If None, check for exact match of t_start/t_stop across the input.
    tolerance : float, optional
        The absolute affordable tolerance for the discrepancies between
        t_start/stop magnitude values across trials.
        Default : 1e-6

    Raises
    ------
    TypeError
        If input objects are not instances of the specified `object_type`.
    ValueError
        If input object units, t_start, or t_stop do not match across trials.
    """
    if not isinstance(neo_objects, (list, tuple)):
        neo_objects = [neo_objects]
    try:
        units = neo_objects[0].units
        start = neo_objects[0].t_start.item()
        stop = neo_objects[0].t_stop.item()
    except (IndexError, AttributeError):
        raise TypeError(f"The input must be a list of {object_type.__name__}")
    if not is_time_quantity(t_start, t_stop, allow_none=True):
        raise TypeError("'t_start' and 't_stop' must be time quantities.")
    if tolerance is None:
        tolerance = 0
    for neo_obj in neo_objects:
        if not isinstance(neo_obj, object_type):
            raise TypeError("The input must be a list of "
                            f"{object_type.__name__}. Got "
                            f"{type(neo_obj).__name__}")
        if neo_obj.units != units:
            raise ValueError("The input must have the same units.")
        if t_start is None and abs(neo_obj.t_start.item() - start) > tolerance:
            raise ValueError("The input must have the same t_start.")
        if t_stop is None and abs(neo_obj.t_stop.item() - stop) > tolerance:
            raise ValueError("The input must have the same t_stop.")


def check_same_units(quantities, object_type=pq.Quantity):
    """
    Check that all input quantities are of the same type and share common
    units. Raise an error if the check is unsuccessful.

    Parameters
    ----------
    quantities : list of pq.Quantity or pq.Quantity
        A list of quantities, neo objects or a single neo object.
    object_type : type, optional
        The common type.
        Default: pq.Quantity

    Raises
    ------
    TypeError
        If input objects are not instances of the specified `object_type`.
    ValueError
        If input objects do not share common units.
    """
    if not isinstance(quantities, (list, tuple)):
        quantities = [quantities]
    try:
        units = quantities[0].units
    except (IndexError, AttributeError):
        raise TypeError(f"The input must be a list of {object_type.__name__}")
    for quantity in quantities:
        if not isinstance(quantity, object_type):
            raise TypeError("The input must be a list of "
                            f"{object_type.__name__}. Got "
                            f"{type(quantity).__name__}")
        if quantity.units != units:
            raise ValueError("The input quantities must have the same units, "
                             "which is achieved with object.rescale('ms') "
                             "operation.")


def round_binning_errors(values, tolerance=1e-8):
    """
    Round the input `values` in-place due to the machine floating point
    precision errors.

    Parameters
    ----------
    values : np.ndarray or float
        An input array or a scalar.
    tolerance : float or None, optional
        The precision error absolute tolerance; acts as ``atol`` in
        :func:`numpy.isclose` function. If None, no rounding is performed.
        Default: 1e-8

    Returns
    -------
    values : np.ndarray or int
        Corrected integer values.

    Examples
    --------
    >>> from elephant.utils import round_binning_errors
    >>> round_binning_errors(0.999999, tolerance=None)
    0
    >>> round_binning_errors(0.999999, tolerance=1e-6)
    1
    """
    if tolerance is None or tolerance == 0:
        if isinstance(values, np.ndarray):
            return values.astype(np.int32)
        return int(values)  # a scalar

    # same as '1 - (values % 1) <= tolerance' but faster
    correction_mask = 1 - tolerance <= values % 1
    if isinstance(values, np.ndarray):
        num_corrections = correction_mask.sum()
        if num_corrections > 0:
            warnings.warn(f'Correcting {num_corrections} rounding errors by '
                          f'shifting the affected spikes into the following '
                          f'bin. You can set tolerance=None to disable this '
                          'behaviour.')
            values[correction_mask] += 0.5
        return values.astype(np.int32)

    if correction_mask:
        warnings.warn('Correcting a rounding error in the calculation '
                      'of the number of bins by incrementing the value by 1. '
                      'You can set tolerance=None to disable this '
                      'behaviour.')
        values += 0.5
    return int(values)


def get_cuda_capability_major():
    """
    Extracts CUDA capability major version of the first available Nvidia GPU
    card, if detected. Otherwise, return 0.

    Returns
    -------
    int
        CUDA capability major version.
    """
    cuda_success = 0
    for libname in ('libcuda.so', 'libcuda.dylib', 'cuda.dll'):
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        # not found
        return 0
    result = cuda.cuInit(0)
    if result != cuda_success:
        return 0
    device = ctypes.c_int()
    # parse the first GPU card only
    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if result != cuda_success:
        return 0

    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cuda.cuDeviceComputeCapability(ctypes.byref(cc_major),
                                   ctypes.byref(cc_minor),
                                   device)
    return cc_major.value
