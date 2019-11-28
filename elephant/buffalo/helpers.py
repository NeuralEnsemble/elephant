# -*- coding: utf-8 -*-
"""
This module implements helper functions used with Buffalo objects.

:copyright: Copyright 2014-2019 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""
import numpy as np
import quantities as pq


def _check_list_size_and_types(value, allowed_types=None):
    """
    Checks if a Python list is not empty, and if the objects are only of the types allowed.

    Parameters
    ----------
    value: list
        List object that will be checked

    allowed_types: tuple
        Allowed types for list objects.
        If None, type checking will be skipped.

    Raises
    -------
    ValueError
        If list empty or if it contains items of not allowed types
    """
    if not isinstance(value, list):
        raise ValueError("Not a list")

    # First check if the list is not empty
    if not len(value):
        raise ValueError("List must not be empty")

    # Now Check that items are of the allowed types
    if allowed_types is not None:
        for item in value:
            if not type(item) in allowed_types:
                raise ValueError(f"List must contain only allowed_types: {', '.join(map(str, allowed_types))}")


def _check_ndarray_size_and_types(value, ndim=1, dtypes=None):
    """
    Checks if a NumPy ndarray is not empty, has the specified number of dimensions, and if the dtype is allowed.

    Parameters
    ----------
    value: np.ndarray
        NumPy array object that will be checked

    ndim: int
        Number of dimensions. Default is 1

    dtypes: tuple
        Allowed dtypes for the NumPy ndarray.
        If None, type checking will be skipped.

    Raises
    ------
    ValueError
        If array is empty, or with the wrong dimensions, or of a not allowed dtype
    """
    # First check that array is not empty
    if not value.size:
        raise ValueError("NumPy array must not be empty")

    # Array must have `ndim` dimensions
    if value.ndim != ndim:
        raise ValueError(f"NumPy array must be {ndim}D. {value.ndim} dimensions were given")

    # Check if dtype is any of the allowed
    if dtypes is not None:
        if value.dtype not in tuple(map(np.dtype, dtypes)):
            raise ValueError(f"NumPy array must be of dtype: {', '.join(map(str, dtypes))}")


def _check_quantities_size_and_unit(value, ndim=1, unit=None):
    """
    Checks that a Quantity array is not empty, has the specified number of dimensions, and has a specific dimensionality

    Parameters
    ----------
    value: Quantity array
        The array being evaluated

    ndim: int
        Number of dimensions. Default is 1

    unit: pq.Quantity
        Unit that should be the same dimensionality as the array under evaluation.
        If None, dimensionality checking will be skipped.

    Raises
    ------
    ValueError
        If array is empty, or with the wrong dimensions, or the quantity has the wrong dimensionality.
    """
    # Check if empty
    if not len(value):
        raise ValueError("Quantity array must not be empty")

    # Check if has `ndim` dimensions
    if value.ndim != ndim:
        raise ValueError(f"Quantity array must be {ndim}D. {value.ndim} dimensions were given")

    # Check if quantity is the desired
    if unit is not None:
        if value.simplified.dimensionality != unit.dimensionality:
            raise ValueError(f"Quantity array input must be of {unit.units} quantity")
