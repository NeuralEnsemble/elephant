# -*- coding: utf-8 -*-
"""
Unit tests for the analyses in the `buffalo.helpers` module.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import print_function, unicode_literals

import unittest
import numpy as np
import quantities as pq

from elephant.buffalo.helpers import (_check_list_size_and_types,
                                      _check_ndarray_size_and_types,
                                      _check_quantities_size_and_unit)


class BuffaloHelpersTestCase(unittest.TestCase):

    def test_list_check(self):
        empty = []
        right_if_numerical = [1, 2, 3.0]
        wrong_if_numerical = ['1', '2', '3']
        allowed_types_num = (int, float)
        allowed_types_str = (str,)

        with self.assertRaises(ValueError):
            _check_list_size_and_types(empty)

        with self.assertRaises(ValueError):
            _check_list_size_and_types(wrong_if_numerical, allowed_types=allowed_types_num)

        with self.assertRaises(ValueError):
            _check_list_size_and_types(right_if_numerical, allowed_types=allowed_types_str)

        # These should pass
        _check_list_size_and_types(wrong_if_numerical)          # No type checking
        _check_list_size_and_types(right_if_numerical)          # No type checking
        _check_list_size_and_types(right_if_numerical, allowed_types=allowed_types_num)   # If num types are allowed
        _check_list_size_and_types(wrong_if_numerical, allowed_types=allowed_types_str)   # If strings are allowed

    def test_array_check(self):
        empty = np.array([])
        right_if_numerical = np.array([1, 2, 3.0])
        wrong_if_numerical = np.array(['1', '2', '3'])
        allowed_types_num = (np.int, np.float)
        allowed_types_str = (np.dtype('<U1'),)

        numerical_2d_array = np.array([[1, 2, 3],
                                       [4, 5, 6]])

        with self.assertRaises(ValueError):
            _check_ndarray_size_and_types(empty)

        with self.assertRaises(ValueError):
            _check_ndarray_size_and_types(wrong_if_numerical, dtypes=allowed_types_num)

        with self.assertRaises(ValueError):
            _check_ndarray_size_and_types(right_if_numerical, dtypes=allowed_types_str)

        with self.assertRaises(ValueError):
            _check_ndarray_size_and_types(numerical_2d_array, ndim=2, dtypes=allowed_types_str)

        with self.assertRaises(ValueError):
            _check_ndarray_size_and_types(numerical_2d_array, ndim=1, dtypes=allowed_types_num)

        # These should pass
        _check_ndarray_size_and_types(wrong_if_numerical)          # No type checking
        _check_ndarray_size_and_types(right_if_numerical)          # No type checking
        _check_ndarray_size_and_types(right_if_numerical, dtypes=allowed_types_num)          # If num types are allowed
        _check_ndarray_size_and_types(wrong_if_numerical, dtypes=allowed_types_str)          # If strings are allowed
        _check_ndarray_size_and_types(numerical_2d_array, ndim=2, dtypes=allowed_types_num)  # Right dimension and types

    def test_quantity_check(self):
        length = pq.m
        time = pq.s

        empty = pq.Quantity([], length)
        right_if_length = pq.Quantity([1, 2, 3.0], length)
        wrong_if_length = pq.Quantity([1, 2, 3.0], time)

        numerical_2d_quantity = pq.Quantity([[1, 2, 3],
                                             [4, 5, 6]], length)

        with self.assertRaises(ValueError):
            _check_quantities_size_and_unit(empty)

        with self.assertRaises(ValueError):
            _check_quantities_size_and_unit(wrong_if_length, unit=length)

        with self.assertRaises(ValueError):
            _check_quantities_size_and_unit(right_if_length, unit=time)

        with self.assertRaises(ValueError):
            _check_quantities_size_and_unit(numerical_2d_quantity, ndim=2, unit=time)

        with self.assertRaises(ValueError):
            _check_quantities_size_and_unit(numerical_2d_quantity, ndim=1, unit=length)

        # These should pass
        _check_quantities_size_and_unit(wrong_if_length)          # No unit checking
        _check_quantities_size_and_unit(right_if_length)          # No unit checking
        _check_quantities_size_and_unit(right_if_length, unit=length)             # If length quantity
        _check_quantities_size_and_unit(wrong_if_length, unit=time)               # If time quantity
        _check_quantities_size_and_unit(numerical_2d_quantity, ndim=2, unit=length)  # Right dimension and unit
