# -*- coding: utf-8 -*-
"""
Unit tests for the Buffalo package.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division

import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

from elephant.buffalo.object_hash import BuffaloObjectHash
import elephant.kernels as kernels
from elephant import statistics
from elephant.spike_train_generation import homogeneous_poisson_process


class ObjectHash_TestCase(unittest.TestCase):

    def setUp(self):
        self.test_list = [1, 2, 3]
        self.test_array = np.array([1, 2, 3])
        self.test_quantities = pq.Quantity([1, 2, 3] * pq.ms)
        self.test_string = "test"
        self.test_float = 5.0
        self.test_integer = 5

    def test_hash_builtins(self):
        for var in [self.test_list, self.test_integer, self.test_float,
                    self.test_string]:
            obj_hash = BuffaloObjectHash(var)
            self.assertEquals(obj_hash.type, "builtins.")
        list_hash = BuffaloObjectHash(self.test_list)
        self.assertIsInstance(hash(list_hash), int)

        str_hash = BuffaloObjectHash(self.test_string)
        self.assertIsInstance(hash(str_hash), int)

        int_hash = BuffaloObjectHash(self.test_integer)
        self.assertIsInstance(hash(int_hash), int)