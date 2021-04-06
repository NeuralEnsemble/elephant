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

import elephant.buffalo as buffalo

try:
    from elephant.buffalo.object_hash import BuffaloObjectHasher
except ImportError:
    pass


@unittest.skipUnless(buffalo.HAVE_PROV, "requirements-prov missing")
class ObjectHasherTestCase(unittest.TestCase):

    def setUp(self):
        self.test_list = [1, 2, 3]
        self.test_array = np.array([1, 2, 3])
        self.test_quantities = pq.Quantity([1, 2, 3] * pq.ms)
        self.test_string = "test"
        self.test_float = 5.0
        self.test_integer = 5

    def test_hash_builtins(self):
        hasher = BuffaloObjectHasher()
        for var in [self.test_list, self.test_integer, self.test_float,
                    self.test_string]:
            obj_hash = hasher.info(var)
            self.assertTrue(obj_hash.type.startswith("builtins."))

        list_hash = hasher.info(self.test_list).hash
        self.assertIsInstance(list_hash, int)

        str_hash = hasher.info(self.test_string).hash
        self.assertIsInstance(str_hash, int)

        int_hash = hasher.info(self.test_integer).hash
        self.assertIsInstance(int_hash, int)
