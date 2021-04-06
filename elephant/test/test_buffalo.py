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
from elephant.buffalo.object_hash import BuffaloObjectHasher


@unittest.skipUnless(buffalo.HAVE_PROV)
class ObjectHasherTestCase(unittest.TestCase):

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
            obj_hash = BuffaloObjectHasher(var)
            self.assertEquals(obj_hash.type, "builtins.")
        list_hash = BuffaloObjectHasher(self.test_list)
        self.assertIsInstance(hash(list_hash), int)

        str_hash = BuffaloObjectHasher(self.test_string)
        self.assertIsInstance(hash(str_hash), int)

        int_hash = BuffaloObjectHasher(self.test_integer)
        self.assertIsInstance(hash(int_hash), int)
