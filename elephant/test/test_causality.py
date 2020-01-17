# -*- coding: utf-8 -*-
"""
Unit tests for the causality module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

import neo
import numpy as np
import scipy.signal as spsig
import scipy.stats
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq
import elephant.signal_processing
from numpy.ma.testutils import assert_array_equal, assert_allclose


class PairwiseGrangerTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_lag_covariances(self):
        pass

    def test_vector_arm(self):
        pass

    def tearDown(self) -> None:
        pass


