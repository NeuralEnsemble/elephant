# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq

import elephant.statistics as es


class isi_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_2d = np.array([[0.3, 0.56, 0.87, 1.23],
                                       [0.02, 0.71, 1.82, 8.46],
                                       [0.03, 0.14, 0.15, 0.92]])
        self.targ_array_2d_0 = np.array([[-0.28,  0.15,  0.95,  7.23],
                                         [0.01, -0.57, -1.67, -7.54]])
        self.targ_array_2d_1 = np.array([[0.26, 0.31, 0.36],
                                         [0.69, 1.11, 6.64],
                                         [0.11, 0.01, 0.77]])
        self.targ_array_2d_default = self.targ_array_2d_1

        self.test_array_1d = self.test_array_2d[0, :]
        self.targ_array_1d = self.targ_array_2d_1[0, :]

    def test_isi_with_spiketrain(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms', t_stop=10.0)
        target = pq.Quantity(self.targ_array_1d, 'ms')
        res = es.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_quantities_1d(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        target = pq.Quantity(self.targ_array_1d, 'ms')
        res = es.isi(st)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_1d(self):
        st = self.test_array_1d
        target = self.targ_array_1d
        res = es.isi(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_default(self):
        st = self.test_array_2d
        target = self.targ_array_2d_default
        res = es.isi(st)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_0(self):
        st = self.test_array_2d
        target = self.targ_array_2d_0
        res = es.isi(st, axis=0)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)

    def test_isi_with_plain_array_2d_1(self):
        st = self.test_array_2d
        target = self.targ_array_2d_1
        res = es.isi(st, axis=1)
        assert not isinstance(res, pq.Quantity)
        assert_array_almost_equal(res, target, decimal=9)


class isi_cv_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_regular = np.arange(1, 6)

    def test_cv_isi_regular_spiketrain_is_zero(self):
        st = neo.SpikeTrain(self.test_array_regular,  units='ms', t_stop=10.0)
        targ = 0.0
        res = es.cv(es.isi(st))
        self.assertEqual(res, targ)

    def test_cv_isi_regular_array_is_zero(self):
        st = self.test_array_regular
        targ = 0.0
        res = es.cv(es.isi(st))
        self.assertEqual(res, targ)



if __name__ == '__main__':
    unittest.main()
