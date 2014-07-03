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

import elephant.conversion as cv


def get_nearest(times, time):
    return (np.abs(times-time)).argmin()


class binarize_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_array_1d = np.array([1.23, 0.3, 0.87, 0.56])

    def test_binarize_with_spiketrain_exact(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=100)
        times = np.arange(0, 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_spiketrain_exact_set_ends(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=100)
        times = np.arange(5., 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_start=5., t_stop=10.)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_spiketrain_round(self):
        st = neo.SpikeTrain(self.test_array_1d, units='ms',
                            t_stop=10.0, sampling_rate=10.0)
        times = np.arange(0, 10.+.1, .1)
        target = np.zeros_like(times).astype('bool')
        for time in np.round(self.test_array_1d, 1):
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_exact(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(0, 1.23+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True,
                                sampling_rate=100.*pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_exact_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(0, 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_stop=10.,
                                sampling_rate=100.*pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_quantities_round_set_ends(self):
        st = pq.Quantity(self.test_array_1d, units='ms')
        times = np.arange(5., 10.+.1, .1)
        target = np.zeros_like(times).astype('bool')
        times = pq.Quantity(times, units='ms')

        res, tres = cv.binarize(st, return_times=True, t_start=5., t_stop=10.,
                                sampling_rate=10.*pq.kHz)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_plain_array_exact(self):
        st = self.test_array_1d
        times = np.arange(0, 1.23+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res, tres = cv.binarize(st, return_times=True, sampling_rate=100)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_with_plain_array_exact_set_ends(self):
        st = self.test_array_1d
        times = np.arange(0, 10.+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res, tres = cv.binarize(st, return_times=True, t_stop=10., sampling_rate=100.)
        assert_array_almost_equal(res, target, decimal=9)
        assert_array_almost_equal(tres, times, decimal=9)

    def test_binarize_no_time(self):
        st = self.test_array_1d
        times = np.arange(0, 1.23+.01, .01)
        target = np.zeros_like(times).astype('bool')
        for time in self.test_array_1d:
            target[get_nearest(times, time)] = True

        res0, tres = cv.binarize(st, return_times=True, sampling_rate=100)
        res1 = cv.binarize(st, return_times=False, sampling_rate=100)
        res2 = cv.binarize(st, sampling_rate=100)
        assert_array_almost_equal(res0, res1, decimal=9)
        assert_array_almost_equal(res0, res2, decimal=9)

    def test_binariz_rate_with_plain_array_and_units_typeerror(self):
        st = self.test_array_1d
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=pq.Quantity(0, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_stop=pq.Quantity(10, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=pq.Quantity(10, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=pq.Quantity(0, 'ms'),
                          t_stop=10.,
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          t_start=0.,
                          t_stop=pq.Quantity(10, 'ms'),
                          sampling_rate=10.)
        self.assertRaises(TypeError, cv.binarize, st,
                          sampling_rate=10.*pq.Hz)

    def test_binariz_without_sampling_rate_valueerror(self):
        st0 = self.test_array_1d
        st1 = pq.Quantity(st0, 'ms')
        self.assertRaises(ValueError, cv.binarize, st0)
        self.assertRaises(ValueError, cv.binarize, st0,
                          t_start=0)
        self.assertRaises(ValueError, cv.binarize, st0,
                          t_stop=10)
        self.assertRaises(ValueError, cv.binarize, st0,
                          t_start=0, t_stop=10)
        self.assertRaises(ValueError, cv.binarize, st1,
                          t_start=pq.Quantity(0, 'ms'), t_stop=10.)
        self.assertRaises(ValueError, cv.binarize, st1,
                          t_start=0., t_stop=pq.Quantity(10, 'ms'))
        self.assertRaises(ValueError, cv.binarize, st1)


if __name__ == '__main__':
    unittest.main()
