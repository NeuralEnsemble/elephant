# -*- coding: utf-8 -*-
"""
Unit tests for the synchrofact detection app
"""

import unittest

import neo
import numpy as np
import quantities as pq

from elephant import utils
from numpy.testing import assert_array_equal


class TestUtils(unittest.TestCase):

    def test_check_neo_consistency(self):
        self.assertRaises(TypeError,
                          utils.check_neo_consistency,
                          [], object_type=neo.SpikeTrain)
        self.assertRaises(TypeError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s),
                           np.arange(2)], object_type=neo.SpikeTrain)
        self.assertRaises(ValueError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.s,
                                          t_start=1*pq.s,
                                          t_stop=2*pq.s),
                           neo.SpikeTrain([1]*pq.s,
                                          t_start=0*pq.s,
                                          t_stop=2*pq.s)],
                          object_type=neo.SpikeTrain)
        self.assertRaises(ValueError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s),
                           neo.SpikeTrain([1]*pq.s, t_stop=3*pq.s)],
                          object_type=neo.SpikeTrain)
        self.assertRaises(ValueError,
                          utils.check_neo_consistency,
                          [neo.SpikeTrain([1]*pq.ms, t_stop=2000*pq.ms),
                           neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s)],
                          object_type=neo.SpikeTrain)

    def test_round_binning_errors(self):
        with self.assertWarns(UserWarning):
            n_bins = utils.round_binning_errors(0.999999, tolerance=1e-6)
            self.assertEqual(n_bins, 1)
        self.assertEqual(utils.round_binning_errors(0.999999, tolerance=None),
                         0)
        array = np.array([0, 0.7, 1 - 1e-8, 1 - 1e-9])
        with self.assertWarns(UserWarning):
            corrected = utils.round_binning_errors(array.copy())
            assert_array_equal(corrected, [0, 0, 1, 1])
        assert_array_equal(
            utils.round_binning_errors(array.copy(), tolerance=None),
            [0, 0, 0, 0])


class CalculateNBinsTestCase(unittest.TestCase):
    def test_calculate_n_bins_not_time_quantity(self):
        # t_start is in Hz
        self.assertRaises(TypeError, utils.calculate_n_bins, t_start=1*pq.Hz,
                          t_stop=5*pq.ms, bin_size=1*pq.ms)

    def test_calculate_n_bins_different_units(self):
        self.assertWarns(UserWarning, utils.calculate_n_bins, t_start=1*pq.s,
                         t_stop=5*pq.ms, bin_size=1*pq.ms)

    def test_calculate_n_bins_int(self):
        n_bins = utils.calculate_n_bins(t_start=0*pq.s, t_stop=10*pq.ms,
                                        bin_size=3*pq.ms)
        self.assertIsInstance(n_bins, int)
        self.assertEqual(n_bins, 3)

    def test_calculate_n_bins_float(self):
        n_bins = utils.calculate_n_bins(t_start=0.0*pq.s, t_stop=9.0*pq.ms,
                                        bin_size=3.0*pq.ms)
        self.assertIsInstance(n_bins, int)
        self.assertEqual(n_bins, 3)

    def test_calculate_n_bins_rounding(self):
        n_bins = utils.calculate_n_bins(t_start=0*pq.s,
                                        t_stop=0.99999999*pq.ms,
                                        bin_size=1.0*pq.ms)
        self.assertIsInstance(n_bins, int)
        self.assertEqual(n_bins, 1)


if __name__ == '__main__':
    unittest.main()
