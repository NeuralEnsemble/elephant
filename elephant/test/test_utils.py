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


if __name__ == '__main__':
    unittest.main()
