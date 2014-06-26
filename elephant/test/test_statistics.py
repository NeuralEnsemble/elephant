"""
docstring goes here

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq
import neo
import elephant.statistics as es


class SpikeTrainStatisticsTest(unittest.TestCase):

    def test_isi_with_spiketrain(self):
        st = neo.SpikeTrain([0.3, 0.56, 0.87, 1.23], units='ms', t_stop=10.0)
        target = pq.Quantity([0.26, 0.31, 0.36], 'ms')
        assert_array_almost_equal(es.isi(st), target, decimal=9)

    def test_isi_with_plain_array_no_units(self):
        st = np.array([0.3, 0.56, 0.87, 1.23])
        target = np.array([0.26, 0.31, 0.36])
        intervals = es.isi(st)
        assert not isinstance(intervals, pq.Quantity)
        assert_array_almost_equal(intervals, target, decimal=9)

    def test_isi_with_plain_array_and_units(self):
        st = np.array([0.3, 0.56, 0.87, 1.23])
        target = pq.Quantity([0.26, 0.31, 0.36], 'ms')
        intervals = es.isi(st, units="ms")
        assert isinstance(intervals, pq.Quantity)
        assert_array_almost_equal(intervals, target, decimal=9)

    def test_isi_with_spiketrain_change_units(self):
        st = neo.SpikeTrain([0.3, 0.56, 0.87, 1.23], units='s', t_stop=10.0)
        target = pq.Quantity([260.0, 310.0, 360.0], 'ms')
        assert_array_almost_equal(es.isi(st, units='ms'), target, decimal=9)

    def test_cv_isi_of_regular_spiketrain_is_zero(self):
        regular_st = neo.SpikeTrain([1, 2, 3, 4, 5], units='ms', t_stop=10.0)
        cvisi = es.cv(es.isi(regular_st))
        self.assertEqual(cvisi, 0.0)

    def test_cv_isi_of_regular_array_is_zero(self):
        regular_st = np.array([1, 2, 3, 4, 5])
        cvisi = es.cv(es.isi(regular_st))
        self.assertEqual(cvisi, 0.0)
