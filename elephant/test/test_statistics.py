"""
docstring goes here

:copyright: Copyright 2014 by the ElePhAnT team, see AUTHORS.txt.
:license: CeCILL, see LICENSE.txt for details.
"""

import unittest
from numpy.testing.utils import assert_array_almost_equal
import quantities as pq
import neo
import elephant.statistics as es


class SpikeTrainStatisticsTest(unittest.TestCase):

    def test_isi(self):
        st = neo.SpikeTrain([0.3, 0.56, 0.87, 1.23], units='ms', t_stop=10.0)
        target = pq.Quantity([0.26, 0.31, 0.36], 'ms')
        assert_array_almost_equal(es.isi(st), target, decimal=9)

    def test_cv_isi_of_regular_spiketrain_is_zero(self):
        regular_st = neo.SpikeTrain([1, 2, 3, 4, 5], units='ms', t_stop=10.0)
        self.assertEqual(es.cv_isi(regular_st), 0.0)
