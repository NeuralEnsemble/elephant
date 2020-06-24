# -*- coding: utf-8 -*-
"""
Unit tests for the synchrofact detection app
"""

import unittest

import neo
import numpy as np
import quantities as pq

from elephant import utils


class checkSpiketrainTestCase(unittest.TestCase):

    def test_wrong_input_errors(self):
        self.assertRaises(ValueError,
                          utils._check_consistency_of_spiketrainlist,
                          [], 1 / pq.s)
        self.assertRaises(TypeError,
                          utils._check_consistency_of_spiketrainlist,
                          neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s))
        self.assertRaises(TypeError,
                          utils._check_consistency_of_spiketrainlist,
                          [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s),
                           np.arange(2)],
                          1 / pq.s)
        self.assertRaises(ValueError,
                          utils._check_consistency_of_spiketrainlist,
                          [neo.SpikeTrain([1]*pq.s,
                                          t_start=1*pq.s,
                                          t_stop=2*pq.s),
                           neo.SpikeTrain([1]*pq.s,
                                          t_start=0*pq.s,
                                          t_stop=2*pq.s)],
                          same_t_start=True)
        self.assertRaises(ValueError,
                          utils._check_consistency_of_spiketrainlist,
                          [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s),
                           neo.SpikeTrain([1]*pq.s, t_stop=3*pq.s)],
                          same_t_stop=True)
        self.assertRaises(ValueError,
                          utils._check_consistency_of_spiketrainlist,
                          [neo.SpikeTrain([1]*pq.ms, t_stop=2000*pq.ms),
                           neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s)],
                          same_units=True)


if __name__ == '__main__':
    unittest.main()
