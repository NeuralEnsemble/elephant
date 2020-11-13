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


if __name__ == '__main__':
    unittest.main()
