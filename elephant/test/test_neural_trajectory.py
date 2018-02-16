# -*- coding: utf-8 -*-
"""
Unit tests for the neural_trajectory analysis.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
import quantities as pq
import neo

from numpy.testing import assert_array_equal

try:
    import sklearn
except ImportError:
    HAVE_SKLEARN = False
else:
    import elephant.neural_trajectory as nt
    HAVE_SKLEARN = True


@unittest.skipUnless(HAVE_SKLEARN, 'requires sklearn')
class NeuralTrajectoryTestCase(unittest.TestCase):
    def setUp(self):
        def gamma_train(k, tetha, t_max):
            x = []

            for i in range(int(t_max * (k * tetha) ** (-1) * 3)):
                x.append(np.random.gamma(k, tetha))

            s = np.cumsum(x)
            idx = np.where(s < t_max)
            s = s[idx]  # Poisson process

            return s

        def h_alt(rate1, rate2, rate3, rate4, c1, c2, c3, T, k1=1, k2=1, k3=1,
                  k4=1):
            teta1 = rate1 ** -1
            teta2 = rate2 ** -1
            teta3 = rate3 ** -1
            teta4 = rate4 ** -1
            s1 = gamma_train(k1, teta1, c1)
            s2 = gamma_train(k2, teta2, c2) + c1
            s3 = gamma_train(k3, teta3, c3) + c1 + c2
            s4 = gamma_train(k4, teta4, T) + c1 + c2 + c3

            return np.concatenate((s1, s2, s3, s4))

        trials = []
        dat_temp = []
        for tr in range(1):
            np.random.seed(tr)
            n1 = neo.SpikeTrain(h_alt(2, 10, 2, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1 * pq.s)
            n2 = neo.SpikeTrain(h_alt(2, 10, 2, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1 * pq.s)
            n3 = neo.SpikeTrain(h_alt(2, 2, 10, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1 * pq.s)
            n4 = neo.SpikeTrain(h_alt(2, 2, 10, 2, 2.5, 2.5, 2.5, 2.5),
                                t_start=0, t_stop=10, units=1 * pq.s)
            dat_temp.append((tr, np.array([n1, n2, n3, n4])))
            trials.append((tr, [n1, n2, n3, n4]))

        self.input = np.array(dat_temp,
                              dtype=[('trialId', 'O'), ('spikes', 'O')])

        self.method = 'gpfa'
        self.x_dim = 4

    def test_input(self):
        dat = [(0, [0, 1, 2])]
        self.assertRaises(ValueError, nt.neural_trajectory, dat)
        result = nt.neural_trajectory(self.input,
                                      x_dim=self.x_dim)
        self.assertEqual(result['cvf'], 0)
        self.assertEqual(result['bin_size'], 20 * pq.ms)
        assert_array_equal(result['hasSpikesBool'], np.array([True] * 4))
        self.assertAlmostEqual(result['log_likelihood'], -26.504094758661424)
        self.assertEqual(result['min_var_frac'], 0.01)


def suite():
    suite = unittest.makeSuite(NeuralTrajectoryTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
