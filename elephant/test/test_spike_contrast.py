import unittest

import numpy as np
from numpy.testing import assert_array_equal
from quantities import Hz, ms

import elephant.spike_contrast as spc
import elephant.spike_train_generation as stgen


class TestUM(unittest.TestCase):

    def test_spike_contrast(self):
        np.random.seed(24)  # to make the results reproducible
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_4 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_5 = stgen.homogeneous_poisson_process(rate=1 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_6 = stgen.homogeneous_poisson_process(rate=1 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_trains = [spike_train_1, spike_train_2, spike_train_3,
                        spike_train_4, spike_train_5, spike_train_6]
        synchrony = spc.spike_contrast(spike_trains, t_start=0, t_stop=10000)
        self.assertEqual(synchrony, 0.2098687702924583)

    def test_spike_contrast_1(self):
        np.random.seed(21)
        spike_train = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                        t_start=0. * ms,
                                                        t_stop=10000. * ms,
                                                        as_array=True)
        spike_trains = np.array([spike_train, spike_train])
        synchrony = spc.spike_contrast(spike_trains, t_start=0, t_stop=10000)
        self.assertEqual(synchrony, 1.0)

    def test_spike_contrast_2(self):
        np.random.seed(19)
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms,
                                                          as_array=True)

        spike_trains = np.array([spike_train_1, spike_train_2, spike_train_3])
        synchrony = spc.spike_contrast(spike_trains, t_start=0, t_stop=20000)
        self.assertEqual(synchrony, 0.5)

    def test_get_theta_and_n_per_bin(self):
        spike_trains = np.array([[1, 1, 1],
                                 [2, 2, 2],
                                 [3, 3, 2.5],
                                 [9, 9, 0]]).T
        theta, n = spc._get_theta_and_n_per_bin(spike_trains,
                                                t_start=0,
                                                t_stop=10,
                                                bin_size=5)
        assert_array_equal(theta, [9, 3, 2])
        assert_array_equal(n, [3, 3, 2])

    def test_binning_half_overlap(self):
        spiketrain = np.array([1, 2, 3, 9])
        histogram = spc._binning_half_overlap(spiketrain,
                                              t_start=0, t_stop=10, bin_size=5)
        assert_array_equal(histogram, [3, 1, 1])


if __name__ == '__main__':
    unittest.main()
