import unittest
import elephant.spike_contrast as spc
import elephant.spike_train_generation as stgen
import numpy as np
from quantities import Hz, ms


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_spike_contrast(self):
        np.random.seed(24)  # to make the results reproducible
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_4 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_5 = stgen.homogeneous_poisson_process(rate=1*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_6 = stgen.homogeneous_poisson_process(rate=1*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_trains = np.array(
                             [spike_train_1, spike_train_2, spike_train_3, spike_train_4, spike_train_5, spike_train_6])
        self.assertEqual((spc.spike_contrast(spike_trains, 0, 10000)), 0.2098687702924583)

    def test_spike_contrast_1(self):
        spike_train = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_trains = np.array([spike_train, spike_train])
        self.assertEqual((spc.spike_contrast(spike_trains, 0, 10000)), 1.0)

    def test_spike_contrast_2(self):
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms, as_array=True)

        spike_trains = np.array([spike_train_1, spike_train_2, spike_train_3])
        self.assertEqual((spc.spike_contrast(spike_trains, 0, 20000)), 0.5)

    def test_get_theta_and_n_per_bin(self):
        spike_trains = np.array([[1, 1, 1],
                                 [2, 2, 2],
                                 [3, 3, 2.5],
                                 [9, 9, 0]])
        spike_trains = np.where(spike_trains == 0, np.nan, spike_trains)
        bin_size = 5
        t_start = 0
        t_stop = 10
        theta, n = spc.get_theta_and_n_per_bin(spike_trains, t_start, t_stop, bin_size)
        expected_theta = np.array([9, 3, 2])
        expected_n = np.array([3, 3, 2])
        self.assertTrue((theta == expected_theta).all())
        self.assertTrue((n == expected_n).all())

    def test_binning_half_overlap(self):
        spike_train_i = np.array([1, 2, 3, 9])
        t_start = 0
        t_stop = 10
        bin_size = 5
        self.assertTrue(([3, 1, 1] == spc.binning_half_overlap(spike_train_i, t_start, t_stop, bin_size)).all())


if __name__ == '__main__':
    unittest.main()
