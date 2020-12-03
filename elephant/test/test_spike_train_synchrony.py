from __future__ import division

import json
import unittest

import neo
import numpy as np
from numpy.testing import assert_array_equal
from quantities import Hz, ms, second

import elephant.spike_train_synchrony as spc
import elephant.spike_train_generation as stgen
from elephant.test.download import download, unzip


class TestUM(unittest.TestCase):

    def test_spike_contrast_random(self):
        # randomly generated spiketrains that share the same t_start and
        # t_stop
        np.random.seed(24)  # to make the results reproducible
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_4 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_5 = stgen.homogeneous_poisson_process(rate=1 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_6 = stgen.homogeneous_poisson_process(rate=1 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_trains = [spike_train_1, spike_train_2, spike_train_3,
                        spike_train_4, spike_train_5, spike_train_6]
        synchrony = spc.spike_contrast(spike_trains)
        self.assertAlmostEqual(synchrony, 0.2098687, places=6)

    def test_spike_contrast_same_signal(self):
        np.random.seed(21)
        spike_train = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                        t_start=0. * ms,
                                                        t_stop=10000. * ms)
        spike_trains = [spike_train, spike_train]
        synchrony = spc.spike_contrast(spike_trains, min_bin=1 * ms)
        self.assertEqual(synchrony, 1.0)

    def test_spike_contrast_double_duration(self):
        np.random.seed(19)
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_3 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)

        spike_trains = [spike_train_1, spike_train_2, spike_train_3]
        synchrony = spc.spike_contrast(spike_trains, t_stop=20000 * ms)
        self.assertEqual(synchrony, 0.5)

    def test_spike_contrast_non_overlapping_spiketrains(self):
        np.random.seed(15)
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=0. * ms,
                                                          t_stop=10000. * ms)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_start=5000. * ms,
                                                          t_stop=10000. * ms)
        spiketrains = [spike_train_1, spike_train_2]
        synchrony = spc.spike_contrast(spiketrains, t_stop=5000 * ms)
        # the synchrony of non-overlapping spiketrains must be zero
        self.assertEqual(synchrony, 0.)

    def test_spike_contrast_trace(self):
        np.random.seed(15)
        spike_train_1 = stgen.homogeneous_poisson_process(rate=20 * Hz,
                                                          t_stop=1000. * ms)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=20 * Hz,
                                                          t_stop=1000. * ms)
        synchrony, trace = spc.spike_contrast([spike_train_1, spike_train_2],
                                              return_trace=True)
        self.assertEqual(synchrony, max(trace.synchrony))
        self.assertEqual(len(trace.contrast), len(trace.active_spiketrains))
        self.assertEqual(len(trace.active_spiketrains), len(trace.synchrony))

    def test_invalid_data(self):
        # invalid spiketrains
        self.assertRaises(TypeError, spc.spike_contrast, [[0, 1], [1.5, 2.3]])
        self.assertRaises(ValueError, spc.spike_contrast,
                          [neo.SpikeTrain([10] * ms, t_stop=1000 * ms),
                           neo.SpikeTrain([20] * ms, t_stop=1000 * ms)])

        # a single spiketrain
        spiketrain_valid = neo.SpikeTrain([0, 1000] * ms, t_stop=1000 * ms)
        self.assertRaises(ValueError, spc.spike_contrast, [spiketrain_valid])

        spiketrain_valid2 = neo.SpikeTrain([500, 800] * ms, t_stop=1000 * ms)
        spiketrains = [spiketrain_valid, spiketrain_valid2]

        # invalid shrink factor
        self.assertRaises(ValueError, spc.spike_contrast, spiketrains,
                          bin_shrink_factor=0.)

        # invalid t_start, t_stop, and min_bin
        self.assertRaises(TypeError, spc.spike_contrast, spiketrains,
                          t_start=0)
        self.assertRaises(TypeError, spc.spike_contrast, spiketrains,
                          t_stop=1000)
        self.assertRaises(TypeError, spc.spike_contrast, spiketrains,
                          min_bin=0.01)

    def test_t_start_agnostic(self):
        np.random.seed(15)
        t_stop = 10 * second
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_stop=t_stop)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_stop=t_stop)
        spiketrains = [spike_train_1, spike_train_2]
        synchrony_target = spc.spike_contrast(spiketrains)
        # a check for developer: test meaningful result
        assert synchrony_target > 0
        t_shift = 20 * second
        spiketrains_shifted = [
            neo.SpikeTrain(st.times + t_shift,
                           t_start=t_shift,
                           t_stop=t_stop + t_shift)
            for st in spiketrains
        ]
        synchrony = spc.spike_contrast(spiketrains_shifted)
        self.assertAlmostEqual(synchrony, synchrony_target)

    def test_get_theta_and_n_per_bin(self):
        spike_trains = [
            [1, 2, 3, 9],
            [1, 2, 3, 9],
            [1, 2, 2.5]
        ]
        theta, n = spc._get_theta_and_n_per_bin(spike_trains,
                                                t_start=0,
                                                t_stop=10,
                                                bin_size=5)
        assert_array_equal(theta, [9, 3, 2])
        assert_array_equal(n, [3, 3, 2])

    def test_binning_half_overlap(self):
        spiketrain = np.array([1, 2, 3, 9])
        bin_step = 5 / 2
        t_start = 0
        t_stop = 10
        edges = np.arange(t_start, t_stop + bin_step, bin_step)
        histogram = spc._binning_half_overlap(spiketrain, edges=edges)
        assert_array_equal(histogram, [3, 1, 1])

    def test_spike_contrast_with_Izhikevich_network_auto(self):
        # This test reproduces the Test data 3 (Izhikevich network), fig. 3,
        # Manuel Ciba et. al, 2018.
        # The data is a dictionary of simulations of different networks.
        # Each simulation of a network is a dictionary with two keys:
        # 'spiketrains' and the ground truth 'synchrony'.
        # The default unit time is seconds. Each simulation lasted 2 seconds,
        # starting from 0.

        izhikevich_url = r"https://web.gin.g-node.org/INM-6/" \
                         r"elephant-data/raw/master/" \
                         r"dataset-3/Data_Izhikevich_network.zip"
        filepath_zip = download(url=izhikevich_url,
                                checksum="70e848500c1d9c6403b66de8c741d849")
        unzip(filepath_zip)
        filepath_json = filepath_zip.with_suffix(".json")
        with open(filepath_json) as read_file:
            data = json.load(read_file)

        # for the sake of compute time, take the first 5 networks
        networks_subset = tuple(data.values())[:5]

        for network_simulations in networks_subset:
            for simulation in network_simulations.values():
                synchrony_true = simulation['synchrony']
                spiketrains = [
                    neo.SpikeTrain(st, t_start=0 * second, t_stop=2 * second,
                                   units=second)
                    for st in simulation['spiketrains']]
                synchrony = spc.spike_contrast(spiketrains)
                self.assertAlmostEqual(synchrony, synchrony_true, places=2)


if __name__ == '__main__':
    unittest.main()
