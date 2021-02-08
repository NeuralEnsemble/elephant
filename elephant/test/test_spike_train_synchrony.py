from __future__ import division

import json
import unittest

import neo
import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal, assert_array_equal
from quantities import Hz, ms, second

import elephant.spike_train_generation as stgen
from elephant.spike_train_synchrony import Synchrotool, spike_contrast, \
    _get_theta_and_n_per_bin, _binning_half_overlap
from elephant.test.download import download, unzip


class TestSpikeContrast(unittest.TestCase):

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
        synchrony = spike_contrast(spike_trains)
        self.assertAlmostEqual(synchrony, 0.2098687, places=6)

    def test_spike_contrast_same_signal(self):
        np.random.seed(21)
        spike_train = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                        t_start=0. * ms,
                                                        t_stop=10000. * ms)
        spike_trains = [spike_train, spike_train]
        synchrony = spike_contrast(spike_trains, min_bin=1 * ms)
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
        synchrony = spike_contrast(spike_trains, t_stop=20000 * ms)
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
        synchrony = spike_contrast(spiketrains, t_stop=5000 * ms)
        # the synchrony of non-overlapping spiketrains must be zero
        self.assertEqual(synchrony, 0.)

    def test_spike_contrast_trace(self):
        np.random.seed(15)
        spike_train_1 = stgen.homogeneous_poisson_process(rate=20 * Hz,
                                                          t_stop=1000. * ms)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=20 * Hz,
                                                          t_stop=1000. * ms)
        synchrony, trace = spike_contrast([spike_train_1, spike_train_2],
                                          return_trace=True)
        self.assertEqual(synchrony, max(trace.synchrony))
        self.assertEqual(len(trace.contrast), len(trace.active_spiketrains))
        self.assertEqual(len(trace.active_spiketrains), len(trace.synchrony))
        self.assertEqual(len(trace.bin_size), len(trace.synchrony))
        self.assertIsInstance(trace.bin_size, pq.Quantity)
        self.assertEqual(trace.bin_size[0], 500 * pq.ms)
        self.assertAlmostEqual(trace.bin_size[-1], 10.1377798 * pq.ms)

    def test_invalid_data(self):
        # invalid spiketrains
        self.assertRaises(TypeError, spike_contrast, [[0, 1], [1.5, 2.3]])
        self.assertRaises(ValueError, spike_contrast,
                          [neo.SpikeTrain([10] * ms, t_stop=1000 * ms),
                           neo.SpikeTrain([20] * ms, t_stop=1000 * ms)])

        # a single spiketrain
        spiketrain_valid = neo.SpikeTrain([0, 1000] * ms, t_stop=1000 * ms)
        self.assertRaises(ValueError, spike_contrast, [spiketrain_valid])

        spiketrain_valid2 = neo.SpikeTrain([500, 800] * ms, t_stop=1000 * ms)
        spiketrains = [spiketrain_valid, spiketrain_valid2]

        # invalid shrink factor
        self.assertRaises(ValueError, spike_contrast, spiketrains,
                          bin_shrink_factor=0.)

        # invalid t_start, t_stop, and min_bin
        self.assertRaises(TypeError, spike_contrast, spiketrains,
                          t_start=0)
        self.assertRaises(TypeError, spike_contrast, spiketrains,
                          t_stop=1000)
        self.assertRaises(TypeError, spike_contrast, spiketrains,
                          min_bin=0.01)

    def test_t_start_agnostic(self):
        np.random.seed(15)
        t_stop = 10 * second
        spike_train_1 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_stop=t_stop)
        spike_train_2 = stgen.homogeneous_poisson_process(rate=10 * Hz,
                                                          t_stop=t_stop)
        spiketrains = [spike_train_1, spike_train_2]
        synchrony_target = spike_contrast(spiketrains)
        # a check for developer: test meaningful result
        assert synchrony_target > 0
        t_shift = 20 * second
        spiketrains_shifted = [
            neo.SpikeTrain(st.times + t_shift,
                           t_start=t_shift,
                           t_stop=t_stop + t_shift)
            for st in spiketrains
        ]
        synchrony = spike_contrast(spiketrains_shifted)
        self.assertAlmostEqual(synchrony, synchrony_target)

    def test_get_theta_and_n_per_bin(self):
        spike_trains = [
            [1, 2, 3, 9],
            [1, 2, 3, 9],
            [1, 2, 2.5]
        ]
        theta, n = _get_theta_and_n_per_bin(spike_trains,
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
        histogram = _binning_half_overlap(spiketrain, edges=edges)
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
                synchrony = spike_contrast(spiketrains)
                self.assertAlmostEqual(synchrony, synchrony_true, places=2)


class SynchrofactDetectionTestCase(unittest.TestCase):

    def _test_template(self, spiketrains, correct_complexities, sampling_rate,
                       spread, deletion_threshold=2, mode='delete',
                       in_place=False, binary=True):

        synchrofact_obj = Synchrotool(
            spiketrains,
            sampling_rate=sampling_rate,
            binary=binary,
            spread=spread)

        # test annotation
        synchrofact_obj.annotate_synchrofacts()

        annotations = [st.array_annotations['complexity']
                       for st in spiketrains]

        assert_array_equal(annotations, correct_complexities)

        if mode == 'extract':
            correct_spike_times = [
                spikes[mask] for spikes, mask
                in zip(spiketrains,
                       correct_complexities >= deletion_threshold)
            ]
        else:
            correct_spike_times = [
                spikes[mask] for spikes, mask
                in zip(spiketrains,
                       correct_complexities < deletion_threshold)
            ]

        # test deletion
        synchrofact_obj.delete_synchrofacts(threshold=deletion_threshold,
                                            in_place=in_place,
                                            mode=mode)

        cleaned_spike_times = [st.times for st in spiketrains]

        for correct_st, cleaned_st in zip(correct_spike_times,
                                          cleaned_spike_times):
            assert_array_almost_equal(cleaned_st, correct_st)

    def test_no_synchrofacts(self):

        # nothing to find here
        # there used to be an error for spread > 0 when nothing was found

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 9, 12, 19] * pq.s,
                                      t_stop=20*pq.s),
                       neo.SpikeTrain([3, 7, 15, 17] * pq.s,
                                      t_stop=20*pq.s)]

        correct_annotations = np.array([[1, 1, 1, 1],
                                        [1, 1, 1, 1]])

        self._test_template(spiketrains, correct_annotations, sampling_rate,
                            spread=1, mode='delete',
                            deletion_threshold=2)

    def test_spread_0(self):

        # basic test with a minimum number of two spikes per synchrofact
        # only taking into account multiple spikes
        # within one bin of size 1 / sampling_rate

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s,
                                      t_stop=20*pq.s),
                       neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s,
                                      t_stop=20*pq.s)]

        correct_annotations = np.array([[2, 1, 1, 1, 2, 1],
                                        [2, 1, 1, 1, 2, 1]])

        self._test_template(spiketrains, correct_annotations, sampling_rate,
                            spread=0, mode='delete', in_place=True,
                            deletion_threshold=2)

    def test_spiketrains_findable(self):

        # same test as `test_spread_0` with the addition of
        # a neo structure: we must not overwrite the spiketrain
        # list of the segment before determining the index

        sampling_rate = 1 / pq.s

        segment = neo.Segment()

        segment.spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s,
                                              t_stop=20*pq.s),
                               neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s,
                                              t_stop=20*pq.s)]

        segment.create_relationship()

        correct_annotations = np.array([[2, 1, 1, 1, 2, 1],
                                        [2, 1, 1, 1, 2, 1]])

        self._test_template(segment.spiketrains, correct_annotations,
                            sampling_rate, spread=0, mode='delete',
                            in_place=True, deletion_threshold=2)

    def test_unidirectional_uplinks(self):

        # same test as `test_spiketrains_findable` but the spiketrains
        # are rescaled first
        # the rescaled spiketrains have a unidirectional uplink to segment
        # check that this does not cause an error
        # check that a UserWarning is issued in this case

        sampling_rate = 1 / pq.s

        segment = neo.Segment()

        segment.spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 16, 19] * pq.s,
                                              t_stop=20*pq.s),
                               neo.SpikeTrain([1, 4, 8, 12, 16, 18] * pq.s,
                                              t_stop=20*pq.s)]

        segment.create_relationship()

        spiketrains = [st.rescale(pq.s) for st in segment.spiketrains]

        correct_annotations = np.array([[2, 1, 1, 1, 2, 1],
                                        [2, 1, 1, 1, 2, 1]])

        with self.assertWarns(UserWarning):
            self._test_template(spiketrains, correct_annotations,
                                sampling_rate, spread=0, mode='delete',
                                in_place=True, deletion_threshold=2)

    def test_spread_1(self):

        # test synchrofact search taking into account adjacent bins
        # this requires an additional loop with shifted binning

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s,
                                      t_stop=21*pq.s)]

        correct_annotations = np.array([[2, 2, 1, 3, 3, 1],
                                        [2, 2, 1, 3, 1, 1]])

        self._test_template(spiketrains, correct_annotations, sampling_rate,
                            spread=1, mode='delete', in_place=True,
                            deletion_threshold=2)

    def test_n_equals_3(self):

        # test synchrofact detection with a minimum number of
        # three spikes per synchrofact

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 1, 5, 10, 13, 16, 17, 19] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 9, 12, 14, 16, 20] * pq.s,
                                      t_stop=21*pq.s)]

        correct_annotations = np.array([[3, 3, 2, 2, 3, 3, 3, 2],
                                        [3, 2, 1, 2, 3, 3, 3, 2]])

        self._test_template(spiketrains, correct_annotations, sampling_rate,
                            spread=1, mode='delete', binary=False,
                            in_place=True, deletion_threshold=3)

    def test_extract(self):

        # test synchrofact search taking into account adjacent bins
        # this requires an additional loop with shifted binning

        sampling_rate = 1 / pq.s

        spiketrains = [neo.SpikeTrain([1, 5, 9, 11, 13, 20] * pq.s,
                                      t_stop=21*pq.s),
                       neo.SpikeTrain([1, 4, 7, 12, 16, 18] * pq.s,
                                      t_stop=21*pq.s)]

        correct_annotations = np.array([[2, 2, 1, 3, 3, 1],
                                        [2, 2, 1, 3, 1, 1]])

        self._test_template(spiketrains, correct_annotations, sampling_rate,
                            spread=1, mode='extract', in_place=True,
                            deletion_threshold=2)

    def test_binning_for_input_with_rounding_errors(self):

        # a test with inputs divided by 30000 which leads to rounding errors
        # these errors have to be accounted for by proper binning;
        # check if we still get the correct result

        sampling_rate = 30000 / pq.s

        spiketrains = [neo.SpikeTrain(np.arange(1000) * pq.s / 30000,
                                      t_stop=.1 * pq.s),
                       neo.SpikeTrain(np.arange(2000, step=2) * pq.s / 30000,
                                      t_stop=.1 * pq.s)]

        first_annotations = np.ones(1000)
        first_annotations[::2] = 2

        second_annotations = np.ones(1000)
        second_annotations[:500] = 2

        correct_annotations = np.array([first_annotations,
                                        second_annotations])

        self._test_template(spiketrains, correct_annotations, sampling_rate,
                            spread=0, mode='delete', in_place=True,
                            deletion_threshold=2)

    def test_correct_transfer_of_spiketrain_attributes(self):

        # for delete=True the spiketrains in the block are changed,
        # test if their attributes remain correct

        sampling_rate = 1 / pq.s

        spiketrain = neo.SpikeTrain([1, 1, 5, 0] * pq.s,
                                    t_stop=10 * pq.s)

        block = neo.Block()

        group = neo.Group(name='Test Group')
        block.groups.append(group)
        group.spiketrains.append(spiketrain)

        segment = neo.Segment()
        block.segments.append(segment)
        segment.block = block
        segment.spiketrains.append(spiketrain)
        spiketrain.segment = segment

        spiketrain.annotate(cool_spike_train=True)
        spiketrain.array_annotate(
            spike_number=np.arange(len(spiketrain.times.magnitude)))
        spiketrain.waveforms = np.sin(
            np.arange(len(spiketrain.times.magnitude))[:, np.newaxis]
            + np.arange(len(spiketrain.times.magnitude))[np.newaxis, :])

        correct_mask = np.array([False, False, True, True])

        # store the correct attributes
        correct_annotations = spiketrain.annotations.copy()
        correct_waveforms = spiketrain.waveforms[correct_mask].copy()
        correct_array_annotations = {key: value[correct_mask] for key, value in
                                     spiketrain.array_annotations.items()}

        # perform a synchrofact search with delete=True
        synchrofact_obj = Synchrotool(
            [spiketrain],
            spread=0,
            sampling_rate=sampling_rate,
            binary=False)
        synchrofact_obj.delete_synchrofacts(
            mode='delete',
            in_place=True,
            threshold=2)

        # Ensure that the spiketrain was not duplicated
        self.assertEqual(len(block.filter(objects=neo.SpikeTrain)), 1)

        cleaned_spiketrain = segment.spiketrains[0]

        # Ensure that the spiketrain is also in the group
        self.assertEqual(len(block.groups[0].spiketrains), 1)
        self.assertIs(block.groups[0].spiketrains[0], cleaned_spiketrain)

        cleaned_annotations = cleaned_spiketrain.annotations
        cleaned_waveforms = cleaned_spiketrain.waveforms
        cleaned_array_annotations = cleaned_spiketrain.array_annotations
        cleaned_array_annotations.pop('complexity')

        self.assertDictEqual(correct_annotations, cleaned_annotations)
        assert_array_almost_equal(cleaned_waveforms, correct_waveforms)
        self.assertTrue(len(cleaned_array_annotations)
                        == len(correct_array_annotations))
        for key, value in correct_array_annotations.items():
            self.assertTrue(key in cleaned_array_annotations.keys())
            assert_array_almost_equal(value, cleaned_array_annotations[key])

    def test_wrong_input_errors(self):
        synchrofact_obj = Synchrotool(
            [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s)],
            sampling_rate=1/pq.s,
            binary=True,
            spread=1)
        self.assertRaises(ValueError,
                          synchrofact_obj.delete_synchrofacts,
                          -1)


if __name__ == '__main__':
    unittest.main()
