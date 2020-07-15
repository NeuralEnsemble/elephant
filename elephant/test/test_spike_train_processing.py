# -*- coding: utf-8 -*-
"""
Unit tests for the synchrofact detection app
"""

import unittest

import neo
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import quantities as pq

from elephant import spike_train_processing


class SynchrofactDetectionTestCase(unittest.TestCase):

    def _test_template(self, spiketrains, correct_complexities, sampling_rate,
                       spread, deletion_threshold=2, invert_delete=False,
                       in_place=False, binary=True):

        synchrofact_obj = spike_train_processing.synchrotool(
            spiketrains,
            sampling_rate=sampling_rate,
            binary=binary,
            spread=spread)

        # test annotation
        synchrofact_obj.annotate_synchrofacts()

        annotations = [st.array_annotations['complexity']
                       for st in spiketrains]

        assert_array_equal(annotations, correct_complexities)

        if invert_delete:
            correct_spike_times = np.array(
                [spikes[mask] for spikes, mask
                 in zip(spiketrains,
                        correct_complexities >= deletion_threshold)
                 ])
        else:
            correct_spike_times = np.array(
                [spikes[mask] for spikes, mask
                 in zip(spiketrains, correct_complexities < deletion_threshold)
                 ])

        # test deletion
        synchrofact_obj.delete_synchrofacts(threshold=deletion_threshold,
                                            in_place=in_place,
                                            invert=invert_delete)

        cleaned_spike_times = np.array(
            [st.times for st in spiketrains])

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
                            spread=1, invert_delete=False,
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
                            spread=0, invert_delete=False, in_place=True,
                            deletion_threshold=2)

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
                            spread=1, invert_delete=False, in_place=True,
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
                            spread=1, invert_delete=False, binary=False,
                            in_place=True, deletion_threshold=3)

    def test_invert_delete(self):

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
                            spread=1, invert_delete=True, in_place=True,
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
                            spread=0, invert_delete=False, in_place=True,
                            deletion_threshold=2)

    def test_correct_transfer_of_spiketrain_attributes(self):

        # for delete=True the spiketrains in the block are changed,
        # test if their attributes remain correct

        sampling_rate = 1 / pq.s

        spiketrain = neo.SpikeTrain([1, 1, 5, 0] * pq.s,
                                    t_stop=10 * pq.s)

        block = neo.Block()

        channel_index = neo.ChannelIndex(name='Channel 1', index=1)
        block.channel_indexes.append(channel_index)

        unit = neo.Unit('Unit 1')
        channel_index.units.append(unit)
        unit.spiketrains.append(spiketrain)
        spiketrain.unit = unit

        segment = neo.Segment()
        block.segments.append(segment)
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
        synchrofact_obj = spike_train_processing.synchrotool(
            [spiketrain],
            spread=0,
            sampling_rate=sampling_rate,
            binary=False)
        synchrofact_obj.delete_synchrofacts(
            invert=False,
            in_place=True,
            threshold=2)

        # Ensure that the spiketrain was not duplicated
        self.assertEqual(len(block.filter(objects=neo.SpikeTrain)), 1)

        cleaned_spiketrain = segment.spiketrains[0]

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
        synchrofact_obj = spike_train_processing.synchrotool(
            [neo.SpikeTrain([1]*pq.s, t_stop=2*pq.s)],
            sampling_rate=1/pq.s,
            binary=True,
            spread=1)
        self.assertRaises(ValueError,
                          synchrofact_obj.delete_synchrofacts,
                          -1)


if __name__ == '__main__':
    unittest.main()
