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


def generate_block(spike_times, segment_edges=[0, 10, 20]*pq.s):
    """
    Generate a block with segments with start and end times given by segment_edges
    and with spike trains given by spike_times.
    """
    n_segments = len(segment_edges) - 1

    # Create Block to contain all generated data
    block = neo.Block()

    # Create multiple Segments
    block.segments = [neo.Segment(index=i,
                                  t_start=segment_edges[i],
                                  t_stop=segment_edges[i+1])
                      for i in range(n_segments)]

    # Create multiple ChannelIndexes
    block.channel_indexes = [neo.ChannelIndex(name='C%d' % i, index=i)
                             for i in range(len(spike_times[0]))]

    # Attach multiple Units to each ChannelIndex
    for i, channel_idx in enumerate(block.channel_indexes):
        channel_idx.units = [neo.Unit('U1')]
        for seg_idx, seg in enumerate(block.segments):
            train = neo.SpikeTrain(spike_times[seg_idx][i],
                                   t_start=segment_edges[seg_idx],
                                   t_stop=segment_edges[seg_idx+1])
            seg.spiketrains.append(train)
            channel_idx.units[0].spiketrains.append(train)

    block.create_many_to_one_relationship()
    return block


class SynchrofactDetectionTestCase(unittest.TestCase):

    def test_no_synchrofacts(self):

        # nothing to find here
        # there was an error for spread > 1 when nothing was found
        # since boundaries is then set to [] and we later check boundaries.shape
        # fixed by skipping the interval merge step when there are no intervals

        sampling_rate = 1 / pq.s

        spike_times = np.array([[[1, 9], [3, 7]], [[12, 19], [15, 17]]]) * pq.s

        block = generate_block(spike_times)

        # test annotation
        spike_train_processing.detect_synchrofacts(block, segment='all', n=2, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=False,
                                                  unit_type='all')

        correct_annotations = [[np.array([False, False]), np.array([False, False])],
                               [np.array([False, False]), np.array([False, False])]]

        annotations = [[st.array_annotations['synchrofacts'] for st in seg.spiketrains]
                       for seg in block.segments]

        assert_array_equal(annotations, correct_annotations)

        # test deletion
        spike_train_processing.detect_synchrofacts(block, segment='all', n=2, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=True,
                                                  unit_type='all')

        correct_spike_times = np.array(
            [[spikes[mask] for spikes, mask in zip(seg_spike_times, seg_mask)]
             for seg_spike_times, seg_mask in zip(spike_times,
                                                  np.logical_not(correct_annotations)
                                                  )
             ])

        cleaned_spike_times = np.array(
            [[st.times for st in seg.spiketrains] for seg in block.segments])

        for correct_seg, cleaned_seg in zip(correct_spike_times, cleaned_spike_times):
            for correct_st, cleaned_st in zip(correct_seg, cleaned_seg):
                assert_array_almost_equal(cleaned_st, correct_st)

    def test_spread_1(self):

        # basic test with a minimum number of two spikes per synchrofact
        # only taking into account multiple spikes
        # within one bin of size 1 / sampling_rate

        sampling_rate = 1 / pq.s

        spike_times = np.array([[[1, 5, 9], [1, 4, 8]],
                                [[11, 16, 19], [12, 16, 18]]]) * pq.s

        block = generate_block(spike_times)

        # test annotation
        spike_train_processing.detect_synchrofacts(block, segment='all', n=2,
                                                  spread=1,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=False,
                                                  unit_type='all')

        correct_annotations = np.array([[[True, False, False], [True, False, False]],
                                       [[False, True, False], [False, True, False]]])

        annotations = [[st.array_annotations['synchrofacts'] for st in seg.spiketrains]
                       for seg in block.segments]

        assert_array_equal(annotations, correct_annotations)

        # test deletion
        spike_train_processing.detect_synchrofacts(block, segment='all', n=2, spread=1,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=True,
                                                  unit_type='all')

        correct_spike_times = np.array([[spikes[mask]
                                         for spikes, mask in zip(seg_spike_times,
                                                                 seg_mask)]
                                        for seg_spike_times, seg_mask
                                        in zip(spike_times,
                                               np.logical_not(correct_annotations))])

        cleaned_spike_times = np.array([[st.times for st in seg.spiketrains]
                                        for seg in block.segments])

        assert_array_almost_equal(cleaned_spike_times, correct_spike_times)

    def test_spread_2(self):

        # test synchrofact search taking into account adjacent bins
        # this requires an additional loop with shifted binning

        sampling_rate = 1 / pq.s

        spike_times = np.array([[[1, 5, 9], [1, 4, 7]],
                                [[10, 12, 19], [11, 15, 17]]]) * pq.s

        block = generate_block(spike_times)

        # test annotation
        spike_train_processing.detect_synchrofacts(block, segment='all',
                                                  n=2, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=False,
                                                  unit_type='all')

        correct_annotations = [[np.array([True, True, False]),
                                np.array([True, True, False])],
                               [np.array([True, True, False]),
                                np.array([True, False, False])]]

        annotations = [[st.array_annotations['synchrofacts'] for st in seg.spiketrains]
                       for seg in block.segments]

        assert_array_equal(annotations, correct_annotations)

        # test deletion
        spike_train_processing.detect_synchrofacts(block, segment='all', n=2, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=True,
                                                  unit_type='all')

        correct_spike_times = np.array([[spikes[mask] for spikes, mask in
                                         zip(seg_spike_times, seg_mask)]
                                        for seg_spike_times, seg_mask in
                                        zip(spike_times,
                                            np.logical_not(correct_annotations))])

        cleaned_spike_times = np.array([[st.times for st in seg.spiketrains]
                                        for seg in block.segments])

        for correct_seg, cleaned_seg in zip(correct_spike_times, cleaned_spike_times):
            for correct_st, cleaned_st in zip(correct_seg, cleaned_seg):
                assert_array_almost_equal(cleaned_st, correct_st)

    def test_n_equals_3(self):

        # test synchrofact detection with a minimum number of
        # three spikes per synchrofact

        sampling_rate = 1 / pq.s

        spike_times = np.array([[[1, 1, 5, 10], [1, 4, 7, 9]],
                                [[12, 15, 16, 18], [11, 13, 15, 19]]]) * pq.s

        block = generate_block(spike_times)

        # test annotation
        spike_train_processing.detect_synchrofacts(block, segment='all', n=3, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=False,
                                                  unit_type='all')

        correct_annotations = [[np.array([True, True, False, False]),
                                np.array([True, False, False, False])],
                               [np.array([True, True, True, False]),
                                np.array([True, True, True, False])]]

        annotations = [[st.array_annotations['synchrofacts'] for st in seg.spiketrains]
                       for seg in block.segments]

        assert_array_equal(annotations, correct_annotations)

        # test deletion
        spike_train_processing.detect_synchrofacts(block, segment='all', n=3, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=True,
                                                  unit_type='all')

        correct_spike_times = np.array([[spikes[mask] for spikes, mask in
                                         zip(seg_spike_times, seg_mask)]
                                        for seg_spike_times, seg_mask in
                                        zip(spike_times,
                                            np.logical_not(correct_annotations))])

        cleaned_spike_times = np.array([[st.times for st in seg.spiketrains]
                                        for seg in block.segments])

        for correct_seg, cleaned_seg in zip(correct_spike_times, cleaned_spike_times):
            for correct_st, cleaned_st in zip(correct_seg, cleaned_seg):
                assert_array_almost_equal(cleaned_st, correct_st)

    def test_binning_for_input_with_rounding_errors(self):

        # redo the test_n_equals_3 with inputs divided by 30000
        # which leads to rounding errors
        # these errors have to be accounted for by proper binning;
        # check if we still get the correct result

        sampling_rate = 30000. / pq.s

        spike_times = np.array([[[1, 1, 5, 10], [1, 4, 7, 9]],
                                [[12, 15, 16, 18], [11, 13, 15, 19]]]) / 30000. * pq.s

        block = generate_block(spike_times,
                               segment_edges=[0./30000., 10./30000., 20./30000.]*pq.s)

        # test annotation
        spike_train_processing.detect_synchrofacts(block, segment='all', n=3, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=False,
                                                  unit_type='all')

        correct_annotations = [[np.array([True, True, False, False]),
                                np.array([True, False, False, False])],
                               [np.array([True, True, True, False]),
                                np.array([True, True, True, False])]]

        annotations = [[st.array_annotations['synchrofacts'] for st in seg.spiketrains]
                       for seg in block.segments]

        assert_array_equal(annotations, correct_annotations)

        # test deletion
        spike_train_processing.detect_synchrofacts(block, segment='all', n=3, spread=2,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=True,
                                                  unit_type='all')

        correct_spike_times = np.array([[spikes[mask] for spikes, mask in
                                         zip(seg_spike_times, seg_mask)]
                                        for seg_spike_times, seg_mask in
                                        zip(spike_times,
                                            np.logical_not(correct_annotations))])

        cleaned_spike_times = np.array([[st.times for st in seg.spiketrains]
                                        for seg in block.segments])

        for correct_seg, cleaned_seg in zip(correct_spike_times, cleaned_spike_times):
            for correct_st, cleaned_st in zip(correct_seg, cleaned_seg):
                assert_array_almost_equal(cleaned_st, correct_st)

    def test_correct_transfer_of_spiketrain_attributes(self):

        # for delete=True the spiketrains in the block are changed,
        # test if their attributes remain correct

        sampling_rate = 1 / pq.s

        spike_times = np.array([[[1, 1, 5, 9]]]) * pq.s

        block = generate_block(spike_times, segment_edges=[0, 10]*pq.s)

        block.segments[0].spiketrains[0].annotate(cool_spike_train=True)
        block.segments[0].spiketrains[0].array_annotate(
            spike_number=np.arange(len(
                block.segments[0].spiketrains[0].times.magnitude)))
        block.segments[0].spiketrains[0].waveforms = np.sin(
            np.arange(len(
                block.segments[0].spiketrains[0].times.magnitude))[:, np.newaxis] +
            np.arange(len(
                block.segments[0].spiketrains[0].times.magnitude))[np.newaxis, :])

        correct_mask = np.array([False, False, True, True])

        # store the correct attributes
        correct_annotations = block.segments[0].spiketrains[0].annotations.copy()
        correct_waveforms = block.segments[0].spiketrains[0].waveforms[
            correct_mask].copy()
        correct_array_annotations = {
            key: value[correct_mask] for key, value in
            block.segments[0].spiketrains[0].array_annotations.items()}

        # perform a synchrofact search with delete=True
        spike_train_processing.detect_synchrofacts(block, segment='all',
                                                  n=2, spread=1,
                                                  sampling_rate=sampling_rate,
                                                  invert=False, delete=True,
                                                  unit_type='all')

        # Ensure that the spiketrain was not duplicated
        self.assertEqual(len(block.filter(objects=neo.SpikeTrain)), 1)

        cleaned_annotations = block.segments[0].spiketrains[0].annotations
        cleaned_waveforms = block.segments[0].spiketrains[0].waveforms
        cleaned_array_annotations = block.segments[0].spiketrains[0].array_annotations

        self.assertDictEqual(correct_annotations, cleaned_annotations)
        assert_array_almost_equal(cleaned_waveforms, correct_waveforms)
        self.assertTrue(len(cleaned_array_annotations)
                        == len(correct_array_annotations))
        for key, value in correct_array_annotations.items():
            self.assertTrue(key in cleaned_array_annotations.keys())
            assert_array_almost_equal(value, cleaned_array_annotations[key])


if __name__ == '__main__':
    unittest.main()
