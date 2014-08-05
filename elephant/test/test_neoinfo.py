# needed for python 3 compatibility
from __future__ import absolute_import
import unittest
import numpy as np
import quantities as pq
from elephant.neoinfo import NeoInfo
from neo.core import Block, Segment, AnalogSignal, SpikeTrain, Unit, \
    RecordingChannelGroup, RecordingChannel


class NeoInfoTestCase(unittest.TestCase):
    def setUp(self):
        self.blk1 = Block()
        self.blk3 = Block()
        self.unit = Unit()
        self.rcg = RecordingChannelGroup(name='all channels')
        self.spk_lst = []
        self.asig_lst = []
        self.setup_block()
        self.setup_signal_lsts()
        self.info = NeoInfo(self.blk1)
        self.info_st = NeoInfo(self.spk_lst)
        self.info_as = NeoInfo(self.asig_lst)

    def setup_block(self):
        """
        Initializes same neo.Block for every test function.

        """
        for ind in range(3):
            seg = Segment(name='segment %d' % ind, index=ind)
            a = AnalogSignal(
                [np.sin(2 * np.pi * 10 * t / 1000.0) for t in range(100)],
                sampling_rate=10 * pq.Hz,
                units='mV')
            st = SpikeTrain([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s,
                            t_stop=10.0 * pq.s)
            chan = RecordingChannel(index=ind)
            self.rcg.recordingchannels.append(chan)
            chan.recordingchannelgroups.append(self.rcg)
            chan.analogsignals.append(a)
            chan.analogsignals.append(a)
            chan.analogsignals.append(a)
            a.recordingchannel = chan
            seg.analogsignals.append(a)
            seg.analogsignals.append(a)
            seg.analogsignals.append(a)
            st.unit = self.unit
            seg.spiketrains.append(st)
            seg.spiketrains.append(st)
            seg.spiketrains.append(st)
            self.unit.spiketrains.append(st)
            self.blk1.segments.append(seg)
        # Append
        self.unit.block = self.blk1
        self.rcg.units.append(self.unit)
        self.blk1.recordingchannelgroups.append(self.rcg)

    def setup_signal_lsts(self):
        j = 0
        for _ in range(1, 4):
            s = np.array([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7])
            a = np.random.rand(3, 10)
            s += j
            a += j
            st = SpikeTrain(s * pq.s,
                            t_start=j * pq.s, t_stop=(j + 10) * pq.s)
            asig = AnalogSignal(a * pq.nA,
                                sampling_rate=10 * pq.kHz)
            self.asig_lst.append(asig)
            self.spk_lst.append(st)
            j += 10

    def tearDown(self):
        self.blk1 = None
        del self.blk1
        self.info = None
        del self.info
        self.info2 = None
        del self.info2
        self.rcg = None
        del self.rcg
        self.spk_lst = None
        del self.spk_lst
        self.asig_lst = None
        del self.asig_lst

    # ########## Trial related methods, ie indices etc. ##########
    def test_trials(self):
        self.assertEqual(self.info.get_input_type(), 'Block')
        self.info.set_trial_conditions(each_st_has_n_spikes=(True, 7))
        self.assertRaises(IndexError,
                          self.info.get_num_spiketrains_of_valid_trial, 4)
        self.assertEqual(self.info.get_trial_ids(), [0, 1, 2])
        self.assertTrue(self.info.has_trials())
        self.assertTrue(self.info.has_units())
        self.assertTrue(self.info.has_spiketrains())
        self.assertTrue(self.info.has_analogsignals())
        self.assertTrue(self.info.has_recordingchannels())
        self.assertTrue(self.info.has_recordingchannelgroup())
        self.assertEqual(self.info.get_num_units(), 1)
        self.assertEqual(self.info.get_num_trials(), 3)
        self.assertEqual(self.info.get_num_recordingchannels(), 3)
        self.assertEqual(self.info.get_num_recordingchannelgroup(), 1)
        self.assertEqual(self.info.get_num_analogsignals(), 9)
        self.assertEqual(self.info.get_num_spiketrains(), 9)

    def test_trials_with_lsts(self):
        self.assertEqual(self.info_st.get_input_type(), 'SpikeTrain List')
        self.info.set_trial_conditions(each_st_has_n_spikes=(True, 10))
        self.assertRaises(IndexError,
                          self.info_st.get_num_spiketrains_of_valid_trial, 4)

        self.assertEqual(self.info_st.get_trial_ids(), [0])
        self.assertTrue(self.info_st.has_trials())
        self.assertFalse(self.info_st.has_units())
        self.assertTrue(self.info_st.has_spiketrains())
        self.assertFalse(self.info_st.has_analogsignals())
        self.assertFalse(self.info_st.has_recordingchannels())
        self.assertEqual(self.info_st.get_num_trials(), 1)
        self.assertEqual(self.info_st.get_num_analogsignals(), 0)
        self.assertEqual(self.info_st.get_num_spiketrains(), 3)

        self.assertEqual(self.info_as.get_input_type(), 'AnalogSignal List')
        self.info.set_trial_conditions(each_st_has_n_spikes=(True, 10))
        self.assertRaises(IndexError,
                          self.info_as.get_num_spiketrains_of_valid_trial, 4)

        self.assertEqual(self.info_as.get_trial_ids(), [0])
        self.assertTrue(self.info_as.has_trials())
        self.assertFalse(self.info_as.has_units())
        self.assertFalse(self.info_as.has_spiketrains())
        self.assertTrue(self.info_as.has_analogsignals())
        self.assertFalse(self.info_as.has_recordingchannels())
        self.assertEqual(self.info_as.get_num_trials(), 1)
        self.assertEqual(self.info_as.get_num_analogsignals(), 3)
        self.assertEqual(self.info_as.get_num_spiketrains(), 0)

    ############# Test Trial Conditions #############
    def test_trial_has_n_spiketrains(self):
        self.info.set_trial_conditions(trial_has_n_st=(True, 3))
        self.assertTrue(self.info.is_valid([0, 1, 2]))
        self.assertTrue(self.info.has_spiketrain_in_vaild_trial())
        self.assertEqual(self.info.valid_trial_ids, [0, 1, 2])
        self.assertEqual(self.info.get_num_spiketrains_of_valid_trial(1),
                         [(1, 3)])
        self.assertEqual(
            self.info.get_num_spiketrains_of_valid_trial(1, with_id=False), 3)
        self.assertEqual(self.info.get_num_spiketrains_of_valid_trial(),
                         [(0, 3), (1, 3), (2, 3)])
        self.assertEqual(self.info.get_num_valid_trials(), 3)
        self.info.set_trial_conditions(trial_has_n_st=(True, 5))
        self.assertEqual(self.info.valid_trial_ids, [])
        self.assertTrue(self.info.has_spiketrains())
        self.assertFalse(self.info.has_spiketrain_in_vaild_trial())
        self.assertFalse(self.info.is_valid([0, 1, 2]))
        self.assertEqual(self.info.get_num_spiketrains_of_valid_trial(), 0)
        self.assertEqual(self.info.get_num_valid_trials(), 0)

        self.info.reset_trial_conditions()
        self.info.set_trial_conditions(trial_has_exact_st=(True, 2))
        self.assertEqual(self.info.valid_trial_ids, [])
        self.info.set_trial_conditions(trial_has_exact_st=(True, 3))
        self.assertEqual(self.info.valid_trial_ids, [0, 1, 2])

    def test_trial_has_spiketrains_lsts(self):
        self.info_st.set_trial_conditions(trial_has_n_st=(True, 3))
        self.assertTrue(self.info_st.is_valid([0]))
        self.assertTrue(self.info_st.has_spiketrain_in_vaild_trial())
        self.assertEqual(self.info_st.valid_trial_ids, [0])
        self.assertEqual(self.info_st.get_num_spiketrains_of_valid_trial(0), 3)
        self.assertEqual(self.info_st.get_num_spiketrains_of_valid_trial(), 3)
        self.assertEqual(
            self.info_st.get_num_spiketrains_of_valid_trial(0, with_id=False),
            3)
        self.assertEqual(self.info_st.get_num_spiketrains_of_valid_trial(), 3)
        self.assertEqual(self.info_st.get_num_valid_trials(), 1)

        self.info.reset_trial_conditions()
        self.info_st.set_trial_conditions(trial_has_exact_st=(True, 1))
        self.assertEqual(self.info_st.valid_trial_ids, [])
        self.info_st.set_trial_conditions(trial_has_exact_st=(True, 3))
        self.assertEqual(self.info_st.valid_trial_ids, [0])

    def test_trial_has_n_analogsignals(self):
        self.info.set_trial_conditions(trial_has_n_as=(True, 3))
        self.assertTrue(self.info.is_valid([0, 1, 2]))
        self.assertEqual(self.info.get_num_analogsignals_of_valid_trial(1),
                         [(1, 3)])
        self.assertEqual(
            self.info.get_num_analogsignals_of_valid_trial(1, with_id=False),
            3)
        self.assertEqual(self.info.get_num_analogsignals_of_valid_trial(),
                         [(0, 3), (1, 3), (2, 3)])
        self.assertTrue(self.info.has_analogsignal_in_valid_trial())
        self.info.set_trial_conditions(trial_has_n_as=(True, 5))
        self.assertEqual(self.info.valid_trial_ids, [])
        self.assertFalse(self.info.has_analogsignal_in_valid_trial())
        self.assertFalse(self.info.is_valid([0, 1, 2]))
        self.assertEqual(self.info.get_num_analogsignals_of_valid_trial(), 0)
        self.info.reset_trial_conditions()
        self.info.set_trial_conditions(trial_has_exact_as=(True, 2))
        self.assertEqual(self.info.valid_trial_ids, [])
        self.info.set_trial_conditions(trial_has_exact_as=(True, 3))
        self.assertEqual(self.info.valid_trial_ids, [0, 1, 2])

    def test_trial_has_n_analogsignals_lsts(self):
        self.info_as.set_trial_conditions(trial_has_n_as=(True, 3))
        self.assertTrue(self.info_as.is_valid([0]))
        self.assertEqual(self.info_as.get_num_analogsignals_of_valid_trial(0),
                         3)
        self.assertEqual(
            self.info_as.get_num_analogsignals_of_valid_trial(0,
                                                              with_id=False),
            3)
        self.assertEqual(self.info_as.get_num_analogsignals_of_valid_trial(),
                         3)
        self.assertTrue(self.info_as.has_analogsignal_in_valid_trial())
        self.info_as.set_trial_conditions(trial_has_n_as=(True, 5))
        self.assertEqual(self.info_as.valid_trial_ids, [])
        self.assertFalse(self.info_as.has_analogsignal_in_valid_trial())
        self.assertFalse(self.info_as.is_valid([0, 1, 2]))
        self.assertEqual(self.info_as.get_num_analogsignals_of_valid_trial(),
                         0)
        self.info.reset_trial_conditions()
        self.info.set_trial_conditions(trial_has_exact_as=(True, 2))
        self.assertEqual(self.info_as.valid_trial_ids, [])
        self.info_as.set_trial_conditions(trial_has_exact_as=(True, 10))
        self.assertEqual(self.info_as.valid_trial_ids, [])

    def test_check_each_st_has_n_spikes(self):
        self.info.set_trial_conditions(each_st_has_n_spikes=(True, 7))
        self.assertTrue(self.info.is_valid([0, 1, 2]))
        self.assertEqual(
            self.info.get_num_spiketrains_of_valid_trial(with_id=False), 3)
        self.info.set_trial_conditions(each_st_has_n_spikes=(True, 10))
        self.assertEqual(self.info.get_num_spiketrains_of_valid_trial(), 0)
        self.assertEqual(self.info.valid_trial_ids, [])
        self.info.set_trial_conditions(each_st_has_n_spikes=(True, 1))
        self.assertEqual(self.info.get_num_spiketrains_of_valid_trial(), 0)
        self.assertEqual(self.info.valid_trial_ids, [])

    def test_check_each_st_has_n_spikes_lsts(self):
        self.info_st.set_trial_conditions(each_st_has_n_spikes=(True, 7))
        self.assertTrue(self.info_st.is_valid([0]))
        self.assertEqual(
            self.info_st.get_num_spiketrains_of_valid_trial(with_id=False), 3)
        self.info_st.set_trial_conditions(each_st_has_n_spikes=(True, 10))
        self.assertEqual(self.info_st.get_num_spiketrains_of_valid_trial(), 0)
        self.assertEqual(self.info_st.valid_trial_ids, [])
        self.info_st.set_trial_conditions(each_st_has_n_spikes=(True, 1))
        self.assertEqual(self.info_st.get_num_spiketrains_of_valid_trial(), 0)
        self.assertEqual(self.info_st.valid_trial_ids, [])

    def test_trial_has_n_units(self):
        self.info.set_trial_conditions(trial_has_n_units=(True, 1))
        self.assertEqual(self.info.get_num_unit_valid_trial(), 1)
        self.assertTrue(self.info.is_valid([0, 1, 2]))
        self.assertEqual([elem[1] for elem in
                          self.info.get_units_from_valid_trial()],
                         self.blk1.list_units)
        self.assertEqual(self.info.get_units(), self.blk1.list_units)
        self.info.set_trial_conditions(trial_has_n_units=(True, 5))
        self.assertEqual(self.info.get_num_unit_valid_trial(), 0)
        self.assertFalse(self.info.is_valid([0, 1, 2]))
        self.assertEqual(self.info.get_units_from_valid_trial(), [])

    def test_trial_has_n_rc(self):
        self.info.set_trial_conditions(trial_has_n_rc=(True, 1))
        self.assertEqual(self.info.get_num_valid_trials(), 3)
        self.assertEqual(self.info.get_num_recordingchannel_from_valid_trial(),
                         3)
        self.assertEqual([elem[1] for elem in
                          self.info.get_recordingchannels_from_valid_trial()],
                         self.blk1.list_recordingchannels)
        self.info.set_trial_conditions(trial_has_n_rc=(True, 2))
        self.assertEqual(self.info.get_num_valid_trials(), 0)
        self.assertEqual(self.info.get_recordingchannels_from_valid_trial(),
                         [])

    def test_trial_has_no_overlap(self):
        self.info.set_trial_conditions(trial_has_no_overlap=(True, False))
        self.assertEqual(self.info.valid_trial_ids, [])

        # Positive test
        blk = Block()
        j = 0
        for i in range(3):
            seg = Segment(name='segment %d' % i, index=i)
            s = np.array([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7])
            st = SpikeTrain((s + j) * pq.s,
                            t_start=j * pq.s, t_stop=(j + 10.0) * pq.s)
            seg.spiketrains.append(st)
            blk.segments.append(seg)
            j += 10
        ni = NeoInfo(blk)
        ni.set_trial_conditions(trial_has_no_overlap=(True, 0))
        self.assertEqual(ni.valid_trial_ids, [0, 1, 2])
        # Test with take first element if there is overlap
        ni.reset_trial_conditions()
        ni.set_trial_conditions(trial_has_no_overlap=(True, True))
        self.assertEqual(ni.valid_trial_ids, [0, 1, 2])

        # Another negative case
        #################################
        # # Structure of trials         #
        # # 0---                        #
        # #   1---                      #
        # #    2---                     #
        # #        3---                 #
        # # Only trial 3 has no overlap #
        #################################
        block = Block()
        seg1 = Segment()
        seg2 = Segment()
        seg3 = Segment()
        seg4 = Segment()
        s = np.array([0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7])
        st1 = SpikeTrain(s * pq.s, t_start=0 * pq.s, t_stop=10.0 * pq.s)
        st2 = SpikeTrain((s + 7) * pq.s, t_start=7 * pq.s, t_stop=17.0 * pq.s)
        st3 = SpikeTrain((s + 11) * pq.s, t_start=11 * pq.s,
                         t_stop=21.0 * pq.s)
        st4 = SpikeTrain((s + 22) * pq.s, t_start=22 * pq.s,
                         t_stop=32.0 * pq.s)
        seg1.spiketrains.append(st1)
        seg2.spiketrains.append(st2)
        seg3.spiketrains.append(st3)
        seg4.spiketrains.append(st4)
        block.segments.append(seg1)
        block.segments.append(seg2)
        block.segments.append(seg3)
        block.segments.append(seg4)
        ni2 = NeoInfo(block)
        ni2.set_trial_conditions(trial_has_no_overlap=(True, False))
        self.assertEqual(ni2.valid_trial_ids, [3])
        self.assertEqual(ni2.get_num_spiketrains_of_valid_trial(), [(3, 1)])
        valid_st = ni2.get_spiketrains_from_valid_trials()[0][1][0].magnitude
        self.assertTrue(np.array_equal(valid_st, st4.magnitude))
        # self.assertCountEqual(valid_st, st4.magnitude)
        self.assertEqual(ni2.get_num_analogsignals_of_valid_trial(), 0)

        # Test with take first element if there is overlap
        ni2.reset_trial_conditions()
        ni2.set_trial_conditions(trial_has_no_overlap=(True, True))
        self.assertEqual(ni2.valid_trial_ids, [0, 3])

    def test_data_aligned(self):
        self.info.set_trial_conditions(data_aligned=(True,))
        self.assertEqual(self.info.valid_trial_ids, [0, 1, 2])
        self.assertTrue(self.info.is_valid([0, 1, 2]))

    def test_trial_contains_each_unit(self):
        self.info.set_trial_conditions(contains_each_unit=(True,))
        self.assertEqual(self.info.valid_trial_ids, [0, 1, 2])
        self.assertEqual(self.info.get_num_units(), 1)
        self.assertEqual(self.info.get_num_unit_valid_trial(), 1)

    def test_trial_contains_each_rc(self):
        self.info.set_trial_conditions(contains_each_rc=(True,))
        self.assertEqual(self.info.valid_trial_ids, [])
        self.assertEqual(self.info.get_num_recordingchannels(), 3)
        self.assertEqual(self.info.get_num_recordingchannel_from_valid_trial(),
                         0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NeoInfoTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
