import random
import unittest
from collections import defaultdict

import neo
import numpy as np
import quantities as pq

from elephant.datasets import download_datasets
from elephant.online import OnlineUnitaryEventAnalysis
from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.unitary_event_analysis import jointJ_window_analysis


def _generate_spiketrains(freq, length, trigger_events, injection_pos,
                          trigger_pre_size, trigger_post_size,
                          time_unit=1*pq.s):
    """
    Generate two spiketrains from a homogeneous Poisson process with
    injected coincidences.
    """
    st1 = StationaryPoissonProcess(rate=freq,
                                   t_start=(0*pq.s).rescale(time_unit),
                                   t_stop=length.rescale(time_unit)
                                   ).generate_spiketrain()
    st2 = StationaryPoissonProcess(rate=freq,
                                   t_start=(0*pq.s.rescale(time_unit)),
                                   t_stop=length.rescale(time_unit)
                                   ).generate_spiketrain()
    # inject 10 coincidences within a 0.1s interval for each trial
    injection = (np.linspace(0, 0.1, 10)*pq.s).rescale(time_unit)
    all_injections = np.array([])
    for i in trigger_events:
        all_injections = np.concatenate(
            (all_injections, (i+injection_pos)+injection), axis=0) * time_unit
    st1 = st1.duplicate_with_new_data(
        np.sort(np.concatenate((st1.times, all_injections)))*time_unit)
    st2 = st2.duplicate_with_new_data(
        np.sort(np.concatenate((st2.times, all_injections)))*time_unit)

    # stack spiketrains by trial
    st1_stacked = [st1.time_slice(
        t_start=i - trigger_pre_size,
        t_stop=i + trigger_post_size).time_shift(-i + trigger_pre_size)
                   for i in trigger_events]
    st2_stacked = [st2.time_slice(
        t_start=i - trigger_pre_size,
        t_stop=i + trigger_post_size).time_shift(-i + trigger_pre_size)
                   for i in trigger_events]
    spiketrains = np.stack((st1_stacked, st2_stacked), axis=1)
    spiketrains = spiketrains.tolist()

    return spiketrains, st1, st2


def _simulate_buffered_reading(n_buffers, ouea, st1, st2,
                               incoming_data_window_size, length_remainder,
                               events=None, st_type="list_of_neo.SpikeTrain"):
    if events is None:
        events = np.array([])
    for i in range(n_buffers):
        buff_t_start = i * incoming_data_window_size

        if length_remainder > 1e-7 and i == n_buffers - 1:
            buff_t_stop = i * incoming_data_window_size + length_remainder
        else:
            buff_t_stop = i * incoming_data_window_size + \
                          incoming_data_window_size

        events_in_buffer = np.array([])
        if len(events) > 0:
            idx_events_in_buffer = (events >= buff_t_start) & \
                                   (events <= buff_t_stop)
            events_in_buffer = events[idx_events_in_buffer]
            events = events[np.logical_not(idx_events_in_buffer)]

        if st_type == "list_of_neo.SpikeTrain":
            ouea.update_uea(
                spiketrains=[
                    st1.time_slice(t_start=buff_t_start, t_stop=buff_t_stop),
                    st2.time_slice(t_start=buff_t_start, t_stop=buff_t_stop)],
                events=events_in_buffer)
        elif st_type == "list_of_numpy_array":
            ouea.update_uea(
                spiketrains=[
                    st1.time_slice(t_start=buff_t_start, t_stop=buff_t_stop
                                   ).magnitude,
                    st2.time_slice(t_start=buff_t_start, t_stop=buff_t_stop
                                   ).magnitude],
                events=events_in_buffer, t_start=buff_t_start,
                t_stop=buff_t_stop, time_unit=st1.units)
        else:
            raise ValueError("undefined type for spiktrains representation! "
                             "Use either list of neo.SpikeTrains or "
                             "list of numpy arrays")
        # print(f"#buffer = {i}")  # DEBUG-aid


def _load_real_data(filepath, n_trials, trial_length, time_unit):
    # load data and extract spiketrains
    io = neo.io.NixIO(f"{filepath}", 'ro')
    block = io.read_block()
    spiketrains = []
    # each segment contains a single trial
    for ind in range(len(block.segments)):
        spiketrains.append(block.segments[ind].spiketrains)
    # for each neuron: concatenate all trials to one long neo.Spiketrain
    st1_long = [spiketrains[i].multiplexed[1][
                    np.where(spiketrains[i].multiplexed[0] == 0)]
                + i * trial_length
                for i in range(len(spiketrains))]
    st2_long = [spiketrains[i].multiplexed[1][
                    np.where(spiketrains[i].multiplexed[0] == 1)]
                + i * trial_length
                for i in range(len(spiketrains))]
    st1_concat = st1_long[0]
    st2_concat = st2_long[0]
    for i in range(1, len(st1_long)):
        st1_concat = np.concatenate((st1_concat, st1_long[i]))
        st2_concat = np.concatenate((st2_concat, st2_long[i]))
    neo_st1 = neo.SpikeTrain((st1_concat / 1000) * pq.s, t_start=0 * pq.s,
                             t_stop=n_trials * trial_length).rescale(time_unit)
    neo_st2 = neo.SpikeTrain((st2_concat / 1000) * pq.s, t_start=0 * pq.s,
                             t_stop=n_trials * trial_length).rescale(time_unit)
    spiketrains = [[st[j].rescale(time_unit) for j in range(len(st))] for st in
                   spiketrains]
    return spiketrains, neo_st1, neo_st2


def _calculate_n_buffers(n_trials, tw_length, noise_length, idw_length):
    _n_buffers_float = n_trials * (tw_length + noise_length) / idw_length
    _n_buffers_int = int(_n_buffers_float)
    _n_buffers_fraction = _n_buffers_float - _n_buffers_int
    n_buffers = _n_buffers_int + 1 if _n_buffers_fraction > 1e-7 else \
        _n_buffers_int
    length_remainder = idw_length * _n_buffers_fraction
    return n_buffers, length_remainder


class TestOnlineUnitaryEventAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(73)
        cls.time_unit = 1 * pq.ms
        cls.last_n_trials = 50

        # download real data once and load it several times later
        repo_path = 'tutorials/tutorial_unitary_event_analysis/' \
                    'data/dataset-1.nix'
        cls.repo_path = repo_path
        cls.filepath = download_datasets(repo_path)

        cls.st_types = ["list_of_neo.SpikeTrain", "list_of_numpy_array"]

    def setUp(self):
        # do nothing
        pass

    def _assert_equality_of_passed_and_saved_trials(
            self, last_n_trials, passed_trials, saved_trials):
        eps_float64 = np.finfo(np.float64).eps
        n_neurons = len(passed_trials[0])
        with self.subTest("test 'trial' equality"):
            for t in range(last_n_trials):
                for n in range(n_neurons):
                    np.testing.assert_allclose(
                        actual=saved_trials[-t][n].rescale(
                            self.time_unit).magnitude,
                        desired=saved_trials[-t][n].rescale(
                            self.time_unit).magnitude,
                        atol=eps_float64, rtol=eps_float64)

    def _assert_equality_of_result_dicts(self, ue_dict_offline, ue_dict_online,
                                         tol_dict_user):
        eps_float64 = np.finfo(np.float64).eps
        eps_float32 = np.finfo(np.float32).eps
        tol_dict = {"atol_Js": eps_float64, "rtol_Js": eps_float64,
                    "atol_indices": eps_float64, "rtol_indices": eps_float64,
                    "atol_n_emp": eps_float64, "rtol_n_emp": eps_float64,
                    "atol_n_exp": eps_float64, "rtol_n_exp": eps_float64,
                    "atol_rate_avg": eps_float32, "rtol_rate_avg": eps_float32}
        tol_dict.update(tol_dict_user)

        with self.subTest("test 'Js' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["Js"], desired=ue_dict_offline["Js"],
                atol=tol_dict["atol_Js"],
                rtol=tol_dict["rtol_Js"])
        with self.subTest("test 'indices' equality"):
            for key in ue_dict_offline["indices"].keys():
                np.testing.assert_allclose(
                    actual=ue_dict_online["indices"][key],
                    desired=ue_dict_offline["indices"][key],
                    atol=tol_dict["atol_indices"],
                    rtol=tol_dict["rtol_indices"])
        with self.subTest("test 'n_emp' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["n_emp"],
                desired=ue_dict_offline["n_emp"],
                atol=tol_dict["atol_n_emp"], rtol=tol_dict["rtol_n_emp"])
        with self.subTest("test 'n_exp' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["n_exp"],
                desired=ue_dict_offline["n_exp"],
                atol=tol_dict["atol_n_exp"],
                rtol=tol_dict["rtol_n_exp"])
        with self.subTest("test 'rate_avg' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["rate_avg"].magnitude,
                desired=ue_dict_offline["rate_avg"].magnitude,
                atol=tol_dict["atol_rate_avg"], rtol=tol_dict["rtol_rate_avg"])
        with self.subTest("test 'input_parameters' equality"):
            for key in ue_dict_offline["input_parameters"].keys():
                np.testing.assert_equal(
                    actual=ue_dict_online["input_parameters"][key],
                    desired=ue_dict_offline["input_parameters"][key])

    def _test_unitary_events_analysis_with_real_data(
            self, idw_length, method="pass_events_at_initialization",
            time_unit=1 * pq.s, st_type="list_of_neo.SpikeTrain"):
        # Fix random seed to guarantee fixed output
        random.seed(1224)

        # set relevant variables of this TestCase
        n_trials = 36
        trial_window_length = (2.1 * pq.s).rescale(time_unit)
        IDW_length = idw_length.rescale(time_unit)
        noise_length = (0. * pq.s).rescale(time_unit)
        trigger_events = (np.arange(0., n_trials * 2.1, 2.1) * pq.s).rescale(
            time_unit)
        n_buffers, length_remainder = _calculate_n_buffers(
            n_trials=n_trials, tw_length=trial_window_length,
            noise_length=noise_length, idw_length=IDW_length)

        # load data and extract spiketrains
        # 36 trials with 2.1s length and 0s background noise in between trials
        spiketrains, neo_st1, neo_st2 = _load_real_data(
            filepath=self.filepath, n_trials=n_trials,
            trial_length=trial_window_length, time_unit=time_unit)

        # perform standard unitary events analysis
        ue_dict = jointJ_window_analysis(
            spiketrains, bin_size=(0.005 * pq.s).rescale(time_unit),
            winsize=(0.1 * pq.s).rescale(time_unit),
            winstep=(0.005 * pq.s).rescale(time_unit), pattern_hash=[3])

        if method == "pass_events_at_initialization":
            init_events = trigger_events
            reading_events = np.array([]) * time_unit
        elif method == "pass_events_while_buffered_reading":
            init_events = np.array([]) * time_unit
            reading_events = trigger_events
        else:
            raise ValueError("Illegal method to pass events!")

        # create instance of OnlineUnitaryEventAnalysis
        _last_n_trials = min(self.last_n_trials, len(spiketrains))
        if st_type == "list_of_neo.SpikeTrain":
            ouea = OnlineUnitaryEventAnalysis(
                bin_window_size=(0.005 * pq.s).rescale(time_unit),
                trigger_pre_size=(0. * pq.s).rescale(time_unit),
                trigger_post_size=(2.1 * pq.s).rescale(time_unit),
                sliding_analysis_window_size=(0.1 * pq.s).rescale(time_unit),
                sliding_analysis_window_step=(0.005 * pq.s).rescale(time_unit),
                trigger_events=init_events,
                time_unit=time_unit,
                save_n_trials=_last_n_trials)
        elif st_type == "list_of_numpy_array":
            ouea = OnlineUnitaryEventAnalysis(
                bin_window_size=(0.005 * pq.s).rescale(time_unit).magnitude,
                trigger_pre_size=(0. * pq.s).rescale(time_unit).magnitude,
                trigger_post_size=(2.1 * pq.s).rescale(time_unit).magnitude,
                sliding_analysis_window_size=(0.1 * pq.s
                                              ).rescale(time_unit).magnitude,
                sliding_analysis_window_step=(0.005 * pq.s
                                              ).rescale(time_unit).magnitude,
                trigger_events=init_events.magnitude,
                time_unit=time_unit.__str__().split(" ")[1],
                save_n_trials=_last_n_trials)
        else:
            raise ValueError("undefined type for spiktrains representation! "
                             "Use either list of neo.SpikeTrains or "
                             "list of numpy arrays")
        # perform online unitary event analysis
        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call update_ue()
        _simulate_buffered_reading(
            n_buffers=n_buffers, ouea=ouea, st1=neo_st1, st2=neo_st2,
            incoming_data_window_size=IDW_length,
            length_remainder=length_remainder, events=reading_events,
            st_type=st_type)
        ue_dict_online = ouea.get_results()

        # assert equality between result dicts of standard / online ue version
        self._assert_equality_of_result_dicts(
            ue_dict_offline=ue_dict, ue_dict_online=ue_dict_online,
            tol_dict_user={})

        self._assert_equality_of_passed_and_saved_trials(
            last_n_trials=_last_n_trials, passed_trials=spiketrains,
            saved_trials=ouea.get_all_saved_trials())

        return ouea

    def _test_unitary_events_analysis_with_artificial_data(
            self, idw_length, method="pass_events_at_initialization",
            time_unit=1 * pq.s, st_type="list_of_neo.SpikeTrain"):
        # fix random seed to guarantee fixed output
        random.seed(1224)

        # set relevant variables of this TestCase
        n_trials = 40
        trial_window_length = (1 * pq.s).rescale(time_unit)
        noise_length = (1.5 * pq.s).rescale(time_unit)
        incoming_data_window_size = idw_length.rescale(time_unit)
        trigger_events = (np.arange(0., n_trials*2.5, 2.5) * pq.s).rescale(
            time_unit)
        trigger_pre_size = (0. * pq.s).rescale(time_unit)
        trigger_post_size = (1. * pq.s).rescale(time_unit)
        n_buffers, length_remainder = _calculate_n_buffers(
            n_trials=n_trials, tw_length=trial_window_length,
            noise_length=noise_length, idw_length=incoming_data_window_size)

        # create two long random homogeneous poisson spiketrains representing
        # 40 trials with 1s length and 1.5s background noise in between trials
        spiketrains, st1_long, st2_long = _generate_spiketrains(
            freq=5*pq.Hz, length=(trial_window_length+noise_length)*n_trials,
            trigger_events=trigger_events,
            injection_pos=(0.6 * pq.s).rescale(time_unit),
            trigger_pre_size=trigger_pre_size,
            trigger_post_size=trigger_post_size,
            time_unit=time_unit)

        # perform standard unitary event analysis
        ue_dict = jointJ_window_analysis(
            spiketrains, bin_size=(0.005 * pq.s).rescale(time_unit),
            win_size=(0.1 * pq.s).rescale(time_unit),
            win_step=(0.005 * pq.s).rescale(time_unit), pattern_hash=[3])

        if method == "pass_events_at_initialization":
            init_events = trigger_events
            reading_events = np.array([]) * time_unit
        elif method == "pass_events_while_buffered_reading":
            init_events = np.array([]) * time_unit
            reading_events = trigger_events
        else:
            raise ValueError("Illegal method to pass events!")

        # create instance of OnlineUnitaryEventAnalysis
        _last_n_trials = min(self.last_n_trials, len(spiketrains))
        ouea = None
        if st_type == "list_of_neo.SpikeTrain":
            ouea = OnlineUnitaryEventAnalysis(
                bin_window_size=(0.005 * pq.s).rescale(time_unit),
                trigger_pre_size=trigger_pre_size,
                trigger_post_size=trigger_post_size,
                sliding_analysis_window_size=(0.1 * pq.s).rescale(time_unit),
                sliding_analysis_window_step=(0.005 * pq.s).rescale(time_unit),
                trigger_events=init_events,
                time_unit=time_unit,
                save_n_trials=_last_n_trials)
        elif st_type == "list_of_numpy_array":
            ouea = OnlineUnitaryEventAnalysis(
                bin_window_size=(0.005 * pq.s).rescale(time_unit).magnitude,
                trigger_pre_size=trigger_pre_size.magnitude,
                trigger_post_size=trigger_post_size.magnitude,
                sliding_analysis_window_size=(0.1 * pq.s
                                              ).rescale(time_unit).magnitude,
                sliding_analysis_window_step=(0.005 * pq.s
                                              ).rescale(time_unit).magnitude,
                trigger_events=init_events.magnitude,
                time_unit=time_unit.__str__().split(" ")[1],
                save_n_trials=_last_n_trials)
        else:
            raise ValueError("undefined type for spiktrains representation! "
                             "Use either list of neo.SpikeTrains or "
                             "list of numpy arrays")
        # perform online unitary event analysis
        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call update_ue()
        _simulate_buffered_reading(
            n_buffers=n_buffers, ouea=ouea, st1=st1_long, st2=st2_long,
            incoming_data_window_size=incoming_data_window_size,
            length_remainder=length_remainder,
            events=reading_events, st_type=st_type)
        ue_dict_online = ouea.get_results()

        # assert equality between result dicts of standard / online ue version
        self._assert_equality_of_result_dicts(
            ue_dict_offline=ue_dict, ue_dict_online=ue_dict_online,
            tol_dict_user={})

        self._assert_equality_of_passed_and_saved_trials(
            last_n_trials=_last_n_trials, passed_trials=spiketrains,
            saved_trials=ouea.get_all_saved_trials())

        return ouea

    # test: trial window > incoming data window
    def test_trial_window_larger_IDW_artificial_data(self):
        """Test, if online UE analysis is correct when the trial window is
        larger than the in-coming data window with artificial data."""
        idw_length = ([0.995, 0.8, 0.6, 0.3, 0.1, 0.05]*pq.s).rescale(
            self.time_unit)
        for idw in idw_length:
            for st_type in self.st_types:
                with self.subTest(f"IDW = {idw} | st_type: {st_type}"):
                    self._test_unitary_events_analysis_with_artificial_data(
                        idw_length=idw, time_unit=self.time_unit,
                        st_type=st_type)
                    self.doCleanups()

    def test_trial_window_larger_IDW_real_data(self):
        """Test, if online UE analysis is correct when the trial window is
                larger than the in-coming data window with real data."""
        idw_length = ([2.05, 2., 1.1, 0.8, 0.1, 0.05]*pq.s).rescale(
            self.time_unit)
        for idw in idw_length:
            for st_type in self.st_types:
                with self.subTest(f"IDW = {idw} | st_type: {st_type}"):
                    self._test_unitary_events_analysis_with_real_data(
                        idw_length=idw, time_unit=self.time_unit,
                        st_type=st_type)
                    self.doCleanups()

    # test: trial window = incoming data window
    def test_trial_window_as_large_as_IDW_real_data(self):
        """Test, if online UE analysis is correct when the trial window is
                as large as the in-coming data window with real data."""
        idw_length = (2.1*pq.s).rescale(self.time_unit)
        for st_type in self.st_types:
            with self.subTest(f"IDW = {idw_length} | st_type: {st_type}"):
                self._test_unitary_events_analysis_with_real_data(
                    idw_length=idw_length, time_unit=self.time_unit,
                    st_type=st_type)
                self.doCleanups()

    def test_trial_window_as_large_as_IDW_artificial_data(self):
        """Test, if online UE analysis is correct when the trial window is
                as large as the in-coming data window with artificial data."""
        idw_length = (1*pq.s).rescale(self.time_unit)
        for st_type in self.st_types:
            with self.subTest(f"IDW = {idw_length} | st_type: {st_type}"):
                self._test_unitary_events_analysis_with_artificial_data(
                    idw_length=idw_length, time_unit=self.time_unit,
                    st_type=st_type)
                self.doCleanups()

    # test: trial window < incoming data window
    def test_trial_window_smaller_IDW_artificial_data(self):
        """Test, if online UE analysis is correct when the trial window is
        smaller than the in-coming data window with artificial data."""
        idw_length = ([1.05, 1.1, 2, 10, 50, 100]*pq.s).rescale(self.time_unit)
        for idw in idw_length:
            for st_type in self.st_types:
                with self.subTest(f"IDW = {idw} | st_type: {st_type}"):
                    self._test_unitary_events_analysis_with_artificial_data(
                        idw_length=idw, time_unit=self.time_unit,
                        st_type=st_type)
                    self.doCleanups()

    def test_trial_window_smaller_IDW_real_data(self):
        """Test, if online UE analysis is correct when the trial window is
                smaller than the in-coming data window with real data."""
        idw_length = ([2.15, 2.2, 3, 10, 50, 75.6]*pq.s).rescale(
            self.time_unit)
        for idw in idw_length:
            for st_type in self.st_types:
                with self.subTest(f"IDW = {idw} | st_type: {st_type}"):
                    self._test_unitary_events_analysis_with_real_data(
                        idw_length=idw, time_unit=self.time_unit,
                        st_type=st_type)
                    self.doCleanups()

    def test_pass_trigger_events_while_buffered_reading_real_data(self):
        idw_length = (2.1*pq.s).rescale(self.time_unit)
        for st_type in self.st_types:
            with self.subTest(f"IDW = {idw_length} | st_type: {st_type}"):
                self._test_unitary_events_analysis_with_real_data(
                    idw_length=idw_length,
                    method="pass_events_while_buffered_reading",
                    time_unit=self.time_unit, st_type=st_type)
                self.doCleanups()

    def test_pass_trigger_events_while_buffered_reading_artificial_data(self):
        idw_length = (1*pq.s).rescale(self.time_unit)
        for st_type in self.st_types:
            with self.subTest(f"IDW = {idw_length} | st_type: {st_type}"):
                self._test_unitary_events_analysis_with_artificial_data(
                    idw_length=idw_length,
                    method="pass_events_while_buffered_reading",
                    time_unit=self.time_unit, st_type=st_type)
                self.doCleanups()

    def test_reset(self):
        idw_length = (2.1*pq.s).rescale(self.time_unit)
        with self.subTest(f"IDW = {idw_length}"):
            ouea = self._test_unitary_events_analysis_with_real_data(
                idw_length=idw_length, time_unit=self.time_unit)
            self.doCleanups()
        # do reset with default parameters
        ouea.reset()
        # check all class attributes
        with self.subTest(f"check 'bw_size'"):
            self.assertEqual(ouea.bin_window_size, 0.005 * pq.s)
        with self.subTest(f"check 'trigger_events'"):
            self.assertEqual(ouea.trigger_events, [])
        with self.subTest(f"check 'trigger_pre_size'"):
            self.assertEqual(ouea.trigger_pre_size, 0.5 * pq.s)
        with self.subTest(f"check 'trigger_post_size'"):
            self.assertEqual(ouea.trigger_post_size, 0.5 * pq.s)
        with self.subTest(f"check 'sliding_analysis_window_size'"):
            self.assertEqual(ouea.sliding_analysis_window_size, 0.1 * pq.s)
        with self.subTest(f"check 'sliding_analysis_window_step'"):
            self.assertEqual(ouea.sliding_analysis_window_step, 0.005 * pq.s)
        with self.subTest(f"check 'n_neurons'"):
            self.assertEqual(ouea.n_neurons, 2)
        with self.subTest(f"check 'pattern_hash'"):
            self.assertEqual(ouea.pattern_hash, [3])
        with self.subTest(f"check 'time_unit'"):
            self.assertEqual(ouea.time_unit, 1*pq.s)
        with self.subTest(f"check 'save_n_trials'"):
            self.assertEqual(ouea.save_n_trials, None)
        with self.subTest(f"check 'data_available_in_memory_window'"):
            self.assertEqual(ouea.data_available_in_memory_window, None)
        with self.subTest(f"check 'waiting_for_new_trigger'"):
            self.assertEqual(ouea.waiting_for_new_trigger, True)
        with self.subTest(f"check 'trigger_events_left_over'"):
            self.assertEqual(ouea.trigger_events_left_over, True)
        with self.subTest(f"check 'mw'"):
            np.testing.assert_equal(ouea.memory_window, [[] for _ in range(2)])
        with self.subTest(f"check 'tw_size'"):
            self.assertEqual(ouea.trial_window_size, 1 * pq.s)
        with self.subTest(f"check 'tw'"):
            np.testing.assert_equal(ouea.trial_window, [[] for _ in range(2)])
        with self.subTest(f"check 'tw_counter'"):
            self.assertEqual(ouea.trial_counter, 0)
        with self.subTest(f"check 'n_bins'"):
            self.assertEqual(ouea.n_bins, None)
        with self.subTest(f"check 'bw'"):
            self.assertEqual(ouea.bin_window, None)
        with self.subTest(f"check 'sliding_analysis_window_pos_counter'"):
            self.assertEqual(ouea.sliding_analysis_window_position, 0)
        with self.subTest(f"check 'n_windows'"):
            self.assertEqual(ouea.n_sliding_analysis_windows, 181)
        with self.subTest(f"check 'n_trials'"):
            self.assertEqual(ouea.n_trials, 0)
        with self.subTest(f"check 'n_hashes'"):
            self.assertEqual(ouea.n_hashes, 1)
        with self.subTest(f"check 'method'"):
            self.assertEqual(ouea.method, 'analytic_TrialByTrial')
        with self.subTest(f"check 'n_surrogates'"):
            self.assertEqual(ouea.n_surrogates, 100)
        with self.subTest(f"check 'input_parameters'"):
            self.assertEqual(ouea.input_parameters["pattern_hash"], [3])
            self.assertEqual(ouea.input_parameters["bin_size"], 5 * pq.ms)
            self.assertEqual(ouea.input_parameters["win_size"], 100 * pq.ms)
            self.assertEqual(ouea.input_parameters["win_step"], 5 * pq.ms)
            self.assertEqual(ouea.input_parameters["method"],
                             'analytic_TrialByTrial')
            self.assertEqual(ouea.input_parameters["t_start"], 0 * pq.s)
            self.assertEqual(ouea.input_parameters["t_stop"], 1 * pq.s)
            self.assertEqual(ouea.input_parameters["n_surrogates"], 100)
        with self.subTest(f"check 'Js'"):
            np.testing.assert_equal(ouea.Js, np.zeros((181, 1),
                                                      dtype=np.float64))
        with self.subTest(f"check 'n_exp'"):
            np.testing.assert_equal(ouea.n_exp, np.zeros((181, 1),
                                                         dtype=np.float64))
        with self.subTest(f"check 'n_emp'"):
            np.testing.assert_equal(ouea.n_emp, np.zeros((181, 1),
                                                         dtype=np.float64))
        with self.subTest(f"check 'rate_avg'"):
            np.testing.assert_equal(ouea.rate_avg, np.zeros((181, 1, 2),
                                                            dtype=np.float64))
        with self.subTest(f"check 'indices'"):
            np.testing.assert_equal(ouea.indices, defaultdict(list))


if __name__ == '__main__':
    unittest.main()
