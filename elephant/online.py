import warnings
from collections import defaultdict, deque

import neo
import numpy as np
import quantities as pq
import scipy.special as sc

import elephant.conversion as conv
from elephant.unitary_event_analysis import hash_from_pattern, jointJ
from elephant.unitary_event_analysis import _winpos, _bintime, _UE


class OnlineUnitaryEventAnalysis:
    """
    Online version of the unitary event analysis (UEA).

    This class facilitates methods to perform the unitary event analysis in an
    online manner, i.e. data generation and data analysis happen concurrently.
    The attributes of this class are eiter partial results or descriptive
    parameters of the UEA.

    Parameters
    ----------
    bin_window_size : pq.Quantity
        Size of the bin window, which is used to bin the spike trains.
    trigger_events : pq.Quantity
        Quantity array of time points around which the trials are defined.
        The time interval of a trial is defined as follows:
        [trigger_event - trigger_pre_size, trigger_event + trigger_post_size]
    trigger_pre_size : pq.Quantity
        Interval size before the trigger event. It is used with
        'trigger_post_size' to define the trial.
    trigger_post_size : pq.Quantity
        Interval size after the trigger event. It is used with
        'trigger_pre_size' to define the trial.
    sliding_analysis_window_size : pq.Quantity
        Size of the sliding analysis window, which is used to perform the UEA
        on the trial segments. It advances with a step size defined by
        'sliding_analysis_window_step'.
    sliding_analysis_window_step : pq.Quantity
        Size of the step which is used to advance the sliding analysis window
        to its next position / next trial segment to analyze.
    n_neurons : int
        Number of neurons which are analyzed with the UEA.
    pattern_hash : int or list of int or None
        A list of interested patterns in hash values (see `hash_from_pattern`
        and `inverse_hash_from_pattern` functions in
        'elephant.unitary_event_analysis'). If None, all neurons are
        participated.
        Default: None
    time_unit : pq.Quantity
        This time unit is used for all calculations which requires a time
        quantity. (Default: [s])
    save_n_trials : (positive) int
        The number of trials `n` which will be saved after their analysis with
        a queue following the FIFO strategy (first in, first out), i.e. only
        the last `n` analyzed trials will be stored. (default: None)

    Attributes
    ----------
    data_available_in_memory_window : boolean
        Reflects the status of spike trains in the memory window. It is `True',
        when spike trains are in the memory window which were not yet analyzed.
        Otherwise, it is `False'.
    waiting_for_new_trigger : boolean
        Reflects the status of the updating-algorithm in 'update_uea()'.
        It is `True`, when the algorithm is in the state of pre- / post-trial
        analysis, i.e. it expects the arrival of the next trigger event which
        will define the next trial. Otherwise, it is `False`, when the
        algorithm is within the analysis of the current trial, i.e. it does
        not need the next trigger event at this moment.
    trigger_events_left_over : boolean
        Reflects the status of the trial defining events in the
        'trigger_events' list. It is `True`, when there are events left, which
        were not analyzed yet. Otherwise, it is `False`.
    memory_window : list of lists
        Contains for each neuron the spikes which are currently available in
        the memory window.
            * 0-axis --> Neurons
            * 1-axis --> Spike times
    trial_window_size : pq.Quantity
        The size of the trial window. It is the sum of 'trigger_pre_size' and
        'trigger_post_size'.
    trial_window : list of lists
        Contains for each neuron the spikes which belong to the current trial
        and are available in the memory window.
            * 0-axis --> Neurons
            * 1-axis --> Spike times
    trial_counter : int
        Counts how many trails are yet analyzed.
    n_bins : int
        Number of bins which are used for the binning of the spikes of a trial.
    bin_window : np.array of booleans
        A binned representation of the current trial window. `True` indicates
        the presence of a spike in the bin and `False` indicates absence of
        a spike.
        * 0-axis --> Neurons
        * 1-axis --> Index position of the bin
    sliding_analysis_window_position : int
        Represents the current position of the sliding analysis window.
    n_sliding_analysis_windows : int
        Total number of positions of the sliding analysis window.
    n_trials : int
        Total number of trials to analyze.
    n_hashes: int
        Number of patterns (coded as hashes) to be analyzed. (see also
        'elephant.unitary_event_analysis.hash_from_pattern()')
    method : string
        The method with which to compute the unitary events:
            * 'analytic_TrialByTrial': calculate the analytical expectancy
                on each trial, then sum over all trials
            * 'analytic_TrialAverage': calculate the expectancy by averaging
                over trials (cf. Gruen et al. 2003);
            * 'surrogate_TrialByTrial': calculate the distribution of expected
                coincidences by spike time randomization in each trial and
                sum over trials
        'analytic_TrialAverage' and 'surrogate_TrialByTrial' are not supported
        yet.
        Default: 'analytic_trialByTrial'
    n_surrogates : int
        Number of surrogates which would be used when 'surrogate_TrialByTrial'
        is chosen as 'method'. Yet 'surrogate_TrialByTrial' is not supported.
    input_parameters : dict
        Dictionary of the input parameters which would be used for calling
        the offline version of UEA for the same data to get the same results.
    Js : np.ndarray
        JointSurprise of different given patterns within each window.
            * 0-axis --> different window
            * 1-axis --> different pattern hash
    n_exp : np.ndarray
        The expected number of coincidences of each pattern within each window.
            * 0-axis --> different window
            * 1-axis --> different pattern hash
    n_emp : np.ndarray
        The empirical number of coincidences of each pattern within each
        window.
            * 0-axis --> different window
            * 1-axis --> different pattern hash
    rate_avg : np.ndarray
        The average firing rate of each neuron of each pattern within
        each window.
            * 0-axis --> different window
            * 1-axis --> different pattern hash
            * 2-axis --> different neuron
    indices : defaultdict
        Dictionary contains for each trial the indices of pattern
        within each window.

    Methods
    -------
    get_results()
        Returns the result dictionary with the following class attribute names
        as keys and the corresponding attribute values as the complementary
        value for the key: (see also Attributes section for respective key
        descriptions)
        * 'Js'
        * 'indices'
        * 'n_emp'
        * 'n_exp'
        * 'rate_avg'
        * 'input_parameters'
    update_uea(spiketrains, events)
        Updates the entries of the result dictionary by processing the
        new arriving 'spiketrains' and trial defining trigger 'events'.
    reset(bin_window_size, trigger_events, trigger_pre_size, trigger_post_size,
            sliding_analysis_window_size, sliding_analysis_window_step,
            n_neurons, pattern_hash, time_unit, save_n_trials)
        Resets all class attributes to their initial (default) value. It is
        actually a re-initialization which allows parameter adjustments.
    get_all_saved_trials()
        Returns the last 'n'-trials which were analyzed according to the FIFO
        strategy (first in, first out).

    Returns
    -------
    see 'get_results()' in Methods section

    """

    def __init__(self, bin_window_size=0.005, trigger_events=None,
                 trigger_pre_size=0.5, trigger_post_size=0.5,
                 sliding_analysis_window_size=0.1,
                 sliding_analysis_window_step=0.005, n_neurons=2,
                 pattern_hash=None, time_unit='s', save_n_trials=None):
        """
        Constructor. Initializes all attributes of the new instance.
        """
        # state controlling booleans for the updating algorithm
        self.data_available_in_memory_window = None
        self.waiting_for_new_trigger = True
        self.trigger_events_left_over = True

        # save constructor parameters
        if time_unit not in ['s', 'ms'] and time_unit not in [pq.s, pq.ms]:
            warnings.warn(message=f"Unusual time units like {time_unit} can "
                                  f"cause numerical imprecise results. "
                                  f"Use `ms` or `s` instead!",
                          category=UserWarning)
        self.time_unit = pq.Quantity(1, time_unit)
        self.bin_window_size = pq.Quantity(bin_window_size, self.time_unit)
        if trigger_events is None:
            self.trigger_events = []
        else:
            self.trigger_events = pq.Quantity(trigger_events,
                                              self.time_unit).tolist()
        self.trigger_pre_size = pq.Quantity(trigger_pre_size, self.time_unit)
        self.trigger_post_size = pq.Quantity(trigger_post_size, self.time_unit)
        self.sliding_analysis_window_size = \
            pq.Quantity(sliding_analysis_window_size, self.time_unit)
        self.sliding_analysis_window_step = \
            pq.Quantity(sliding_analysis_window_step, self.time_unit)
        self.n_neurons = n_neurons
        if pattern_hash is None:
            pattern = [1] * n_neurons
            self.pattern_hash = hash_from_pattern(pattern)
        if np.issubdtype(type(self.pattern_hash), np.integer):
            self.pattern_hash = [int(self.pattern_hash)]
        self.save_n_trials = save_n_trials

        # initialize the memory window
        self.memory_window = [[] for _ in range(self.n_neurons)]

        # initialize the trial window and helper variables
        self.trial_window_size = self.trigger_pre_size + self.trigger_post_size
        self.trial_window = [[] for _ in range(self.n_neurons)]
        self.trial_counter = 0

        # initialize the bin window and helper variable
        self.n_bins = None
        self.bin_window = None

        # initialize the sliding analysis window and helper variable
        self.sliding_analysis_window_position = 0
        self.n_sliding_analysis_windows = \
            int(np.round((self.trial_window_size -
                          self.sliding_analysis_window_size +
                          self.sliding_analysis_window_step) /
                         self.sliding_analysis_window_step))

        # determine the number trials and the number of patterns (hashes)
        self.n_trials = len(self.trigger_events)
        self.n_hashes = len(self.pattern_hash)
        # (optional) save last `n` analysed trials for visualization
        if self.save_n_trials is not None:
            self.all_trials = deque(maxlen=self.save_n_trials)

        # save input parameters as dict like the offline version of UEA it does
        # to facilitate a later comparison of the used parameters
        self.method = 'analytic_TrialByTrial'
        self.n_surrogates = 100
        self.input_parameters = \
            dict(pattern_hash=self.pattern_hash,
                 bin_size=self.bin_window_size.rescale(pq.ms),
                 win_size=self.sliding_analysis_window_size.rescale(pq.ms),
                 win_step=self.sliding_analysis_window_step.rescale(pq.ms),
                 method=self.method,
                 t_start=0 * self.time_unit,
                 t_stop=self.trial_window_size,
                 n_surrogates=self.n_surrogates)

        # initialize the intermediate result arrays for the joint surprise
        # (Js), number of expected coincidences (n_exp), number of empirically
        # found coincidences (n_emp), rate average of the analyzed neurons
        # (rate_avg), as well as the indices of the sliding analysis window
        # position where coincidences appear
        self.Js, self.n_exp, self.n_emp = np.zeros(
            (3, self.n_sliding_analysis_windows, self.n_hashes),
            dtype=np.float64)
        self.rate_avg = np.zeros(
            (self.n_sliding_analysis_windows, self.n_hashes, self.n_neurons),
            dtype=np.float64)
        self.indices = defaultdict(list)

    def get_all_saved_trials(self):
        """
        Return the last `n`-trials which were analyzed.

        `n` is the number of trials which were saved after their analysis
        using a queue with the FIFO strategy (first in, first out).

        Returns
        -------
            : list of list of neo.SpikeTrain
            A nested list of trials, neurons and their neo.SpikeTrain objects,
            respectively.

        """
        return list(self.all_trials)

    def get_results(self):
        """
        Return result dictionary.

        Prepares the dictionary entries by reshaping them into the correct
        shape with the correct datatype. Moreover, it calculates 'Js' (could
        not be updated continuously) and scales 'n_exp' and 'rate_avg'.

        Returns
        -------
            : dict
            Dictionary with the following class attribute names
            as keys and the corresponding attribute values as the complementary
            value for the key: (see also Attributes section for respective key
            descriptions)
            * 'Js'
            * 'indices'
            * 'n_emp'
            * 'n_exp'
            * 'rate_avg'
            * 'input_parameters'

        """
        for key in self.indices.keys():
            self.indices[key] = np.hstack(self.indices[key]).flatten()
        self.n_exp /= (self.sliding_analysis_window_size /
                       self.bin_window_size)
        p = self._pval(self.n_emp.astype(np.float64),
                       self.n_exp.astype(np.float64)).flatten()
        self.Js = jointJ(p)
        self.rate_avg = self.rate_avg / (self.bin_window_size *
                                         self.trial_counter)
        return {
            'Js': self.Js.reshape((self.n_sliding_analysis_windows,
                                   self.n_hashes)).astype(np.float32),
            'indices': self.indices,
            'n_emp': self.n_emp.reshape((self.n_sliding_analysis_windows,
                                         self.n_hashes)).astype(np.float32),
            'n_exp': self.n_exp.reshape((self.n_sliding_analysis_windows,
                                         self.n_hashes)).astype(np.float32),
            'rate_avg': self.rate_avg.reshape(
                (self.n_sliding_analysis_windows,
                 self.n_hashes, self.n_neurons)).astype(np.float32),
            'input_parameters': self.input_parameters}

    def reset(self, bin_window_size=0.005 * pq.s, trigger_events=None,
              trigger_pre_size=0.5 * pq.s, trigger_post_size=0.5 * pq.s,
              sliding_analysis_window_size=0.1 * pq.s,
              sliding_analysis_window_step=0.005 * pq.s, n_neurons=2,
              pattern_hash=None, time_unit=1 * pq.s, save_n_trials=None):
        """
        Resets all class attributes to their initial value.

        This reset is actually a re-initialization which allows parameter
        adjustments, so that one instance of 'OnlineUnitaryEventAnalysis' can
        be flexibly adjusted to changing experimental circumstances.

        Parameters
        ----------
        (same as for the constructor; see docstring of constructor for details)

        """
        self.__init__(bin_window_size, trigger_events, trigger_pre_size,
                      trigger_post_size, sliding_analysis_window_size,
                      sliding_analysis_window_step, n_neurons,
                      pattern_hash, time_unit, save_n_trials)

    def update_uea(self, spiketrains, events=None, t_start=None, t_stop=None,
                   time_unit=None):
        """
        Update unitary event analysis (UEA) with new arriving spike data from
        the incoming data window (incoming data window).

        This method orchestrates the updating process. It saves the incoming
        'spiketrains' into the memory window (MW) and adds also the new
        trigger 'events' into the 'trigger_events' list. Then depending on
        the state in which the algorithm is, it processes the new
        'spiketrains' respectively. There are two major states with each two
        substates between the algorithm is switching.

        Parameters
        ----------
        spiketrains : list of neo.SpikeTrain or list of numpy.ndarray
            Spike times of the analysed neurons.
        events : list of pq.Quantity or list of numpy.ndarray
            Time points of the trial defining trigger events.
        t_start : float
            Start time of the incoming data window.
            Required if 'spiketrains' is a list of numpy.ndarray.
        t_stop : float
            Stop time of the incoming data window.
            Required if 'spiketrains' is a list of numpy.ndarray.
        time_unit : string
            Name of the time unit used for representing the spikes in the
            incoming data window, e.g. 's', 'ms'.

        Warns
        -----
        UserWarning
            * if an overlap between successive trials exists, spike data
                of these trials will be analysed twice. The user should adjust
                the trigger events and/or the trial window size to increase
                the interval between successive trials to avoid an overlap.

        Notes
        -----
        Short summary of the different algorithm major states / substates:
        1. pre/post trial analysis: algorithm is waiting for
            incoming data window with new trigger event
            1.1. incoming data window contains new trigger event
            1.2. incoming data window does not contain new trigger event
        2. within trial analysis: algorithm is waiting for incoming data window
            with spikes of current trial
            2.1. incoming data window contains new trigger event
            2.2. incoming data window does not contain new trigger event, it
                just has new spikes of the current trial

        """
        if isinstance(spiketrains[0], neo.SpikeTrain):
            # rescale spiketrains to time_unit
            spiketrains = [st.rescale(self.time_unit)
                           if st.t_start.units == st.units == st.t_stop
                           else st.rescale(st.units).rescale(self.time_unit)
                           for st in spiketrains]
        elif isinstance(spiketrains[0], np.ndarray):
            if t_start is None or t_stop is None or time_unit is None:
                raise ValueError("'spiketrains' is a list of np.array(), thus"
                                 "'t_start', 't_stop' and 'time_unit' must be"
                                 "specified!")

            spiketrains = [neo.SpikeTrain(times=st, t_start=t_start,
                                          t_stop=t_stop, units=time_unit
                                          ).rescale(self.time_unit)
                           for st in spiketrains]
        # extract relevant time information
        t_start_incoming_data_window = spiketrains[0].t_start
        t_stop_incoming_data_window = spiketrains[0].t_stop

        if events is None:
            events = np.array([])
        if len(events) > 0:
            for event in events:
                if isinstance(events, np.ndarray):
                    event = pq.Quantity(event, time_unit)
                if event.rescale(self.time_unit) not in self.trigger_events:
                    self.trigger_events.append(event.rescale(self.time_unit))
            self.trigger_events.sort()
            self.n_trials = len(self.trigger_events)

        # save spikes of incoming data window into memory window
        self._save_incoming_data_window_into_memory_window(spiketrains)

        # analyse all trials which are available in the memory
        self.data_available_in_memory_window = True
        while self.data_available_in_memory_window:
            if self.trial_counter == self.n_trials:
                break
            if self.n_trials == 0:
                current_trigger_event = np.inf * self.time_unit
                next_trigger_event = np.inf * self.time_unit
            else:
                current_trigger_event = self.trigger_events[self.trial_counter]
                if self.trial_counter <= self.n_trials - 2:
                    next_trigger_event = self.trigger_events[
                        self.trial_counter + 1]
                else:
                    next_trigger_event = np.inf * self.time_unit

            # # case 1: pre/post trial analysis,
            # i.e. waiting for incoming data window with new trigger event
            if self.waiting_for_new_trigger:
                # # subcase 1: incoming data window contains trigger event
                if (t_start_incoming_data_window <= current_trigger_event) & \
                        (current_trigger_event <= t_stop_incoming_data_window):
                    self.waiting_for_new_trigger = False
                    if self.trigger_events_left_over:
                        # define trial window around trigger event
                        self._define_trial_window(
                            trigger_event=current_trigger_event)
                        # apply bin window to available data in trial window
                        self._binning_of_trial_window()
                        # move sliding analysis window over available data
                        # in trial window
                        self._move_sliding_analysis_window_over_trial_window(
                            t_stop_incoming_data_window=
                            t_stop_incoming_data_window)

                # # subcase 2: incoming data window does not contain
                # trigger event
                else:
                    self._move_memory_window(
                        new_t_start=t_stop_incoming_data_window -
                        self.trigger_pre_size)

            # # Case 2: within trial analysis, i.e. waiting for
            # new incoming data window with spikes of current trial
            else:
                # # Subcase 3: incoming data window contains new trigger event
                if (t_start_incoming_data_window <= next_trigger_event) & \
                        (next_trigger_event <= t_stop_incoming_data_window):
                    # check if an overlap between current / next trial exists
                    if self._check_trial_window_overlap(
                            current_trigger_event=current_trigger_event,
                            next_trigger_event=next_trigger_event):
                        warnings.warn(
                            f"Data in trial {self.trial_counter} will be "
                            f"analysed twice! Adjust the trigger events and/or"
                            f" the trial window size.", UserWarning)
                    else:  # no overlap exists
                        pass
                # # Subcase 4: incoming data window does not contain
                # trigger event, i.e. just new spikes of the current trial

                if self.trigger_events_left_over:
                    # define trial window around trigger event
                    self._define_trial_window(
                        trigger_event=current_trigger_event)
                    # apply bin window to available data in trial window
                    self._binning_of_trial_window()
                    # move sliding analysis window over available data
                    # in trial window
                    self._move_sliding_analysis_window_over_trial_window(
                        t_stop_incoming_data_window=
                        t_stop_incoming_data_window)

    def _pval(self, n_emp, n_exp):
        """
        Calculates the probability of detecting 'n_emp' or more coincidences
        based on a distribution with sole parameter 'n_exp'.

        To calculate this probability, the upper incomplete gamma function is
        used.

        Parameters
        ----------
        n_emp : int
            Number of empirically observed coincidences.
        n_exp : float
            Number of theoretically expected coincidences.

        Returns
        -------
        p : float
            Probability of finding 'n_emp' or more coincidences based on a
            distribution with sole parameter 'n_exp'

        """
        p = 1. - sc.gammaincc(n_emp, n_exp)
        return p

    def _save_incoming_data_window_into_memory_window(self,
                                                      incoming_data_window):
        """
        Save in-incoming data window into memory window.

        This method appends for each neuron all the spikes which are arriving
        with 'incoming_data_window' into the respective  sub-list of
        'memory_window'.

        Parameters
        ---------
        incoming_data_window : list of pq.Quantity arrays
            * 0-axis --> Neurons
            * 1-axis --> Spike times

        """
        for i in range(self.n_neurons):
            self.memory_window[i] += incoming_data_window[i].tolist()

    def _move_memory_window(self, new_t_start):
        """
        Move memory window.

        This method moves the memory window, i.e. it removes for each neuron
        all the spikes that occurred before the time point 'new_t_start'.
        Spikes which occurred after 'new_t_start' will be kept.

        Parameters
        ----------
        new_t_start : pq.Quantity
            New start point in time of the memory window. Spikes which occurred
            after this time point will be kept, otherwise removed.

        """
        for i in range(self.n_neurons):
            idx = np.where(new_t_start > self.memory_window[i])[0]
            # print(f"idx = {idx}")
            if len(idx) != 0:  # move memory_window
                self.memory_window[i] = self.memory_window[i][idx[-1] + 1:]
            else:  # keep memory_window
                self.data_available_in_memory_window = False

    def _define_trial_window(self, trigger_event):
        """
        Define trial window based on a trigger event.

        This method defines the trial window around the 'trigger_event', i.e.
        it sets the start and stop of the trial, so that it covers the
        following interval:
        [trigger_event - trigger_pre_size, trigger_event + trigger_post_size]
        Then it collects for each neuron all spike times from the memory window
        which are within this interval and puts them into the trial window.

        Parameters
        ----------
        trigger_event : pq.Quantity
            Time point around which the trial will be defined.

        """
        self.trial_start = trigger_event - self.trigger_pre_size
        self.trial_stop = trigger_event + self.trigger_post_size
        for i in range(self.n_neurons):
            self.trial_window[i] = [t for t in self.memory_window[i]
                                    if (self.trial_start <= t) &
                                    (t <= self.trial_stop)]

    def _check_trial_window_overlap(self, current_trigger_event,
                                    next_trigger_event):
        """
        Check if successive trials do overlap each other.

        This method checks whether two successive trials are overlapping
        each other. To do this it compares the stop time of the precedent
        trial and the start time of the subsequent trial. An overlap is present
        if start time of the subsequent trial is before the stop time
        of the precedent trial.

        Parameters
        ----------
        current_trigger_event : pq.Quantity
            Time point around which the current / precedent trial was defined.
        next_trigger_event : pq.Quantity
            Time point around which the next / subsequent trial will be
            defined.

        Returns
        -------
        : boolean
            If an overlap exists, return `True`. Otherwise, `False`.

        """
        if current_trigger_event + self.trigger_post_size > \
                next_trigger_event - self.trigger_pre_size:
            return True
        else:
            return False

    def _binning_of_trial_window(self):
        """
        Apply bin window to trial window.

        Perform the binning and clipping procedure on the trial window, i.e.
        if at least one spike is within a bin, it is occupied and
        if no spike is within a bin, it is empty.

        """
        self.n_bins = int(((self.trial_stop - self.trial_start) /
                           self.bin_window_size).simplified.item())
        self.bin_window = np.zeros((1, self.n_neurons, self.n_bins),
                                   dtype=np.int32)
        spiketrains = [neo.SpikeTrain(np.array(st) * self.time_unit,
                                      t_start=self.trial_start,
                                      t_stop=self.trial_stop)
                       for st in self.trial_window]
        bst = conv.BinnedSpikeTrain(spiketrains, t_start=self.trial_start,
                                    t_stop=self.trial_stop,
                                    bin_size=self.bin_window_size)
        self.bin_window = bst.to_bool_array()

    def _set_sliding_analysis_windows_positions(self, t_start, t_stop,
                                                win_size, win_step, bin_size):
        """
        Set positions of the sliding analysis window.

        Determines the positions of the sliding analysis window with respect to
        the used window size 'win_size' and the advancing step 'win_step'. Also
        converts this time points into bin-units, i.e. into multiple of the
        'bin_size' which facilitates indexing in upcoming analysis steps.

        Parameters
        ----------
        t_start : pq.Quantity
            Time point at which the current trial starts.
        t_stop : pq.Quantity
            Time point at which the current trial ends.
        win_size : pq.Quantity
            Temporal length of the sliding analysis window.
        win_step : pq.Quantity
            Temporal size of the advancing step of the sliding analysis window.
        bin_size : pq.Quantity
            Temporal length of the histogram bins, which were used to bin
            the 'spiketrains' in  '_apply_bw_to_tw()'.

        Warns
        -----
        UserWarning:
            * if the ratio between the 'win_size' and 'bin_size' is not
                an integer
            * if the ratio between the 'win_step' and 'bin_size' is not
                an integer

        """
        self.t_winpos = _winpos(t_start, t_stop, win_size, win_step,
                                position='left-edge')
        while len(self.t_winpos) != self.n_sliding_analysis_windows:
            if len(self.t_winpos) > self.n_sliding_analysis_windows:
                self.t_winpos = _winpos(t_start, t_stop - win_step / 2,
                                        win_size,
                                        win_step, position='left-edge')
            else:
                self.t_winpos = _winpos(t_start, t_stop + win_step / 2,
                                        win_size,
                                        win_step, position='left-edge')
        self.t_winpos_bintime = _bintime(self.t_winpos, bin_size)
        self.winsize_bintime = _bintime(win_size, bin_size)
        self.winstep_bintime = _bintime(win_step, bin_size)
        if self.winsize_bintime * bin_size != win_size:
            warnings.warn(f"The ratio between the win_size ({win_size}) and "
                          f"the bin_size ({bin_size}) is not an integer")
        if self.winstep_bintime * bin_size != win_step:
            warnings.warn(f"The ratio between the win_step ({win_step}) and "
                          f"the bin_size ({bin_size}) is not an integer")

    def _move_sliding_analysis_window_over_trial_window(
            self, t_stop_incoming_data_window):
        """
        Move sliding analysis window over trial window.

        This method iterates over each sliding analysis window position and
        applies at each position the unitary event analysis, i.e. within each
        window it counts the empirically found coincidences and saves their
        indices where they appeared, calculates the expected number of
        coincidences and determines the firing rates of the neurons.
        The respective results are then used to update the class attributes
        'n_emp', 'n_exp', 'rate_avg' and 'indices'.

        Parameters
        ----------
        t_stop_incoming_data_window : pq.Quantity
            Time point at which the current incoming data window ends.

        Notes
        -----
        The 'Js' attribute is not continuously updated, because the
        joint-surprise is determined just when the user calls 'get_results()'.
        This is due to the dependency of the distribution from which 'Js' is
        calculated on the attributes 'n_emp' and 'n_exp'. Updating / changing
        'n_emp' and 'n_exp' changes also this distribution, so that it is not
        possible anymore to simply sum the joint-surprise values of different
        trials at the same sliding analysis window position, because they were
        based on different distributions.

        """
        # define sliding analysis window positions
        self._set_sliding_analysis_windows_positions(
            t_start=self.trial_start, t_stop=self.trial_stop,
            win_size=self.sliding_analysis_window_size,
            win_step=self.sliding_analysis_window_step,
            bin_size=self.bin_window_size)

        # iterate over sliding analysis window positions
        for i in range(self.sliding_analysis_window_position,
                       self.n_sliding_analysis_windows):
            p_realtime = self.t_winpos[i]
            p_bintime = self.t_winpos_bintime[i] - self.t_winpos_bintime[0]
            # check if sliding analysis window is filled with data
            # case 1: sliding analysis window is filled
            if p_realtime + self.sliding_analysis_window_size <= \
                    t_stop_incoming_data_window:
                mat_win = np.zeros((1, self.n_neurons, self.winsize_bintime))
                n_bins_in_current_sliding_analysis_window = self.bin_window[
                    :, p_bintime:p_bintime + self.winsize_bintime].shape[1]
                if n_bins_in_current_sliding_analysis_window < \
                        self.winsize_bintime:
                    mat_win[0] += np.pad(
                        self.bin_window[:, p_bintime:p_bintime +
                                        self.winsize_bintime],
                        (0, self.winsize_bintime -
                         n_bins_in_current_sliding_analysis_window),
                        "minimum")[0:2]
                else:
                    mat_win[0] += \
                        self.bin_window[
                        :, p_bintime:p_bintime + self.winsize_bintime]
                Js_win, rate_avg, n_exp_win, n_emp_win, indices_lst = _UE(
                    mat_win, pattern_hash=self.pattern_hash,
                    method=self.method, n_surrogates=self.n_surrogates)
                self.rate_avg[i] += rate_avg
                self.n_exp[i] += (np.round(
                    n_exp_win * (self.sliding_analysis_window_size /
                                 self.bin_window_size))).astype(int)
                self.n_emp[i] += n_emp_win
                self.indices_lst = indices_lst
                if len(self.indices_lst[0]) > 0:
                    self.indices[f"trial{self.trial_counter}"].append(
                        self.indices_lst[0] + p_bintime)
            # case 2: sliding analysis window is empty / partially filled
            else:  # -> pause iteration
                self.sliding_analysis_window_position = i
                self.data_available_in_memory_window = False
                break
            # last sliding analysis window position finished
            if i == self.n_sliding_analysis_windows - 1:
                self.sliding_analysis_window_position = 0
                # move memory window after sliding analysis window is finished
                #  with analysis of one trial
                self._move_memory_window(
                    new_t_start=self.trigger_events[self.trial_counter]
                    + self.trigger_post_size)
                # save analysed trial for visualization
                if self.save_n_trials:
                    _trial_start = 0 * pq.s
                    _trial_stop = self.trial_window_size
                    _offset = self.trigger_events[self.trial_counter] - \
                        self.trigger_pre_size
                    normalized_spike_times = []
                    for n in range(self.n_neurons):
                        normalized_spike_times.append(
                            np.array(self.trial_window[n]) * self.time_unit -
                            _offset)
                    self.all_trials.append(
                        [neo.SpikeTrain(normalized_spike_times[m],
                                        t_start=_trial_start,
                                        t_stop=_trial_stop,
                                        units=self.time_unit)
                         for m in range(self.n_neurons)])
                # reset bin window
                self.bin_window = np.zeros_like(self.bin_window)
                if self.trial_counter <= self.n_trials - 1:
                    self.trial_counter += 1
                else:
                    self.waiting_for_new_trigger = True
                    self.trigger_events_left_over = False
                    self.data_available_in_memory_window = False
                # print(f"trial_counter = {self.trial_counter}")  # DEBUG-aid
