import numpy as np
from neo.core import Block, Segment, SpikeTrain, AnalogSignal, Epoch, \
    RecordingChannel, RecordingChannelGroup, Unit


class NeoInfo(object):
    """
    This class introduces a trial concept for elephant based on the Neo
    framework, and provides general information about Neo objects.

    In its simplest use, NeoInfo extracts information about arbitrary Neo
    objects. From the given input it traverses the hierarchical Neo tree and
    provides methods to extract its objects.

    The main focus is to define the concept of trials in the elephant project
    and link trials to objects of the Neo framework. Trials can be represented
    in a number of different ways. The purpose of this class is to simplify
    interaction with these diverse trial representations, and propose a unified
    (and tested) approach to access trial-cut data. Internally, each trial is
    assigned a unique integer trial ID, which can be used to access the data of
    that trial.

    Multiple trials may be represented as individual **neo.core.Segment**
    objects attached to a **neo.core.Block**. By default, the order of trials,
    given by the trial ID, is given by the order in which they appear in
    **neo.core.Block.segments**.

    A single trial may be represented by a single **neo.core.Segment**, a
    single **neo.core.AnalogSignal** or **neo.core.SpikeTrain** object, or a
    list of **neo.core.AnalogSignal** or **neo.core.SpikeTrain** objects. The
    trial ID of a single trial is always 0.

    The class provides methods to extract the data objects that belong to a
    given trial. In addition, the class allows to define conditions on each
    individual trial that must be met in order for the trial to be considered,
    referred to as a **valid trial**. For example, it is possible to require
    that each valid trial has a specific number of **neo.core.SpikeTrain**
    objects. Given a set of conditions, the class is able to identify those
    trials for which all conditions are met, and extract its objects. For a
    list of currently supported conditions, see **set_trial_conditions()**.

    In addition, the class can test whether the complete set of trials (or
    valid trials) fulfills a certain criterion, e.g., if all trials contain
    data of equal length.

    See also
    --------
    set_trial_conditions()

    Notes
    -----
    TODOs -- to be discussed!:
        * allow for different trial orders based on
            * the index property of segments
            * an arbitrary annotation (with sortable values)
        * allow for a list of Segments
        * allow lists of SpikeTrain or AnalogSignals to be interpreted as
          trials containing one signal each
        * allow for other list-based or dict-based representations of trials
        * allow for representations of trials as single segment or single
          AnalogSignal or single SpikeTrain, together with an Epoch?
        * add more conditions
        * improve error checking for setting conditions

    """

    def __init__(self, x):
        self._input = x
        # Bools for input defining
        self._is_block = False
        self._is_segment = False
        # SpikeTrain or list of SpikeTrains
        self._is_spike_train = False
        self._is_spike_train_lst = False
        # AnalogSignal or list of AnalogSignals
        self._is_analog_signal = False
        self._is_analog_signal_lst = False
        self._is_epoch = False
        self._is_epoch_lst = False
        self._is_recording_channel = False
        self._is_recording_channel_group = False
        self._is_unit = False
        # #########################################
        # Dictionary containing the conditions
        self.d_conditions = dict()
        # Dictionary which stores the invalid trials
        self.d_invalid_trials = {}
        # Set default conditions
        self.__set_default_conditions()
        # #########################################
        # List containing valid trial ids
        self.__valid_trials = None
        # Check the type of input now
        self.__check_input_type()
        # Set conditions now to default state and init the dictionary
        self.set_trial_conditions()

    def __check_input_type(self):
        """
        Checks what type the given input is and stores the result in a private
        boolean variable.

        The following types are supported:
            neo.Block
            neo.Unit
            neo.Segment
            neo.SpikeTrain
            neo.AnalogSignal
            neo.Epoch
            neo.RecordingChannel
            neo.RecordingChannelGroup
            List of neo.SpikeTrain objects
            List of neo.AnalogSignal objects

        Raises
        ------
        TypeError:
            If no input is given, or the input is not known.

        """
        if type(self._input) is Block:
            self._is_block = True
        elif type(self._input) is Segment:
            self._is_segment = True
        elif type(self._input) is list:
            if type(self._input[0]) is SpikeTrain:
                self._is_spike_train_lst = True
            elif type(self._input[0]) is AnalogSignal:
                self._is_analog_signal_lst = True
            elif type(self._input[0]) is Epoch:
                self._is_epoch_lst = True
        elif type(self._input) is SpikeTrain:
            self._is_spike_train = True
        elif type(self._input) is AnalogSignal:
            self._is_analog_signal = True
        elif type(self._input) is Epoch:
            self._is_epoch = True
        elif type(self._input) is RecordingChannel:
            self._is_recording_channel = True
        elif type(self._input) is RecordingChannelGroup:
            self._is_recording_channel_group = True
        elif type(self._input) is Unit:
            self._is_unit = True
        elif type(self._input) is None:
            raise TypeError('No input given.')
        else:
            raise TypeError('No known input.')

    def get_input_type(self):
        """
        Returns the type of given input.

        Returns the type of given input as a String.
        The following types are supported:
            neo.Block
            neo.Unit
            neo.Segment
            neo.SpikeTrain
            neo.AnalogSignal
            neo.Epoch
            neo.RecordingChannel
            neo.RecordingChannelGroup
            List of neo.SpikeTrain objects
            List of neo.AnalogSignal objects

        Returns
        -------
        out : String
            Each of the above mentioned types do have a String:
            neo.Block : "Block"
            neo.Unit : "Unit"
            neo.Segment : "Segment"
            neo.SpikeTrain : "SpikeTrain"
            neo.AnalogSignal : "AnalogSignal"
            neo.Epoch : "Epoch"
            neo.RecordingChannel : "RecordingChannel"
            neo.RecordingChannelGroup : "RecordingChannelGroup"
            List of neo.SpikeTrain objects : 'SpikeTrain List"
            List of neo.AnalogSignal objects : "AnalogSignal List"

        """
        if self._is_block:
            return "Block"
        elif self._is_segment:
            return "Segment"
        elif self._is_spike_train:
            return "SpikeTrain"
        elif self._is_spike_train_lst:
            return "SpikeTrain List"
        elif self._is_analog_signal:
            return "AnalogSignal"
        elif self._is_analog_signal_lst:
            return "AnalogSignal List"
        elif self._is_epoch:
            if type(self._input) is list:
                return "Epoch List"
            elif type(self._input) is Epoch:
                return "Epoch"
        elif self._is_recording_channel:
            return "RecordingChannel"
        elif self._is_recording_channel_group:
            return "RecordingChannelGroup"
        elif self._is_unit:
            return "Unit"
        else:
            return None

    def __set_default_conditions(self):
        self.d_conditions = {
            "trial_has_n_st": (False, 0),
            "trial_has_n_as": (False, 0),
            "trial_has_n_units": (False, 0),
            "trial_has_exact_st": (False, 0),
            "trial_has_exact_as": (False, 0),
            "trial_has_n_rc": (False, 0),
            "trial_has_no_overlap": (False, 0),
            "each_st_has_n_spikes": (False, 0),
            "contains_each_unit": (False, 0),
            "contains_each_rc": (False, 0),
            "contains_each_rcg": False,
            "data_aligned": (False, 0)
        }
        # Set the dict for invalid trials
        self.d_invalid_trials = dict.fromkeys(self.d_conditions.keys(), [])

    def get_trial_conditions(self):
        """
        Returns the trial conditions.

        Returns
        -------
        dict : {}
            Returns a dictionary with the trial conditions. The values of the
            dictionary are Boolean and Tuple of Boolean and Integer.
            If no parameter were changed by user input, the default parameter
            will be returned.

        See Also
        --------
        elephant.core.NeoInfo.set_trial_conditions : Sets the trial conditions.
        """
        return self.d_conditions

    def set_trial_conditions(self, **kwargs):
        """
        Sets the trial conditions.

        Trial conditions determine which trials are valid, and which are not
        valid. All conditions can be enabled or disabled. Some conditions have
        an additional parameter.

        The trial conditions are passed as keyword arguments. The value of each
        argument is a tuple. The first entry of the tuple indicates whether the
        condition is enabled (True) or disabled (False). The following entries
        of the tuple are parameters of the condition. If no argument is given,
        the default parameters will be used (all conditions are disabled).

        The available conditions and their default parameters are (key/value):
            :key                    :var
            trial_has_n_st:         (False, 0)
            trial_has_n_as:         (False, 0)
            trial_has_exact_st:     (False, 0)
            trial_has_exact_as:     (False, 0)
            trial_has_n_rc:         (False, 0)
            trial_has_n_units:      (False, 0)
            trial_has_no_overlap:   (False, 0)
            each_st_has_n_spikes:   (False, 0)
            contains_each_unit:     (False,  )
            contains_each_rc:       (False,  )
            data_aligned:           (False,  )

        Detailed description of conditions:
            trial_has_n_st (parameter n):
                Trial is considered if it has n or more **neo.core.SpikeTrain**
                objects.
            trial_has_n_as (parameter n):
                Trial is considered if it has n or more
                **neo.core.AnalogSignal** objects.
            trial_has_exact_st (parameter n):
                Trial is considered if it has exactly n **neo.core.SpikeTrain**
                objects.
            trial_has_exact_as (parameter n):
                Trial is considered if it has exactly n
                **neo.core.AnalogSignal** objects.
            trial_has_n_rc (parameter n):
                Trial is considered if it has n or more
                **neo.core.RecordingChannel** objects.
            trial_has_n_units (parameter n):
                Trial is considered if it has n or more **neo.core.Unit**
                objects.
            trial_has_no_overlap (optional parameter b):
                Trial has no temporal overlap with other trials. If b is True,
                the first trial in a series of overlapping trials is valid. If
                b is False, none of the overlapping trials is valid.
            each_st_has_n_spikes (parameter n):
                Each SpikeTrain of the trial must have n or more spikes.
            each_st_has_exact_spikes (parameter n):
                Each SpikeTrain of the trial must exactly n spikes.
            contains_each_unit:
                The trial contains data of each **neo.core.Unit** of the
                **neo.core.Block**
            contains_each_rc:
                The trial contains data of each **neo.core.RecodingChannel** of
                the **neo.core.Block**
            data_aligned:
                All data in the trial share a common time axis.


        Examples
        --------
        In this example, we show how to use conditions to ensure that a
        trial contains exactly one spike train with 7 spikes or more.

        >>> import neo.core
        >>> import quantities as pq
        >>> blk = neo.Block()
        >>> seg = neo.Segment()
        >>> st = SpikeTrain(
        >>>     [0.5, 0.7, 1.2, 3.1, 4.3, 5.5, 6.7] * pq.s, t_stop=10.0 * pq.s)
        >>> seg.spiketrains.append(st)
        >>> blk.segments.append(seg)
        >>> ni = NeoInfo(blk)
        >>> ni.set_trial_conditions(
        >>>     trial_has_exact_st=(True, 1),
        >>>     each_st_has_n_spikes=(True, 7))

        """
        # Set conditions in dictionary
        for (cond, default) in self.d_conditions.items():
            self.d_conditions[cond] = kwargs.get(cond, default)

        # Now apply the conditions
        self._apply_conditions()

    def reset_trial_conditions(self):
        """
        Resets the trial conditions to default state.

        Resets the values of the trial condition dictionary and the valid
        trials to the default state.
        Resets also the dictionary for invalid trials.

        See also:
        __set_default_conditions : The method reset_trial_condition calls the
        __set_default_conditions() method to reset above mentioned dictionaries
        to default their state.
        """
        self.__valid_trials = None
        self.__set_default_conditions()

    @property
    def valid_trial_ids(self):
        """
        Returns the trial IDs which respect the trial conditions

        Returns
        -------
        valid_trials : list
            Trial IDs corresponding to trial, which respect the trial
            conditions.
        """
        return self.__valid_trials

    @property
    def invalid_trial_ids(self):
        """
        Return the invalid trial IDs.

        Returns a dictionary containing the trial condition as key and a list
        of invalid IDs as value.

        Returns
        -------
        invalid_trial_ids: dict with String as key and list of integers
        as value
            Dictionary containing trial condition as key (string) and a list
            of invalid IDs (int) as value.
        """
        return self.d_invalid_trials

    def _apply_conditions(self):
        """
        Applies the trial conditions.
        This method is called automatically after setting the trial conditions.

        See also
        --------
        elephant.core.NeoInfo.set_trial_conditions()

        Notes
        -----
        This method does not need to be called again after setting the trial
        conditions.

        """
        # Get all trial ids
        if self.__valid_trials is None or not self.valid_trial_ids:
            self.__valid_trials = self.get_trial_ids()
        # Copy valid trial list to compare and to extract the invalid IDs
        trial_ids = self.valid_trial_ids[:]
        # ### Check and get valid trials ###
        # Check for number of spiketrains
        if self.d_conditions["trial_has_n_st"][0]:
            self.__valid_trials = self._intersect(self.__valid_trials,
                                                  self.__check_trial_n_st(
                                                      self.__valid_trials))
            self.d_invalid_trials["trial_has_n_st"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check for number of spikes
        if self.d_conditions["each_st_has_n_spikes"][0]:
            self.__valid_trials = self._intersect(self.__valid_trials,
                                                  self.
                                                  __check_each_st_has_n_spikes(
                                                      self.__valid_trials))
            self.d_invalid_trials["each_st_has_n_spikes"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check for number of analogsignals
        if self.d_conditions["trial_has_n_as"][0]:
            self.__valid_trials = self._intersect(self.__valid_trials,
                                                  self.__check_trial_n_as(
                                                      self.__valid_trials))
            self.d_invalid_trials["trial_has_n_as"] = self._difference(
                trial_ids,
                self.__valid_trials)

        # Check for exact numbers of spiketrains
        if self.d_conditions["trial_has_exact_st"][0]:
            self.__valid_trials = self._intersect(self.__valid_trials,
                                                  self.__check_trial_exact_st(
                                                      self.__valid_trials))
            self.d_invalid_trials["trial_has_exact_st"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check for exact numbers of analogsignals
        if self.d_conditions["trial_has_exact_as"][0]:
            self.__valid_trials = self._intersect(self.__valid_trials,
                                                  self.__check_trial_exact_as(
                                                      self.__valid_trials))
            self.d_invalid_trials["trial_has_exact_as"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check for number of units
        if self.d_conditions["trial_has_n_units"][0]:
            self.__valid_trials = self._intersect(
                self.__valid_trials,
                self.__check_trial_has_n_units(self.__valid_trials))
            self.d_invalid_trials["trial_has_n_units"] = self._difference(
                self.d_invalid_trials["trial_has_n_units"],
                self.__valid_trials)

        # Check for number of recording channels
        if self.d_conditions["trial_has_n_rc"][0]:
            self.__valid_trials = self._intersect(
                self.__valid_trials,
                self.__check_trial_has_n_rc(self.__valid_trials))
            self.d_invalid_trials["trial_has_n_rc"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check if trial contains each unit
        if self.d_conditions["contains_each_unit"][0]:
            self.__valid_trials = self._intersect(
                self.__valid_trials,
                self.__check_contains_each_unit(self.__valid_trials))
            self.d_invalid_trials["contains_each_unit"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check if trial contains each recording channel
        if self.d_conditions["contains_each_rc"][0]:
            self.__valid_trials = self._intersect(
                self.__valid_trials,
                self.__check_contains_each_rc(self.__valid_trials))
            self.d_invalid_trials["contains_each_rc"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check if all signals per trial are aligned at the same time line,
        # regarding (start, stop)
        if self.d_conditions["data_aligned"][0]:
            self.__valid_trials = self._intersect(self.__valid_trials,
                                                  self.__check_data_aligned(
                                                      self.__valid_trials))
            self.d_invalid_trials["data_aligned"] = self._difference(
                trial_ids, self.__valid_trials)

        # Check that trial has no overlap with other trials
        if self.d_conditions["trial_has_no_overlap"][0]:
            try:
                if self.d_conditions["trial_has_no_overlap"][1]:
                    pass
            except IndexError:
                self.d_conditions["trial_has_no_overlap"] = (
                    self.d_conditions["trial_has_no_overlap"][0], 1)
            self.__valid_trials = self._intersect(
                self.__valid_trials,
                self.__check_trial_has_no_overlap(
                    self.__valid_trials,
                    self.d_conditions["trial_has_no_overlap"][1]))
            self.d_invalid_trials["trial_has_no_overlap"] = self._difference(
                trial_ids, self.__valid_trials)

    # ###################################
    # ## Private functions ##############
    # ###################################
    def __check_trial_n_st(self, list_trials):
        """
        Check if in given trial(s) the number of neo.SpikeTrain objects are
        greater than `n`.

        Parameters
        ---------
        list_trials: list of int
            List of trial IDs, which will be iterated trough.

        Returns
        -------
        list: list of int
            Trial IDs according to trial condition (equal or more than **n**
            neo.SpikeTrain objects in trial).
        """
        if self.d_conditions["trial_has_n_st"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, when setting "
                "the condition for a minimal number "
                "of SpikeTrains in each trial."
                % str(self.d_conditions["trial_has_n_st"][0]))
        # List of valid trial, will be returned
        st_lst_valid_trials = []
        if self._is_block:
            for tr_id in list_trials:
                seg = self._input.segments[tr_id]
                if len(seg.spiketrains) >= self.d_conditions[
                        "trial_has_n_st"][1]:
                    st_lst_valid_trials.append(tr_id)
        elif self._is_segment:
            if len(self._input.spiketrains) >= \
                    self.d_conditions["trial_has_n_st"][1]:
                # Append only trial with Index 0
                st_lst_valid_trials.append(0)
        elif self._is_spike_train_lst:
            if len(self._input) >= self.d_conditions["trial_has_n_st"][1]:
                st_lst_valid_trials.append(0)
        return st_lst_valid_trials

    def __check_each_st_has_n_spikes(self, list_trials):
        """
        Check if in given SpikeTrain objects the number of spikes are equal
        to `n`.

        Parameters
        ---------
         list_trials: list of int
            List of trial IDs, which will be iterated trough.

        Returns
        -------
        list: list of int
            Trial IDs according to trial condition (equal to **n**
            spikes in trial).
        """
        if self.d_conditions["each_st_has_n_spikes"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, "
                "when setting the condition for a minimal number "
                "of Spikes in each SpikeTrain."
                % str(self.d_conditions["each_st_has_n_spikes"][0]))
        spikes_lst_valid_trials = []
        if self._is_block:
            for tr_id in list_trials:
                seg = self._input.segments[tr_id]
                valid = True
                for st in seg.spiketrains:
                    if np.size(st) != \
                            self.d_conditions["each_st_has_n_spikes"][1]:
                        valid = False
                        break
                if valid:
                    spikes_lst_valid_trials.append(tr_id)
        elif self._is_segment:
            seg = self._input
            valid = True
            for st in seg.spiketrains:
                if np.size(st) != self.d_conditions["each_st_has_n_spikes"][1]:
                    valid = False
                    break
            if valid:
                spikes_lst_valid_trials.append(0)
        elif self._is_spike_train_lst:
            valid = True
            for st in self._input:
                if np.size(st) != self.d_conditions["each_st_has_n_spikes"][1]:
                    valid = False
                    break
            if valid:
                spikes_lst_valid_trials.append(0)
        elif self._is_spike_train:
            if np.size(self._input) == \
                    self.d_conditions["each_st_has_n_spikes"][1]:
                spikes_lst_valid_trials.append(0)
        return spikes_lst_valid_trials

    def __check_trial_n_as(self, list_trials):
        """
        Check if in given trial(s) the number of neo.AnalogSignal objects are
        greater than `n`.

        Parameters
        ---------
         list_trials: list of int
            List of trial IDs, which will be iterated trough.

        Returns
        -------
        list: list of int
            Trial IDs according to trial condition (more than **n**
            neo.AnalogSignal objects in trial).
        """
        if self.d_conditions["trial_has_n_as"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, "
                "when setting the condition for a minimal number "
                "of AnalogSignals in each trial."
                % str(self.d_conditions["trial_has_n_as"][1]))
        # List of valid trial, will be returned
        as_lst_valid_trials = []
        if self._is_block:
            for tr_id in list_trials:
                seg = self._input.segments[tr_id]
                if len(seg.analogsignals) >= \
                        self.d_conditions["trial_has_n_as"][1]:
                    as_lst_valid_trials.append(tr_id)
        elif self._is_segment:
            if len(self._input.analogsignals) >= \
                    self.d_conditions["trial_has_n_as"][1]:
                # Append only trial with Index 0
                as_lst_valid_trials.append(0)
        elif self._is_analog_signal_lst:
            if len(self._input) >= self.d_conditions["trial_has_n_as"][1]:
                # Append only trial with Index 0
                as_lst_valid_trials.append(0)
        return as_lst_valid_trials

    def __check_trial_exact_st(self, list_trials):
        """
            Check if in given trial(s) the number of neo.SpikeTrain objects are
            equal to `n`.

            Parameters
            ---------
             list_trials: list of int
                List of trial IDs, which will be iterated trough.

            Returns
            -------
            list: list of int
                Trial IDs according to trial condition (equal to **n**
                neo.SpikeTrain objects in trial).
            """
        if self.d_conditions["trial_has_exact_st"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, when setting "
                "the condition for a minimal number "
                "of SpikeTrains in each trial."
                % str(self.d_conditions["trial_has_exact_st"][0]))
        # List of valid trial, will be returned
        st_lst_valid_trials = []
        if self._is_block:
            for tr_id in list_trials:
                seg = self._input.segments[tr_id]
                if len(seg.spiketrains) == \
                        self.d_conditions["trial_has_exact_st"][1]:
                    st_lst_valid_trials.append(tr_id)
        elif self._is_segment:
            if len(self._input.spiketrains) == \
                    self.d_conditions["trial_has_exact_st"][1]:
                # Append only trial with Index 0
                st_lst_valid_trials.append(0)
        elif self._is_spike_train_lst:
            if len(self._input) == self.d_conditions["trial_has_exact_st"][1]:
                # Append only trial with Index 0
                st_lst_valid_trials.append(0)
        return st_lst_valid_trials

    def __check_trial_exact_as(self, list_trials):
        """
        Check if in given trial(s) the number of neo.AnalogSignal objects are
        equal to `n`.

        Parameters
        ---------
         list_trials: list of int
            List of trial IDs, which will be iterated trough.

        Returns
        -------
        list: list of int
            Trial IDs according to trial condition (equal to  **n**
            neo.AnalogSignal objects in trial).
        """
        if self.d_conditions["trial_has_exact_as"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, "
                "when setting the condition for a minimal number "
                "of AnalogSignals in each trial."
                % str(self.d_conditions["trial_has_exact_as"][1]))
        # List of valid trial, will be returned
        as_lst_valid_trials = []
        if self._is_block:
            for tr_id in list_trials:
                seg = self._input.segments[tr_id]
                if len(seg.analogsignals) == \
                        self.d_conditions["trial_has_exact_as"][1]:
                    as_lst_valid_trials.append(tr_id)
        elif self._is_segment:
            if len(self._input.anlogsignals) == \
                    self.d_conditions["trial_has_exact_as"][1]:
                # Append only trial with Index 0
                as_lst_valid_trials.append(0)
        elif self._is_analog_signal_lst:
            if len(self._input) == self.d_conditions["trial_has_exact_as"][1]:
                # Append only trial with Index 0
                as_lst_valid_trials.append(0)
        return as_lst_valid_trials

    def __check_trial_has_n_units(self, trial_list):
        """
        Checks if the number of neo.Unit objects per trial is greater or equal
        to given `n`.

        Parameters
        ----------
         trial_list : list of int
            List of trial IDs, which will be iterated trough.

        list: list of int
            Trial IDs according to trial condition (equal or more than **n**
            neo.Units objects in trial).
        """
        if self.d_conditions["trial_has_n_units"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, "
                "when setting the condition for a minimal number "
                "of Units in each trial."
                % str(self.d_conditions["trial_has_n_units"][1]))
        counter = 0
        unit_lst_valid_trials = []
        unit_lst = []
        if self._is_block:
            for tr_id in trial_list:
                seg = self._input.segments[tr_id]
                for st in seg.spiketrains:
                    if st.unit and st.unit not in unit_lst:
                        counter += 1
                        unit_lst.append(st.unit)
                if counter >= self.d_conditions["trial_has_n_units"][1]:
                    unit_lst_valid_trials.append(tr_id)
        return unit_lst_valid_trials

    def __check_trial_has_n_rc(self, trial_list):
        """
        Checks if the number of neo,RecordingChannel objects per trial is
        greater or equal to given `n`.

        Parameters
        ----------
         trial_list : list of int
            List of trial IDs, which will be iterated trough.

        list: list of int
            Trial IDs according to trial condition (equal or more than **n**
            neo.Units objects in trial).
        """
        if self.d_conditions["trial_has_n_rc"][1] < 1:
            raise ValueError(
                "Please provide a number greater than %s, "
                "when setting the condition for a minimal number "
                "of RecordingChannel in each trial."
                % str(self.d_conditions["trial_has_n_rc"][1]))
        rc_lst_valid_trials = []
        rc_lst = []
        if self._is_block:
            for tr_id in trial_list:
                counter = 0
                seg = self._input.segments[tr_id]
                for asig in seg.analogsignals:
                    if asig.recordingchannel not in rc_lst:
                        counter += 1
                        rc_lst.append(asig.recordingchannel)
                        if counter >= self.d_conditions["trial_has_n_rc"][1]:
                            rc_lst_valid_trials.append(tr_id)
                            break
        return rc_lst_valid_trials

    def __check_contains_each_unit(self, trial_list):
        """
        Checks if each trial contains each neo.Unit object.

         Parameters
        ----------
         trial_list : list of int
            List of trial IDs, which will be iterated trough.

        list: list of int
            Trial IDs according to trial condition (each trial each
            neo.Unit object).
        """
        units = self.get_units()
        lst_valid_trials = []
        visited_units = {}
        for tr_id in trial_list:
            if self._is_block:
                seg = self._input.segments[tr_id]
                counter = 0
                for st in seg.spiketrains:
                    if st.unit not in visited_units:
                        counter += 1
                        visited_units[st.unit] = True
                if len(units) == counter:
                    lst_valid_trials.append(tr_id)
                visited_units.clear()
            elif self._is_segment:
                seg = self._input
                counter = 0
                for st in seg.spiketrains:
                    if st.unit not in visited_units:
                        counter += 1
                        visited_units[st.unit] = True
                if len(units) == counter:
                    lst_valid_trials.append(tr_id)
                visited_units.clear()
        return lst_valid_trials

    def __check_contains_each_rc(self, trial_list):
        """
        Checks if each trial contains each neo.RecordingChannel
         object.

         Parameters
        ----------
         trial_list : list of int
            List of trial IDs, which will be iterated trough.

        list: list of int
            Trial IDs according to trial condition (each trial each
            neo.RecordingChannel object).
        """
        rc_list = self.get_recordingchannels()
        lst_valid_trials = []
        visited_rc = {}
        for tr_id in trial_list:
            if self._is_block:
                seg = self._input.segments[tr_id]
                counter = 0
                for asig in seg.analogsignals:
                    if asig.recordingchannel not in visited_rc:
                        counter += 1
                        visited_rc[asig.recordingchannel] = True
                if counter == len(rc_list):
                    lst_valid_trials.append(tr_id)
                visited_rc.clear()
            elif self._is_segment:
                seg = self._input
                counter = 0
                for asig in seg.analogsignals:
                    if asig.recordingchannel not in visited_rc:
                        counter += 1
                        visited_rc[asig.recordingchannel] = True
                if counter == len(rc_list):
                    lst_valid_trials.append(tr_id)
                visited_rc.clear()
        return lst_valid_trials

    def __check_data_aligned(self, trial_list):
        """
        Check if all signals per trial are aligned at the same time line,
        regarding (start, stop).

        Parameters
        ----------
        trial_list : list of int
            List of trial IDs, which will be iterated trough.

        list: list of int
            Trial IDs according to trial condition (data is aligned).

        Raises
        ------
        ValueError
            If no neo.SpikeTrain or neo.AnalogSignal objects are found.
        """
        as_time_list = []
        st_time_list = []
        lst_valid_trials = []
        if self.has_spiketrains() is False and self.has_spiketrains() is False:
            raise ValueError("No SpikeTrain and AnalogSignal objects "
                             "in input.")
        for tr_id in trial_list:
            if self._is_block:
                seg = self._input.segments[tr_id]
                for st in seg.spiketrains:
                    st_time_list.append((st.t_start, st.t_stop))
                for asig in seg.analogsignals:
                    as_time_list.append((asig.t_start, asig.t_stop))
                # Check if start and stop times are equal
                if len(as_time_list) == 0:
                    if st_time_list.count(st_time_list[0]) == len(
                            st_time_list):
                        lst_valid_trials.append(tr_id)
                elif len(st_time_list) == 0:
                    if as_time_list.count(as_time_list[0]) == len(
                            as_time_list):
                        lst_valid_trials.append(tr_id)
                else:
                    if np.equal(st_time_list, as_time_list).all():
                        lst_valid_trials.append(tr_id)
                del as_time_list[:]
                del st_time_list[:]
            elif self._is_segment:
                seg = self._input
                for st in seg.spiketrains:
                    st_time_list.append((st.t_start, st.t_stop))
                for asig in seg.analogsignals:
                    as_time_list.append((asig.t_start, asig.t_stop))
                # Check if start and stop times are equal
                if len(as_time_list) == 0:
                    if st_time_list.count(st_time_list[0]) == len(
                            st_time_list):
                        lst_valid_trials.append(tr_id)
                elif len(st_time_list) == 0:
                    if as_time_list.count(as_time_list[0]) == len(
                            as_time_list):
                        lst_valid_trials.append(tr_id)
                else:
                    if np.equal(st_time_list, as_time_list).all():
                        lst_valid_trials.append(tr_id)
                del as_time_list[:]
                del st_time_list[:]
            elif self._is_spike_train_lst:
                for st in self._input:
                    st_time_list.append((st.t_start, st.t_stop))
                if st_time_list.count(st_time_list[0]) == len(
                        st_time_list):
                    lst_valid_trials.append(tr_id)
            elif self._is_analog_signal_lst:
                for asig in self._input:
                    as_time_list.append((asig.t_start, asig.t_stop))
                if as_time_list.count(as_time_list[0]) == len(
                        as_time_list):
                    lst_valid_trials.append(tr_id)
        return lst_valid_trials

    def __check_trial_has_no_overlap(self, trial_list, take_first=False):
        """
        Checks if trial overlap with other trials. If it has overlap the trial
        ID won't be considered as valid trial.

        Parameters
        ----------
        trial_list: list of int
            List of trial IDs, which will be iterated trough.
        take_first: 0 or 1
            True: Even if a overlap appears the first trial within all the
            overlapping trials will be taken.
            False: None of the overlapping trials will be taken.

        Returns
        -------
        list: list of int
            Trial IDs according to trial condition (trial has no overlap).
        """
        lst_valid_trials = []
        trials_time = {}
        # Functions to define the min/max of a list of tuples
        # in each trial;
        # first unzips the list, then finds the min/max for each tuple position

        def x_min(x): return list(map(min, zip(*x)))[0]

        def y_max(y): return list(map(max, zip(*y)))[1]

        if self._is_block:
            for tr_id in trial_list:
                seg = self._input.segments[tr_id]
                st_start_stop = [(st.t_start, st.t_stop) for st in
                                 seg.spiketrains]
                as_start_stop = [(asig.t_start, asig.t_stop) for asig in
                                 seg.analogsignals]
                if st_start_stop and as_start_stop:
                    min_start = min(x_min(st_start_stop), x_min(as_start_stop))
                    max_stop = max(y_max(st_start_stop), y_max(as_start_stop))
                elif st_start_stop:
                    min_start = x_min(st_start_stop)
                    max_stop = y_max(st_start_stop)
                elif as_start_stop:
                    min_start = x_min(as_start_stop)
                    max_stop = y_max(as_start_stop)
                else:
                    raise ValueError("No min start, max stop times found.")
                trials_time[tr_id] = (min_start, max_stop)
            # Store in list
            tr_time_lst = list(trials_time.items())
            # Order items according to smallest start time
            # Result is a sorted list of following form: (int, (int, int))
            # sorting by first item in tuple
            tr_time_lst.sort(key=lambda x: x[1][0])
            # Create binary mask for indexing,
            # in order not to visit invalid trials again
            bin_mask = np.ones(len(tr_time_lst), dtype=int)
            # Iterate over trial times list
            for idx, tpl in enumerate(tr_time_lst):
                # Get trial id
                ids = tpl[0]
                # Boolean to indicate if trial is valid or not
                valid = True
                # Check if in binary mask
                if not bin_mask[idx] == 0:
                    valid = self.__overlap(bin_mask, tr_time_lst, idx, valid)
                    if valid and bool(take_first) is False:
                        lst_valid_trials.append(ids)
                    elif take_first:
                        lst_valid_trials.append(ids)
        elif self._is_segment:
            for tr_id in trial_list:
                lst_valid_trials.append(tr_id)
        elif self._is_spike_train_lst:
            lst_valid_trials.append(0)
        elif self._is_analog_signal_lst:
            lst_valid_trials.append(0)
        return lst_valid_trials

    def __overlap(self, bin_mask, lst, i, b):
        """
        Recursive, helper method, to compare the start and stop time points of
        two neighbouring trials
        (actual and next element in list).

        Parameters
        ----------
        bin_mask: numpy.ndarray
            A mask with 1's or 0's, to see which trial is overlapping. 1 means
            no overlap and 0 means overlap.
            Corresponds to trial ID.
        lst: list of tuples of int
            A list with the trials.
        i: int
            Actual position in `lst`.
        b: bool
            A boolean variable which will be returned. Indicates if a trial is
            valid or not.

        Returns
        -------
        valid: bool
            A boolean variable to indicate if a trial is valid or not.

        Notes
        -----
        Algorithm compares actual element i (trial_1) with next element i+1
        (trial_2) from the list. List contains
        tuple of trial IDs and tuple of start and stop points per trial.
        The list is ordered according to the smallest
        start point.
        Each step in the calling function (__check_trial_has_no_overlap()) an
        element of the list will be picked out.
        If it is not flagged as an overlapping trial.
        """
        valid = b
        if i + 1 >= len(lst):
            return valid
        else:
            # Get the actual and next element of list
            trial_1 = lst[i]
            trial_2 = lst[i + 1]
            # If start time of second trial is smaller
            # than stop time of actual trial
            if trial_2[1][0] < trial_1[1][1]:
                bin_mask[i] = 0
                bin_mask[i + 1] = 0
                valid = False
                self.__overlap(bin_mask, lst, i + 1, valid)
        return valid

    @staticmethod
    def _intersect(a, b):
        return list(set(a) & set(b))

    @staticmethod
    def _difference(a, b):
        return list(set(a) - set(b))

    def is_valid(self, trial_id=None):
        """
        Checks if a trial ID belongs to a valid trial

        Parameters
        ----------
        trial_id: int, list or numpy.ndarray
            Checks for a Integer, List or numpy array if the IDs correspond to
            a valid trial.

        Returns
        -------
        bool: bool
            True if ID corresponds to a valid trial.
            False, otherwise.
        """
        if trial_id:
            tr_id = trial_id
        else:
            tr_id = self.valid_trial_ids
        if type(trial_id) is int:
            tr_id = np.array([trial_id])
        for ids in tr_id:
            if ids in self.valid_trial_ids:
                return True
        return False

    # ## Get Num methods ###
    def get_num_trials(self):
        """
        Returns the number of trials.

        Returns
        --------
        counter : int
            The number of trials.

        """
        if self._is_block:
            return len(self._input.segments)
        elif self._is_segment:
            return 1
        elif self._is_spike_train_lst:
            return 1
        elif self._is_analog_signal_lst:
            return 1

    def get_num_spiketrains(self):
        """
        Returns the total number of all SpikeTrain objects.

        Returns
        -------
        num: int
            Number of neo.SpikeTrain objects.
        """
        if self.has_spiketrains():
            if self._is_block:
                counter = 0
                for seg in self._input.segments:
                    counter += len(seg.spiketrains)
                return counter
            elif self._is_segment:
                return len(self._input.spiketrains)
            elif self._is_spike_train:
                return 1
            elif self._is_unit:
                return len(self._input.spiketrains)
            elif self._is_spike_train_lst:
                return len(self._input)
        return 0

    def get_num_analogsignals(self):
        """
        Returns the total number of all AnalogSignal objects.

        Returns
        -------
        num: int
            Number of neo.AnalogSignal objects.
        """
        if self.has_analogsignals():
            if self._is_block:
                counter = 0
                for seg in self._input.segments:
                    counter += len(seg.analogsignals)
                return counter
            elif self._is_segment:
                return len(self._input.analogsignals)
            elif self._is_analog_signal:
                return 1
            elif self._is_recording_channel:
                return len(self._input.analogsignals)
            elif self._is_recording_channel_group:
                counter = 0
                rc_lst = []
                for rc in self._input.recordingchannels:
                    if rc not in rc_lst:
                        rc_lst.append(rc)
                        counter += len(rc.analogsignals)
                return counter
            elif self._is_analog_signal_lst:
                return len(self._input)
        return 0

    def get_num_units(self):
        """
        Returns the number of neo.Unit objects.

        Returns
        -------
        num : int
            Number of neo.Unit objects from given input.
        """
        if self._is_block:
            if self._input.list_units:
                return len(self._input.list_units)
            else:
                counter = 0
                for rcg in self._input.recordingchannelgroups:
                    counter += len(rcg.units)
                return counter
        elif self._is_recording_channel_group:
            return len(self._input.units)
        elif self._is_unit:
            return 1

    def get_num_recordingchannels(self):
        """
        Returns the number of RecordingChannels.

        Returns
        -------
        num : int
            Number of neo.RecordingChannel objects from given input.
        """
        if self._is_block:
            # Block knows (has property of) the list of RecordingChannels
            return len(self._input.list_recordingchannels)
        elif self._is_recording_channel_group:
            rc_lst = []
            for rc in self._input.recordingchannels:
                if rc not in rc_lst:
                    rc_lst.append(rc)
            return rc_lst
        elif self._is_recording_channel:
            return 1
        elif self._is_recording_channel_group:
            return len(self._input.recordingchannels)

    def get_num_recordingchannelgroup(self):
        """
        Returns the number of RecordingChannelsGroups.

         Returns
        -------
        num : int
            Number of neo.RecordingChannelGroup objects from given input.
        """
        if self._is_block:
            return len(self._input.recordingchannelgroups)
        elif self._is_recording_channel_group:
            return 1

    def get_trial_ids(self):
        """
        Returns IDs of trials.

        The Index which corresponds to the trial IDs will be returned.
        If the given input is a neo.Segment it will regarded as trial with
        index 0.

        Returns
        -------
        indices : list
            The indices corresponding to trial ID of trials.
        """
        idx_lst = []
        if self._is_block:
            for idx in range(len(self._input.segments)):
                idx_lst.append(idx)
                # idx_lst.append(seg.index)
        elif self._is_segment:
            idx_lst.append(0)
            # idx_lst.append(self._input.index)
        elif self._is_spike_train:
            idx_lst.append(0)
        elif self._is_analog_signal:
            idx_lst.append(0)
        elif self._is_spike_train_lst:
            idx_lst.append(0)
        elif self._is_analog_signal_lst:
            idx_lst.append(0)
        else:
            raise AttributeError('No suitable input %s' % type(self._input))
        return idx_lst

    def get_spike_trains(self, trial_id=None):
        """
        Extracts SpikeTrain objects from given input and returns them.
        If no SpikeTrain objects are found an empty list is returned.

        Parameters
        ----------
        trial_id : int, list or None
            If an integer is given all SpikeTrain objects according to the `
            trial_id` will be returned.
            If a list is given all SpikeTrain objects with corresponding
            trial ID will be returned.
            If no `trial_id` is given, all SpikeTrain objects from given input
            will be returned.
            None is Default.

        Returns
        -------
        list : list of tuples of (int, list of SpikeTrain objects)
            A list containing tuples with corresponding trial ID and list of
            SpikeTrain objects will be returned.
        """
        st_lst = []
        if self.has_spiketrains():
            # Return all SpikeTrain objects
            if trial_id is None:
                if self._is_block:
                    for idx, seg in enumerate(self._input.segments):
                        st_lst.append((idx, seg.spiketrains))
                elif self._is_segment:
                    st_lst.append((0, self._input.spiketrains))
                elif self._is_spike_train:
                    st_lst.append((0, [self._input]))
                elif self._is_spike_train_lst:
                    st_lst.append((0, [self._input]))
                elif self._is_unit:
                    st_lst.append((0, self._input.spiketrains))
            else:
                if self._is_block:
                    if type(trial_id) is int:
                        try:
                            return st_lst.append(
                                (trial_id,
                                    [st for st in self._input.segments[
                                        trial_id].spiketrains]))
                        except IndexError:
                            raise IndexError("ID of trials are: %s" % str(
                                self.get_trial_ids()))
                    else:
                        for i in trial_id:
                            try:
                                st_lst.append((i, [st for st in
                                                   self._input.segments[
                                                       i].spiketrains]))
                            except IndexError:
                                raise IndexError("ID of trials are: %s" % str(
                                    self.get_trial_ids()))
        return st_lst

    def get_analog_signals(self, trial_id=None):
        """
        Extracts all AnalogSignal objects from given input and returns them.
        If no AnalogSignal objects are found an empty list is returned.

        Returns
        --------
        list : list of neo.AnalogSignal objects.
        analogsignal: neo.AnalogSignal object if the input is
        neo.AnalogSignal object.
        """
        asig_lst = []
        if self.has_analogsignals():
            # Return all AnalogSignal objects
            if trial_id is None:
                if self._is_block:
                    for idx, seg in enumerate(self._input.segments):
                        asig_lst.append((idx, seg.analogsignals))
                elif self._is_segment:
                    asig_lst.append((0, self._input.analogsignals))
                elif self._is_analog_signal:
                    asig_lst.append((0, [self._input]))
                elif self._is_analog_signal_lst:
                    asig_lst.append((0, [self._input]))
                elif self._is_recording_channel:
                    asig_lst.append((0, self._input.analogsignals))
            else:
                if self._is_block:
                    if type(trial_id) is int:
                        try:
                            return trial_id, [asig for asig in
                                              self._input.segments[
                                                  trial_id].analogsignals]
                        except IndexError:
                            raise IndexError("ID of trials are: %s" % str(
                                self.get_trial_ids()))
                    else:
                        for i in trial_id:
                            try:
                                asig_lst.append((i, [asig for asig in
                                                     self._input.segments[
                                                         i].analogsignals]))
                            except IndexError:
                                raise IndexError("ID of trials are: %s" % str(
                                    self.get_trial_ids()))
        return asig_lst

    def get_units(self):
        """
        Returns neo.Unit objects.

        Returns
        -------
        list: list of neo.Unit objects
        unit: neo.Unit object if the input is a neo.Unit object.
        """
        if self._is_block:
            return self._input.list_units
        elif self._is_unit:
            return self._input
        elif self._is_recording_channel_group:
            return self._input.units

    def get_recordingchannels(self):
        """
        Returns neo.RecordingChannel objects.

        Returns
        -------
        list: list of neo.RecordingChannel objects
            Default case.
            A list of neo.RecordingChannel object is returned if the input is
            a neo.RecordingChannel object.
        """
        rc_lst = []
        if self._is_block:
            return self._input.list_recordingchannels
        elif self._is_segment:
            for asig in self._input.analogsignals:
                if asig.recordingchannel not in rc_lst:
                    rc_lst.append(asig.recordingchannel)
        elif self._is_analog_signal:
            return self._input.recordingchannel
        elif self._is_recording_channel:
            return [self._input]
        elif self._is_recording_channel_group:
            for rc in self._input.recordingchannels:
                if rc not in rc_lst:
                    rc_lst.append(rc)
        return rc_lst

    def get_recording_channel_group(self):
        """
        Returns neo.RecordingChannelGroup objects.

        Returns
        -------
        list: list of neo.RecordingChannelGroup objects
            Default case.
            A list of neo.RecordingChannelGroup object is returned if the
            input is a neo.RecordingChannelGroup object.
        """
        if self._is_block:
            return self._input.recordingchannelgroups
        elif self._is_recording_channel_group:
            return [self._input]

    def get_num_analogsignals_of_valid_trial(self, trial_id=None,
                                             with_id=True):
        """
        Returns the number of AnalogSignal objects of trial with `trial_ID`,
        respecting trial conditions.

        The number of AnalogSignals per valid trial with given trial ID will
        be returned.

        Parameters
        ----------
        trial_id : int, list, numpy.ndarray or None
            If an integer is given all AnalogSignal objects according to the
            `trial_id` will be returned.
            If a list is given all AnalogSignal objects with corresponding
            trial ID will be returned.
            If the parameter is **None**, all valid trials will be considered.
            Default is None.
        with_id: bool
            If True number of neo.AnalogSignal objects with corresponding ID
            will be returned.
            If False number of all AnalogSignal objects from all valid trials
            will be returned.
            Default is True.

        Returns
        -------
        counter: int
            Number of AnalogSignal objects with respect to trial conditions.
        numbers : list of tuples of (int, int)
            If a list as parameter is given, a list of tuples with
            corresponding trial ID and number of AnalogSignal objects
            will be returned.

        """
        counter = 0
        if trial_id is None:
            tr_id = self.valid_trial_ids
        elif type(trial_id) is int:
            # Convert to array
            tr_id = np.array([trial_id])
        else:
            tr_id = trial_id
        # Check if there are valid analogsignals per trial ID
        if self.has_analogsignal_in_valid_trial(trial_id):
            if self._is_block:
                numbers_lst = []
                for ids in tr_id:
                    counter = 0
                    seg = self._input.segments[ids]
                    counter += len(seg.analogsignals)
                    if with_id:
                        numbers_lst.append((ids, counter))
                if with_id:
                    return numbers_lst
                else:
                    return counter
            elif self._is_segment:
                if self.is_valid():
                    return [(0, self.get_num_analogsignals())]
            elif self._is_analog_signal_lst:
                if self.is_valid():
                    return len(self._input)
        return counter

    def get_num_spiketrains_of_valid_trial(self, trial_id=None, with_id=True):
        """
        Returns the number of SpikeTrain objects of trial with `trial_ID`,
        respecting trial conditions.

        The number of neo.Spiketrain objects per valid trial with
        corresponding trial ID will be returned.

        Parameters
        ----------
        trial_id : int, list, numpy.ndarray or None
            If an integer is given all SpikeTrain objects according to the
            `trial_id` will be returned.
            If a list is given all SpikeTrain objects with corresponding
            trial ID will be returned.
            If the parameter is **None**, all valid trials will be considered.
            Default is None.
        with_id: bool
            If True number of neo.SpikeTrain objects with corresponding ID will
            be returned.
            If False number of all SpikeTrain objects from all valid trials
            will be returned.
            Default is True.

        Returns
        -------
        counter: int
            Number of SpikeTrain objects with respect to trial conditions.
        numbers : list of tuples of (int, int)
            If a list as parameter is given, a list of tuples with
            corresponding trial ID and number of SpikeTrain objects will
            be returned.

        """
        counter = 0
        if trial_id is None:
            tr_id = self.valid_trial_ids
        elif type(trial_id) is int:
            # Convert to array
            tr_id = np.array([trial_id])
        else:
            tr_id = trial_id
        # Check if there are valid spiketrains per trial ID
        if self.has_spiketrain_in_vaild_trial(tr_id):
            if self._is_block:
                numbers_lst = []
                for ids in tr_id:
                    counter = 0
                    seg = self._input.segments[ids]
                    counter += len(seg.spiketrains)
                    if with_id:
                        numbers_lst.append((ids, counter))
                if with_id:
                    return numbers_lst
                else:
                    return counter
            elif self._is_segment:
                if self.is_valid():
                    return self.get_num_spiketrains()
            elif self._is_spike_train_lst:
                if self.is_valid():
                    return len(self._input)
        return counter

    def get_num_valid_trials(self):
        """
        Returns the number of valid trials, according to trial conditions

        Returns
        -------
        valid_trials : int
            Number of valid trials

        """
        return len(self.valid_trial_ids)

    def get_num_unit_valid_trial(self, trial_id=None):
        """
        Returns the number of neo.Unit objects, regarding trials which respect
        the conditions.

        Parameters
        ----------
        trial_id: int, list, None
            The trial_id determines which trials to be considered, it can be
            either an integer, a list or None.
            If `trial_id` is None, all valid trials will be considered.
            Default is None.
        Returns
        -------
        number: int
            Number of units according to trials with respect to trial
            conditions.
        """
        return len(self.get_units_from_valid_trial(trial_id))

    def get_num_recordingchannel_from_valid_trial(self, trial_id=None):
        """
        Returns the number of neo.RecordingChannel objects, regarding trials
        which respect the conditions.

        Parameters
        ----------
        trial_id: int, list, None
            The trial_id determines which trials to be considered, it can be
            either an integer, a list or None.
            If `trial_id` is None, all valid trials will be considered.
            Default is None.
        Returns
        -------
        number: int
            Number of recordingchannels according to trials with respect to
            trial conditions.
        """
        return len(self.get_recordingchannels_from_valid_trial(trial_id))

    def get_recordingchannels_from_valid_trial(self, trial_id=None):
        """
         Returns  neo.RecordingChannel objects, regarding trials which respect
         the conditions.

         Parameters
         ----------
         trial_id: int, list, None
             The trial_id determines which trials to be considered, it can be
             either an integer, a list or None.
             If `trial_id` is None, all valid trials will be considered.
             Default is None.

         Returns
         -------
         rc_lst: List of tuples of int and list of neo.RecordingChannel objects
             Returns a list of tuples with ID and corresponding list of
             neo.RecordingChannel objects.
         """
        if trial_id is None:
            trial_id = self.valid_trial_ids
        else:
            trial_id = np.array([trial_id])
        rc_lst = []
        tmp_lst = []
        if self.has_analogsignal_in_valid_trial(trial_id):
            if self._is_block:
                for tr_id in trial_id:
                    seg = self._input.segments[tr_id]
                    for asig in seg.analogsignals:
                        if asig.recordingchannel not in tmp_lst:
                            rc_lst.append((tr_id, asig.recordingchannel))
                            tmp_lst.append(asig.recordingchannel)
            elif self._is_segment:
                seg = self._input
                for asig in seg.analogsignals:
                    if asig.recordingchannel not in tmp_lst:
                        rc_lst.append((0, asig.recordingchannel))
                        tmp_lst.append(asig.recordingchannel)
        del tmp_lst
        return rc_lst

    def get_units_from_valid_trial(self, trial_id=None):
        """
         Returns  neo.Unit objects, regarding trials which respect
         the conditions.

         Parameters
         ----------
         trial_id: int, list, None
             The trial_id determines which trials to be considered, it can be
             either an integer, a list or None.
             If `trial_id` is None, all valid trials will be considered.
             Default is None.

         Returns
         -------
         units: List of tuples of int and list of neo.Unit objects
             Returns a list of tuples with ID and corresponding list of
             neo.Unit objects.
         """
        if trial_id is None:
            trial_id = self.valid_trial_ids
        else:
            trial_id = np.array([trial_id])
        unit_lst = []
        tmp_lst = []
        if self.has_spiketrain_in_vaild_trial(trial_id):
            if self._is_block:
                for tr_id in trial_id:
                    seg = self._input.segments[tr_id]
                    for st in seg.spiketrains:
                        if st.unit not in tmp_lst:
                            unit_lst.append((tr_id, st.unit))
                            tmp_lst.append(st.unit)
                    # del tmp_lst[:]
            elif self._is_segment:
                seg = self._input
                for st in seg.spiketrains:
                    if st.unit not in tmp_lst:
                        unit_lst.append((0, st.unit))
                        tmp_lst.append(st.unit)
            elif self._is_recording_channel_group:
                for unit in self._input.units:
                    for st in unit.spiketrains:
                        if st.unit not in unit_lst:
                            unit_lst.append(st.unit)
        del tmp_lst
        return unit_lst

    def get_spiketrains_from_valid_trials(self, trial_id=None):
        """
        Returns SpikeTrain objects of valid trials.

        Parameters
        ----------
        trial_id: int or list
            Only valid trials with given ID will be considered.
            If `trial_id` is **None** all valid trials will be considered.
            Default is None.

        Returns
        -------
        trials_st_list : list of tuples of trial ID
                        and list of neo.SpikeTrain objects
            When input is a neo.Block a dictionary containing keys with trial
            ID and corresponding list of neo.SpikeTrain objects as values will
            be returned. When input is a neo.Segment a list of tuple with trial
            ID and a list of neo.SpikeTrain objects will be returned.
        """
        st_trials = []
        if trial_id is None:
            trial_id = self.valid_trial_ids
        elif type(trial_id) is int:
            trial_id = np.array([trial_id])
        if self.has_spiketrain_in_vaild_trial(trial_id):
            for tr_id in trial_id:
                if self._is_block:
                    seg = self._input.segments[tr_id]
                    st_trials.append((tr_id, seg.spiketrains))
                elif self._is_segment:
                    if self.is_valid():
                        return [(0, self._input.spiketrains)]
                elif self._is_spike_train_lst:
                    if self.is_valid():
                        return [(0, self._input)]
        return st_trials

    def get_analogsignals_from_valid_trials(self, trial_id=None):
        """
        Returns AnalogSignal objects of valid trials.

        Parameters
        ----------
        trial_id: int or list
            Only valid trials with given ID will be considered.
            If `trial_id` is **None** all valid trials will be considered.
            Default is None.

        Returns
        -------
        trials_st_list : list of tuples of trial ID
                         and list of neo.AnalogSignal objects
            When input is a neo.Block a a dictionary containing keys with
            trial ID and corresponding list of
            neo.AnalogSignal objects as values will be returned.
            When input is a neo.Segment a list of tuple with trial ID and a
            list of neo.AnalogSignal objects will be
            returned.
        """
        asig_trials = []
        if trial_id is None:
            trial_id = self.valid_trial_ids()
        else:
            trial_id = np.array([trial_id])
        if self.has_analogsignal_in_valid_trial(trial_id):
            for tr_id in trial_id:
                if self._is_block:
                    for seg in self._input[tr_id]:
                        asig_trials.append((tr_id, seg.analogsignals))
                elif self._is_segment:
                    if self.is_valid():
                        return [(0, self._input.analogsignals)]
                elif self._is_analog_signal_lst:
                    if self.is_valid():
                        return [(0, self._input)]
        return asig_trials

    def has_spiketrains(self):
        """
        Checks if given input has SpikeTrain Objects.

        Returns
        -------
        bool :
            True, if given input has one non empty SpikeTrain object or is a
            SpikeTrain object.
            False, otherwise.
        """
        if self._is_spike_train:
            return True
        elif self._is_block:
            for seg in self._input.segments:
                for st in seg.spiketrains:
                    if np.size(st) > 0:
                        return True
        elif self._is_segment:
            for st in self._input.spiketrains:
                if np.size(st) > 0:
                    return True
        elif self._is_unit:
            for st in self._input.spiketrains:
                if np.size(st) > 0:
                    return True
        elif self._is_recording_channel_group:
            for ut in self._input.units:
                for st in ut.spiketrains:
                    if np.size(st) > 0:
                        return True
        elif self._is_spike_train_lst:
            for st in self._input:
                if np.size(st) > 0:
                    return True
        return False

    def has_analogsignals(self):
        """
        Checks if given input has AnalogSignal Objects.

        Return
        ------
        bool : boolean
            True, if given input has one non empty AnalogSignal object or is
            an AnalogSignal object.
            False, otherwise.
        """
        if self._is_analog_signal:
            return True
        elif self._is_analog_signal_lst:
            for asig in self._input:
                if np.size(asig) > 0:
                    return True
        elif self._is_block:
            for seg in self._input.segments:
                for asig in seg.analogsignals:
                    if np.size(asig) > 0:
                        return True
        elif self._is_segment:
            for asig in self._input.analogsignals:
                if np.size(asig) > 0:
                    return True
        elif self._is_recording_channel:
            for asig in self._input.analogsignals:
                if np.size(asig) > 0:
                    return True
        elif self._is_recording_channel_group:
            for rc in self._input.recordingchannels:
                for asig in rc.analogsignals:
                    if np.size(asig) > 0:
                        return True
        return False

    def has_epochs(self):
        """
        Checks if given input has Epochs.

        Return
        ------
        bool :
            True, if given input has a list of Epochs or is an Epoch
            object.
            False, otherwise.
        """
        # Is already an Epoch
        if self._is_epoch:
            return True
        elif self._is_block:
            for seg in self._input.segments:
                for ep in seg.epochs:
                    if type(ep) is Epoch and np.size(ep.times) > 0:
                        return True
        elif self._is_segment:
            for ep in self._input.epochs:
                if type(ep) is Epoch and np.size(ep.times) > 0:
                    return True
        return False

    def has_units(self):
        """
        Checks if given input has neo.Unit Objects.

        Returns
        -------
        bool :
            True, if given input has one non empty neo.Unit object or is a
            neo.Unit object.
            False, otherwise.
        """
        if self._is_block or self._is_recording_channel_group or self._is_unit:
            if self.has_spiketrains() and self.get_num_units() > 0:
                return True
        return False

    def has_trials(self):
        """
        Checks if given input has trials.

        Returns
        -------
        bool :
            True, if given input has at least one trial.
            False, otherwise.
        """
        if (self.has_spiketrains() or self.has_analogsignals()) \
                and self.get_num_trials() > 0:
            return True

        return False

    def has_recordingchannels(self):
        """
        Checks if given input has neo.RecordingChannel Objects.

        Returns
        -------
        bool :
            True, if given input has one non empty neo.RecordingChannel object
            or is a neo.RecordingChannel object.
            False, otherwise.
        """
        if self._is_block or self._is_recording_channel or \
                self._is_recording_channel_group:
            if self.has_analogsignals() \
                    and self.get_num_recordingchannels() > 0:
                return True
        return False

    def has_recordingchannelgroup(self):
        """
        Checks if given input has neo.RecordingChannelGroup objects.

        Returns
        -------
        bool :
            True, if given input has one non empty neo.RecordingChannelGroup
            object or is a neo.RecordingChannelGroup.
            False, otherwise.
        """
        if self._is_block or self._is_recording_channel_group:
            if (self.has_analogsignals() or self.has_spiketrains()) \
                    and self.get_num_recordingchannelgroup() > 0:
                return True
        return False

    def has_analogsignal_in_valid_trial(self, trial_id=None):
        """
        Checks if a neo.AnalogSignal object exists for given set of trial IDs,
        and if all of the trials are valid.

        Parameters
        ----------
        trial_id : int, list, None
            The trial_id determines which trials to be considered, it can be
            either an integer, a list or None.
            If `trial_id` is None, all valid trials will be considered.
            Default is None.

        Returns
        -------
        bool: bool
            True, if for each trial ID, a AnalogSignal objects exists, and the
            trial is valid.
            False, otherwise.
        dict: {key: trial Index, value: bool}
            If the input is a neo.Block and there is at least one AnalogSignal
            in a trial which is not valid, a dictionary is returned. The trial
            indices are the keys and the value is of type bool. The boolean
            value is True if the trial conditions are met, otherwise False.

        Raises
        ------
        IndexError:
            Raises an IndexError if given trial index does not belong to a
            trial.
        """
        if trial_id is None:
            tr_id = self.valid_trial_ids
        elif type(trial_id) is int:
            tr_id = np.array([trial_id])
        else:
            tr_id = trial_id
        if self.is_valid(tr_id) is False and len(self.valid_trial_ids) > 0:
            raise IndexError("Given ID is not a valid ID regarding "
                             "trial conditions, valid trial IDs are: %s"
                             % str(self.valid_trial_ids))
        elif len(self.valid_trial_ids) == 0:
            return False
        take_trial = np.zeros(len(tr_id), dtype=int)
        for i, ids in enumerate(tr_id):
            if self._is_block:
                checked_trials = {}
                valid = False
                try:
                    seg = self._input.segments[ids]
                except IndexError:
                    raise IndexError(
                        "IndexError: list index out of range. Trial IDs are %s"
                        % str(self.valid_trial_ids))
                for asig in seg.analogsignals:
                    if np.size(asig) > 0:
                        take_trial[i] = 1
                        checked_trials[ids] = True
                        valid = True
                    if valid is False:
                        checked_trials[ids] = False
                if take_trial.all():
                    return True
                else:
                    return checked_trials
            elif self._is_segment:
                seg = self._input
                for asig in seg.analogsignals:
                    if np.size(asig) > 0:
                        return True
            elif self._is_analog_signal_lst:
                for asig in self._input:
                    if np.size(asig) > 0:
                        return True
        return False

    def has_spiketrain_in_vaild_trial(self, trial_id=None):
        """
        Checks if a neo.SpikeTrain object exists for given set of trial IDs,
        and if all of the trials are valid.

        Parameters
        ----------
        trial_id : int, list, None
            The trial_id determines which trials to be considered, it can be
            either an integer, a list or None.
            If `trial_id` is None, all valid trials will be considered.
            Default is None.

        Returns
        -------
        bool: bool
            True, if for each trial ID, a SpikeTrain objects exists, and the
            trial is valid.
            False, otherwise.
        dict: {key: trial Index, value: bool}
            If the input is a neo.Block and there is at least one SpikeTrain in
            a trial which is not valid, a dictionary is returned. The trial
            indices are the keys and the value is of type bool. The boolean
            value is True if the trial conditions are met, otherwise False.

        Raises
        ------
        IndexError:
            Raises an IndexError if given trial index does not belong to a
            trial.
        """
        if trial_id is None:
            tr_id = self.valid_trial_ids
        elif type(trial_id) is int:
            tr_id = np.array([trial_id])
        else:
            tr_id = trial_id
        if self.is_valid(tr_id) is False and len(self.valid_trial_ids) > 0:
            raise IndexError("Given ID is not a valid ID regarding "
                             "trial conditions, valid trial IDs are: %s"
                             % str(self.valid_trial_ids))
        elif len(self.valid_trial_ids) == 0:
            return False
        take_trial = np.zeros(len(tr_id), dtype=int)
        for i, ids in enumerate(tr_id):
            if self._is_block:
                checked_trials = {}
                valid = False
                try:
                    seg = self._input.segments[ids]
                except IndexError:
                    raise IndexError(
                        "IndexError: list index out of range. "
                        "Trial IDs are %s" % str(self.valid_trial_ids))
                for st in seg.spiketrains:
                    if np.size(st) > 0:
                        take_trial[i] = 1
                        checked_trials[ids] = True
                        valid = True
                if valid is False:
                    checked_trials[ids] = False
                if take_trial.all():
                    return True
                else:
                    return checked_trials
            elif self._is_segment:
                seg = self._input
                for st in seg.spiketrains:
                    if np.size(st) > 0:
                        return True
            elif self._is_spike_train_lst:
                for st in self._input:
                    if np.size(st) > 0:
                        return True
        return False

    def is_trials_equal_len(self, trial_id=None):
        """
        Checks if all trial have equal length.

        Parameters
        ----------
        trial_id : list of int or None
            List of trial IDs.
            If None, all trials are considered.
            Default is `None`.

        Returns
        -------
        result : bool
            True, if all trials have equal length.
            False, otherwise.
        """
        min_start = 0
        max_stop = 0
        if trial_id is None:
            trial_id = self.get_trial_ids()
        if self._is_block:
            # Functions to define the min/max of a list of tuples
            # in each trial;
            # first unzips the list, then finds the min/max for each tuple
            # position
            def x_min(x): return list(map(min, zip(*x)))[0]

            def y_max(y): return list(map(max, zip(*y)))[1]

            for i, idx in enumerate(trial_id):
                seg = self._input.segments[idx]
                # Store old start, stops
                tmp_start = min_start
                tmp_stop = max_stop
                st_start_stop = [(st.t_start, st.t_stop) for st in
                                 seg.spiketrains]
                as_start_stop = [(asig.t_start, asig.t_stop) for asig in
                                 seg.analogsignals]
                if st_start_stop and as_start_stop:
                    min_start = min(x_min(st_start_stop), x_min(as_start_stop))
                    max_stop = max(y_max(st_start_stop), y_max(as_start_stop))
                elif st_start_stop:
                    min_start = x_min(st_start_stop)
                    max_stop = y_max(st_start_stop)
                elif as_start_stop:
                    min_start = x_min(as_start_stop)
                    max_stop = y_max(as_start_stop)
                if i >= 1:
                    if tmp_start != min_start or tmp_stop != max_stop:
                        return False
            return True
