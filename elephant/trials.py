"""
This module defines the basic classes that represent trials in Elephant.

Many neuroscience methods rely on the concept of repeated trials to improve the
estimate of quantities measured from the data. Typically, trials are
considered as fixed time intervals tied to a specific event in the experiment,
such as the onset of a stimulus. In the simplest case, results from multiple
trials are averaged. In other scenarios, more intricate steps must be taken
in order to pool information from each repetition of a trial.

Neo does not impose a specific way to represent trial data. A natural way to
represent trials is to have a :class:`neo.Block` containing multiple
:class:`neo.Segment` objects, each representing the data of one trial. Another
popular option is to store trials as a list of lists, where the outer refers to
the collection of trials, and inner lists contain Neo data objects
(:class:`neo.SpikeTrain` and :class:`neo.AnalogSignal`) containing the
individual data of each trial.

The classes of this module abstract from these specific data representations
by introducing a set of :class:`Trials` classes with a common API. These
classes are initialized by a supported way of structuring trials, e.g.,
:class:`TrialsFromBlock` for the first method described above. Internally, the
:class:`Trials` class will not convert this representation, but provide access
to data in specific trials (e.g., all spike trains in trial 5) or general
information about the trial structure (e.g., how many trials are there?) via a
fixed API. Trials are indexed consecutively starting from 0.

In the current implementation, classes :class:`TrialsFromBlock` and
:class:`TrialsFromLists` provide this unified way to access trial data.

.. autosummary::
    :toctree: _toctree/trials
    :template: trials_class.rst

    TrialsFromBlock
    TrialsFromLists


:copyright: Copyright 2014-2025 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from typing import List

from functools import wraps
import numpy as np

import neo
from neo.core import Segment, Block, SpikeTrain, AnalogSignal
from neo.core.spiketrainlist import SpikeTrainList
from elephant.utils import deprecated_alias


def trials_to_list_of_spiketrainlist(function):
    """
    Decorator that converts each argument passed as a :class:`Trials` object
    into a list of :class:`neo.SpikeTrainList` before calling the wrapped
    function.

    Parameters
    ----------
    function: callable
        The function to be decorated.

    Returns
    -------
    callable
        The decorated function.

    Examples
    --------
    The decorator can be used as follows:

        >>> @trials_to_list_of_spiketrainlist
        ... def process_data(self, spiketrains):
        ...     return None
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        new_args = tuple(
            [
                arg.get_spiketrains_from_trial_as_list(idx)
                for idx in range(arg.n_trials)
            ]
            if isinstance(arg, Trials)
            else arg
            for arg in args
        )
        new_kwargs = {
            key: (
                [
                    value.get_spiketrains_from_trial_as_list(idx)
                    for idx in range(value.n_trials)
                ]
                if isinstance(value, Trials)
                else value
            )
            for key, value in kwargs.items()
        }

        return function(*new_args, **new_kwargs)

    return wrapper


class Trials(ABC):
    """
    Abstract base class for handling trial-based data in Elephant.

    The `Trials` class defines a standardized interface for accessing and
    manipulating trial data. It provides a unified set of attributes and
    methods for trial handling, and serves as the base class for all
    data-structure-specific implementations.

    Child classes such as :class:`TrialsFromBlock` and :class:`TrialsFromLists`
    support specific input data structures. Usage details and examples are
    provided in their respective documentation.

    Parameters
    ----------
    description : str, optional
        Textual description of the set of trials, accessible via the
        `description` attribute.
        Default: None

    See Also
    --------
    :class:`TrialsFromBlock`
    :class:`TrialsFromLists`
    """

    def __init__(self, description: str = None):
        self.description = description

    @abstractmethod
    def __getitem__(self, trial_index: int) -> Segment:
        # Get a specific trial by its index as a Segment
        raise NotImplementedError

    @property
    @abstractmethod
    def n_trials(self) -> int:
        """
        Number of trials.

        Returns
        -------
        int
            Total number of trials.
        """
        raise NotImplementedError

    @abstractmethod
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        """
        Number of spike trains per trial.

        Returns
        -------
        List[int]
            Number of :class:`neo.SpikeTrain` objects in each trial, ordered by
            trial index in ascending order starting from zero.
        """
        raise NotImplementedError

    @abstractmethod
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        """
        Number of analog signals per trial.

        Returns
        -------
        List[int]
            Number of :class:`neo.AnalogSignal` objects in each trial, ordered
            by trial index in ascending order starting from zero.
        """
        raise NotImplementedError

    @deprecated_alias(trial_id="trial_index")
    def get_trial_as_segment(self, trial_index: int) -> Segment:
        """
        Return a single trial as a :class:`neo.Segment`.

        Parameters
        ----------
        trial_index : int
            Zero-based index of the trial to retrieve.

        Returns
        -------
        :class:`neo.Segment`
            Segment containing all spike trains and analog signals associated
            with the specified trial. Spike trains and analog signals are
            accessed via the `spiketrains` and `analogsignals` attributes,
            respectively. Their order corresponds to their index within these
            collections (e.g., `spiketrains[0]` is the first spike train).
        """
        return self.__getitem__(trial_index)

    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_block(self, trial_indexes: List[int] = None) -> Block:
        """
        Return multiple trials grouped into a :class:`neo.Block`.

        Parameters
        ----------
        trial_indexes : List[int], optional
            Zero-based indices of the trials to include in the block.
            If None, all trials are returned.
            Default: None

        Returns
        -------
        :class:`neo.Block`
            Block containing one :class:`neo.Segment` per trial. The trials are
            accessed via the `segments` attribute. If all trials are included,
            element indices correspond to trial indices. If a subset is
            specified, the order matches that of `trial_indexes`.

        See Also
        --------
        :method:`get_trial_as_segment()`
        """
        block = Block()
        if not trial_indexes:
            trial_indexes = list(range(self.n_trials))
        for trial_index in trial_indexes:
            block.segments.append(self.get_trial_as_segment(trial_index))
        return block

    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_list(self, trial_indexes: List[int] = None
                           ) -> List[Segment]:
        """
        Return multiple trials as a list of :class:`neo.Segment` objects.

        Parameters
        ----------
        trial_indexes : List[int], optional
            Zero-based indices of the trials to include in the list.
            If None, all trials are returned.
            Default: None

        Returns
        -------
        List[Segment]
            List containing one :class:`neo.Segment` per selected trial. If all
            trials are returned, list indices correspond to trial indices.
            If a subset is specified, the order matches that of
            `trial_indexes`.
        """
        if not trial_indexes:
            trial_indexes = list(range(self.n_trials))
        return [self.get_trial_as_segment(trial_index)
                for trial_index in trial_indexes]

    @abstractmethod
    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_list(self, trial_index: int
                                           ) -> SpikeTrainList:
        """
        Return all spike trains from a single trial as a list.

        Parameters
        ----------
        trial_index : int
            Zero-based index of the trial.

        Returns
        -------
        :class:`neo.SpikeTrainList`
            List-like container with all :class:`neo.SpikeTrain` objects from
            the specified trial.
        """
        raise NotImplementedError

    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_segment(self, trial_index: int
                                              ) -> Segment:
        """
        Return all spike trains from a single trial as a :class:`neo.Segment`.

        Parameters
        ----------
        trial_index : int
            Zero-based index of the trial.

        Returns
        -------
        :class:`neo.Segment`
            Segment containing all :class:`neo.SpikeTrain` objects from the
            specified trial, accessible via the `spiketrains` attribute.
        """
        segment = Segment()
        for spiketrain in self.get_spiketrains_from_trial_as_list(trial_index):
            segment.spiketrains.append(spiketrain)
        return segment

    @abstractmethod
    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_list(self, trial_index: int
                                             ) -> List[AnalogSignal]:
        """
        Return all analog signals from a single trial as a list.

        Parameters
        ----------
        trial_index : int
            Zero-based index of the trial.

        Returns
        -------
        List[AnalogSignal]
            List containing all :class:`neo.AnalogSignal` objects from the
            specified trial.
        """
        raise NotImplementedError

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_segment(self, trial_index: int
                                                ) -> Segment:
        """
        Return all analog signals from a single trial as a :class:`neo.Segment`.

        Parameters
        ----------
        trial_index : int
            Zero-based index of the trial.

        Returns
        -------
        :class:`neo.Segment`
            Segment containing all :class:`neo.AnalogSignal` objects from the
            specified trial, accessible via the `analogsignals` attribute.
        """
        segment = Segment()
        for analogsignal in self.get_analogsignals_from_trial_as_list(
                trial_index):
            segment.analogsignals.append(analogsignal)
        return segment

    @abstractmethod
    def get_spiketrains_trial_by_trial(self, spiketrain_index: int) -> (
                                       SpikeTrainList):
        """
        Return spike train across all its trial repetitions.

        This method returns a list containing :class:`neo.SpikeTrain` objects
        corresponding to the same spike train (e.g., from a consistent
        recording channel or neuronal source) across multiple trials.

        Parameters
        ----------
        spiketrain_index : int
            Zero-based index of the spike train to retrieve across trials.

        Returns
        -------
        :class:`neo.SpikeTrainList`
            List-like container storing the :class:`neo.SpikeTrain` objects
            corresponding to the specified `spiketrain_index`, ordered by trial
            index from zero to `n_trials - 1`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_analogsignals_trial_by_trial(self, signal_index: int
                                         ) -> List[AnalogSignal]:
        """
        Return an analog signal across all its trial repetitions.

        This method returns a list containing :class:`neo.AnalogSignal`
        objects corresponding to a continuous signal recorded from a consistent
        recording channel or neuronal source across multiple trials.

        Parameters
        ----------
        signal_index : int
            Zero-based index of the analog signal to retrieve across trials.

        Returns
        -------
        List[AnalogSignal]
            List storing :class:`neo.AnalogSignal` objects corresponding to the
            specified `signal_index`, ordered by trial index from zero to
            `n_trials - 1`.
        """
        raise NotImplementedError


class TrialsFromBlock(Trials):
    """
    This class handles trial data organized within a :class:`neo.Block` object.

    In this representation, each trial is stored as a separate
    :class:`neo.Segment` within the block. All trial segments are accessible
    through the `segments` attribute. The data for a specific trial can be
    accessed by its index, e.g., `segments[0]` corresponds to the first trial.

    Each :class:`neo.Segment` contains collections for spike trains and analog
    signals, accessible via the `spiketrains` and `analogsignals` attributes,
    respectively. When accessing data of individual :class:`neo.SpikeTrain` and
    :class:`neo.AnalogSignal` objects, the indexes within these collections is
    used. For instance, `spiketrains[0]` refers to the first spike train, and
    `analogsignals[0]` to the first analog signal.

    Parameters
    ----------
    block : :class:`neo.Block`
        An instance of :class:`neo.Block` containing the trial data. The block
        contains multiple :class:`neo.Segment` objects, each containing the
        data of a single trial.
    description : str, optional
        Textual description of the set of trials, accessible via the
        :attr:`description` attribute.
        Default: None

    Attributes
    ----------
    description : str
        The description of the set of trials.
    n_trials : int
        The total number of trials.
    n_spiketrains_trial_by_trial : List[int]
        The number of spike trains in each trial.
    n_analogsignals_trial_by_trial : List[int]
        The number of analog signals in each trial.

        Examples
    --------
    1. Generate `TrialFromBlock` object to handle data from two trials, each
       containing three spike trains and one analog signal.

    >>> import numpy as np
    >>> import quantities as pq
    >>> import neo
    >>> from elephant.spike_train_generation import StationaryPoissonProcess
    >>> from elephant.trials import TrialsFromBlock
    >>>
    >>> st_generator = StationaryPoissonProcess(rate=10*pq.Hz, t_stop=1*pq.s)
    >>> trial_block = neo.Block()
    >>> for _ in range(2):
    ...     trial = neo.Segment()
    ...     trial.spiketrains = st_generator.generate_n_spiketrains(3)
    ...     signal = np.sin(np.arange(0, 6*np.pi, 2*np.pi/1000))
    ...     signal += np.random.normal(size=signal.shape)
    ...     trial.analogsignals.append(
    ...         neo.AnalogSignal(signal, units=pq.mV,
    ...                          t_stop=1*pq.s,
    ...                          sampling_rate=(1/3000)*pq.Hz)
    ...     )
    ...     trial_block.segments.append(trial)
    >>>
    >>> trials = TrialsFromBlock(trial_block)

    2. Retrieve overall information.

    >>> print(trials.n_trials)
    2
    >>> print(trials.n_spiketrains_trial_by_trial)
    [3, 3]
    >>> print(trials.n_analogsignals_trial_by_trial)
    [1, 1]

    3. Access data in the first trial.

    >>> first_trial = trials[0]
    >>> first_spike_train = first_trial.spiketrains[0]
    >>> analog_signal = first_trial.analogsignals[0]
    """

    def __init__(self, block: Block, **kwargs):
        self.block = block
        super().__init__(**kwargs)

    def __getitem__(self, trial_index: int) -> Segment:
        # Get a specific trial by its index as a Segment
        return self.block.segments[trial_index]

    @property
    def n_trials(self) -> int:
        # Get the number of trials
        return len(self.block.segments)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        # Get the number of SpikeTrain objects in each trial
        return [len(trial.spiketrains) for trial in self.block.segments]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        # Get the number of AnalogSignal objects in each trial
        return [len(trial.analogsignals) for trial in self.block.segments]

    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_list(self, trial_index: int = 0
                                           ) -> SpikeTrainList:
        # Return a list of all spike trains from a trial
        return SpikeTrainList(
            items=[spiketrain for spiketrain in
                   self.block.segments[trial_index].spiketrains]
        )

    def get_spiketrains_trial_by_trial(self, spiketrain_index: int
                                       ) -> SpikeTrainList:
        # Return a list of all spike train repetitions across trials
        return SpikeTrainList(
            items=[segment.spiketrains[spiketrain_index] for
                   segment in self.block.segments]
        )

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_list(self, trial_index: int
                                             ) -> List[AnalogSignal]:
        # Return a list of all analog signals from a trial
        return [analogsignal for analogsignal in
                self.block.segments[trial_index].analogsignals]

    def get_analogsignals_trial_by_trial(self, signal_index: int
                                         ) -> List[AnalogSignal]:
        # Return a list of all analog signal repetitions across trials
        return [segment.analogsignals[signal_index]
                for segment in self.block.segments]


class TrialsFromLists(Trials):
    """
    This class handles trial data structured as a list of lists.

    In this representation, each inner list represents a single trial and
    includes one or more data elements, such as spike trains
    (:class:`neo.SpikeTrain`) and/or analog signals
    (:class:`neo.AnalogSignal`). The order of these elements must remain
    consistent across all trial repetitions.

    The index identifying each spike train or analog signal is determined by
    its position within the list. For example, if each trial contains two
    elements such as a spike train followed by an analog signal, the spike
    train at index 0 corresponds to the first element, and the analog signal
    at index 0 corresponds to the second element.

    Parameters
    ----------
    list_of_trials : list
        A list of lists containing trial data. The inner lists must contain
        spike train (:class:`neo.SpikeTrain`) or analog signal
        (:class:`neo.AnalogSignal`) objects.
    description : str, optional
        Textual description of the set of trials, accessible via the
        :attr:`description` attribute.
        Default: None

    Attributes
    ----------
    description : str
        The description of the set of trials.
    n_trials : int
        The total number of trials.
    n_spiketrains_trial_by_trial : List[int]
        The number of spike trains in each trial.
    n_analogsignals_trial_by_trial : List[int]
        The number of analog signals in each trial.

    Examples
    --------
    1. Generate `TrialFromLists` object to handle data from three trials, each
       containing two spike trains and one analog signal.

    >>> import numpy as np
    >>> import quantities as pq
    >>> import neo
    >>> from elephant.spike_train_generation import StationaryPoissonProcess
    >>> from elephant.trials import TrialsFromLists
    >>>
    >>> st_generator = StationaryPoissonProcess(rate=10*pq.Hz, t_stop=1*pq.s)
    >>> trial_list = [st_generator.generate_n_spiketrains(2) for _ in range(3)]
    >>> for trial in trial_list:
    ...     signal = np.sin(np.arange(0, 6*np.pi, 2*np.pi/1000))
    ...     signal += np.random.normal(size=signal.shape)
    ...     trial.append(
    ...         neo.AnalogSignal(signal, units=pq.mV,
    ...                          t_stop=1*pq.s,
    ...                          sampling_rate=(1/3000)*pq.Hz)
    ...     )
    >>>
    >>> trials = TrialsFromLists(trial_list)

    2. Retrieve overall information.

    >>> print(trials.n_trials)
    3
    >>> print(trials.n_spiketrains_trial_by_trial)
    [2, 2, 2]
    >>> print(trials.n_analogsignals_trial_by_trial)
    [1, 1, 1]

    3. Access data in the first trial.

    >>> first_trial = trials[0]
    >>> first_spike_train = first_trial.spiketrains[0]
    >>> analog_signal = first_trial.analogsignals[0]
    """

    def __init__(self, list_of_trials: list, **kwargs):
        self.list_of_trials = list_of_trials
        super().__init__(**kwargs)

        # Save indexes for quick search of spike trains or analog signals
        # in a trial. The order of elements in the inner list must be
        # consistent across all trials (using the first list, corresponding
        # to the first trial, to fetch the indexes).
        if list_of_trials:
            is_spiketrain = np.array([isinstance(data_element, SpikeTrain)
                                     for data_element in list_of_trials[0]])
            self._spiketrain_index = is_spiketrain.nonzero()[0]
            self._analogsignal_index = (~is_spiketrain).nonzero()[0]
        else:
            self._spiketrain_index = []
            self._analogsignal_index = []

    def __getitem__(self, trial_index: int) -> Segment:
        # Get a specific trial by its index as a Segment
        segment = Segment()
        for element in self.list_of_trials[trial_index]:
            if isinstance(element, SpikeTrain):
                segment.spiketrains.append(element)
            if isinstance(element, AnalogSignal):
                segment.analogsignals.append(element)
        return segment

    @property
    def n_trials(self) -> int:
        # Get the number of trials
        return len(self.list_of_trials)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        # Get the number of SpikeTrain objects in each trial
        return [sum(map(lambda x: isinstance(x, SpikeTrain), trial))
                for trial in self.list_of_trials]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        # Get the number of AnalogSignal objects in each trial
        return [sum(map(lambda x: isinstance(x, AnalogSignal), trial))
                for trial in self.list_of_trials]

    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_list(self, trial_index: int = 0
                                           ) -> SpikeTrainList:
        # Return a list of all spike trains from a trial
        return SpikeTrainList(items=[
            spiketrain for spiketrain in self.list_of_trials[trial_index]
            if isinstance(spiketrain, SpikeTrain)])

    def get_spiketrains_trial_by_trial(self, spiketrain_index: int
                                       ) -> SpikeTrainList:
        # Return a list of all spike train repetitions across trials
        list_idx = self._spiketrain_index[spiketrain_index]
        return SpikeTrainList(items=[trial[list_idx]
                                     for trial in self.list_of_trials])

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_list(self, trial_index: int
                                             ) -> List[AnalogSignal]:
        # Return a list of all analog signals from a trial
        return [analogsignal for analogsignal in
                self.list_of_trials[trial_index]
                if isinstance(analogsignal, AnalogSignal)]

    def get_analogsignals_trial_by_trial(self, signal_index: int
                                         ) -> AnalogSignal:
        # Return a list of all analog signal repetitions across trials
        list_idx = self._analogsignal_index[signal_index]
        return [trial[list_idx] for trial in self.list_of_trials]
