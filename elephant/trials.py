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
popular option is to store trials as lists of lists, where the outer refers to
individual lists, and inner lists contain Neo data objects
(:class:`neo.SpikeTrain` and :class:`neo.AnalogSignal`) containing individual
data of each trial.

The classes of this module abstract from these individual data representations
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

Tutorial
--------
For a detailed example on the classes usage and trial handling for analyses
using Elephant, check the :doc:`tutorial <../tutorials/trials>`.

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/trials.ipynb

:copyright: Copyright 2014-2025 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from typing import List

from functools import wraps
import numpy as np

import neo
from neo.core import Segment, Block
from neo.core.spiketrainlist import SpikeTrainList
from elephant.utils import deprecated_alias


def trials_to_list_of_spiketrainlist(method):
    """
    Decorator to convert `Trials` object to a list of `SpikeTrainList` before
    calling the wrapped method.

    Parameters
    ----------
    method: callable
        The method to be decorated.

    Returns
    -------
    callable
        The decorated method.

    Examples
    --------
    The decorator can be used as follows:

        >>> @trials_to_list_of_spiketrainlist
        ... def process_data(self, spiketrains):
        ...     return None
    """

    @wraps(method)
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

        return method(*new_args, **new_kwargs)

    return wrapper


class Trials(ABC):
    """
    Base class for handling trials.

    This is the base class from which all trial objects inherit.
    This class implements support for universally recommended arguments.

    Parameters
    ----------
    description : string, optional
        A textual description of the set of trials. Can be accessed via the
        class attribute `description`.
        Default: None.

    """


    def __init__(self, description: str = "Trials"):
        """Create an instance of the trials class."""
        self.description = description

    @abstractmethod
    def __getitem__(self, trial_index: int) -> neo.core.Segment:
        # Get a specific trial by its index as a Segment
        raise NotImplementedError

    @abstractmethod
    def n_trials(self) -> int:
        """Get the number of trials.

        Returns
        -------
        int: Number of trials
        """
        raise NotImplementedError

    @abstractmethod
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        """Get the number of spike trains in each trial as a list.

        Returns
        -------
        list of int: For each trial, contains the number of spike trains in the
            trial.
        """
        raise NotImplementedError

    @abstractmethod
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        """Get the number of analogsignal objects in each trial as a list.

        Returns
        -------
        list of int: For each trial, contains the number of analogsignal objects
            in the trial.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        """Get trial as segment.
    @deprecated_alias(trial_id="trial_index")
    def get_trial_as_segment(self, trial_index: int) -> neo.core.Segment:

        Parameters
        ----------
        trial_index : int
            Index of the trial to retrieve (zero-based).

        Returns
        -------
        class:`neo.Segment`: a segment containing all spike trains and
            analogsignal objects of the trial.
        """
        pass

    @abstractmethod
    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_block(self, trial_indexes: List[int] = None
                            ) -> neo.core.Block:
        """Get trials as block.

        Parameters
        ----------
        trial_indexes : list of int
            Indexes of the trials to include in the Block (zero-based).
            If None is specified, all trials are returned.
            Default: None

        Returns
        -------
        class:`neo.Block`: a Block  containing :class:`neo.Segment` objects for
            each of the selected trials, each containing all spike trains and
            analogsignal objects of the corresponding trial.
        """
        pass

    @abstractmethod
    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_list(self, trial_indexes: List[int] = None
                           ) -> neo.core.spiketrainlist.SpikeTrainList:
        """Get trials as list of segments.

        Parameters
        ----------
        trial_indexes : list of int
            Indexes of the trials to include in the list (zero-based).
            If None is specified, all trials are returned.
            Default: None

        Returns
        -------
        list of :class:`neo.Segment`: a list  containing :class:`neo.Segment`
            objects for each of the selected trials, each containing all spike
            trains and analogsignal objects of the corresponding trial.
        """
        pass

    @abstractmethod
    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_list(self, trial_index: int) -> (
                                       neo.core.spiketrainlist.SpikeTrainList):
        """
        Get all spike trains from a specific trial and return a list.

        Parameters
        ----------
        trial_index : int
            Index of the trial to get the spike trains from (zero-based).

        Returns
        -------
        list of :class:`neo.SpikeTrain`
            List of all spike trains of the trial.
        """
        raise NotImplementedError

    @abstractmethod
    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_segment(self, trial_index: int
                                              ) -> neo.core.Segment:
        """
        Get all spike trains from a specific trial and return a Segment.

        Parameters
        ----------
        trial_index : int
            Index of the trial to get the spike trains from (zero-based).

        Returns
        -------
        :class:`neo.Segment`: Segment containing all spike trains of the trial.
        """
        pass

    @abstractmethod
    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_list(self, trial_index: int
                                             ) -> List[neo.core.AnalogSignal]:
        """
        Get all analogsignals from a specific trial and return a list.

        Parameters
        ----------
        trial_index : int
            Index of the trial to get the analog signals from (zero-based).

        Returns
        -------
        list of :class`neo.AnalogSignal`: list of all analogsignal objects of
            the trial.
        """
        raise NotImplementedError

    @abstractmethod
    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_segment(self, trial_index: int
                                                ) -> neo.core.Segment:
        """
        Get all analogsignal objects from a specific trial and return a
        :class:`neo.Segment`.

        Parameters
        ----------
        trial_index : int
            Index of the trial to get the analog signals from (zero-based).

        Returns
        -------
        class:`neo.Segment`: segment containing all analogsignal objects of
            the trial.
        """

    @abstractmethod
    def get_spiketrains_trial_by_trial(self, spiketrain_index: int) -> (
                                       neo.core.spiketrainlist.SpikeTrainList):
        """
        Retrieve a spike train across all its trial repetitions.

        This method returns a list containing :class:`neo.core.SpikeTrain`
        objects corresponding to the same spike train (e.g., from a consistent
        recording channel or neuronal source) across multiple trials.

        Parameters
        ----------
        spiketrain_index : int
            Index of the spike train to retrieve across trials. Indexing
            starts at 0, so `spiketrain_index == 0` corresponds to the first
            spike train in the trial data.

        Returns
        -------
        list of :class:`neo.core.SpikeTrain`
            A list-like container with the :class:`neo.core.SpikeTrain`
            objects for the specified `spiketrain_id`, ordered from the first
            trial (ID 0) to the last (ID `n_trials - 1`).
        """
        raise NotImplementedError

    @abstractmethod
    def get_analogsignals_trial_by_trial(self, signal_index: int
                                         ) -> List[neo.core.AnalogSignal]:
        """
        Retrieve an analog signal across all its trial repetitions.

        This method returns a list containing :class:`neo.core.AnalogSignal`
        objects corresponding to a continuous signal recorded from a consistent
        recording channel or neuronal source across multiple trials.

        Parameters
        ----------
        signal_index : int
            Index of the analog signal to retrieve across trials. Indexing
            starts at 0, so `signal_index == 0` corresponds to the first
            analog signal in the trial data.

        Returns
        -------
        list of :class:`neo.core.AnalogSignal`
            A list with the :class:`neo.core.AnalogSignal` objects for the
            specified `signal_id`, ordered from the first trial (ID 0) to the
            last (ID `n_trials - 1`).
        """
        raise NotImplementedError


class TrialsFromBlock(Trials):
    """
    This class implements support for handling trials from neo.Block.

    Parameters
    ----------
    block : neo.Block
        An instance of neo.Block containing the trials.
        The structure is assumed to follow the neo representation:
        A block contains multiple segments which are considered to contain the
        single trials.
    description : string, optional
        A textual description of the set of trials. Can be accessed via the
        class attribute `description`.
        Default: None.
    """

    def __init__(self, block: neo.core.block, **kwargs):
        self.block = block
        super().__init__(**kwargs)

    def __getitem__(self, trial_index: int) -> neo.core.segment:
        # Get a specific trial by its index as a Segment
        return self.block.segments[trial_index]

    @deprecated_alias(trial_id="trial_index")
    def get_trial_as_segment(self, trial_index: int) -> neo.core.Segment:
        # Get a specific trial by its index as a Segment
        return self.__getitem__(trial_index)

    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_block(self, trial_indexes: List[int] = None
                            ) -> neo.core.Block:
        # Get a set of trials by their indexes as a Block
        block = Block()
        if not trial_indexes:
            trial_indexes = list(range(self.n_trials))
        for trial_index in trial_indexes:
            block.segments.append(self.get_trial_as_segment(trial_index))
        return block

    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_list(self, trial_indexes: List[int] = None
                           ) -> List[neo.core.Segment]:
        # Get a set of trials by their indexes as a list of Segment
        if not trial_indexes:
            trial_indexes = list(range(self.n_trials))
        return [self.get_trial_as_segment(trial_index)
                for trial_index in trial_indexes]

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
    def get_spiketrains_from_trial_as_list(self, trial_index: int = 0) -> (
                                       neo.core.spiketrainlist.SpikeTrainList):
        # Return a list of all spike trains from a trial
        return SpikeTrainList(items=[spiketrain for spiketrain in
                                     self.block.segments[trial_index].spiketrains])

    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_segment(self, trial_index: int) -> (
            neo.core.Segment):
        # Return a Segment with all spike trains from a trial
        segment = neo.core.Segment()
        for spiketrain in self.get_spiketrains_from_trial_as_list(trial_index
                                                                  ):
            segment.spiketrains.append(spiketrain)
        return segment

    def get_spiketrains_trial_by_trial(self, spiketrain_index: int) -> (
            neo.core.spiketrainlist.SpikeTrainList):
        # Return a list of all spike train repetitions across trials
        return SpikeTrainList(items=[segment.spiketrains[spiketrain_index] for
                                     segment in self.block.segments])

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_list(self, trial_index: int) -> (
            List[neo.core.AnalogSignal]):
        # Return a list of all analog signals from a trial
        return [analogsignal for analogsignal in
                self.block.segments[trial_index].analogsignals]

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_segment(self, trial_index: int) -> (
            neo.core.Segment):
        # Return a Segment with all analog signals from a trial
        segment = neo.core.Segment()
        for analogsignal in self.get_analogsignals_from_trial_as_list(
                trial_index):
            segment.analogsignals.append(analogsignal)
        return segment

    def get_analogsignals_trial_by_trial(self, signal_index: int) -> (
            List[neo.core.AnalogSignal]):
        # Return a list of all analog signal repetitions across trials
        return [segment.analogsignals[signal_index]
                for segment in self.block.segments]


class TrialsFromLists(Trials):
    """
    This class implements support for handling trials from list of lists.

    Parameters
    ----------
    list_of_trials : list of lists
        A list of lists. Each list entry contains a list of neo.SpikeTrains
        and/or neo.AnalogSignals.
    description : string, optional
        A textual description of the set of trials. Can be accessed via the
        class attribute `description`.
        Default: None.
    """

    def __init__(self, list_of_trials: list, **kwargs):
        self.list_of_trials = list_of_trials
        super().__init__(**kwargs)

        # Save indexes for quick search of spike trains or analog signals
        # in a trial. The order of elements in the inner list must be
        # consistent across all trials (using the first list, corresponding
        # to the first trial, to fetch the indexes).
        if list_of_trials:
            is_spiketrain = np.array([isinstance(data_element, neo.SpikeTrain)
                                     for data_element in list_of_trials[0]])
            self._spiketrain_index = is_spiketrain.nonzero()[0]
            self._analogsignal_index = (~is_spiketrain).nonzero()[0]
        else:
            self._spiketrain_index = []
            self._analogsignal_index = []

    def __getitem__(self, trial_index: int) -> neo.core.Segment:
        # Get a specific trial by its index as a Segment
        segment = Segment()
        for element in self.list_of_trials[trial_index]:
            if isinstance(element, neo.core.SpikeTrain):
                segment.spiketrains.append(element)
            if isinstance(element, neo.core.AnalogSignal):
                segment.analogsignals.append(element)
        return segment

    @deprecated_alias(trial_id="trial_index")
    def get_trial_as_segment(self, trial_index: int) -> neo.core.Segment:
        # Get a specific trial by its index as a Segment
        return self.__getitem__(trial_index)

    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_block(self, trial_indexes: List[int] = None
                            ) -> neo.core.Block:
        # Get a set of trials by their indexes as a Block
        if not trial_indexes:
            trial_indexes = list(range(self.n_trials))
        block = Block()
        for trial_index in trial_indexes:
            block.segments.append(self.get_trial_as_segment(trial_index))
        return block

    @deprecated_alias(trial_ids="trial_indexes")
    def get_trials_as_list(self, trial_indexes: List[int] = None
                           ) -> List[neo.core.Segment]:
        # Get a set of trials by their indexes as a list of Segment
        if not trial_indexes:
            trial_indexes = list(range(self.n_trials))
        return [self.get_trial_as_segment(trial_index)
                for trial_index in trial_indexes]

    @property
    def n_trials(self) -> int:
        # Get the number of trials
        return len(self.list_of_trials)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        # Get the number of SpikeTrain objects in each trial
        return [sum(map(lambda x: isinstance(x, neo.core.SpikeTrain), trial))
                for trial in self.list_of_trials]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        # Get the number of AnalogSignal objects in each trial
        return [sum(map(lambda x: isinstance(x, neo.core.AnalogSignal), trial))
                for trial in self.list_of_trials]

    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_list(self, trial_index: int = 0) -> (
                                       neo.core.spiketrainlist.SpikeTrainList):
        # Return a list of all spike trains from a trial
        return SpikeTrainList(items=[
            spiketrain for spiketrain in self.list_of_trials[trial_index]
            if isinstance(spiketrain, neo.core.SpikeTrain)])

    @deprecated_alias(trial_id="trial_index")
    def get_spiketrains_from_trial_as_segment(self, trial_index: int) -> (
            neo.core.Segment):
        # Return a Segment with all spike trains from a trial
        segment = neo.core.Segment()
        for spiketrain in self.get_spiketrains_from_trial_as_list(trial_index):
            segment.spiketrains.append(spiketrain)
        return segment

    def get_spiketrains_trial_by_trial(self, spiketrain_index: int) -> (
            neo.core.spiketrainlist.SpikeTrainList):
        # Return a list of all spike train repetitions across trials
        list_idx = self._spiketrain_index[spiketrain_index]
        return SpikeTrainList(items=[trial[list_idx]
                                     for trial in self.list_of_trials])

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_list(self, trial_index: int) -> (
            List[neo.core.AnalogSignal]):
        # Return a list of all analog signals from a trial
        return [analogsignal for analogsignal in
                self.list_of_trials[trial_index]
                if isinstance(analogsignal, neo.core.AnalogSignal)]

    @deprecated_alias(trial_id="trial_index")
    def get_analogsignals_from_trial_as_segment(self, trial_index: int) -> (
            neo.core.Segment):
        # Return a Segment with all analog signals from a trial
        segment = neo.core.Segment()
        for analogsignal in self.get_analogsignals_from_trial_as_list(
                trial_index):
            segment.analogsignals.append(analogsignal)
        return segment

    def get_analogsignals_trial_by_trial(self, signal_index: int) -> (
            List[neo.core.AnalogSignal]):
        # Return a list of all analog signal repetitions across trials
        list_idx = self._analogsignal_index[signal_index]
        return [trial[list_idx] for trial in self.list_of_trials]
