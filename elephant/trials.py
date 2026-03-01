"""
This module defines the basic classes that represent trials in Elephant.

Many neuroscience methods rely on the concept of repeated trials to improve the
estimate of quantities measured from the data. In the simplest case, results
from multiple trials are averaged, in other scenarios more intricate steps must
be taken in order to pool information from each repetition of a trial. Typically,
trials are considered as fixed time intervals tied to a specific event in the
experiment, such as the onset of a stimulus.

Neo does not impose a specific way in which trials are to be represented. A
natural way to represent trials is to have a :class:`neo.Block` containing multiple
:class:`neo.Segment` objects, each representing the data of one trial. Another popular
option is to store trials as lists of lists, where the outer refers to
individual lists, and inner lists contain Neo data objects (:class:`neo.SpikeTrain`
and :class:`neo.AnalogSignal` containing individual data of each trial.

The classes of this module abstract from these individual data representations
by introducing a set of :class:`Trials` classes with a common API. These classes
are initialized by a supported way of structuring trials, e.g.,
:class:`TrialsFromBlock` for the first method described above. Internally,
:class:`Trials` class will not convert this representation, but provide access
to data in specific trials (e.g., all spike trains in trial 5) or general
information about the trial structure (e.g., how many trials are there?)  via a
fixed API. Trials are consecutively numbered, starting at a trial ID of 0.

In the release, the classes :class:`TrialsFromBlock` and
:class:`TrialsFromLists` provide this unified way to access trial data.

.. autosummary::
    :toctree: _toctree/trials
    :template: trials_class.rst

    TrialsFromBlock
    TrialsFromLists

:copyright: Copyright 2014-2024 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from abc import ABCMeta, abstractmethod
from typing import List

import neo.utils
from neo.core import Segment, Block
from neo.core.spiketrainlist import SpikeTrainList


class Trials:
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

    __metaclass__ = ABCMeta

    def __init__(self, description: str = "Trials"):
        """Create an instance of the trials class."""
        self.description = description

    @abstractmethod
    def __getitem__(self, trial_number: int) -> neo.core.Segment:
        """Get a specific trial by number."""
        pass

    @abstractmethod
    def n_trials(self) -> int:
        """Get the number of trials.

        Returns
        -------
        int: Number of trials
        """
        pass

    @abstractmethod
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        """Get the number of spike trains in each trial as a list.

        Returns
        -------
        list of int: For each trial, contains the number of spike trains in the
            trial.
        """
        pass

    @abstractmethod
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        """Get the number of analogsignal objects in each trial as a list.

        Returns
        -------
        list of int: For each trial, contains the number of analogsignal objects
            in the trial.
        """
        pass

    @abstractmethod
    def get_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        """Get trial as segment.

        Parameters
        ----------
        trial_id : int
            Trial number to get (starting at trial ID 0).

        Returns
        -------
        class:`neo.Segment`: a segment containing all spike trains and
            analogsignal objects of the trial.
        """
        pass

    @abstractmethod
    def get_trials_as_block(self, trial_ids: List[int] = None) -> neo.core.Block:
        """Get trials as block.

        Parameters
        ----------
        trial_ids : list of int
            Trial IDs to include in the Block (starting at trial ID 0).
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
    def get_trials_as_list(
        self, trial_ids: List[int] = None
    ) -> neo.core.spiketrainlist.SpikeTrainList:
        """Get trials as list of segments.

        Parameters
        ----------
        trial_ids : list of int
            Trial IDs to include in the list (starting at trial ID 0).
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
    def get_spiketrains_from_trial_as_list(
        self, trial_id: int
    ) -> neo.core.spiketrainlist.SpikeTrainList:
        """
        Get all spike trains from a specific trial and return a list.

        Parameters
        ----------
        trial_id : int
            Trial ID to get the spike trains from (starting at trial ID 0).

        Returns
        -------
        list of :class:`neo.SpikeTrain`
            List of all spike trains of the trial.
        """
        pass

    @abstractmethod
    def get_spiketrains_from_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        """
        Get all spike trains from a specific trial and return a Segment.

        Parameters
        ----------
        trial_id : int
            Trial ID to get the spike trains from (starting at trial ID 0).

        Returns
        -------
        :class:`neo.Segment`: Segment containing all spike trains of the trial.
        """
        pass

    @abstractmethod
    def get_analogsignals_from_trial_as_list(
        self, trial_id: int
    ) -> List[neo.core.AnalogSignal]:
        """
        Get all analogsignals from a specific trial and return a list.

        Parameters
        ----------
        trial_id : int
            Trial ID to get the analogsignals from (starting at trial ID 0).

        Returns
        -------
        list of :class`neo.AnalogSignal`: list of all analogsignal objects of
            the trial.
        """
        pass

    @abstractmethod
    def get_analogsignals_from_trial_as_segment(
        self, trial_id: int
    ) -> neo.core.Segment:
        """
        Get all analogsignal objects from a specific trial and return a
        :class:`neo.Segment`.

        Parameters
        ----------
        trial_id : int
            Trial ID to get the analogsignals from (starting at trial ID 0).

        Returns
        -------
        class:`neo.Segment`: segment containing all analogsignal objects of
            the trial.
        """


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

    def __getitem__(self, trial_number: int) -> neo.core.segment:
        return self.block.segments[trial_number]

    def get_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        # Get a specific trial by number as a segment
        return self.__getitem__(trial_id)

    def get_trials_as_block(self, trial_ids: List[int] = None) -> neo.core.Block:
        # Get a block of trials by trial numbers
        block = Block()
        if not trial_ids:
            trial_ids = list(range(self.n_trials))
        for trial_number in trial_ids:
            block.segments.append(self.get_trial_as_segment(trial_number))
        return block

    def get_trials_as_list(self, trial_ids: List[int] = None) -> List[neo.core.Segment]:
        if not trial_ids:
            trial_ids = list(range(self.n_trials))
        # Get a list of segments by trial numbers
        return [self.get_trial_as_segment(trial_number) for trial_number in trial_ids]

    @property
    def n_trials(self) -> int:
        # Get the number of trials.
        return len(self.block.segments)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        # Get the number of SpikeTrain instances in each trial.
        return [len(trial.spiketrains) for trial in self.block.segments]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        # Get the number of AnalogSignals instances in each trial.
        return [len(trial.analogsignals) for trial in self.block.segments]

    def get_spiketrains_from_trial_as_list(
        self, trial_id: int = 0
    ) -> neo.core.spiketrainlist.SpikeTrainList:
        # Return a list of all spike trains from a trial
        return SpikeTrainList(
            items=[
                spiketrain for spiketrain in self.block.segments[trial_id].spiketrains
            ]
        )

    def get_spiketrains_from_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        # Return a segment with all spiketrains from a trial
        segment = neo.core.Segment()
        for spiketrain in self.get_spiketrains_from_trial_as_list(trial_id):
            segment.spiketrains.append(spiketrain)
        return segment

    def get_analogsignals_from_trial_as_list(
        self, trial_id: int
    ) -> List[neo.core.AnalogSignal]:
        # Return a list of all analogsignals from a trial
        return [
            analogsignal for analogsignal in self.block.segments[trial_id].analogsignals
        ]

    def get_analogsignals_from_trial_as_segment(
        self, trial_id: int
    ) -> neo.core.Segment:
        # Return a segment with all analogsignals from a trial
        segment = neo.core.Segment()
        for analogsignal in self.get_analogsignals_from_trial_as_list(trial_id):
            segment.analogsignals.append(analogsignal)
        return segment


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
        # Constructor
        # (actual documentation is in class documentation, see above!)
        self.list_of_trials = list_of_trials
        super().__init__(**kwargs)

    def __getitem__(self, trial_number: int) -> neo.core.Segment:
        # Get a specific trial by number
        segment = Segment()
        for element in self.list_of_trials[trial_number]:
            if isinstance(element, neo.core.SpikeTrain):
                segment.spiketrains.append(element)
            if isinstance(element, neo.core.AnalogSignal):
                segment.analogsignals.append(element)
        return segment

    def get_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        # Get a specific trial by number as a segment
        return self.__getitem__(trial_id)

    def get_trials_as_block(self, trial_ids: List[int] = None) -> neo.core.Block:
        if not trial_ids:
            trial_ids = list(range(self.n_trials))
        # Get a block of trials by trial numbers
        block = Block()
        for trial_number in trial_ids:
            block.segments.append(self.get_trial_as_segment(trial_number))
        return block

    def get_trials_as_list(self, trial_ids: List[int] = None) -> List[neo.core.Segment]:
        if not trial_ids:
            trial_ids = list(range(self.n_trials))
        # Get a list of segments by trial numbers
        return [self.get_trial_as_segment(trial_number) for trial_number in trial_ids]

    @property
    def n_trials(self) -> int:
        # Get the number of trials.
        return len(self.list_of_trials)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        # Get the number of spiketrains in each trial.
        return [
            sum(map(lambda x: isinstance(x, neo.core.SpikeTrain), trial))
            for trial in self.list_of_trials
        ]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        # Get the number of analogsignals in each trial.
        return [
            sum(map(lambda x: isinstance(x, neo.core.AnalogSignal), trial))
            for trial in self.list_of_trials
        ]

    def get_spiketrains_from_trial_as_list(
        self, trial_id: int = 0
    ) -> neo.core.spiketrainlist.SpikeTrainList:
        # Return a list of all spiketrains from a trial
        return SpikeTrainList(
            items=[
                spiketrain
                for spiketrain in self.list_of_trials[trial_id]
                if isinstance(spiketrain, neo.core.SpikeTrain)
            ]
        )

    def get_spiketrains_from_trial_as_segment(self, trial_id: int) -> neo.core.Segment:
        # Return a segment with all spiketrains from a trial
        segment = neo.core.Segment()
        for spiketrain in self.get_spiketrains_from_trial_as_list(trial_id):
            segment.spiketrains.append(spiketrain)
        return segment

    def get_analogsignals_from_trial_as_list(
        self, trial_id: int
    ) -> List[neo.core.AnalogSignal]:
        # Return a list of all analogsignals from a trial
        return [
            analogsignal
            for analogsignal in self.list_of_trials[trial_id]
            if isinstance(analogsignal, neo.core.AnalogSignal)
        ]

    def get_analogsignals_from_trial_as_segment(
        self, trial_id: int
    ) -> neo.core.Segment:
        # Return a segment with all analogsignals from a trial
        segment = neo.core.Segment()
        for analogsignal in self.get_analogsignals_from_trial_as_list(trial_id):
            segment.analogsignals.append(analogsignal)
        return segment
