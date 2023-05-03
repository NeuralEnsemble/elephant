"""
This module defines :class:`Trials`, the abstract base class
used by all :module:`elephant.trials` classes.

:copyright: Copyright 2014-2023 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import neo.utils
from abc import ABCMeta, abstractmethod
from typing import List
from neo.core import Segment

class Trials:
    """
    Base class for handling trials.

    This is the base class from which all trial objects inherit.
    This class implements support for universally recommended arguments.

    Attributes
    ----------
    description : string, optional
        The description of the trials.
        Default: None.

    """

    __metaclass__ = ABCMeta

    def __init__(self, description: str = "Trials"):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.description = description

    @abstractmethod
    def __getitem__(self, trial_number: int) -> neo.core.Segment:
        """Get a specific trial by number"""
        pass

    @abstractmethod
    def n_trials(self) -> int:
        """Get the number of trials."""
        pass

    @abstractmethod
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        """Get the number of spiketrains in each trial as a list."""
        pass

    @abstractmethod
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        """Get the number of analogsignals in each trial as a list."""
        pass

    @abstractmethod
    def get_spiketrains_from_trial_as_list(self, trial_number: int
                                   ) -> List[neo.core.SpikeTrain]:
        """
        Get all spiketrains from a specific trial and return a list.

        Parameters
        ----------
        trial_number : int
            Trial number to get the spiketrains from, e.g. choose
            0 for the first trial.

        Returns
        -------
        list of spiketrains: List[neo.core.SpikeTrain]
        """
        pass

    @abstractmethod
    def get_analogsignals_from_trial_as_list(self, trial_number: int
                                   ) -> List[neo.core.AnalogSignal]:
        """
        Get all analogsignals from a specific trial and return a list.

        Parameters
        ----------
        trial_number : int
            Trial number to get the spiketrains from, e.g. choose
            0 for the first trial.

        Returns
        -------
        list of analogsignals: List[neo.core.AnalogSignal]
        """
        pass


class TrialsFromLists(Trials):
    """
    This class implements support for handling trials from list of lists.

    Parameters
    ----------
    list_of_trials : list of lists
        A list of lists. Each list entry contains a list of neo.SpikeTrains
        or neo.AnalogSignals.

    Properties
    ----------
    see Trials class


    """
    def __init__(self, list_of_trials: list, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.list_of_trials = list_of_trials
        super().__init__(**kwargs)

    def __getitem__(self, trial_number: int) -> neo.core.Segment:
        """Get a specific trial by number"""
        segment=Segment()
        for element in self.list_of_trials[trial_number]:
            if isinstance(element, neo.core.SpikeTrain):
                segment.spiketrains.append(element)
            if isinstance(element, neo.core.AnalogSignal):
                segment.analogsignals.append(element)
        return segment

    @property
    def n_trials(self) -> int:
        """Get the number of trials."""
        return len(self.list_of_trials)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        """Get the number of spiketrains in each trial."""
        return[sum(map(lambda x: isinstance(x,  neo.core.SpikeTrain), trial))
               for trial in self.list_of_trials]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        """Get the number of analogsignals in each trial."""
        return [sum(map(lambda x: isinstance(x, neo.core.AnalogSignal), trial))
                for trial in self.list_of_trials]

    def get_spiketrains_from_trial_as_list(self, trial_number: int =0
                                   ) -> List[neo.core.SpikeTrain]:
        """Return a list of all spiketrains from a trial"""
        return [spiketrain for spiketrain in self.list_of_trials[trial_number]
                if isinstance(spiketrain, neo.core.SpikeTrain)]

    def get_analogsignals_from_trial_as_list(self, trial_number: int =0
                                   ) -> List[neo.core.AnalogSignal]:
        """Return a list of all analogsignals from a trial"""
        return [analogsignal for analogsignal in self.list_of_trials[trial_number]
                if isinstance(analogsignal, neo.core.AnalogSignal)]


class TrialsFromBlock(Trials):
    """
    This class implements support for handling trials from neo.Block

    Parameters
    ----------
    block : neo.Block
        An instance of neo.Block containing the trials.
        The structure is assumed to follow the neo representation:
        A block contains multiple segments which are considered to contain the
        single trials.

    Properties
    ----------
    See Trials Class

    """
    def __init__(self, block: neo.core.block, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.block = block
        super().__init__(**kwargs)

    def __getitem__(self, trial_number: int) -> neo.core.segment:
        return self.block.segments[trial_number]

    @property
    def n_trials(self) -> int:
        """Get the number of trials."""
        return len(self.block.segments)

    @property
    def n_spiketrains_trial_by_trial(self) -> List[int]:
        """Get the number of SpikeTrain instances in each trial."""
        return[len(trial.spiketrains) for trial in self.block.segments]

    @property
    def n_analogsignals_trial_by_trial(self) -> List[int]:
        """Get the number of AnalogSignals instances in each trial."""
        return[len(trial.analogsignals) for trial in self.block.segments]

    def get_spiketrains_from_trial_as_list(self, trial_number: int =0
                                   ) -> List[neo.core.SpikeTrain]:
        """Return a list of all spiketrains from a trial"""
        return [spiketrain for spiketrain in
                self.block.segments[trial_number].spiketrains]

    def get_analogsignals_from_trial_as_list(self, trial_number: int =0
                                   ) -> List[neo.core.AnalogSignal]:
        """Return a list of all analogsignals from a trial"""
        return [analogsignal for analogsignal in
                self.block.segments[trial_number].analogsignals]
