"""
This module defines :class:`Trials`, the abstract base class
used by all :module:`elephant.trials` classes.

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import neo.utils
from abc import ABCMeta, abstractmethod
from typing import List

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
    def __getitem__(self, trial_number: int):
        """Get a specific trial by number"""
        pass

    @abstractmethod
    def n_trials(self) -> int:
        """Get the number of trials."""
        pass

    @abstractmethod
    def n_spiketrains(self) -> List[int]:
        """Get the number of spiketrains in each trial."""
        pass

    @abstractmethod
    def n_analogsignals(self) -> List[int]:
        """Get the number of analogsignals in each trial."""
        pass

    @abstractmethod
    def get_spiketrains_from_trial(self, trial_number: int
                                   ) -> List[neo.core.SpikeTrain]:
        """Get all spiketrains from a specific trial and return a list"""
        pass


class TrialsFromLists(Trials):
    """
    This class implements support for handling trials from list of lists.

    Parameters
    ----------
    list_of_trials : list of lists
        A list of lists. Each list entry contains a list of neo.SpikeTrains
        or neo.AnalogSignals.
    """
    def __init__(self, list_of_trials: list, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.list_of_trials = list_of_trials
        super().__init__(**kwargs)

    def __getitem__(self, trial_number: int):
        """Get a specific trial by number"""
        return self.list_of_trials[trial_number]

    @property
    def n_trials(self) -> int:
        """Get the number of trials."""
        return len(self.list_of_trials)

    @property
    def n_spiketrains(self) -> List[int]:
        """Get the number of spiketrains in each trial."""
        return[sum(map(lambda x: isinstance(x,  neo.core.SpikeTrain), trial))
               for trial in self.list_of_trials]

    @property
    def n_analogsignals(self) -> List[int]:
        """Get the number of analogsignals in each trial."""
        return [sum(map(lambda x: isinstance(x, neo.core.AnalogSignal), trial))
                for trial in self.list_of_trials]

    def get_spiketrains_from_trial(self, trial_number: int =0
                                   ) -> List[neo.core.SpikeTrain]:
        """Return a list of all spiketrians from a trial"""
        return self.list_of_trials[trial_number]


class TrialsFromBlock(Trials):
    """
    This class implements support for handling trials from neo.Block

    Parameters
    ----------
    block : neo.Block
        An instance of neo.Block containing the trials.

    Properties
    ----------
    n_trials : int
        Calculated based on the number of segments in the block.

    """
    def __init__(self, block: neo.core.block, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.block = block
        super().__init__(**kwargs)

    def __getitem__(self, trial_number) -> neo.core.segment:
        return self.block.segments[trial_number]

    @property
    def n_trials(self) -> int:
        """Get the number of trials."""
        return len(self.block.segments)

    @property
    def n_spiketrains(self) -> List[int]:
        """Get the number of SpikeTrain instances in each trial."""
        return[len(trial.spiketrains) for trial in self.block.segments]

    @property
    def n_analogsignals(self) -> List[int]:
        """Get the number of AnalogSignals instances in each trial."""
        return[len(trial.analogsignals) for trial in self.block.segments]

    def get_spiketrains_from_trial(self, trial_number: int =0
                                   ) -> List[neo.core.SpikeTrain]:
        """Return a list of all spiketrians from a trial"""
        return self.block.segments[trial_number].spiketrains
