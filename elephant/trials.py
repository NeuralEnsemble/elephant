"""
This module defines :class:`Trials`, the abstract base class
used by all :module:`elephant.trials` classes.

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
import quantities as pq
import neo.utils


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
    def __init__(self, description=None):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.description = description


class TrialsFromLists(Trials):
    """
    This class implements support for handling trials from list of lists.

    Parameters
    ----------
    list_of_trials : list of lists
        A list of lists. Each list entry contains a list of neo.SpikeTrains
        or neo.AnalogSignals.
    """
    def __init__(self, list_of_trials, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.list_of_trials = list_of_trials
        super().__init__(**kwargs)

    def __getitem__(self, trial_number):
        return self.list_of_trials[trial_number]

    @property
    def n_trials(self):
        """
        Get the number of trials.
        """
        return len(self.list_of_trials)

    def n_spiketrains(self):
        """
        Get the number of spiketrains in each trial.
        """
        return[sum(map(lambda x: isinstance(x,  neo.core.SpikeTrain), trial))
               for trial in self.list_of_trials]

    def n_analogsignals(self):
        """
        Get the number of analogsignals in each trial.
        """
        return [sum(map(lambda x: isinstance(x, neo.core.AnalogSignal), trial))
                for trial in self.list_of_trials]


class TrialsFromBlock(Trials):
    """
    This class implements support for handling trials from neo.Block

    Parameters
    ----------
    block : neo.Block
        An instance of neo.Block containing the trials.

    Attributes
    ----------
    n_trials : int
        Calculated based on the number of segments in the block.

    """
    def __init__(self, block, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.block = block
        super().__init__(**kwargs)

    def __getitem__(self, trial_number):
        return self.block.segments[trial_number]

    @property
    def n_trials(self):
        """
        Get the number of trials.
        """
        return len(self.block.segments)

    def n_spiketrains(self):
        """
        Get the number of SpikeTrain instances in each trial.
        """
        return[len(trial.spiketrains) for trial in self.block.segments]

    def n_analogsignals(self):
        """
        Get the number of AnalogSignals instances in each trial.
        """
        return[len(trial.analogsignals) for trial in self.block.segments]
