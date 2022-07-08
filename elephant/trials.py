"""
This module defines :class:`Trials`, the abstract base class
used by all :module:`elephant.trials` classes.
"""


class Trials:
    """
    Base class for handling trials.

    This is the base class from which all trial objects inherit.
    This class implements support for universally recommended arguments.

    Parameters
    ----------
    description : string, optional
        The description of the trials.
        Default: None.

    Attributes
    ----------
    n_trials : int
        The total number of trials. Calculated attribute.
    """
    def __init__(self, description=None):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.description = description
        self.n_trials = None


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


class TrialsFromBlock(Trials):
    """
    This class implements support for handling trials from neo.Block

    Parameters
    ----------
    block : neo.Block object
        An instance of neo.Block.

    """
    def __init__(self, block, **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.block = block
        super().__init__(**kwargs)
