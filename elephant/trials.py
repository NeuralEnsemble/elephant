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


class TrialsFromBlock(Trials):
    """
    This class implements support for handling trials from neo.Block

    Parameters
    ----------
    block : neo.Block
        An instance of neo.Block containing the trials.
    cut_events: list of neo.Event objects, optional
        Events used to cut the trials.
    pre: pq.Quantity, optional
        Defines the time period before the cut_events, which will be added
        to the trial.
    post: pq.Quantity, optional
        defines the time period after the cut_events, which will be added
        to the trial.

    Attributes
    ----------
    n_trials : int
        Calculated based on the number of segments in the block.

    """
    def __init__(self, block,
                 cut_events=None, pre=None, post=None,
                 **kwargs):
        """
        Constructor
        (actual documentation is in class documentation, see above!)
        """
        self.block = block
        self.cut_events = cut_events
        self.pre = pre
        self.post = post
        super().__init__(**kwargs)

    def __getitem__(self, trial_number):
        return self.block.segments[trial_number]

    @property
    def n_trials(self):
        return len(self.block.segments)

    def cut_trials(self, reset_time=True):
        """
        cut trials

        Parameters
        ---------
        reset_time : bool, optional
            Reset time base for each trial, so each trial starts at 0 seconds
        """
        # Create epochs
        cut_epochs = neo.utils.add_epoch(
            self.block.segments[0],
            event1=self.cut_events, event2=None,
            pre=self.pre, post=self.post,
            attach_result=False,
            name='trial_epochs')

        # Create the new block
        trials_block = neo.Block()

        # Cut the recording segment into the trials, as defined by the epochs
        trials_block.segments = neo.utils.cut_segment_by_epoch(
            self.block.segments[0],
            cut_epochs,
            reset_time=reset_time)

        self.block = trials_block
