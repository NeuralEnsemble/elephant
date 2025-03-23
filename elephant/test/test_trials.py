# -*- coding: utf-8 -*-
"""
Unit tests for the trials objects.

:copyright: Copyright 2014-2024 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import neo.utils
import quantities as pq
from neo.core import AnalogSignal

from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.trials import TrialsFromBlock, TrialsFromLists


def _create_trials_block(
    n_trials: int = 0, n_spiketrains: int = 2, n_analogsignals: int = 2
) -> neo.core.Block:
    """Create block with n_trials, n_spiketrains and n_analog_signals"""
    block = neo.Block(name="test_block")
    for trial in range(n_trials):
        segment = neo.Segment(name=f"No. {trial}")
        spiketrains = StationaryPoissonProcess(
            rate=50.0 * pq.Hz, t_start=0 * pq.ms, t_stop=1000 * pq.ms
        ).generate_n_spiketrains(n_spiketrains=n_spiketrains)
        analogsignals = [
            AnalogSignal(signal=[0.01, 3.3, 9.3], units="uV", sampling_rate=1 * pq.Hz)
            for _ in range(n_analogsignals)
        ]
        for spiketrain in spiketrains:
            segment.spiketrains.append(spiketrain)
        for analogsignal in analogsignals:
            segment.analogsignals.append(analogsignal)
        block.segments.append(segment)
    return block


#########
# Tests #
#########


class TrialsFromBlockTestCase(unittest.TestCase):
    """Tests for elephant.trials.TrialsFromBlock class"""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Run once before tests:
        """

        block = _create_trials_block(n_trials=36)
        cls.block = block
        cls.trial_object = TrialsFromBlock(block, description="trials are segments")

    def setUp(self) -> None:
        """
        Run before every test:
        """

    def test_trials_from_block_description(self) -> None:
        """
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, "trials are segments")

    def test_trials_from_block_get_item(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(self.trial_object[0], neo.core.Segment)

    def test_trials_from_block_get_trial_as_segment(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0), neo.core.Segment
        )
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).spiketrains[0],
            neo.core.SpikeTrain,
        )
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).analogsignals[0],
            neo.core.AnalogSignal,
        )

    def test_trials_from_block_get_trials_as_block(self) -> None:
        """
        Test get a block from list of trials.
        """
        block = self.trial_object.get_trials_as_block([0, 3, 5])
        self.assertIsInstance(block, neo.core.Block)
        self.assertIsInstance(self.trial_object.get_trials_as_block(), neo.core.Block)
        self.assertEqual(len(block.segments), 3)

    def test_trials_from_block_get_trials_as_list(self) -> None:
        """
        Test get a list of segments from list of trials.
        """
        list_of_trials = self.trial_object.get_trials_as_list([0, 3, 5])
        self.assertIsInstance(list_of_trials, list)
        self.assertIsInstance(self.trial_object.get_trials_as_list(), list)
        self.assertIsInstance(list_of_trials[0], neo.core.Segment)
        self.assertEqual(len(list_of_trials), 3)

    def test_trials_from_block_n_trials(self) -> None:
        """
        Test get number of trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.block.segments))

    def test_trials_from_block_n_spiketrains_trial_by_trial(self) -> None:
        """
        Test get number of spiketrains per trial.
        """
        self.assertEqual(
            self.trial_object.n_spiketrains_trial_by_trial,
            [len(trial.spiketrains) for trial in self.block.segments],
        )

    def test_trials_from_block_n_analogsignals_trial_by_trial(self) -> None:
        """
        Test get number of analogsignals per trial.
        """
        self.assertEqual(
            self.trial_object.n_analogsignals_trial_by_trial,
            [len(trial.analogsignals) for trial in self.block.segments],
        )

    def test_trials_from_block_get_spiketrains_from_trial_as_list(self) -> None:
        """
        Test get spiketrains from trial as list
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0),
            neo.core.spiketrainlist.SpikeTrainList,
        )
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0)[0],
            neo.core.SpikeTrain,
        )

    def test_trials_from_list_get_spiketrains_from_trial_as_segment(self) -> None:
        """
        Test get spiketrains from trial as segment
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(0), neo.core.Segment
        )
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(0).spiketrains[0],
            neo.core.SpikeTrain,
        )

    def test_trials_from_block_get_analogsignals_from_trial_as_list(self) -> None:
        """
        Test get analogsignals from trial as list
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0), list
        )
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0)[0],
            neo.core.AnalogSignal,
        )

    def test_trials_from_list_get_analogsignals_from_trial_as_segment(self) -> None:
        """
        Test get spiketrains from trial as segment
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0),
            neo.core.Segment,
        )
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0).analogsignals[
                0
            ],
            neo.core.AnalogSignal,
        )


class TrialsFromListTestCase(unittest.TestCase):
    """Tests for elephant.trials.TrialsFromList class"""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Run once before tests:
        Download the dataset from elephant_data
        """
        block = _create_trials_block(n_trials=36)

        # Create Trialobject as list of lists
        # add spiketrains
        trial_list = [
            [spiketrain for spiketrain in trial.spiketrains] for trial in block.segments
        ]
        # add analogsignals
        for idx, trial in enumerate(block.segments):
            for analogsignal in trial.analogsignals:
                trial_list[idx].append(analogsignal)
        cls.trial_list = trial_list

        cls.trial_object = TrialsFromLists(trial_list, description="trial is a list")

    def setUp(self) -> None:
        """
        Run before every test:
        """

    def test_trials_from_list_description(self) -> None:
        """
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, "trial is a list")

    def test_trials_from_list_get_item(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(self.trial_object[0], neo.core.Segment)
        self.assertIsInstance(self.trial_object[0].spiketrains[0], neo.core.SpikeTrain)
        self.assertIsInstance(
            self.trial_object[0].analogsignals[0], neo.core.AnalogSignal
        )

    def test_trials_from_list_get_trial_as_segment(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0), neo.core.Segment
        )
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).spiketrains[0],
            neo.core.SpikeTrain,
        )
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).analogsignals[0],
            neo.core.AnalogSignal,
        )

    def test_trials_from_list_get_trials_as_block(self) -> None:
        """
        Test get a block from list of trials.
        """
        block = self.trial_object.get_trials_as_block([0, 3, 5])
        self.assertIsInstance(block, neo.core.Block)
        self.assertIsInstance(self.trial_object.get_trials_as_block(), neo.core.Block)
        self.assertEqual(len(block.segments), 3)

    def test_trials_from_list_get_trials_as_list(self) -> None:
        """
        Test get a list of segments from list of trials.
        """
        list_of_trials = self.trial_object.get_trials_as_list([0, 3, 5])
        self.assertIsInstance(list_of_trials, list)
        self.assertIsInstance(self.trial_object.get_trials_as_list(), list)
        self.assertIsInstance(list_of_trials[0], neo.core.Segment)
        self.assertEqual(len(list_of_trials), 3)

    def test_trials_from_list_n_trials(self) -> None:
        """
        Test get number of trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.trial_list))

    def test_trials_from_list_n_spiketrains_trial_by_trial(self) -> None:
        """
        Test get number of spiketrains per trial.
        """
        self.assertEqual(
            self.trial_object.n_spiketrains_trial_by_trial,
            [
                sum(map(lambda x: isinstance(x, neo.core.SpikeTrain), trial))
                for trial in self.trial_list
            ],
        )

    def test_trials_from_list_n_analogsignals_trial_by_trial(self) -> None:
        """
        Test get number of analogsignals per trial.
        """
        self.assertEqual(
            self.trial_object.n_analogsignals_trial_by_trial,
            [
                sum(map(lambda x: isinstance(x, neo.core.AnalogSignal), trial))
                for trial in self.trial_list
            ],
        )

    def test_trials_from_list_get_spiketrains_from_trial_as_list(self) -> None:
        """
        Test get spiketrains from trial as list
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0),
            neo.core.spiketrainlist.SpikeTrainList,
        )
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0)[0],
            neo.core.SpikeTrain,
        )

    def test_trials_from_list_get_spiketrains_from_trial_as_segment(self) -> None:
        """
        Test get spiketrains from trial as segment
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(0), neo.core.Segment
        )
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(0).spiketrains[0],
            neo.core.SpikeTrain,
        )

    def test_trials_from_list_get_analogsignals_from_trial_as_list(self) -> None:
        """
        Test get analogsignals from trial as list
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0), list
        )
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0)[0],
            neo.core.AnalogSignal,
        )

    def test_trials_from_list_get_analogsignals_from_trial_as_segment(self) -> None:
        """
        Test get spiketrains from trial as segment
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0),
            neo.core.Segment,
        )
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0).analogsignals[
                0
            ],
            neo.core.AnalogSignal,
        )


if __name__ == "__main__":
    unittest.main()
