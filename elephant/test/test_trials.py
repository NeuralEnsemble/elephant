# -*- coding: utf-8 -*-
"""
Unit tests for the trials objects.

:copyright: Copyright 2014-2022 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import neo.utils
import quantities as pq

from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.trials import TrialsFromBlock, TrialsFromLists


def _create_trials_block(n_trials: int = 0, n_spiketrains: int = 2):
    """ Create block with n_trials and n_spiketrains """
    block = neo.Block(name='test_block')
    for trial in range(n_trials):
        segment = neo.Segment(name=f'No. {trial}')
        spiketrains = StationaryPoissonProcess(rate=50.*pq.Hz, t_start=0*pq.ms,
                                               t_stop=1000*pq.ms
                                               ).generate_n_spiketrains(
                                                n_spiketrains=n_spiketrains)
        for spiketrain in spiketrains:
            segment.spiketrains.append(spiketrain)
        block.segments.append(segment)
    return block


class TrialsFromBlockTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Run once before tests:
        Download the dataset from elephant_data
        """

        block = _create_trials_block(n_trials=36)
        cls.block = block
        cls.trial_object = TrialsFromBlock(block,
                                           description='trials are segments')

    def setUp(self) -> None:
        """
        Run before every test:
        Load the dataset with neo.NixIO
        """

    def test_trials_from_block_description(self) -> None:
        """
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, 'trials are segments')

    def test_trials_from_block_get_trial(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(self.trial_object[0], neo.core.Segment)

    def test_trials_from_block_n_trials(self) -> None:
        """
        Test get number of trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.block.segments))


class TrialsFromListTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Run once before tests:
        Download the dataset from elephant_data
        """
        block = _create_trials_block(n_trials=36)

        # Create Trialobject as list of lists
        trial_list = [trial.spiketrains for trial in block.segments]
        cls.trial_list = trial_list

        cls.trial_object = TrialsFromLists(trial_list,
                                           description='trial is a list')

    def setUp(self) -> None:
        """
        Run before every test:
        Load the dataset with neo.NixIO
        """

    def test_trials_from_list_description(self) -> None:
        """
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, 'trial is a list')

    def test_trials_from_list_get_trial(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(self.trial_object[0],
                              neo.core.spiketrainlist.SpikeTrainList)

    def test_trials_from_list_n_trials(self) -> None:
        """
        Test get number of trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.trial_list))


if __name__ == '__main__':
    unittest.main()
