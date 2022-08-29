# -*- coding: utf-8 -*-
"""
Unit tests for the trials objects.

:copyright: Copyright 2014-2022 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo.utils
import quantities as pq

import elephant.datasets
from elephant.trials import TrialsFromBlock


class TrialsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Run once before tests:
        Download the dataset from elephant_data
        """
        cls.filepath = elephant.datasets.download_datasets(
            'tutorials/tutorial_unitary_event_analysis/data/dataset-1.nix')

    def setUp(self):
        """
        Run before every test:
        Load the dataset with neo.NixIO
        """
        with neo.io.NixIO(
                self.filepath, 'ro') as io:
            self.block = io.read_block()

        self.trial_object = TrialsFromBlock(self.block,
                                            description='successful trials')

    def test_trials_from_block_description(self):
        """
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, 'successful trials')

    def test_trials_from_block_get_trial(self):
        """
        Test get a trial from the trials.
        """
        self.assertEqual(type(self.trial_object[0]), neo.core.Segment)

    def test_trials_from_block_n_trials(self):
        """
        Test get a trial from the trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.block.segments))
