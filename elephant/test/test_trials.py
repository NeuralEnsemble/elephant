# -*- coding: utf-8 -*-
"""
Unit tests for the trials objects.

:copyright: Copyright 2014-2022 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo.utils
import quantities as pq
from elephant.trials import TrialsFromBlock


class TrialsTestCase(unittest.TestCase):
    def setUp(self):
        with neo.io.NixIO(
                'reach_to_grasp_material/i140703-001.nix', 'ro') as io:
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
        self.assertEqual(self.trial_object.n_trials,
                         len(self.block.segments))

    def test_trials_from_block_cut_trials(self):
        """
        Test cutting of the trials
        """
        cut_events = neo.utils.get_events(
            self.block.segments[0],
            trial_event_labels='TS-ON',
            performance_in_trial_str='correct_trial')

        self.trial_object.pre = 0 * pq.ms
        self.trial_object.post = 2105 * pq.ms
        self.trial_object.cut_events = cut_events[0]


        self.trial_object.cut_trials()
        self.assertEqual(self.trial_object.n_trials, 11)
