# -*- coding: utf-8 -*-
"""
Unit tests for the trials objects.

:copyright: Copyright 2014-2022 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
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
