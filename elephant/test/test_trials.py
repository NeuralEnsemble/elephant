# -*- coding: utf-8 -*-
"""
Unit tests for the trials objects.

:copyright: Copyright 2014-2022 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest

import neo
from elephant.trials import TrialsFromBlock
import numpy as np
import quantities as pq

from elephant.spike_train_generation import StationaryPoissonProcess


class TrialsTestCase(unittest.TestCase):
    def setUp(self):
        with neo.io.NixIO(
                'reach_to_grasp_material/i140703-001.nix', 'ro') as io:
            self.block = io.read_block()

    def test_trials_from_block(self):
        """
        Test elephant.trials TrialsFromBlock class.
        """
        trial_object = TrialsFromBlock(self.block,
                                       description='successful trials')
        # Test that the trials description is correct
        self.assertEqual(trial_object.description, 'successful trials')
