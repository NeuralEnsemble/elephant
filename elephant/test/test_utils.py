# -*- coding: utf-8 -*-
"""
Unit tests for elephant.utils
"""

import unittest

import neo
import numpy as np
import quantities as pq

from elephant import utils
from numpy.testing import assert_array_equal

from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.trials import TrialsFromLists
from neo.core.spiketrainlist import SpikeTrainList
from neo.core import SpikeTrain


class TestUtils(unittest.TestCase):
    def test_check_neo_consistency(self):
        self.assertRaises(
            TypeError, utils.check_neo_consistency, [], object_type=neo.SpikeTrain
        )
        self.assertRaises(
            TypeError,
            utils.check_neo_consistency,
            [neo.SpikeTrain([1] * pq.s, t_stop=2 * pq.s), np.arange(2)],
            object_type=neo.SpikeTrain,
        )
        self.assertRaises(
            ValueError,
            utils.check_neo_consistency,
            [
                neo.SpikeTrain([1] * pq.s, t_start=1 * pq.s, t_stop=2 * pq.s),
                neo.SpikeTrain([1] * pq.s, t_start=0 * pq.s, t_stop=2 * pq.s),
            ],
            object_type=neo.SpikeTrain,
        )
        self.assertRaises(
            ValueError,
            utils.check_neo_consistency,
            [
                neo.SpikeTrain([1] * pq.s, t_stop=2 * pq.s),
                neo.SpikeTrain([1] * pq.s, t_stop=3 * pq.s),
            ],
            object_type=neo.SpikeTrain,
        )
        self.assertRaises(
            ValueError,
            utils.check_neo_consistency,
            [
                neo.SpikeTrain([1] * pq.ms, t_stop=2000 * pq.ms),
                neo.SpikeTrain([1] * pq.s, t_stop=2 * pq.s),
            ],
            object_type=neo.SpikeTrain,
        )

    def test_round_binning_errors(self):
        n_bins = utils.round_binning_errors(0.999999, tolerance=1e-6)
        self.assertEqual(n_bins, 1)
        self.assertEqual(utils.round_binning_errors(0.999999, tolerance=None), 0)
        array = np.array([0, 0.7, 1 - 1e-8, 1 - 1e-9])
        corrected = utils.round_binning_errors(array.copy())
        assert_array_equal(corrected, [0, 0, 1, 1])
        assert_array_equal(
            utils.round_binning_errors(array.copy(), tolerance=None), [0, 0, 0, 0]
        )


class DecoratorTest:
    """
    This class is used as a mock for testing the decorator.
    """

    @utils.trials_to_list_of_spiketrainlist
    def method_to_decorate(self, trials=None, trials_obj=None):
        # This is just a mock implementation for testing purposes
        if trials_obj:
            return trials_obj
        else:
            return trials


class TestTrialsToListOfSpiketrainlist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_channels = 10
        cls.n_trials = 5
        cls.list_of_list_of_spiketrains = [
            StationaryPoissonProcess(
                rate=5 * pq.Hz, t_stop=1000.0 * pq.ms
            ).generate_n_spiketrains(cls.n_channels)
            for _ in range(cls.n_trials)
        ]
        cls.trial_object = TrialsFromLists(cls.list_of_list_of_spiketrains)

    def test_decorator_applied(self):
        # Test that the decorator is applied correctly
        self.assertTrue(hasattr(DecoratorTest.method_to_decorate, "__wrapped__"))

    def test_decorator_return_with_trials_input_as_arg(self):
        # Test if decorator takes in trial-object and returns
        # list of spiketrainlists
        new_class = DecoratorTest()
        list_of_spiketrainlists = new_class.method_to_decorate(self.trial_object)
        self.assertEqual(len(list_of_spiketrainlists), self.n_trials)
        for spiketrainlist in list_of_spiketrainlists:
            self.assertIsInstance(spiketrainlist, SpikeTrainList)

    def test_decorator_return_with_list_of_lists_input_as_arg(self):
        # Test if decorator takes in list of lists of spiketrains
        # and does not change input
        new_class = DecoratorTest()
        list_of_list_of_spiketrains = new_class.method_to_decorate(
            self.list_of_list_of_spiketrains
        )
        self.assertEqual(len(list_of_list_of_spiketrains), self.n_trials)
        for list_of_spiketrains in list_of_list_of_spiketrains:
            self.assertIsInstance(list_of_spiketrains, list)
            for spiketrain in list_of_spiketrains:
                self.assertIsInstance(spiketrain, SpikeTrain)

    def test_decorator_return_with_trials_input_as_kwarg(self):
        # Test if decorator takes in trial-object and returns
        # list of spiketrainlists
        new_class = DecoratorTest()
        list_of_spiketrainlists = new_class.method_to_decorate(
            trials_obj=self.trial_object
        )
        self.assertEqual(len(list_of_spiketrainlists), self.n_trials)
        for spiketrainlist in list_of_spiketrainlists:
            self.assertIsInstance(spiketrainlist, SpikeTrainList)

    def test_decorator_return_with_list_of_lists_input_as_kwarg(self):
        # Test if decorator takes in list of lists of spiketrains
        # and does not change input
        new_class = DecoratorTest()
        list_of_list_of_spiketrains = new_class.method_to_decorate(
            trials_obj=self.list_of_list_of_spiketrains
        )
        self.assertEqual(len(list_of_list_of_spiketrains), self.n_trials)
        for list_of_spiketrains in list_of_list_of_spiketrains:
            self.assertIsInstance(list_of_spiketrains, list)
            for spiketrain in list_of_spiketrains:
                self.assertIsInstance(spiketrain, SpikeTrain)


if __name__ == "__main__":
    unittest.main()
