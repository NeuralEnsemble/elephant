# -*- coding: utf-8 -*-
"""
nit tests for the objects of the API handling trial data in Elephant.

:copyright: Copyright 2014-2025 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
import quantities as pq
from neo.core import Block, Segment, AnalogSignal, SpikeTrain
from neo.core.spiketrainlist import SpikeTrainList

from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.trials import (TrialsFromBlock, TrialsFromLists,
                             trials_to_list_of_spiketrainlist)


def _create_trials_block(n_trials: int = 0,
                         n_spiketrains: int = 2,
                         n_analogsignals: int = 2) -> Block:
    """
    Create Neo `Block` with `n_trials`, `n_spiketrains` and `n_analogsignals`.
    """
    block = Block(name='test_block')
    for trial in range(n_trials):
        segment = Segment(name=f'No. {trial}')
        spiketrains = StationaryPoissonProcess(rate=50. * pq.Hz,
                                               t_start=0 * pq.ms,
                                               t_stop=1000 * pq.ms
                                               ).generate_n_spiketrains(
            n_spiketrains=n_spiketrains)
        for idx, st in enumerate(spiketrains):
            st.name = f"Spiketrain {idx}"
            st.description = f"Trial {trial}"

        analogsignals = [AnalogSignal(signal=[.01, 3.3, 9.3], units='uV',
                                      sampling_rate=1 * pq.Hz,
                                      name=f"Signal {idx}",
                                      description=f"Trial {trial}")
                         for idx in range(n_analogsignals)]
        for spiketrain in spiketrains:
            segment.spiketrains.append(spiketrain)
        for analogsignal in analogsignals:
            segment.analogsignals.append(analogsignal)
        block.segments.append(segment)
    return block


##########################
# Tests - helper classes #
##########################

class TrialsBaseTestCase(unittest.TestCase):
    """
    This is a base `unitest.TestCase` class to act as a helper when
    constructing the specific test cases for each implementation of `Trials`.

    This helper class facilitates comparing Neo objects, as custom assertions
    are implemented to perform a series of tests to ensure two Neo objects are
    indeed equal (e.g., checking metadata or contents of collections such as
    spiketrains in a `Segment`).

    As the `Trials` objects are based on references to the input data
    structures, checks for `Segment`, `SpikeTrain`, and `AnalogSignal` objects
    are enforcing instance equality (i.e., `(a is b) == True`).
    """

    def assertSegmentEqual(self, segment_1, segment_2) -> None:
        self.assertIsInstance(segment_1, Segment)
        self.assertIsInstance(segment_2, Segment)
        self.assertIs(segment_1, segment_2)
        self.assertEqual(segment_1.name, segment_2.name)
        self.assertEqual(segment_2.description, segment_2.description)
        self.assertDictEqual(segment_1.annotations, segment_2.annotations)
        self.assertSpikeTrainListEqual(segment_1.spiketrains,
                                       segment_2.spiketrains)
        self.assertAnalogSignalListEqual(segment_1.analogsignals,
                                         segment_2.analogsignals)

    def assertSpikeTrainEqual(self, spiketrain_1, spiketrain_2) -> None:
        self.assertIsInstance(spiketrain_1, SpikeTrain)
        self.assertIsInstance(spiketrain_2, SpikeTrain)
        self.assertIs(spiketrain_1, spiketrain_2)
        self.assertTrue(np.all(spiketrain_1 == spiketrain_2))
        self.assertEqual(spiketrain_1.name, spiketrain_2.name)
        self.assertEqual(spiketrain_1.description, spiketrain_2.description)
        self.assertDictEqual(spiketrain_1.annotations,
                             spiketrain_2.annotations)

    def assertSpikeTrainListEqual(self, spiketrains_1, spiketrains_2) -> None:
        self.assertIsInstance(spiketrains_1, SpikeTrainList)
        self.assertIsInstance(spiketrains_2, SpikeTrainList)
        self.assertEqual(len(spiketrains_1), len(spiketrains_2))
        for st1, st2 in zip(spiketrains_1, spiketrains_2):
            self.assertSpikeTrainEqual(st1, st2)

    def assertAnalogSignalEqual(self, signal_1, signal_2) -> None:
        self.assertIsInstance(signal_1, AnalogSignal)
        self.assertIsInstance(signal_2, AnalogSignal)
        self.assertIs(signal_1, signal_2)
        self.assertTrue(np.all(signal_1 == signal_2))
        self.assertEqual(signal_1.name, signal_2.name)
        self.assertEqual(signal_1.description, signal_2.description)
        self.assertDictEqual(signal_1.annotations, signal_2.annotations)

    def assertAnalogSignalListEqual(self, signals_1, signals_2) -> None:
        # Not enforcing object type as `Segment.analogsignals` are
        # `ObjectList`, and some of the functions return pure Python lists
        # containing the `AnalogSignal` objects. Therefore, the type checking
        # must be done in each test case accordingly.
        self.assertEqual(len(signals_1), len(signals_2))
        for signal_1, signal_2 in zip(signals_1, signals_2):
            self.assertAnalogSignalEqual(signal_1, signal_2)


class DecoratorTest:
    """
    This class is used as a mock for testing the decorator.
    """
    @trials_to_list_of_spiketrainlist
    def method_to_decorate(self, trials=None, trials_obj=None):
        # This is just a mock implementation for testing purposes
        if trials_obj:
            return trials_obj
        return trials


######################
# Tests - test cases #
######################

class TestTrialsToListOfSpiketrainlist(TrialsBaseTestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_channels = 10
        cls.n_trials = 5
        cls.list_of_list_of_spiketrains = [
            StationaryPoissonProcess(rate=5 * pq.Hz, t_stop=1000.0 * pq.ms
                                     ).generate_n_spiketrains(cls.n_channels)
            for _ in range(cls.n_trials)]
        cls.trial_object = TrialsFromLists(cls.list_of_list_of_spiketrains)

    def test_decorator_applied(self) -> None:
        """
        Test that the decorator is applied correctly.
        """
        self.assertTrue(hasattr(
            DecoratorTest.method_to_decorate, '__wrapped__'
            ))

    def test_decorator_return_with_trials_input_as_arg(self) -> None:
        """
        Test if the decorator takes in a `Trials` object and returns a list of
        `SpikeTrainList`.
        """
        new_class = DecoratorTest()
        list_of_spiketrainlists = new_class.method_to_decorate(
            self.trial_object)
        self.assertEqual(len(list_of_spiketrainlists), self.n_trials)
        for spiketrainlist, expected_list in zip(
                list_of_spiketrainlists, self.list_of_list_of_spiketrains):
            self.assertSpikeTrainListEqual(spiketrainlist,
                                           SpikeTrainList(expected_list))

    def test_decorator_return_with_list_of_lists_input_as_arg(self) -> None:
        """
        Test if the decorator takes in a list of lists of `SpikeTrain` and
        does not change the input.
        """
        new_class = DecoratorTest()
        list_of_list_of_spiketrains = new_class.method_to_decorate(
            self.list_of_list_of_spiketrains)
        self.assertEqual(len(list_of_list_of_spiketrains), self.n_trials)
        for list_of_spiketrains, expected_list in (
                zip(list_of_list_of_spiketrains,
                    self.list_of_list_of_spiketrains)):
            self.assertIsInstance(list_of_spiketrains, list)
            for spiketrain, expected_spiketrain in (
                    zip(list_of_spiketrains, expected_list)):
                self.assertSpikeTrainEqual(spiketrain, expected_spiketrain)

    def test_decorator_return_with_trials_input_as_kwarg(self) -> None:
        """
        Test if the decorator takes in a `Trials` object and returns a list of
        `SpikeTrainList` when passed as kwarg.
        """
        new_class = DecoratorTest()
        list_of_spiketrainlists = new_class.method_to_decorate(
            trials_obj=self.trial_object)
        self.assertEqual(len(list_of_spiketrainlists), self.n_trials)
        for spiketrainlist, expected_list in zip(
                list_of_spiketrainlists, self.list_of_list_of_spiketrains):
            self.assertSpikeTrainListEqual(spiketrainlist,
                                           SpikeTrainList(expected_list))

    def test_decorator_return_with_list_of_lists_input_as_kwarg(self) -> None:
        """
        Test if the decorator takes in a list of lists of `SpikeTrain`and does
        not change the input if passed as a kwarg.
        """
        new_class = DecoratorTest()
        list_of_list_of_spiketrains = new_class.method_to_decorate(
            trials_obj=self.list_of_list_of_spiketrains)
        self.assertEqual(len(list_of_list_of_spiketrains), self.n_trials)
        for list_of_spiketrains, expected_list in (
                zip(list_of_list_of_spiketrains,
                    self.list_of_list_of_spiketrains)):
            self.assertIsInstance(list_of_spiketrains, list)
            for spiketrain, expected_spiketrain in (
                    zip(list_of_spiketrains, expected_list)):
                self.assertSpikeTrainEqual(spiketrain, expected_spiketrain)


class TrialsFromBlockTestCase(TrialsBaseTestCase):
    """
    Tests for :class:`elephant.trials.TrialsFromBlock`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        block = _create_trials_block(n_trials=36)
        cls.block = block
        cls.trial_object = TrialsFromBlock(block,
                                           description='trial is Segment')

    def test_deprecations(self) -> None:
        """
        Test if all expected deprecation warnings are triggered.
        """
        trial_object = self.trial_object
        with self.assertWarns(DeprecationWarning):
            trial_object.get_trial_as_segment(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_trials_as_block(trial_ids=[0, 1])
        with self.assertWarns(DeprecationWarning):
            trial_object.get_trials_as_list(trial_ids=[0, 1])
        with self.assertWarns(DeprecationWarning):
            trial_object.get_spiketrains_from_trial_as_list(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_spiketrains_from_trial_as_segment(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_analogsignals_from_trial_as_list(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_analogsignals_from_trial_as_segment(trial_id=0)

    def test_trials_from_block_description(self) -> None:
        """
        Test the description of the `Trials` object.
        """
        self.assertEqual(self.trial_object.description, 'trial is Segment')

    def test_trials_from_block_get_item(self) -> None:
        """
        Test to get a single trial from the `Trials` object using indexing
        with brackets. Return is a `Segment`.
        """
        self.assertIsInstance(self.trial_object[0], Segment)

    def test_trials_from_block_get_trial_as_segment(self) -> None:
        """
        Test to get a single trial from the `Trials` object as a `Segment`.
        """
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).spiketrains[0],
            SpikeTrain)
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).analogsignals[0],
            AnalogSignal)

    def test_trials_from_block_get_trials_as_block(self) -> None:
        """
        Test to get a set of specific trials grouped as a `Block`. Each trial
        is a `Segment` containing all the data in the trial.
        """
        block = self.trial_object.get_trials_as_block([0, 3, 5])
        self.assertIsInstance(block, Block)
        self.assertIsInstance(self.trial_object.get_trials_as_block(), Block)
        self.assertEqual(len(block.segments), 3)

    def test_trials_from_block_get_trials_as_list(self) -> None:
        """
        Test to get a set of specific trials grouped as list of `Segment`.
        Each trial is a single `Segment` containing all the data in the trial.
        """
        list_of_trials = self.trial_object.get_trials_as_list([0, 3, 5])
        self.assertIsInstance(list_of_trials, list)
        self.assertIsInstance(self.trial_object.get_trials_as_list(), list)
        self.assertIsInstance(list_of_trials[0], Segment)
        self.assertEqual(len(list_of_trials), 3)

    def test_trials_from_block_n_trials(self) -> None:
        """
        Test to get the number of trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.block.segments))

    def test_trials_from_block_n_spiketrains_trial_by_trial(self) -> None:
        """
        Test to get the number of `SpikeTrain` objects per trial.
        """
        self.assertEqual(self.trial_object.n_spiketrains_trial_by_trial,
                         [len(trial.spiketrains) for trial in
                          self.block.segments])

    def test_trials_from_block_n_analogsignals_trial_by_trial(self) -> None:
        """
        Test to get the number of `AnalogSignal` objects per trial.
        """
        self.assertEqual(self.trial_object.n_analogsignals_trial_by_trial,
                         [len(trial.analogsignals) for trial in
                          self.block.segments])

    def test_trials_from_block_get_spiketrains_from_trial_as_list(self
                                                                  ) -> None:
        """
        Test to get all spiketrains from a single trial as a `SpikeTrainList`.
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0),
            SpikeTrainList)
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0)[0],
            SpikeTrain)

    def test_trials_from_block_get_spiketrains_from_trial_as_segment(self
                                                                     ) -> None:
        """
        Test to get the all spiketrains from a single trial as a `Segment`.
        The `Segment.spiketrains` collection contains the spiketrains.
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(
                0).spiketrains[0], SpikeTrain)

    def test_trials_from_block_get_analogsignals_from_trial_as_list(self
                                                                    ) -> None:
        """
        Test to get all analog signals from a single trial as a list.
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0), list)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0)[0],
            AnalogSignal)

    def test_trials_from_block_get_analogsignals_from_trial_as_segment(self) \
            -> None:
        """
        Test to get all analog signals from a single trial as a `Segment`.
        The `Segment.analogsignals` collection contains the signals.
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(
                0).analogsignals[0], AnalogSignal)

    def test_trials_from_block_get_spiketrains_trial_by_trial(self) -> None:
        """
        Test to access all the `SpikeTrain` objects corresponding to the
        repetitions of a spiketrain across the trials.
        """
        for st_id in (0, 1):
            spiketrains = self.trial_object.get_spiketrains_trial_by_trial(st_id)

            # Return is neo.SpikeTrainList
            self.assertIsInstance(spiketrains, SpikeTrainList)

            # All elements are neo.SpikeTrain
            self.assertTrue(all(map(lambda x: isinstance(x, SpikeTrain),
                                    spiketrains)
                                )
                            )

            # Data for all trials is returned
            self.assertEqual(len(spiketrains), self.trial_object.n_trials)

            # Each trial-specific SpikeTrain object is from the same spiketrain
            self.assertTrue(all([st.name == f"Spiketrain {st_id}"
                                 for st in spiketrains]
                                )
                            )

            # Order of spiketrains is the order of the trials
            expected_trials = [f"Trial {i}"
                               for i in range(self.trial_object.n_trials)]
            self.assertListEqual([st.description for st in spiketrains],
                                 expected_trials)

    def test_trials_from_block_get_analogsignals_trial_by_trial(self) -> None:
        """
        Test to access all the `AnalogSignal` objects corresponding to the
        repetitions of an analog signal across the trials.
        """
        for as_id in (0, 1):
            signals = self.trial_object.get_analogsignals_trial_by_trial(as_id)

            # Return is list
            self.assertIsInstance(signals, list)

            # All elements are neo.AnalogSignal
            self.assertTrue(all(map(lambda x: isinstance(x, AnalogSignal),
                                    signals)
                                )
                            )
            # Data for all trials returned
            self.assertEqual(len(signals), self.trial_object.n_trials)

            # Each trial-specific AnalogSignal object is from the same signal
            self.assertTrue(all([signal.name == f"Signal {as_id}"
                                 for signal in signals]
                                )
                            )

            # Order in the list is the order of the trials
            expected_trials = [f"Trial {i}"
                               for i in range(self.trial_object.n_trials)]
            self.assertListEqual([signal.description for signal in signals],
                                 expected_trials)


class TrialsFromListTestCase(TrialsBaseTestCase):
    """
    Tests for :class:`elephant.trials.TrialsFromList`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.n_spiketrains = 2
        cls.n_analogsignals = 3
        block = _create_trials_block(n_trials=36,
                                     n_spiketrains=cls.n_spiketrains,
                                     n_analogsignals=cls.n_analogsignals)

        # Create trial data as list of lists
        # 1. Add spiketrains
        trial_list = [[spiketrain for spiketrain in trial.spiketrains]
                      for trial in block.segments]
        # 2. Add analog signals
        for idx, trial in enumerate(block.segments):
            for analogsignal in trial.analogsignals:
                trial_list[idx].append(analogsignal)
        cls.trial_list = trial_list

        # Create TrialsFromLists object
        cls.trial_object = TrialsFromLists(trial_list,
                                           description='trial is a list')

    def assertSegmentEqualToList(self, segment, list_data, n_spiketrains,
                                 n_analogsignals):
        """
        This function compares trial data in a Segment to trial data in
        a Python list. The order of objects is: `SpikeTrain`, `AnalogSignal`.
        The number of spiketrains and analog signals must be informed to split
        the data in the list.
        """
        self.assertIsInstance(segment, Segment)
        self.assertIsInstance(list_data, list)

        self.assertEqual(len(list_data), n_spiketrains + n_analogsignals)
        self.assertEqual(len(segment.spiketrains), n_spiketrains)
        self.assertEqual(len(segment.analogsignals), n_analogsignals)

        spiketrains = list_data[:n_spiketrains]
        signals = list_data[n_spiketrains:]
        self.assertSpikeTrainListEqual(segment.spiketrains,
                                       SpikeTrainList(spiketrains))
        self.assertAnalogSignalListEqual(segment.analogsignals, signals)

    def test_deprecations(self) -> None:
        """
        Test if all expected deprecation warnings are triggered.
        """
        trial_object = self.trial_object
        with self.assertWarns(DeprecationWarning):
            trial_object.get_trial_as_segment(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_trials_as_block(trial_ids=[0, 1])
        with self.assertWarns(DeprecationWarning):
            trial_object.get_trials_as_list(trial_ids=[0, 1])
        with self.assertWarns(DeprecationWarning):
            trial_object.get_spiketrains_from_trial_as_list(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_spiketrains_from_trial_as_segment(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_analogsignals_from_trial_as_list(trial_id=0)
        with self.assertWarns(DeprecationWarning):
            trial_object.get_analogsignals_from_trial_as_segment(trial_id=0)

    def test_trials_from_list_description(self) -> None:
        """
        Test the description of the `Trials` object.
        """
        self.assertEqual(self.trial_object.description, 'trial is a list')

    def test_trials_from_list_get_item(self) -> None:
        """
        Test to get a single trial from the `Trials` object using indexing
        with brackets. Return is a `Segment`.
        """
        self.assertIsInstance(self.trial_object[0], Segment)
        self.assertIsInstance(self.trial_object[0].spiketrains[0], SpikeTrain)
        self.assertIsInstance(self.trial_object[0].analogsignals[0],
                              AnalogSignal)

    def test_trials_from_list_get_trial_as_segment(self) -> None:
        """
        Test to get a single trial from the `Trials` object as a `Segment`.
        """
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0), Segment)
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).spiketrains[0],
            SpikeTrain)
        self.assertIsInstance(
            self.trial_object.get_trial_as_segment(0).analogsignals[0],
            AnalogSignal)

    def test_trials_from_list_get_trials_as_block(self) -> None:
        """
        Test to get a set of specific trials grouped as a `Block`. Each trial
        is a `Segment` containing all the data in the trial.
        """
        block = self.trial_object.get_trials_as_block([0, 3, 5])
        self.assertIsInstance(block, Block)
        self.assertIsInstance(self.trial_object.get_trials_as_block(), Block)
        self.assertEqual(len(block.segments), 3)

    def test_trials_from_list_get_trials_as_list(self) -> None:
        """
        Test to get a set of specific trials grouped as list of `Segment`.
        Each trial is a single `Segment` containing all the data in the trial.
        """
        list_of_trials = self.trial_object.get_trials_as_list([0, 3, 5])
        self.assertIsInstance(list_of_trials, list)
        self.assertIsInstance(self.trial_object.get_trials_as_list(), list)
        self.assertIsInstance(list_of_trials[0], Segment)
        self.assertEqual(len(list_of_trials), 3)

    def test_trials_from_list_n_trials(self) -> None:
        """
        Test to get the number of trials.
        """
        self.assertEqual(self.trial_object.n_trials, len(self.trial_list))

    def test_trials_from_list_n_spiketrains_trial_by_trial(self) -> None:
        """
        Test to get the number of `SpikeTrain` objects per trial.
        """
        self.assertEqual(self.trial_object.n_spiketrains_trial_by_trial,
                         [sum(map(lambda x: isinstance(x, SpikeTrain),
                                  trial)) for trial in self.trial_list])

    def test_trials_from_list_n_analogsignals_trial_by_trial(self) -> None:
        """
        Test to get the number of `AnalogSignal` objects per trial.
        """
        self.assertEqual(self.trial_object.n_analogsignals_trial_by_trial,
                         [sum(map(lambda x: isinstance(x, AnalogSignal),
                                  trial)) for trial in self.trial_list])

    def test_trials_from_list_get_spiketrains_from_trial_as_list(self) -> None:
        """
        Test to get all spiketrains from a single trial as a `SpikeTrainList`.
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0),
            SpikeTrainList)
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_list(0)[0],
            SpikeTrain)

    def test_trials_from_list_get_spiketrains_from_trial_as_segment(self
                                                                    ) -> None:
        """
        Test to get the all spiketrains from a single trial as a `Segment`.
        The `Segment.spiketrains` collection contains the spiketrains.
        """
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_spiketrains_from_trial_as_segment(
                0).spiketrains[0], SpikeTrain)

    def test_trials_from_list_get_analogsignals_from_trial_as_list(self
                                                                   ) -> None:
        """
        Test to get all analog signals from a single trial as a list.
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0), list)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0)[0],
            AnalogSignal)

    def test_trials_from_list_get_analogsignals_from_trial_as_segment(self
                                                                      ) \
            -> None:
        """
        Test to get all analog signals from a single trial as a `Segment`.
        The `Segment.analogsignals` collection contains the signals.
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(
                0).analogsignals[0], AnalogSignal)

    def test_trials_from_list_get_spiketrains_trial_by_trial(self) -> None:
        """
        Test to access all the `SpikeTrain` objects corresponding to the
        repetitions of a spiketrain across the trials.
        """
        for st_id in (0, 1):
            spiketrains = self.trial_object.get_spiketrains_trial_by_trial(
                st_id)

            # Return is neo.SpikeTrainList
            self.assertIsInstance(spiketrains, SpikeTrainList)

            # All elements are neo.SpikeTrain
            self.assertTrue(all(map(lambda x: isinstance(x, SpikeTrain),
                                    spiketrains)
                                )
                            )

            # Data for all trials is returned
            self.assertEqual(len(spiketrains), self.trial_object.n_trials)

            # Each trial-specific SpikeTrain object is from the same spiketrain
            self.assertTrue(all([st.name == f"Spiketrain {st_id}"
                                 for st in spiketrains]
                                )
                            )

            # Order of spiketrains is the order of the trials
            expected_trials = [f"Trial {i}"
                               for i in range(self.trial_object.n_trials)]
            self.assertListEqual([st.description for st in spiketrains],
                                 expected_trials)

    def test_trials_from_list_get_analogsignals_trial_by_trial(self) -> None:
        """
        Test to access all the `AnalogSignal` objects corresponding to the
        repetitions of an analog signal across the trials.
        """
        for as_id in (0, 1):
            signals = self.trial_object.get_analogsignals_trial_by_trial(as_id)

            # Return is list
            self.assertIsInstance(signals, list)

            # All elements are neo.AnalogSignal
            self.assertTrue(all(map(lambda x: isinstance(x, AnalogSignal),
                                    signals)
                                )
                            )
            # Data for all trials returned
            self.assertEqual(len(signals), self.trial_object.n_trials)

            # Each trial-specific AnalogSignal object is from the same signal
            self.assertTrue(all([signal.name == f"Signal {as_id}"
                                 for signal in signals]
                                )
                            )

            # Order in the list is the order of the trials
            expected_trials = [f"Trial {i}"
                               for i in range(self.trial_object.n_trials)]
            self.assertListEqual([signal.description for signal in signals],
                                 expected_trials)


if __name__ == '__main__':
    unittest.main()
