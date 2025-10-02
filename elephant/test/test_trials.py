# -*- coding: utf-8 -*-
"""
nit tests for the objects of the API handling trial data in Elephant.

:copyright: Copyright 2014-2025 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import quantities as pq
from neo.core import Block, Segment, AnalogSignal, SpikeTrain
from neo.core.spiketrainlist import SpikeTrainList

from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.trials import (TrialsFromBlock, TrialsFromLists,
                             trials_to_list_of_spiketrainlist)


def _create_trials_block(n_trials: int = 0,
                         n_spiketrains: int = 2,
                         n_analogsignals: int = 2) -> Block:
    """ Create block with n_trials, n_spiketrains and n_analog_signals """
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


#########
# Tests #
#########

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


class TestTrialsToListOfSpiketrainlist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_channels = 10
        cls.n_trials = 5
        cls.list_of_list_of_spiketrains = [
            StationaryPoissonProcess(rate=5 * pq.Hz, t_stop=1000.0 * pq.ms
                                     ).generate_n_spiketrains(cls.n_channels)
            for _ in range(cls.n_trials)]
        cls.trial_object = TrialsFromLists(cls.list_of_list_of_spiketrains)

    def test_decorator_applied(self):
        # Test that the decorator is applied correctly
        self.assertTrue(hasattr(
            DecoratorTest.method_to_decorate, '__wrapped__'
            ))

    def test_decorator_return_with_trials_input_as_arg(self):
        # Test if decorator takes in trial-object and returns
        # list of spiketrainlists
        new_class = DecoratorTest()
        list_of_spiketrainlists = new_class.method_to_decorate(
            self.trial_object)
        self.assertEqual(len(list_of_spiketrainlists), self.n_trials)
        for spiketrainlist in list_of_spiketrainlists:
            self.assertIsInstance(spiketrainlist, SpikeTrainList)

    def test_decorator_return_with_list_of_lists_input_as_arg(self):
        # Test if decorator takes in list of lists of spiketrains
        # and does not change input
        new_class = DecoratorTest()
        list_of_list_of_spiketrains = new_class.method_to_decorate(
            self.list_of_list_of_spiketrains)
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
            trials_obj=self.trial_object)
        self.assertEqual(len(list_of_spiketrainlists), self.n_trials)
        for spiketrainlist in list_of_spiketrainlists:
            self.assertIsInstance(spiketrainlist, SpikeTrainList)

    def test_decorator_return_with_list_of_lists_input_as_kwarg(self):
        # Test if decorator takes in list of lists of spiketrains
        # and does not change input
        new_class = DecoratorTest()
        list_of_list_of_spiketrains = new_class.method_to_decorate(
            trials_obj=self.list_of_list_of_spiketrains)
        self.assertEqual(len(list_of_list_of_spiketrains), self.n_trials)
        for list_of_spiketrains in list_of_list_of_spiketrains:
            self.assertIsInstance(list_of_spiketrains, list)
            for spiketrain in list_of_spiketrains:
                self.assertIsInstance(spiketrain, SpikeTrain)


class TrialsFromBlockTestCase(unittest.TestCase):
    """Tests for elephant.trials.TrialsFromBlock class"""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Run once before tests:
        """

        block = _create_trials_block(n_trials=36)
        cls.block = block
        cls.trial_object = TrialsFromBlock(block,
                                           description='trials are segments')

    def setUp(self) -> None:
        """
        Run before every test:
        """

    def test_deprecations(self):
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
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, 'trials are segments')

    def test_trials_from_block_get_item(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(self.trial_object[0], Segment)

    def test_trials_from_block_get_trial_as_segment(self) -> None:
        """
        Test get a trial from the trials.
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
        Test get a block from list of trials.
        """
        block = self.trial_object.get_trials_as_block([0, 3, 5])
        self.assertIsInstance(block, Block)
        self.assertIsInstance(self.trial_object.get_trials_as_block(), Block)
        self.assertEqual(len(block.segments), 3)

    def test_trials_from_block_get_trials_as_list(self) -> None:
        """
        Test get a list of segments from list of trials.
        """
        list_of_trials = self.trial_object.get_trials_as_list([0, 3, 5])
        self.assertIsInstance(list_of_trials, list)
        self.assertIsInstance(self.trial_object.get_trials_as_list(), list)
        self.assertIsInstance(list_of_trials[0], Segment)
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
        self.assertEqual(self.trial_object.n_spiketrains_trial_by_trial,
                         [len(trial.spiketrains) for trial in
                          self.block.segments])

    def test_trials_from_block_n_analogsignals_trial_by_trial(self) -> None:
        """
        Test get number of analogsignals per trial.
        """
        self.assertEqual(self.trial_object.n_analogsignals_trial_by_trial,
                         [len(trial.analogsignals) for trial in
                          self.block.segments])

    def test_trials_from_block_get_spiketrains_from_trial_as_list(self
                                                                  ) -> None:
        """
        Test get spiketrains from trial as list
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
        Test get spiketrains from trial as segment
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
        Test get analogsignals from trial as list
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0), list)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_list(0)[0],
            AnalogSignal)

    def test_trials_from_block_get_analogsignals_from_trial_as_segment(self) \
            -> None:
        """
        Test get spiketrains from trial as segment
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(
                0).analogsignals[0], AnalogSignal)

    def test_trials_from_block_get_spiketrains_trial_by_trial(self) -> None:
        """
        Test accessing all the SpikeTrain objects corresponding to the
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
        Test accessing all the AnalogSignal objects corresponding to the
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
        trial_list = [[spiketrain for spiketrain in trial.spiketrains]
                      for trial in block.segments]
        # add analogsignals
        for idx, trial in enumerate(block.segments):
            for analogsignal in trial.analogsignals:
                trial_list[idx].append(analogsignal)
        cls.trial_list = trial_list

        cls.trial_object = TrialsFromLists(trial_list,
                                           description='trial is a list')

    def setUp(self) -> None:
        """
        Run before every test:
        """

    def test_deprecations(self):
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
        Test description of the trials object.
        """
        self.assertEqual(self.trial_object.description, 'trial is a list')

    def test_trials_from_list_get_item(self) -> None:
        """
        Test get a trial from the trials.
        """
        self.assertIsInstance(self.trial_object[0], Segment)
        self.assertIsInstance(self.trial_object[0].spiketrains[0], SpikeTrain)
        self.assertIsInstance(self.trial_object[0].analogsignals[0],
                              AnalogSignal)

    def test_trials_from_list_get_trial_as_segment(self) -> None:
        """
        Test get a trial from the trials.
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
        Test get a block from list of trials.
        """
        block = self.trial_object.get_trials_as_block([0, 3, 5])
        self.assertIsInstance(block, Block)
        self.assertIsInstance(self.trial_object.get_trials_as_block(), Block)
        self.assertEqual(len(block.segments), 3)

    def test_trials_from_list_get_trials_as_list(self) -> None:
        """
        Test get a list of segments from list of trials.
        """
        list_of_trials = self.trial_object.get_trials_as_list([0, 3, 5])
        self.assertIsInstance(list_of_trials, list)
        self.assertIsInstance(self.trial_object.get_trials_as_list(), list)
        self.assertIsInstance(list_of_trials[0], Segment)
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
        self.assertEqual(self.trial_object.n_spiketrains_trial_by_trial,
                         [sum(map(lambda x: isinstance(x, SpikeTrain),
                                  trial)) for trial in self.trial_list])

    def test_trials_from_list_n_analogsignals_trial_by_trial(self) -> None:
        """
        Test get number of analogsignals per trial.
        """
        self.assertEqual(self.trial_object.n_analogsignals_trial_by_trial,
                         [sum(map(lambda x: isinstance(x, AnalogSignal),
                                  trial)) for trial in self.trial_list])

    def test_trials_from_list_get_spiketrains_from_trial_as_list(self) -> None:
        """
        Test get spiketrains from trial as list
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
        Test get spiketrains from trial as segment
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
        Test get analogsignals from trial as list
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
        Test get spiketrains from trial as segment
        """
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(0),
            Segment)
        self.assertIsInstance(
            self.trial_object.get_analogsignals_from_trial_as_segment(
                0).analogsignals[0], AnalogSignal)

    def test_trials_from_list_get_spiketrains_trial_by_trial(self) -> None:
        """
        Test accessing all the SpikeTrain objects corresponding to the
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
        Test accessing all the AnalogSignal objects corresponding to the
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
