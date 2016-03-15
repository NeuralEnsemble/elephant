# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division
import unittest
import os
import warnings

import neo
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from scipy.stats import kstest, expon
from quantities import ms, second, Hz, kHz, mV

import elephant.spike_train_generation as stgen
from elephant.statistics import isi


def pdiff(a, b):
    """Difference between a and b as a fraction of a

    i.e. abs((a - b)/a)
    """
    return abs((a - b)/a)


class AnalogSignalSpikeExtractionTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_threshold_detection(self):
        # Test whether spikes are extracted at the correct times from
        # an analog signal.

        # Load membrane potential simulated using Brian2
        # according to make_spike_extraction_test_data.py.
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        npz_file_loc = os.path.join(curr_dir,'spike_extraction_test_data.npz')
        iom2 = neo.io.PyNNNumpyIO(npz_file_loc)
        data = iom2.read()
        vm = data[0].segments[0].analogsignals[0]
        spike_train = stgen.threshold_detection(vm)
        try:
            len(spike_train)
        except TypeError: # Handles an error in Neo related to some zero length
                          # spike trains being treated as unsized objects.
            warnings.warn(("The spike train may be an unsized object. This may be related "
                            "to an issue in Neo with some zero-length SpikeTrain objects. "
                            "Bypassing this by creating an empty SpikeTrain object."))
            spike_train = neo.core.SpikeTrain([],t_start=spike_train.t_start,
                                                 t_stop=spike_train.t_stop,
                                                 units=spike_train.units)

        # Correct values determined previously.
        true_spike_train = [0.0123, 0.0354, 0.0712, 0.1191,
                            0.1694, 0.22, 0.2711]

        # Does threshold_detection gives the correct number of spikes?
        self.assertEqual(len(spike_train),len(true_spike_train))
        # Does threshold_detection gives the correct times for the spikes?
        try:
            assert_array_almost_equal(spike_train,spike_train)
        except AttributeError: # If numpy version too old to have allclose
            self.assertTrue(np.array_equal(spike_train,spike_train))

class HomogeneousPoissonProcessTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_statistics(self):
        # There is a statistical test and has a non-zero chance of failure during normal operation.
        # Re-run the test to see if the error persists.
        for rate in [123.0*Hz, 0.123*kHz]:
            for t_stop in [2345*ms, 2.345*second]:
                spiketrain = stgen.homogeneous_poisson_process(rate, t_stop=t_stop)
                intervals = isi(spiketrain)

                expected_spike_count = int((rate * t_stop).simplified)
                self.assertLess(pdiff(expected_spike_count, spiketrain.size), 0.2)  # should fail about 1 time in 1000

                expected_mean_isi = (1/rate)
                self.assertLess(pdiff(expected_mean_isi, intervals.mean()), 0.2)

                expected_first_spike = 0*ms
                self.assertLess(spiketrain[0] - expected_first_spike, 7*expected_mean_isi)

                expected_last_spike = t_stop
                self.assertLess(expected_last_spike - spiketrain[-1], 7*expected_mean_isi)

                # Kolmogorov-Smirnov test
                D, p = kstest(intervals.rescale(t_stop.units),
                              "expon",
                              args=(0, expected_mean_isi.rescale(t_stop.units)),  # args are (loc, scale)
                              alternative='two-sided')
                self.assertGreater(p, 0.001)
                self.assertLess(D, 0.12)

    def test_low_rates(self):
        spiketrain = stgen.homogeneous_poisson_process(0*Hz, t_stop=1000*ms)
        self.assertEqual(spiketrain.size, 0)
        # not really a test, just making sure that all code paths are covered
        for i in range(10):
            spiketrain = stgen.homogeneous_poisson_process(1*Hz, t_stop=1000*ms)

    def test_buffer_overrun(self):
        np.random.seed(6085)  # this seed should produce a buffer overrun
        t_stop=1000*ms
        rate = 10*Hz
        spiketrain = stgen.homogeneous_poisson_process(rate, t_stop=t_stop)
        expected_last_spike = t_stop
        expected_mean_isi = (1/rate).rescale(ms)
        self.assertLess(expected_last_spike - spiketrain[-1], 4*expected_mean_isi)


class HomogeneousGammaProcessTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_statistics(self):
        # There is a statistical test and has a non-zero chance of failure during normal operation.
        # Re-run the test to see if the error persists.
        a = 3.0
        for b in (67.0*Hz, 0.067*kHz):
            for t_stop in (2345*ms, 2.345*second):
                spiketrain = stgen.homogeneous_gamma_process(a, b, t_stop=t_stop)
                intervals = isi(spiketrain)

                expected_spike_count = int((b/a * t_stop).simplified)
                self.assertLess(pdiff(expected_spike_count, spiketrain.size), 0.25)  # should fail about 1 time in 1000

                expected_mean_isi = (a/b).rescale(ms)
                self.assertLess(pdiff(expected_mean_isi, intervals.mean()), 0.3)

                expected_first_spike = 0*ms
                self.assertLess(spiketrain[0] - expected_first_spike, 4*expected_mean_isi)

                expected_last_spike = t_stop
                self.assertLess(expected_last_spike - spiketrain[-1], 4*expected_mean_isi)

                # Kolmogorov-Smirnov test
                D, p = kstest(intervals.rescale(t_stop.units),
                              "gamma",
                              args=(a, 0, (1/b).rescale(t_stop.units)),  # args are (a, loc, scale)
                              alternative='two-sided')
                self.assertGreater(p, 0.001)
                self.assertLess(D, 0.25)


class cppTestCase(unittest.TestCase):
    def test_cpp_hom(self):
        # testing output with generic inputs
        A = [0, .9, .1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = 3 * Hz
        cpp_hom = stgen.cpp(rate, A, t_stop, t_start=t_start)
        # testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_hom], [neo.SpikeTrain]*len(cpp_hom))
        self.assertEqual(cpp_hom[0].simplified.units, 1000 * ms)
        self.assertEqual(type(cpp_hom), list)
        # testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_hom], [1000 * ms]*len(
                cpp_hom))
        # testing output t_start t_stop
        for st in cpp_hom:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        self.assertEqual(len(cpp_hom), len(A) - 1)

        # testing the units
        A = [0, 0.9, 0.1]
        t_stop = 10000*ms
        t_start = 5 * 1000 * ms
        rate = 3 * Hz
        cpp_unit = stgen.cpp(rate, A, t_stop, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)

        # testing output without copy of spikes
        A = [1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = 3 * Hz
        cpp_hom_empty = stgen.cpp(rate, A, t_stop, t_start=t_start)

        self.assertEqual(
            [len(train) for train in cpp_hom_empty], [0]*len(cpp_hom_empty))

        # testing output with rate equal to 0
        A = [0, .9, .1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = 0 * Hz
        cpp_hom_empty_r = stgen.cpp(rate, A, t_stop, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cpp_hom_empty_r], [0]*len(
                cpp_hom_empty_r))

        # testing output with same spike trains in output
        A = [0, 0, 1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = 3 * Hz
        cpp_hom_eq = stgen.cpp(rate, A, t_stop, t_start=t_start)

        self.assertTrue(
            np.allclose(cpp_hom_eq[0].magnitude, cpp_hom_eq[1].magnitude))

    def test_cpp_hom_errors(self):
        # testing raises of ValueError (wrong inputs)
        # testing empty amplitude
        self.assertRaises(
            ValueError, stgen.cpp, A=[], t_stop=10*1000 * ms, rate=3*Hz)

        # testing sum of amplitude>1
        self.assertRaises(
            ValueError, stgen.cpp, A=[1, 1, 1], t_stop=10*1000 * ms, rate=3*Hz)
        # testing negative value in the amplitude
        self.assertRaises(
            ValueError, stgen.cpp, A=[-1, 1, 1], t_stop=10*1000 * ms,
            rate=3*Hz)
        # test negative rate
        self.assertRaises(
            AssertionError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms,
            rate=-3*Hz)
        # test wrong unit for rate
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms,
            rate=3*1000 * ms)

        # testing raises of AttributeError (missing input units)
        # Testing missing unit to t_stop
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10, rate=3*Hz)
        # Testing missing unit to t_start
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms, rate=3*Hz,
            t_start=3)
        # testing rate missing unit
        self.assertRaises(
            AttributeError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms,
            rate=3)

    def test_cpp_het(self):
        # testing output with generic inputs
        A = [0, .9, .1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = [3, 4] * Hz
        cpp_het = stgen.cpp(rate, A, t_stop, t_start=t_start)
        # testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_het], [neo.SpikeTrain]*len(cpp_het))
        self.assertEqual(cpp_het[0].simplified.units, 1000 * ms)
        self.assertEqual(type(cpp_het), list)
        # testing units
        self.assertEqual(
            [train.simplified.units for train in cpp_het], [1000 * ms]*len(
                cpp_het))
        # testing output t_start and t_stop
        for st in cpp_het:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        # testing the number of output spiketrains
        self.assertEqual(len(cpp_het), len(A) - 1)
        self.assertEqual(len(cpp_het), len(rate))

        # testing the units
        A = [0, 0.9, 0.1]
        t_stop = 10000*ms
        t_start = 5 * 1000 * ms
        rate = [3, 4] * Hz
        cpp_unit = stgen.cpp(rate, A, t_stop, t_start=t_start)

        self.assertEqual(cpp_unit[0].units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_stop.units, t_stop.units)
        self.assertEqual(cpp_unit[0].t_start.units, t_stop.units)
        # testing without copying any spikes
        A = [1, 0, 0]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = [3, 4] * Hz
        cpp_het_empty = stgen.cpp(rate, A, t_stop, t_start=t_start)

        self.assertEqual(len(cpp_het_empty[0]), 0)

        # testing output with rate equal to 0
        A = [0, .9, .1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = [0, 0] * Hz
        cpp_het_empty_r = stgen.cpp(rate, A, t_stop, t_start=t_start)
        self.assertEqual(
            [len(train) for train in cpp_het_empty_r], [0]*len(
                cpp_het_empty_r))

        # testing completely sync spiketrains
        A = [0, 0, 1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = [3, 3] * Hz
        cpp_het_eq = stgen.cpp(rate, A, t_stop, t_start=t_start)

        self.assertTrue(np.allclose(
            cpp_het_eq[0].magnitude, cpp_het_eq[1].magnitude))

    def test_cpp_het_err(self):
        # testing raises of ValueError (wrong inputs)
        # testing empty amplitude
        self.assertRaises(
            ValueError, stgen.cpp, A=[], t_stop=10*1000 * ms, rate=[3, 4]*Hz)
        # testing sum amplitude>1
        self.assertRaises(
            ValueError, stgen.cpp, A=[1, 1, 1], t_stop=10*1000 * ms,
            rate=[3, 4]*Hz)
        # testing amplitude negative value
        self.assertRaises(
            ValueError, stgen.cpp, A=[-1, 1, 1], t_stop=10*1000 * ms,
            rate=[3, 4]*Hz)
        # testing negative rate
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms,
            rate=[-3, 4]*Hz)
        # testing empty rate
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms, rate=[]*Hz)
        # testing empty amplitude
        self.assertRaises(
            ValueError, stgen.cpp, A=[], t_stop=10*1000 * ms, rate=[3, 4]*Hz)
        # testing different len(A)-1 and len(rate)
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1], t_stop=10*1000 * ms, rate=[3, 4]*Hz)
        # testing rate with different unit from Hz
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1], t_stop=10*1000 * ms,
            rate=[3, 4]*1000 * ms)
        # Testing analytical constrain between amplitude and rate
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 0, 1], t_stop=10*1000 * ms,
            rate=[3, 4]*Hz, t_start=3)

        # testing raises of AttributeError (missing input units)
        # Testing missing unit to t_stop
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10, rate=[3, 4]*Hz)
        # Testing missing unit to t_start
        self.assertRaises(
            ValueError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms,
            rate=[3, 4]*Hz, t_start=3)
        # Testing missing unit to rate
        self.assertRaises(
            AttributeError, stgen.cpp, A=[0, 1, 0], t_stop=10*1000 * ms,
            rate=[3, 4])

    def test_cpp_jttered(self):
        # testing output with generic inputs
        A = [0, .9, .1]
        t_stop = 10 * 1000 * ms
        t_start = 5 * 1000 * ms
        rate = 3 * Hz
        cpp_shift = stgen.cpp(
            rate, A, t_stop, t_start=t_start, shift=3*ms)
        # testing the ouput formats
        self.assertEqual(
            [type(train) for train in cpp_shift], [neo.SpikeTrain]*len(
                cpp_shift))
        self.assertEqual(cpp_shift[0].simplified.units, 1000 * ms)
        self.assertEqual(type(cpp_shift), list)
        # testing quantities format of the output
        self.assertEqual(
            [train.simplified.units for train in cpp_shift],
            [1000 * ms]*len(cpp_shift))
        # testing output t_start t_stop
        for st in cpp_shift:
            self.assertEqual(st.t_stop, t_stop)
            self.assertEqual(st.t_start, t_start)
        self.assertEqual(len(cpp_shift), len(A) - 1)


if __name__ == '__main__':
    unittest.main()
