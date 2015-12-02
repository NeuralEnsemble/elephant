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
            self.assertTrue(np.allclose(spike_train,spike_train))
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


if __name__ == '__main__':
    unittest.main()
