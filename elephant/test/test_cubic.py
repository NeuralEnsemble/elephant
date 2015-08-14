# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:35:49 2014

@author: quaglio
"""

import unittest
import elephant.cubic as cubic
import quantities as pq
import neo
import numpy


class CubicTestCase(unittest.TestCase):
    '''
    This test is constructed to check the implementation of the CuBIC
    method [1].
    In the setup function is constructed an neo.AnalogSignalArray, that
    represents the Population Histogram of a population of neurons with order
    of correlation equal to ten. Since the population contu is either equal to
    0 or 10 means that the embedded order of correlation is exactly 10.
    In test_cubic() the format of all the output and the order of correlation
    of the function jelephant.cubic.cubic() are tested.

    References
    ----------
    [1]Staude, Rotter, Gruen, (2009) J. Comp. Neurosci
    '''
    def setUp(self):
        n2 = 300
        n0 = 100000-n2
        self.xi = 10
        self.data_signal = neo.AnalogSignalArray(
            numpy.array([self.xi] * n2 + [0] * n0).reshape(n0 + n2, 1) *
            pq.dimensionless, sampling_period=1*pq.s)
        self.data_array = numpy.array([self.xi] * n2 + [0] * n0)
        self.alpha = 0.05
        self.ximax = 10

    def test_cubic(self):

        # Computing the output of CuBIC for the test data AnalogSignal
        xi, p_vals, k, test_aborted = cubic.cubic(
            self.data_signal, alpha=self.alpha)

        # Check the types of the outputs
        self.assertIsInstance(xi, int)
        self.assertIsInstance(p_vals, list)
        self.assertIsInstance(k, list)

        # Check that the number of tests is the output order of correlation
        self.assertEqual(xi, len(p_vals))

        # Check that all the first  xi-1 tests have not passed the
        # significance level alpha
        for p in p_vals[:-1]:
            self.assertGreater(self.alpha, p)

        # Check that the last p-value has passed the significance level
        self.assertGreater(p_vals[-1], self.alpha)

        # Check that the number of cumulant of the output is 3
        self.assertEqual(3, len(k))

        # Check the analytical constrain of the cumulants for which K_1<K_2
        self.assertGreater(k[1], k[0])

        # Check the computed order of correlation is the expected
        # from the test data
        self.assertEqual(xi, self.xi)

        # Computing the output of CuBIC for the test data Array
        xi, p_vals, k, test_aborted = cubic.cubic(
            self.data_array, alpha=self.alpha)

        # Check the types of the outputs
        self.assertIsInstance(xi, int)
        self.assertIsInstance(p_vals, list)
        self.assertIsInstance(k, list)

        # Check that the number of tests is the output order of correlation
        self.assertEqual(xi, len(p_vals))

        # Check that all the first  xi-1 tests have not passed the
        # significance level alpha
        for p in p_vals[:-1]:
            self.assertGreater(self.alpha, p)

        # Check that the last p-value has passed the significance level
        self.assertGreater(p_vals[-1], self.alpha)

        # Check that the number of cumulant of the output is 3
        self.assertEqual(3, len(k))

        # Check the analytical constrain of the cumulants for which K_1<K_2
        self.assertGreater(k[1], k[0])

        # Check the computed order of correlation is the expected
        # from the test data
        self.assertEqual(xi, self.xi)

        # Check the output for test_aborted
        self.assertEqual(test_aborted, False)

    def test_cubic_ximax(self):
        # Test exceeding ximax
        xi_ximax, p_vals_ximax, k_ximax, test_aborted = cubic.cubic(
            self.data_signal, alpha=1, ximax=self.ximax)

        self.assertEqual(test_aborted, True)
        self.assertEqual(xi_ximax - 1, self.ximax)

    def test_cubic_errors(self):

        # Check error ouputs for mis-settings of the parameters

        # Empty signal
        self.assertRaises(
            ValueError, cubic.cubic, neo.AnalogSignalArray(
                []*pq.dimensionless, sampling_period=10*pq.ms))

        # Multidimensional array
        self.assertRaises(ValueError, cubic.cubic, neo.AnalogSignalArray(
            [[1, 2, 3], [1, 2, 3]] * pq.dimensionless,
            sampling_period=10 * pq.ms))
        self.assertRaises(ValueError, cubic.cubic, numpy.array(
            [[1, 2, 3], [1, 2, 3]]))

        # Negative alpha
        self.assertRaises(ValueError, cubic.cubic, self.data_array, alpha=-0.1)

        # Negative number of iterations ximax
        self.assertRaises(ValueError, cubic.cubic, self.data_array, ximax=-100)

        # Checking case in which the second cumulant of the signal is smaller
        # than the first cumulant (analitycal constrain of the method)
        self.assertRaises(ValueError, cubic.cubic, neo.AnalogSignalArray(
            numpy.array([1]*1000).reshape(1000, 1), units=pq.dimensionless,
            sampling_period=10*pq.ms), alpha=self.alpha)


def suite():
    suite = unittest.makeSuite(CubicTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
