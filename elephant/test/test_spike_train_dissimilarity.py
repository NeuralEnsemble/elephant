# -*- coding: utf-8 -*-
"""
Tests for the spike train dissimilarity measures module.

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
from neo import SpikeTrain
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.integrate as spint
from quantities import ms, s, Hz
import elephant.kernels as kernels
import elephant.spike_train_generation as stg
import elephant.spike_train_dissimilarity as stds


class TimeScaleDependSpikeTrainDissimMeasures_TestCase(unittest.TestCase):
    def setUp(self):
        self.st00 = SpikeTrain([], units='ms', t_stop=1000.0)
        self.st01 = SpikeTrain([1], units='ms', t_stop=1000.0)
        self.st02 = SpikeTrain([2], units='ms', t_stop=1000.0)
        self.st03 = SpikeTrain([2.9], units='ms', t_stop=1000.0)
        self.st04 = SpikeTrain([3.1], units='ms', t_stop=1000.0)
        self.st05 = SpikeTrain([5], units='ms', t_stop=1000.0)
        self.st06 = SpikeTrain([500], units='ms', t_stop=1000.0)
        self.st07 = SpikeTrain([12, 32], units='ms', t_stop=1000.0)
        self.st08 = SpikeTrain([32, 52], units='ms', t_stop=1000.0)
        self.st09 = SpikeTrain([42], units='ms', t_stop=1000.0)
        self.st10 = SpikeTrain([18, 60], units='ms', t_stop=1000.0)
        self.st11 = SpikeTrain([10, 20, 30, 40], units='ms', t_stop=1000.0)
        self.st12 = SpikeTrain([40, 30, 20, 10], units='ms', t_stop=1000.0)
        self.st13 = SpikeTrain([15, 25, 35, 45], units='ms', t_stop=1000.0)
        self.st14 = SpikeTrain([10, 20, 30, 40, 50], units='ms', t_stop=1000.0)
        self.st15 = SpikeTrain([0.01, 0.02, 0.03, 0.04, 0.05],
                               units='s', t_stop=1000.0)
        self.st16 = SpikeTrain([12, 16, 28, 30, 42], units='ms', t_stop=1000.0)
        self.st21 = stg.homogeneous_poisson_process(50 * Hz, 0 * ms, 1000 * ms)
        self.st22 = stg.homogeneous_poisson_process(40 * Hz, 0 * ms, 1000 * ms)
        self.st23 = stg.homogeneous_poisson_process(30 * Hz, 0 * ms, 1000 * ms)
        self.rd_st_list = [self.st21, self.st22, self.st23]
        self.st31 = SpikeTrain([12.0], units='ms', t_stop=1000.0)
        self.st32 = SpikeTrain([12.0, 12.0], units='ms', t_stop=1000.0)
        self.st33 = SpikeTrain([20.0], units='ms', t_stop=1000.0)
        self.st34 = SpikeTrain([20.0, 20.0], units='ms', t_stop=1000.0)
        self.array1 = np.arange(1, 10)
        self.array2 = np.arange(1.2, 10)
        self.qarray1 = self.array1 * Hz
        self.qarray2 = self.array2 * Hz
        self.tau0 = 0.0 * ms
        self.q0 = np.inf / ms
        self.tau1 = 0.000000001 * ms
        self.q1 = 1.0 / self.tau1
        self.tau2 = 1.0 * ms
        self.q2 = 1.0 / self.tau2
        self.tau3 = 10.0 * ms
        self.q3 = 1.0 / self.tau3
        self.tau4 = 100.0 * ms
        self.q4 = 1.0 / self.tau4
        self.tau5 = 1000000000.0 * ms
        self.q5 = 1.0 / self.tau5
        self.tau6 = np.inf * ms
        self.q6 = 0.0 / ms
        self.tau7 = 0.01 * s
        self.q7 = 1.0 / self.tau7
        self.t = np.linspace(0, 200, 20000001) * ms

    def test_wrong_input(self):
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.array1, self.array2], self.q3)
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.qarray1, self.qarray2], self.q3)
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.qarray1, self.qarray2], 5.0 * ms)

        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.array1, self.array2], self.q3,
                          algorithm='intuitive')
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.qarray1, self.qarray2], self.q3,
                          algorithm='intuitive')
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.qarray1, self.qarray2], 5.0 * ms,
                          algorithm='intuitive')

        self.assertRaises(TypeError, stds.van_rossum_distance,
                          [self.array1, self.array2], self.tau3)
        self.assertRaises(TypeError, stds.van_rossum_distance,
                          [self.qarray1, self.qarray2], self.tau3)
        self.assertRaises(TypeError, stds.van_rossum_distance,
                          [self.qarray1, self.qarray2], 5.0 * Hz)

        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.st11, self.st13], self.tau2)
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.st11, self.st13], 5.0)
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.st11, self.st13], self.tau2,
                          algorithm='intuitive')
        self.assertRaises(TypeError, stds.victor_purpura_distance,
                          [self.st11, self.st13], 5.0,
                          algorithm='intuitive')
        self.assertRaises(TypeError, stds.van_rossum_distance,
                          [self.st11, self.st13], self.q4)
        self.assertRaises(TypeError, stds.van_rossum_distance,
                          [self.st11, self.st13], 5.0)

        self.assertRaises(NotImplementedError, stds.victor_purpura_distance,
                          [self.st01, self.st02], self.q3,
                          kernel=kernels.Kernel(2.0 / self.q3))
        self.assertRaises(NotImplementedError, stds.victor_purpura_distance,
                          [self.st01, self.st02], self.q3,
                          kernel=kernels.SymmetricKernel(2.0 / self.q3))
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st02], self.q1,
            kernel=kernels.TriangularKernel(
                2.0 / (np.sqrt(6.0) * self.q2)))[0, 1],
            stds.victor_purpura_distance(
            [self.st01, self.st02], self.q3,
            kernel=kernels.TriangularKernel(
                2.0 / (np.sqrt(6.0) * self.q2)))[0, 1])
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st02],
            kernel=kernels.TriangularKernel(
                2.0 / (np.sqrt(6.0) * self.q2)))[0, 1], 1.0)
        self.assertNotEqual(stds.victor_purpura_distance(
            [self.st01, self.st02],
            kernel=kernels.AlphaKernel(
                2.0 / (np.sqrt(6.0) * self.q2)))[0, 1], 1.0)

        self.assertRaises(NameError, stds.victor_purpura_distance,
                          [self.st11, self.st13], self.q2, algorithm='slow')

    def test_victor_purpura_distance_fast(self):
        # Tests of distances of simplest spike trains:
        self.assertEqual(stds.victor_purpura_distance(
            [self.st00, self.st00], self.q2)[0, 1], 0.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st00, self.st01], self.q2)[0, 1], 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st00], self.q2)[0, 1], 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st01], self.q2)[0, 1], 0.0)
        # Tests of distances under elementary spike operations
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st02], self.q2)[0, 1], 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st03], self.q2)[0, 1], 1.9)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st04], self.q2)[0, 1], 2.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st05], self.q2)[0, 1], 2.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st00, self.st07], self.q2)[0, 1], 2.0)
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st07, self.st08], self.q4)[0, 1], 0.4)
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st07, self.st10], self.q3)[0, 1], 0.6 + 2)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st11, self.st14], self.q2)[0, 1], 1)
        # Tests on timescales
        self.assertEqual(stds.victor_purpura_distance(
            [self.st11, self.st14], self.q1)[0, 1],
            stds.victor_purpura_distance(
            [self.st11, self.st14], self.q5)[0, 1])
        self.assertEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q0)[0, 1], 6.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q1)[0, 1], 6.0)
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q5)[0, 1], 2.0, 5)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q6)[0, 1], 2.0)
        # Tests on unordered spiketrains
        self.assertEqual(stds.victor_purpura_distance(
            [self.st11, self.st13], self.q4)[0, 1],
            stds.victor_purpura_distance(
            [self.st12, self.st13], self.q4)[0, 1])
        self.assertNotEqual(stds.victor_purpura_distance(
            [self.st11, self.st13], self.q4,
            sort=False)[0, 1],
            stds.victor_purpura_distance(
            [self.st12, self.st13], self.q4,
            sort=False)[0, 1])
        # Tests on metric properties with random spiketrains
        # (explicit calculation of second metric axiom in particular case,
        # because from dist_matrix it is trivial)
        dist_matrix = stds.victor_purpura_distance(
            [self.st21, self.st22, self.st23], self.q3)
        for i in range(3):
            for j in range(3):
                self.assertGreaterEqual(dist_matrix[i, j], 0)
                if dist_matrix[i, j] == 0:
                    assert_array_equal(self.rd_st_list[i], self.rd_st_list[j])
        assert_array_equal(stds.victor_purpura_distance(
            [self.st21, self.st22], self.q3),
            stds.victor_purpura_distance(
            [self.st22, self.st21], self.q3))
        self.assertLessEqual(dist_matrix[0, 1],
                             dist_matrix[0, 2] + dist_matrix[1, 2])
        self.assertLessEqual(dist_matrix[0, 2],
                             dist_matrix[1, 2] + dist_matrix[0, 1])
        self.assertLessEqual(dist_matrix[1, 2],
                             dist_matrix[0, 1] + dist_matrix[0, 2])
        # Tests on proper unit conversion
        self.assertAlmostEqual(
            stds.victor_purpura_distance([self.st14, self.st16],
                                         self.q3)[0, 1],
            stds.victor_purpura_distance([self.st15, self.st16],
                                         self.q3)[0, 1])
        self.assertAlmostEqual(
            stds.victor_purpura_distance([self.st16, self.st14],
                                         self.q3)[0, 1],
            stds.victor_purpura_distance([self.st16, self.st15],
                                         self.q3)[0, 1])
        self.assertAlmostEqual(
            stds.victor_purpura_distance([self.st01, self.st05],
                                         self.q3)[0, 1],
            stds.victor_purpura_distance([self.st01, self.st05],
                                         self.q7)[0, 1])
        # Tests on algorithmic behaviour for equal spike times
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st31, self.st34], self.q3)[0, 1], 0.8 + 1.0)
        self.assertAlmostEqual(
            stds.victor_purpura_distance([self.st31, self.st34],
                                         self.q3)[0, 1],
            stds.victor_purpura_distance([self.st32, self.st33],
                                         self.q3)[0, 1])
        self.assertAlmostEqual(
            stds.victor_purpura_distance(
                [self.st31, self.st33], self.q3)[0, 1] * 2.0,
            stds.victor_purpura_distance(
                [self.st32, self.st34], self.q3)[0, 1])
        # Tests on spike train list lengthes smaller than 2
        self.assertEqual(stds.victor_purpura_distance(
            [self.st21], self.q3)[0, 0], 0)
        self.assertEqual(len(stds.victor_purpura_distance([], self.q3)), 0)

    def test_victor_purpura_distance_intuitive(self):
        # Tests of distances of simplest spike trains
        self.assertEqual(stds.victor_purpura_distance(
            [self.st00, self.st00], self.q2,
            algorithm='intuitive')[0, 1], 0.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st00, self.st01], self.q2,
            algorithm='intuitive')[0, 1], 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st00], self.q2,
            algorithm='intuitive')[0, 1], 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st01], self.q2,
            algorithm='intuitive')[0, 1], 0.0)
        # Tests of distances under elementary spike operations
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st02], self.q2,
            algorithm='intuitive')[0, 1], 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st03], self.q2,
            algorithm='intuitive')[0, 1], 1.9)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st04], self.q2,
            algorithm='intuitive')[0, 1], 2.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st05], self.q2,
            algorithm='intuitive')[0, 1], 2.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st00, self.st07], self.q2,
            algorithm='intuitive')[0, 1], 2.0)
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st07, self.st08], self.q4,
            algorithm='intuitive')[0, 1], 0.4)
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st07, self.st10], self.q3,
            algorithm='intuitive')[0, 1], 2.6)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st11, self.st14], self.q2,
            algorithm='intuitive')[0, 1], 1)
        # Tests on timescales
        self.assertEqual(stds.victor_purpura_distance(
            [self.st11, self.st14], self.q1,
            algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st11, self.st14], self.q5,
            algorithm='intuitive')[0, 1])
        self.assertEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q0,
            algorithm='intuitive')[0, 1], 6.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q1,
            algorithm='intuitive')[0, 1], 6.0)
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q5,
            algorithm='intuitive')[0, 1], 2.0, 5)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st07, self.st11], self.q6,
            algorithm='intuitive')[0, 1], 2.0)
        # Tests on unordered spiketrains
        self.assertEqual(stds.victor_purpura_distance(
            [self.st11, self.st13], self.q4,
            algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st12, self.st13], self.q4,
            algorithm='intuitive')[0, 1])
        self.assertNotEqual(stds.victor_purpura_distance(
            [self.st11, self.st13], self.q4,
            sort=False, algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st12, self.st13], self.q4,
            sort=False, algorithm='intuitive')[0, 1])
        # Tests on metric properties with random spiketrains
        # (explicit calculation of second metric axiom in particular case,
        # because from dist_matrix it is trivial)
        dist_matrix = stds.victor_purpura_distance(
            [self.st21, self.st22, self.st23],
            self.q3, algorithm='intuitive')
        for i in range(3):
            for j in range(3):
                self.assertGreaterEqual(dist_matrix[i, j], 0)
                if dist_matrix[i, j] == 0:
                    assert_array_equal(self.rd_st_list[i], self.rd_st_list[j])
        assert_array_equal(stds.victor_purpura_distance(
            [self.st21, self.st22], self.q3,
            algorithm='intuitive'),
            stds.victor_purpura_distance(
            [self.st22, self.st21], self.q3,
            algorithm='intuitive'))
        self.assertLessEqual(dist_matrix[0, 1],
                             dist_matrix[0, 2] + dist_matrix[1, 2])
        self.assertLessEqual(dist_matrix[0, 2],
                             dist_matrix[1, 2] + dist_matrix[0, 1])
        self.assertLessEqual(dist_matrix[1, 2],
                             dist_matrix[0, 1] + dist_matrix[0, 2])
        # Tests on proper unit conversion
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st14, self.st16], self.q3,
            algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st15, self.st16], self.q3,
            algorithm='intuitive')[0, 1])
        self.assertAlmostEqual(stds.victor_purpura_distance(
            [self.st16, self.st14], self.q3,
            algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st16, self.st15], self.q3,
            algorithm='intuitive')[0, 1])
        self.assertEqual(stds.victor_purpura_distance(
            [self.st01, self.st05], self.q3,
            algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st01, self.st05], self.q7,
            algorithm='intuitive')[0, 1])
        # Tests on algorithmic behaviour for equal spike times
        self.assertEqual(stds.victor_purpura_distance(
            [self.st31, self.st34], self.q3,
            algorithm='intuitive')[0, 1],
            0.8 + 1.0)
        self.assertEqual(stds.victor_purpura_distance(
            [self.st31, self.st34], self.q3,
            algorithm='intuitive')[0, 1],
            stds.victor_purpura_distance(
            [self.st32, self.st33], self.q3,
            algorithm='intuitive')[0, 1])
        self.assertEqual(stds.victor_purpura_distance(
            [self.st31, self.st33], self.q3,
            algorithm='intuitive')[0, 1] * 2.0,
            stds.victor_purpura_distance(
            [self.st32, self.st34], self.q3,
            algorithm='intuitive')[0, 1])
        # Tests on spike train list lengthes smaller than 2
        self.assertEqual(stds.victor_purpura_distance(
            [self.st21], self.q3,
            algorithm='intuitive')[0, 0], 0)
        self.assertEqual(len(stds.victor_purpura_distance(
                             [], self.q3, algorithm='intuitive')), 0)

    def test_victor_purpura_algorithm_comparison(self):
        assert_array_almost_equal(
            stds.victor_purpura_distance([self.st21, self.st22, self.st23],
                                         self.q3),
            stds.victor_purpura_distance([self.st21, self.st22, self.st23],
                                         self.q3, algorithm='intuitive'))

    def test_van_rossum_distance(self):
        # Tests of distances of simplest spike trains
        self.assertEqual(stds.van_rossum_distance(
            [self.st00, self.st00], self.tau2)[0, 1], 0.0)
        self.assertEqual(stds.van_rossum_distance(
            [self.st00, self.st01], self.tau2)[0, 1], 1.0)
        self.assertEqual(stds.van_rossum_distance(
            [self.st01, self.st00], self.tau2)[0, 1], 1.0)
        self.assertEqual(stds.van_rossum_distance(
            [self.st01, self.st01], self.tau2)[0, 1], 0.0)
        # Tests of distances under elementary spike operations
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st01, self.st02], self.tau2)[0, 1],
            float(np.sqrt(2 * (1.0 - np.exp(-np.absolute(
                ((self.st01[0] - self.st02[0]) /
                 self.tau2).simplified))))))
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st01, self.st05], self.tau2)[0, 1],
            float(np.sqrt(2 * (1.0 - np.exp(-np.absolute(
                ((self.st01[0] - self.st05[0]) /
                 self.tau2).simplified))))))
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st01, self.st05], self.tau2)[0, 1],
            np.sqrt(2.0), 1)
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st01, self.st06], self.tau2)[0, 1],
            np.sqrt(2.0), 20)
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st00, self.st07], self.tau1)[0, 1],
            np.sqrt(0 + 2))
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st07, self.st08], self.tau4)[0, 1],
            float(np.sqrt(2 * (1.0 - np.exp(-np.absolute(
                ((self.st07[0] - self.st08[-1]) /
                 self.tau4).simplified))))))
        f_minus_g_squared = (
            (self.t > self.st08[0]) * np.exp(
                -((self.t - self.st08[0]) / self.tau3).simplified) +
            (self.t > self.st08[1]) * np.exp(
                -((self.t - self.st08[1]) / self.tau3).simplified) -
            (self.t > self.st09[0]) * np.exp(
                -((self.t - self.st09[0]) / self.tau3).simplified))**2
        distance = np.sqrt(2.0 * spint.cumtrapz(
                           y=f_minus_g_squared, x=self.t.magnitude)[-1] /
                           self.tau3.rescale(self.t.units).magnitude)
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st08, self.st09], self.tau3)[0, 1], distance, 5)
        self.assertAlmostEqual(stds.van_rossum_distance(
            [self.st11, self.st14], self.tau2)[0, 1], 1)
        # Tests on timescales
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st11, self.st14], self.tau1)[0, 1],
            stds.van_rossum_distance([self.st11, self.st14], self.tau5)[0, 1])

        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st11], self.tau0)[0, 1],
            np.sqrt(len(self.st07) + len(self.st11)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st14], self.tau0)[0, 1],
            np.sqrt(len(self.st07) + len(self.st14)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st11], self.tau1)[0, 1],
            np.sqrt(len(self.st07) + len(self.st11)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st14], self.tau1)[0, 1],
            np.sqrt(len(self.st07) + len(self.st14)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st11], self.tau5)[0, 1],
            np.absolute(len(self.st07) - len(self.st11)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st14], self.tau5)[0, 1],
            np.absolute(len(self.st07) - len(self.st14)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st11], self.tau6)[0, 1],
            np.absolute(len(self.st07) - len(self.st11)))
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st07, self.st14], self.tau6)[0, 1],
            np.absolute(len(self.st07) - len(self.st14)))
        # Tests on unordered spiketrains
        self.assertEqual(
            stds.van_rossum_distance([self.st11, self.st13], self.tau4)[0, 1],
            stds.van_rossum_distance([self.st12, self.st13], self.tau4)[0, 1])
        self.assertNotEqual(
            stds.van_rossum_distance([self.st11, self.st13],
                                     self.tau4, sort=False)[0, 1],
            stds.van_rossum_distance([self.st12, self.st13],
                                     self.tau4, sort=False)[0, 1])
        # Tests on metric properties with random spiketrains
        # (explicit calculation of second metric axiom in particular case,
        # because from dist_matrix it is trivial)
        dist_matrix = stds.van_rossum_distance(
            [self.st21, self.st22, self.st23], self.tau3)
        for i in range(3):
            for j in range(3):
                self.assertGreaterEqual(dist_matrix[i, j], 0)
                if dist_matrix[i, j] == 0:
                    assert_array_equal(self.rd_st_list[i], self.rd_st_list[j])
        assert_array_equal(
            stds.van_rossum_distance([self.st21, self.st22], self.tau3),
            stds.van_rossum_distance([self.st22, self.st21], self.tau3))
        self.assertLessEqual(dist_matrix[0, 1],
                             dist_matrix[0, 2] + dist_matrix[1, 2])
        self.assertLessEqual(dist_matrix[0, 2],
                             dist_matrix[1, 2] + dist_matrix[0, 1])
        self.assertLessEqual(dist_matrix[1, 2],
                             dist_matrix[0, 1] + dist_matrix[0, 2])
        # Tests on proper unit conversion
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st14, self.st16], self.tau3)[0, 1],
            stds.van_rossum_distance([self.st15, self.st16], self.tau3)[0, 1])
        self.assertAlmostEqual(
            stds.van_rossum_distance([self.st16, self.st14], self.tau3)[0, 1],
            stds.van_rossum_distance([self.st16, self.st15], self.tau3)[0, 1])
        self.assertEqual(
            stds.van_rossum_distance([self.st01, self.st05], self.tau3)[0, 1],
            stds.van_rossum_distance([self.st01, self.st05], self.tau7)[0, 1])
        # Tests on algorithmic behaviour for equal spike times
        f_minus_g_squared = (
            (self.t > self.st31[0]) * np.exp(
                -((self.t - self.st31[0]) / self.tau3).simplified) -
            (self.t > self.st34[0]) * np.exp(
                -((self.t - self.st34[0]) / self.tau3).simplified) -
            (self.t > self.st34[1]) * np.exp(
                -((self.t - self.st34[1]) / self.tau3).simplified))**2
        distance = np.sqrt(2.0 * spint.cumtrapz(
                           y=f_minus_g_squared, x=self.t.magnitude)[-1] /
                           self.tau3.rescale(self.t.units).magnitude)
        self.assertAlmostEqual(stds.van_rossum_distance([self.st31, self.st34],
                                                        self.tau3)[0, 1],
                               distance, 5)
        self.assertEqual(stds.van_rossum_distance([self.st31, self.st34],
                                                  self.tau3)[0, 1],
                         stds.van_rossum_distance([self.st32, self.st33],
                                                  self.tau3)[0, 1])
        self.assertEqual(stds.van_rossum_distance([self.st31, self.st33],
                                                  self.tau3)[0, 1] * 2.0,
                         stds.van_rossum_distance([self.st32, self.st34],
                                                  self.tau3)[0, 1])
        # Tests on spike train list lengthes smaller than 2
        self.assertEqual(stds.van_rossum_distance(
            [self.st21], self.tau3)[0, 0], 0)
        self.assertEqual(len(stds.van_rossum_distance([], self.tau3)), 0)


if __name__ == '__main__':
    unittest.main()
