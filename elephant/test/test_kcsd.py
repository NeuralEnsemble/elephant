# -*- coding: utf-8 -*-
"""
Unit tests for the kCSD methods

This was written by :
Chaitanya Chintaluri,
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.

:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import neo
import numpy as np
import quantities as pq
from elephant import current_source_density as CSD
import elephant.current_source_density_src.utility_functions as utils


class KCSD1D_TestCase(unittest.TestCase):
    def setUp(self):
        self.ele_pos = utils.generate_electrodes(dim=1).reshape(5, 1)
        self.csd_profile = utils.gauss_1d_dipole
        pots = CSD.generate_lfp(self.csd_profile, self.ele_pos)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD1D'
        self.test_params = {'h': 50.}

        temp_signals = []
        for ii in range(len(self.pots)):
            temp_signals.append(self.pots[ii])
        self.an_sigs = neo.AnalogSignal(np.array(temp_signals).T * pq.mV,
                                        sampling_rate=1000 * pq.Hz)
        chidx = neo.ChannelIndex(range(len(self.pots)))
        chidx.analogsignals.append(self.an_sigs)
        chidx.coordinates = self.ele_pos * pq.mm

        chidx.create_relationship()

    def test_kcsd1d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = CSD.estimate_csd(self.an_sigs, method=self.test_method,
                                  **self.test_params)
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(result.times, [0.] * pq.s)
        self.assertEqual(len(result.annotations.keys()), 1)
        true_csd = self.csd_profile(result.annotations['x_coords'])
        rms = np.linalg.norm(np.array(result[0, :]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS between trueCSD and estimate > 0.5')

    def test_valid_inputs(self):
        self.test_method = 'InvalidMethodName'
        self.assertRaises(ValueError, self.test_kcsd1d_estimate)
        self.test_method = 'KCSD1D'
        self.test_params = {'src_type': 22}
        self.assertRaises(KeyError, self.test_kcsd1d_estimate)
        self.test_method = 'KCSD1D'
        self.test_params = {'InvalidKwarg': 21}
        self.assertRaises(TypeError, self.test_kcsd1d_estimate)
        cv_params = {'InvalidCVArg': np.array((0.1, 0.25, 0.5))}
        self.assertRaises(TypeError, self.test_kcsd1d_estimate, cv_params)


class KCSD2D_TestCase(unittest.TestCase):
    def setUp(self):
        xx_ele, yy_ele = utils.generate_electrodes(dim=2, res=9,
                                                   xlims=[0.05, 0.95],
                                                   ylims=[0.05, 0.95])
        self.ele_pos = np.vstack((xx_ele, yy_ele)).T
        self.csd_profile = utils.large_source_2D
        pots = CSD.generate_lfp(
            self.csd_profile,
            xx_ele,
            yy_ele,
            resolution=100)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD2D'
        self.test_params = {'gdx': 0.25, 'gdy': 0.25, 'R_init': 0.08,
                            'h': 50., 'xmin': 0., 'xmax': 1.,
                            'ymin': 0., 'ymax': 1.}
        temp_signals = []
        for ii in range(len(self.pots)):
            temp_signals.append(self.pots[ii])
        self.an_sigs = neo.AnalogSignal(np.array(temp_signals).T * pq.mV,
                                        sampling_rate=1000 * pq.Hz)
        chidx = neo.ChannelIndex(range(len(self.pots)))
        chidx.analogsignals.append(self.an_sigs)
        chidx.coordinates = self.ele_pos * pq.mm

        chidx.create_relationship()

    def test_kcsd2d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = CSD.estimate_csd(self.an_sigs, method=self.test_method,
                                  **self.test_params)
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(result.times, [0.] * pq.s)
        self.assertEqual(len(result.annotations.keys()), 2)
        true_csd = self.csd_profile(result.annotations['x_coords'],
                                    result.annotations['y_coords'])
        rms = np.linalg.norm(np.array(result[0, :]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS ' + str(rms) +
                        'between trueCSD and estimate > 0.5')

    def test_moi_estimate(self):
        result = CSD.estimate_csd(self.an_sigs, method='MoIKCSD',
                                  MoI_iters=10, lambd=0.0,
                                  gdx=0.2, gdy=0.2)
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(result.times, [0.] * pq.s)
        self.assertEqual(len(result.annotations.keys()), 2)

    def test_valid_inputs(self):
        self.test_method = 'InvalidMethodName'
        self.assertRaises(ValueError, self.test_kcsd2d_estimate)
        self.test_method = 'KCSD2D'
        self.test_params = {'src_type': 22}
        self.assertRaises(KeyError, self.test_kcsd2d_estimate)
        self.test_params = {'InvalidKwarg': 21}
        self.assertRaises(TypeError, self.test_kcsd2d_estimate)
        cv_params = {'InvalidCVArg': np.array((0.1, 0.25, 0.5))}
        self.assertRaises(TypeError, self.test_kcsd2d_estimate, cv_params)


class KCSD3D_TestCase(unittest.TestCase):
    def setUp(self):
        xx_ele, yy_ele, zz_ele = utils.generate_electrodes(dim=3, res=5,
                                                           xlims=[0.15, 0.85],
                                                           ylims=[0.15, 0.85],
                                                           zlims=[0.15, 0.85])
        self.ele_pos = np.vstack((xx_ele, yy_ele, zz_ele)).T
        self.csd_profile = utils.gauss_3d_dipole
        pots = CSD.generate_lfp(self.csd_profile, xx_ele, yy_ele, zz_ele)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD3D'
        self.test_params = {'gdx': 0.05, 'gdy': 0.05, 'gdz': 0.05,
                            'lambd': 5.10896977451e-19, 'src_type': 'step',
                            'R_init': 0.31, 'xmin': 0., 'xmax': 1., 'ymin': 0.,
                            'ymax': 1., 'zmin': 0., 'zmax': 1.}

        temp_signals = []
        for ii in range(len(self.pots)):
            temp_signals.append(self.pots[ii])
        self.an_sigs = neo.AnalogSignal(np.array(temp_signals).T * pq.mV,
                                        sampling_rate=1000 * pq.Hz)
        chidx = neo.ChannelIndex(range(len(self.pots)))
        chidx.analogsignals.append(self.an_sigs)
        chidx.coordinates = self.ele_pos * pq.mm

        chidx.create_relationship()

    def test_kcsd3d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = CSD.estimate_csd(self.an_sigs, method=self.test_method,
                                  **self.test_params)
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(result.times, [0.] * pq.s)
        self.assertEqual(len(result.annotations.keys()), 3)
        true_csd = self.csd_profile(result.annotations['x_coords'],
                                    result.annotations['y_coords'],
                                    result.annotations['z_coords'])
        rms = np.linalg.norm(np.array(result[0, :]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS ' + str(rms) +
                        ' between trueCSD and estimate > 0.5')

    def test_valid_inputs(self):
        self.test_method = 'InvalidMethodName'
        self.assertRaises(ValueError, self.test_kcsd3d_estimate)
        self.test_method = 'KCSD3D'
        self.test_params = {'src_type': 22}
        self.assertRaises(KeyError, self.test_kcsd3d_estimate)
        self.test_params = {'InvalidKwarg': 21}
        self.assertRaises(TypeError, self.test_kcsd3d_estimate)
        cv_params = {'InvalidCVArg': np.array((0.1, 0.25, 0.5))}
        self.assertRaises(TypeError, self.test_kcsd3d_estimate, cv_params)


if __name__ == '__main__':
    unittest.main()
