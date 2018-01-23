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
import numpy as np
import quantities as pq
from elephant import current_source_density as csd
import elephant.current_source_density_src.utility_functions as utils

available_1d = ['StandardCSD', 'DeltaiCSD', 'StepiCSD', 'SplineiCSD', 'KCSD1D']
available_2d = ['KCSD2D', 'MoIKCSD']
available_3d = ['KCSD3D']
kernel_methods = ['KCSD1D', 'KCSD2D', 'KCSD3D', 'MoIKCSD']
icsd_methods = ['DeltaiCSD', 'StepiCSD', 'SplineiCSD']
py_iCSD_toolbox = ['StandardCSD', 'DeltaiCSD', 'StepiCSD', 'SplineiCSD']


class LFP_TestCase(unittest.TestCase):
    def test_lfp1d_electrodes(self):
        ele_pos = utils.generate_electrodes(dim=1).reshape(5, 1)
        lfp = csd.generate_lfp(utils.gauss_1d_dipole, ele_pos)
        self.assertEqual(ele_pos.shape[1], 1)
        self.assertEqual(ele_pos.shape[0], lfp.shape[1])

    def test_lfp2d_electrodes(self):
        ele_pos = utils.generate_electrodes(dim=2)
        xx_ele, yy_ele = ele_pos
        lfp = csd.generate_lfp(utils.large_source_2D, xx_ele, yy_ele)
        self.assertEqual(len(ele_pos), 2)
        self.assertEqual(xx_ele.shape[0], lfp.shape[1])

    def test_lfp3d_electrodes(self):
        ele_pos = utils.generate_electrodes(dim=3, res=3)
        xx_ele, yy_ele, zz_ele = ele_pos
        lfp = csd.generate_lfp(utils.gauss_3d_dipole, xx_ele, yy_ele, zz_ele)
        self.assertEqual(len(ele_pos), 3)
        self.assertEqual(xx_ele.shape[0], lfp.shape[1])


class CSD1D_TestCase(unittest.TestCase):
    def setUp(self):
        self.ele_pos = utils.generate_electrodes(dim=1).reshape(5, 1)
        self.lfp = csd.generate_lfp(utils.gauss_1d_dipole, self.ele_pos)
        self.csd_method = csd.estimate_csd

        self.params = {}  # Input dictionaries for each method
        self.params['DeltaiCSD'] = {'sigma_top': 0. * pq.S / pq.m,
                                    'diam': 500E-6 * pq.m}
        self.params['StepiCSD'] = {'sigma_top': 0. * pq.S / pq.m, 'tol': 1E-12,
                                   'diam': 500E-6 * pq.m}
        self.params['SplineiCSD'] = {'sigma_top': 0. * pq.S / pq.m,
                                     'num_steps': 201, 'tol': 1E-12,
                                     'diam': 500E-6 * pq.m}
        self.params['StandardCSD'] = {}
        self.params['KCSD1D'] = {'h': 50., 'Rs': np.array((0.1, 0.25, 0.5))}

    def test_validate_inputs(self):
        self.assertRaises(TypeError, self.csd_method, lfp=[[1], [2], [3]])
        self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                          coords=self.ele_pos * pq.mm)
        # inconsistent number of electrodes
        self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                          coords=[1, 2, 3, 4] * pq.mm, method='StandardCSD')
        # bad method name
        self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                          method='InvalidMethodName')
        self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                          method='KCSD2D')
        self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                          method='KCSD3D')

    def test_inputs_standardcsd(self):
        method = 'StandardCSD'
        result = self.csd_method(self.lfp, method=method)
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(result.shape[0], 1)

    def test_inputs_deltasplineicsd(self):
        methods = ['DeltaiCSD', 'SplineiCSD']
        for method in methods:
            self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                              method=method)
            result = self.csd_method(self.lfp, method=method,
                                     **self.params[method])
            self.assertEqual(result.t_start, 0.0 * pq.s)
            self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
            self.assertEqual(result.times.shape[0], 1)

    def test_inputs_stepicsd(self):
        method = 'StepiCSD'
        self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
                          method=method)
        self.assertRaises(AssertionError, self.csd_method, lfp=self.lfp,
                          method=method, **self.params[method])
        self.params['StepiCSD'].update({'h': np.ones(5) * 100E-6 * pq.m})
        result = self.csd_method(self.lfp, method=method,
                                 **self.params[method])
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(result.times.shape[0], 1)

    def test_inuts_kcsd(self):
        method = 'KCSD1D'
        result = self.csd_method(self.lfp, method=method,
                                 **self.params[method])
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(len(result.times), 1)


class CSD2D_TestCase(unittest.TestCase):
    def setUp(self):
        xx_ele, yy_ele = utils.generate_electrodes(dim=2)
        self.lfp = csd.generate_lfp(utils.large_source_2D, xx_ele, yy_ele)
        self.params = {}  # Input dictionaries for each method
        self.params['KCSD2D'] = {'sigma': 1., 'Rs': np.array((0.1, 0.25, 0.5))}

    def test_kcsd2d_init(self):
        method = 'KCSD2D'
        result = csd.estimate_csd(lfp=self.lfp, method=method,
                                  **self.params[method])
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(len(result.times), 1)


class CSD3D_TestCase(unittest.TestCase):
    def setUp(self):
        xx_ele, yy_ele, zz_ele = utils.generate_electrodes(dim=3)
        self.lfp = csd.generate_lfp(utils.gauss_3d_dipole,
                                    xx_ele, yy_ele, zz_ele)
        self.params = {}
        self.params['KCSD3D'] = {'gdx': 0.1, 'gdy': 0.1, 'gdz': 0.1,
                                 'src_type': 'step',
                                 'Rs': np.array((0.1, 0.25, 0.5))}

    def test_kcsd2d_init(self):
        method = 'KCSD3D'
        result = csd.estimate_csd(lfp=self.lfp, method=method,
                                  **self.params[method])
        self.assertEqual(result.t_start, 0.0 * pq.s)
        self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
        self.assertEqual(len(result.times), 1)


if __name__ == '__main__':
    unittest.main()
