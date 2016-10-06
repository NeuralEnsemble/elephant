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
import quantities as pq
from elephant import csd
import elephant.csd_methods.utility_functions as utils

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
        self.assertEqual(ele_pos.shape[0], len(lfp))

    def test_lfp2d_electrodes(self):
        ele_pos = utils.generate_electrodes(dim=2)
        xx_ele, yy_ele = ele_pos
        lfp = csd.generate_lfp(utils.large_source_2D, xx_ele, yy_ele)
        self.assertEqual(len(ele_pos), 2)
        self.assertEqual(xx_ele.shape[0], len(lfp))

    def test_lfp3d_electrodes(self):
        ele_pos = utils.generate_electrodes(dim=3, res=3)
        xx_ele, yy_ele, zz_ele = ele_pos
        lfp = csd.generate_lfp(utils.gauss_3d_dipole, xx_ele, yy_ele, zz_ele)
        self.assertEqual(len(ele_pos), 3)
        self.assertEqual(xx_ele.shape[0], len(lfp))


# class CSD1D_TestCase(unittest.TestCase):
#     def setUp(self):
#         self.ele_pos = utils.generate_electrodes(dim=1).reshape(5, 1)
#         self.lfp = csd.generate_lfp(utils.gauss_1d_dipole, self.ele_pos)
#         self.params = {}  # Input dictionaries for each method
#         self.params['DeltaiCSD'] = {'sigma_top': 0. * pq.S / pq.m,
#                                     'diam': 500E-6 * pq.m}
#         self.params['StepiCSD'] = {'sigma_top': 0. * pq.S / pq.m, 'tol': 1E-12,
#                                    'diam': 500E-6 * pq.m}
#         self.params['SplineiCSD'] = {'sigma_top': 0. * pq.S / pq.m,
#                                      'num_steps': 201, 'tol': 1E-12,
#                                      'diam': 500E-6 * pq.m}
#         self.params['StandardCSD'] = {}
#         self.params['KCSD1D'] = {'h': 50., 'Rs': np.array((0.1, 0.25, 0.5))}

#     def test_valid_method(self):
#         self.assert

#     def test_valid_inputs(self):
#         self.params[.update({})
#         self.assertRaises(ValueError,  self.test_kcsd3d_estimate)
#         self.test_method = 'KCSD3D'
#         self.test_params = {'src_type':22}
#         self.assertRaises(KeyError, self.test_kcsd3d_estimate)
#         self.test_params = {'InvalidKwarg':21}
#         self.assertRaises(TypeError, self.test_kcsd3d_estimate)
#         cv_params = {'InvalidCVArg':np.array((0.1,0.25,0.5))}
#         self.assertRaises(TypeError, self.test_kcsd3d_estimate, cv_params)

#     def test_method(self, method='StandardCSD'):
#         result = csd.estimate_csd(self.lfp, method=method, **self.params[method])
#         self.assertEqual(result.t_start, 0.0)
#         self.assertEqual(len(result.times), 1)


if __name__ == '__main__':
    unittest.main()
