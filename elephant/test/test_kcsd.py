# -*- coding: utf-8 -*-
"""
Unit tests for the kCSD methods

:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import neo
import numpy as np
import quantities as pq
import elephant.csd as CSD

class KCSD1D_TestCase(unittest.TestCase):
    def setUp(self):
        self.ele_pos = CSD.generate_electrodes(dim=1).reshape(5,1)
        pots = CSD.FWD(CSD.gauss_1d_dipole, self.ele_pos) 
        self.pots = np.reshape(pots, (-1,1))
        self.test_method = 'KCSD1D'
        self.test_params = {'h':50.}
        self.an_sigs=[]
        for ii in range(len(self.pots)):
            rc = neo.RecordingChannel()
            rc.coordinate = self.ele_pos[ii]*pq.mm
            asig = neo.AnalogSignal(self.pots[ii]*pq.mV, sampling_rate=1000*pq.Hz)
            rc.analogsignals = [asig]
            rc.create_relationship()
            self.an_sigs.append(asig)
    
    def test_kcsd1d_estimate(self, cv_params={}):
        result = CSD.CSD(self.an_sigs, method=self.test_method, 
                         params=self.test_params, cv_params=cv_params)        
        self.assertEqual(result.t_start, 0.0*pq.s)
        self.assertEqual(result.sampling_rate, 1000*pq.Hz)
        self.assertEqual(result.times, [0.]*pq.s)
        self.assertEqual(len(result.annotations.keys()), 1)

    def test_valid_inputs(self):
        self.test_method = 'InvalidMethodName'
        self.assertRaises(ValueError,  self.test_kcsd1d_estimate)
        self.test_method = 'KCSD1D'
        self.test_params = {'src_type':22} 
        self.assertRaises(KeyError, self.test_kcsd1d_estimate)
        self.test_method = 'KCSD1D'
        self.test_params = {'InvalidKwarg':21} 
        self.assertRaises(TypeError, self.test_kcsd1d_estimate)
        cv_params = {'InvalidCVArg':np.array((0.1,0.25,0.5))} 
        self.assertRaises(TypeError, self.test_kcsd1d_estimate, cv_params)

class KCSD2D_TestCase(unittest.TestCase):
    def setUp(self):
        xx_ele, yy_ele = CSD.generate_electrodes(dim=2)
        self.ele_pos = np.vstack((xx_ele, yy_ele)).T
        pots = CSD.FWD(CSD.small_source_2D, xx_ele, yy_ele) 
        self.pots = np.reshape(pots, (-1,1))
        self.test_method = 'KCSD2D'
        self.test_params = {'gdx':0.2, 'gdy':0.2}
        self.an_sigs=[]
        for ii in range(len(self.pots)):
            rc = neo.RecordingChannel()
            rc.coordinate = self.ele_pos[ii]*pq.mm
            asig = neo.AnalogSignal(self.pots[ii]*pq.mV, sampling_rate=1000*pq.Hz)
            rc.analogsignals = [asig]
            rc.create_relationship()
            self.an_sigs.append(asig)
    
    def test_kcsd2d_estimate(self, cv_params={}):
        result = CSD.CSD(self.an_sigs, method=self.test_method, 
                         params=self.test_params, cv_params=cv_params)        
        self.assertEqual(result.t_start, 0.0*pq.s)
        self.assertEqual(result.sampling_rate, 1000*pq.Hz)
        self.assertEqual(result.times, [0.]*pq.s)
        self.assertEqual(len(result.annotations.keys()), 2)

    def test_moi_estimate(self):
        result = CSD.CSD(self.an_sigs, method='MoIKCSD', 
                         params={'MoI_iters':10, 'lambd':0.0, 
                                 'gdx':0.2, 'gdy':0.2})
        self.assertEqual(result.t_start, 0.0*pq.s)
        self.assertEqual(result.sampling_rate, 1000*pq.Hz)
        self.assertEqual(result.times, [0.]*pq.s)
        self.assertEqual(len(result.annotations.keys()), 2)

    def test_valid_inputs(self):
        self.test_method = 'InvalidMethodName'
        self.assertRaises(ValueError,  self.test_kcsd2d_estimate)
        self.test_method = 'KCSD2D'
        self.test_params = {'src_type':22} 
        self.assertRaises(KeyError, self.test_kcsd2d_estimate)
        self.test_params = {'InvalidKwarg':21} 
        self.assertRaises(TypeError, self.test_kcsd2d_estimate)
        cv_params = {'InvalidCVArg':np.array((0.1,0.25,0.5))} 
        self.assertRaises(TypeError, self.test_kcsd2d_estimate, cv_params)

class KCSD3D_TestCase(unittest.TestCase):
    def setUp(self):
        xx_ele, yy_ele, zz_ele = CSD.generate_electrodes(dim=3, res=3)
        self.ele_pos = np.vstack((xx_ele, yy_ele, zz_ele)).T
        pots = CSD.FWD(CSD.gauss_3d_dipole, xx_ele, yy_ele, zz_ele) 
        self.pots = np.reshape(pots, (-1,1))
        self.test_method = 'KCSD3D'
        self.test_params = {'gdx':0.3, 'gdy':0.3, 'gdz':0.3, 'src_type':'step'}
        self.an_sigs=[]
        for ii in range(len(self.pots)):
            rc = neo.RecordingChannel()
            rc.coordinate = self.ele_pos[ii]*pq.mm
            asig = neo.AnalogSignal(self.pots[ii]*pq.mV, sampling_rate=1000*pq.Hz)
            rc.analogsignals = [asig]
            rc.create_relationship()
            self.an_sigs.append(asig)
    
    def test_kcsd3d_estimate(self, cv_params={}):
        result = CSD.CSD(self.an_sigs, method=self.test_method, 
                         params=self.test_params, cv_params=cv_params)        
        self.assertEqual(result.t_start, 0.0*pq.s)
        self.assertEqual(result.sampling_rate, 1000*pq.Hz)
        self.assertEqual(result.times, [0.]*pq.s)
        self.assertEqual(len(result.annotations.keys()), 3)

    def test_valid_inputs(self):
        self.test_method = 'InvalidMethodName'
        self.assertRaises(ValueError,  self.test_kcsd3d_estimate)
        self.test_method = 'KCSD3D'
        self.test_params = {'src_type':22} 
        self.assertRaises(KeyError, self.test_kcsd3d_estimate)
        self.test_params = {'InvalidKwarg':21} 
        self.assertRaises(TypeError, self.test_kcsd3d_estimate)
        cv_params = {'InvalidCVArg':np.array((0.1,0.25,0.5))} 
        self.assertRaises(TypeError, self.test_kcsd3d_estimate, cv_params)




if __name__ == '__main__':
    unittest.main()
