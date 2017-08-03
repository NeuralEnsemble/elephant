# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:23:11 2017

@author: emanuele
"""

import unittest

import neo
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal
import quantities as pq
import elephant.multiple_filter_test as mft

  

class Gth_TestCase(unittest.TestCase):
   def setUp(self):       
       self.test_array = [0.4, 0.5, 0.65, 0.7,    0.9, 1.15, 1.2, 1.9]
       '''
       spks_ri = [0.9, 1.15, 1.2]
       spk_le = [0.4, 0.5, 0.65, 0.7]
       '''
       mu_ri = (0.25 + 0.05)/2
       mu_le = (0.1 + 0.15 + 0.05)/3
       sigma_ri = ((0.25 - 0.15)**2 + (0.05 - 0.15)**2)/2
       sigma_le = ((0.1 - 0.1)**2 + (0.15 - 0.1)**2 + (0.05 - 0.1)**2)/3
       self.targ_t08_h025 = 0
       self.targ_t08_h05 = (3 - 4)/ np.sqrt((sigma_ri/mu_ri**(3))*0.5 + (sigma_le/mu_le**(3))*0.5)
       
       ### Window Large ####
   def test_Gth_with_spiketrain_h05(self):
       st = neo.SpikeTrain(self.test_array, units='s', t_stop = 2.0)
       target = self.targ_t08_h05
       res = mft.Gth(0.8 *pq.s, 0.5*pq.s, st)
       assert_array_almost_equal(res, target, decimal=9)
           
   def test_isi_with_quantities_h05(self):
       st = pq.Quantity(self.test_array, units='s')
       target = self.targ_t08_h05
       res = mft.Gth(0.8 *pq.s, 0.5*pq.s, st)
       assert_array_almost_equal(res, target, decimal=9)
   
   def test_isi_with_plain_array_h05(self):
       st = self.test_array
       target = self.targ_t08_h05
       res = mft.Gth(0.8 *pq.s, 0.5*pq.s, st*pq.s)
       assert not isinstance(res, pq.Quantity)
       assert_array_almost_equal(res, target, decimal=9)
   
   ### Window Small ####
   def test_Gth_with_spiketrain_h025(self):
       st = neo.SpikeTrain(self.test_array, units='s', t_stop = 2.0)
       target = self.targ_t08_h025
       res = mft.Gth(0.8 *pq.s, 0.25*pq.s, st)
       assert_array_almost_equal(res, target, decimal=9)
       
   def test_Gth_with_quantities_h025(self):
       st = pq.Quantity(self.test_array, units='s')
       target = self.targ_t08_h025
       res = mft.Gth(0.8 *pq.s, 0.25*pq.s, st)
       assert_array_almost_equal(res, target, decimal=9)
   
   def test_Gth_with_plain_array_h025(self):
       st = self.test_array
       target = self.targ_t08_h025
       res = mft.Gth(0.8 *pq.s, 0.25*pq.s, st*pq.s)
       assert_array_almost_equal(res, target, decimal=9)
    
class Rth_TestCase(unittest.TestCase):
   def setUp(self):       
       self.test_array = [1.1,1.2,1.4,  1.6,1.7,1.75,1.8,1.85,1.9,1.95]
       x = (7 - 3)/np.sqrt( (0.0025/0.15**3)*0.5+ (0.0003472/0.05833**3)*0.5 )
       self.targ_h05 = [[0.5, 1, 1.5], [(0-1.7)/np.sqrt(0.4),(0-1.7)/np.sqrt(0.4),(x-1.7)/np.sqrt(0.4)]]

   def test_Rth_with_spiketrain_h05(self):
       st = neo.SpikeTrain(self.test_array, units='s', t_stop = 2.1)
       target = self.targ_h05
       res = mft.Rth(0.5*pq.s, 0.5*pq.s, st, 2.01*pq.s,np.array([[0.5],[1.7],[0.4]]))
       assert_array_almost_equal(res[1], target[1], decimal=3)
           
   def test_Rth_with_quantities_h05(self):
       st = pq.Quantity(self.test_array, units='s')
       target = self.targ_h05
       res = mft.Rth(0.5*pq.s, 0.5*pq.s, st, 2.01*pq.s,np.array([[0.5],[1.7],[0.4]]))
       assert_array_almost_equal(res[0], target[0], decimal=3)
           
   def test_Rth_with_plain_array_h05(self):
       st = self.test_array
       target = self.targ_h05
       res = mft.Rth(0.5*pq.s, 0.5*pq.s, st*pq.s,2.01*pq.s,np.array([[0.5],[1.7],[0.4]]))
       assert not isinstance(res, pq.Quantity)
       assert_array_almost_equal(res, target, decimal=3) 
       
           
class MultipleFilterAlgorithm_TestCase(unittest.TestCase):
   def setUp(self):       
       self.test_array = [1.1,1.2,1.4,  1.6,1.7,1.75,1.8,1.85,1.9,1.95]
       self.targ_h05_dt05 = [1.5*pq.s]
       
   def test_MultipleFilterAlgorithm_with_spiketrain_h05(self):
       st = neo.SpikeTrain(self.test_array, units='s', t_stop = 2.1)
       target = [self.targ_h05_dt05]
       res = mft.MultipleFilterAlgorithm([0.5]*pq.s, st, 2.1*pq.s, 5, 10000, dt = 0.5*pq.s)
       assert_array_almost_equal(res, target, decimal=9)
           
   def test_MultipleFilterAlgorithm_with_quantities_h05(self):
       st = pq.Quantity(self.test_array, units='s')
       target = [self.targ_h05_dt05]
       res = mft.MultipleFilterAlgorithm([0.5]*pq.s, st, 2.1*pq.s, 5, 10000, dt = 0.5*pq.s)
       assert_array_almost_equal(res, target, decimal=9)
           
   def test_MultipleFilterAlgorithm_with_plain_array_h05(self):
       st = self.test_array
       target = [self.targ_h05_dt05]
       res = mft.MultipleFilterAlgorithm([0.5]*pq.s, st*pq.s, 2.1*pq.s, 5, 10000, dt = 0.5*pq.s)
       assert not isinstance(res, pq.Quantity)
       assert_array_almost_equal(res, target, decimal=9)

def suite():
    suite = unittest.makeSuite(Rth_TestCase, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

'''     
if __name__ == '__main__':
    unittest.main()
   
suite = unittest.TestLoader().loadTestsFromTestCase(Rth_TestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
'''