# -*- coding: utf-8 -*-
"""
Unit tests for the kCSD methods

:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import neo
import numpy as np
import quantities as pq
from elephant.current_source_density import CSD

class kcsd1d_TestCase(unittest.TestCase):
    def setUp(self):
        ele_pos = CSD.generate_electrodes(dim=1).reshape(5,1)
        pots = CSD.FWD(gauss_1d_dipole, ele_pos) 
        pots = np.reshape(pots, (-1,1))
        test_method = 'KCSD1D'
        test_params = {'h':50.}

        an_sigs=[]
        for ii in range(len(pots)):
            rc = neo.RecordingChannel()
            rc.coordinate = ele_pos[ii]*pq.mm
            asig = neo.AnalogSignal(pots[ii]*pq.mV,sampling_rate=1000*pq.Hz)
            rc.analogsignals = [asig]
            rc.create_relationship()
            an_sigs.append(asig)
   
        result = CSD.CSD(an_sigs, method=test_method, params=test_params, cv_params={'Rs':np.array((0.1,0.25,0.5))})

if __name__ == '__main__':
    unittest.main()
