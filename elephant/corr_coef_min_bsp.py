# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:56:27 2017

@author: papen
"""

# Nx, Ny, N, ii, and jj have to be calculated for each tau separately in order 
# to account for border effects, namely counting spikes that are not part of
# the cch calculation

from quantities import ms
from neo import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cch

times = [[51.,   229.],
         [24.,    65.,   101.,   138.,   155.,   283.]]

sts = [SpikeTrain(t, units='ms', t_stop=300*ms) for t in times]

bsts = [BinnedSpikeTrain(st, binsize=50*ms) for st in sts]

print cch(bsts[0], bsts[1], cross_corr_coef=True, border_correction=True)[0].T[0]

#>> [-1.22474487 -0.61237244 -0.61237244  0.          0.         -0.61237244
#  0.61237244 -0.61237244 -1.22474487 -0.61237244 -1.22474487] dimensionless