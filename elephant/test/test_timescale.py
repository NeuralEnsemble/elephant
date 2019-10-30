# -*- coding: utf-8 -*-
"""
Unit tests for the timescale calculation.

:copyright: Copyright 2014-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import numpy as np
import quantities as pq

from elephant.spike_train_generation import homogeneous_gamma_process
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import spike_train_timescale


def test_timescale_calculation():
    '''
    Test the timescale generation using an alpha-shaped ISI distribution,
    see [1, eq. 1.68]. This is equivalent to a homogeneous gamma process
    with alpha=2 and beta=2*nu where nu is the rate.

    For this process, the autocorrelation function is given by a sum of a
    delta peak and a (negative) exponential, see [1, eq. 1.69].
    The exponential decays with \tau_corr = 1 / (4*nu), thus this fixes
    timescale.

    [1] Lindner, B. (2009). A brief introduction to some simple stochastic
        processes. Stochastic Methods in Neuroscience, 1.
    '''
    nu = 25/pq.s
    T = 1000*pq.s
    binsize = 1*pq.ms
    timescale = 1 / (4*nu)
    timescale.units = pq.ms

    timescale_num = []
    for _ in range(50):
        spikes = homogeneous_gamma_process(2, 2*nu, 0*pq.ms, T)
        spikes_bin = BinnedSpikeTrain(spikes, binsize)
        timescale_i = spike_train_timescale(spikes_bin, 10*timescale)
        timescale_i.units = timescale.units
        timescale_num.append(timescale_i.magnitude)
    correct = np.isclose(timescale.magnitude, timescale_num, rtol=1e-1)
    assert correct.mean() > 0.9


if __name__ == '__main__':
    test_timescale_calculation()
