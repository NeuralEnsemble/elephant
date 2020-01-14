import sys
import unittest

import neo
import numpy as np
import quantities as pq

import spike_train_generation as gen
import spade


class PVTestCase(unittest.TestCase):

    def test_pvalue_spec_2d(self):
        rate = 40 * pq.Hz
        refr_period = 4 * pq.ms
        t_start = 0. * pq.ms
        t_stop = 1000. * pq.ms
        num_spiketrains = 20

        binsize = 3 * pq.ms
        winlen = 5
        dither = 10 * pq.ms
        n_surr = 10
        min_spikes = 2
        min_occ = 2
        max_spikes = 10
        max_occ = None
        min_neu = 2
        alpha = 0.05
        spectrum = '#'
        playing_it_safe = False

        np.random.seed(0)
        hpr = gen.homogeneous_poisson_process_with_refr_period
        sts = [hpr(rate, refr_period, t_start, t_stop)
               for ind in range(num_spiketrains)]

        np.random.seed(0)
        spade_results = spade.spade(
            sts, binsize, winlen, min_spikes=min_spikes, min_occ=min_occ,
            max_spikes=max_spikes, max_occ=max_occ, min_neu=min_neu,
            n_surr=n_surr, dither=dither, spectrum=spectrum,
            alpha=alpha, stat_corr='fdr_bh', psr_param=None,
            output_format='concepts')
        print(spade_results)

    def test_pvalue_spec_3d(self):
        rate = 40 * pq.Hz
        refr_period = 4 * pq.ms
        t_start = 0. * pq.ms
        t_stop = 1000. * pq.ms
        num_spiketrains = 20

        binsize = 3 * pq.ms
        winlen = 5
        dither = 10 * pq.ms
        n_surr = 10
        min_spikes = 2
        min_occ = 2
        max_spikes = 10
        max_occ = None
        min_neu = 2
        alpha = 0.05
        spectrum = '3d#'

        np.random.seed(0)
        hpr = gen.homogeneous_poisson_process_with_refr_period
        sts = [hpr(rate, refr_period, t_start, t_stop)
               for ind in range(num_spiketrains)]

        np.random.seed(0)

        spade_results = spade.spade(
            sts, binsize, winlen, min_spikes=min_spikes, min_occ=min_occ,
            max_spikes=max_spikes, max_occ=max_occ, min_neu=min_neu,
            n_surr=n_surr, dither=dither, spectrum=spectrum,
            alpha=alpha, stat_corr='fdr_bh', psr_param=None,
            output_format='concepts')
        print(spade_results)


def suite():
    suite = unittest.makeSuite(PVTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())