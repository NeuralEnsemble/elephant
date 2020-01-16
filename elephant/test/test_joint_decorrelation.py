# -*- coding: utf-8 -*-
"""
Unit tests for the JD analysis.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import unittest

import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from numpy.testing import assert_array_equal, assert_array_almost_equal
import inspect

from elephant.joint_decorrelation import JD
from elephant.signal_processing import butter
from elephant.spectral import welch_psd

python_version_major = sys.version_info.major

class JDTestCase(unittest.TestCase):
    def setUp(self):
        self.n_channels = 3
        self.n_samples = 1000
        self.sampling_rate = 1000*pq.Hz
        self.freq_noise = 50*pq.Hz
        self.time = np.arange(self.n_samples) / self.sampling_rate
        np.random.seed(27)
        self.data = np.random.normal(size=(self.n_channels, self.n_samples))
        self.data += np.sin(2*np.pi*(self.freq_noise*self.time).magnitude)

        self.filtered_data = butter(self.data, self.freq_noise - 1*pq.Hz,
                                    self.freq_noise + 1*pq.Hz,
                                    fs=self.sampling_rate)

        self.data_covmat = np.cov(self.data)
        self.filtered_data_covmat = np.cov(self.filtered_data)
        self.covmat_asymmetric = np.random.random(size=(self.n_channels, self.n_channels))

        self.expected_power_ratio = [0.2912283695676657, 0.001297459126086244, 4.284646320734217e-05]
        self.expected_power_transform = [0.3838421077931539, 0.0037819100713350285, 7.062686318122588e-05]
        self.expected_power_after_cleaning = [0.0025457376782979093, 4.800646996067096e-05, 0.0012307234552476575]
        self.expected_power_line_noise = [0.2977021880595121, 0.3149944535009742, 0.36930326882490905]

    def test_fit(self):
        jd = JD()
        jd.fit(baseline_data=self.data, bias_filtered_data=self.filtered_data)
        self.assertTrue(np.all(jd.power_ratio == self.expected_power_ratio))

        jd_cov = JD()
        jd_cov.fit(baseline_covmat=self.data_covmat,
                   bias_filtered_covmat=self.filtered_data_covmat)

        self.assertTrue(np.all(jd.todss == jd_cov.todss))
        self.assertTrue(np.all(jd.fromdss == jd_cov.fromdss))
        self.assertTrue(np.all(jd.pwr0 == jd_cov.pwr0))
        self.assertTrue(np.all(jd.pwr1 == jd_cov.pwr1))

        # data given, covmats needs to be calculated
        # check data shape compatibility
        with self.assertRaises(ValueError):
            jd.fit(baseline_data=self.data,
                   bias_filtered_data=self.filtered_data[1:])

        # covmats given
        # check baseline data is symmetric
        with self.assertRaises(ValueError):
            jd_cov.fit(baseline_covmat=self.covmat_asymmetric,
                       bias_filtered_covmat=self.filtered_data_covmat)
        # check filtered data is symmetric
        with self.assertRaises(ValueError):
            jd_cov.fit(baseline_covmat=self.data_covmat,
                       bias_filtered_covmat=self.covmat_asymmetric)

        # check data shape compatibility
        with self.assertRaises(ValueError):
            jd_cov.fit(baseline_covmat=self.data_covmat,
                       bias_filtered_covmat=self.filtered_data_covmat[1:, 1:])
        with self.assertRaises(ValueError):
            jd_cov.fit(baseline_covmat=self.data_covmat[1:, 1:],
                       bias_filtered_covmat=self.filtered_data_covmat)


    def test_transform(self):
        jd = JD()
        jd.fit(self.data, self.filtered_data)
        result = jd.transform(self.data)

        with self.assertRaises(ValueError):
            jd.transform(self.data[1:])


        calculated_power = []
        for i, dat in enumerate(result):
            freqs, psd = welch_psd(dat, fs=self.sampling_rate, freq_res=1.0)
            idx_line_noise = np.where(freqs == self.freq_noise.magnitude)[0][0]
            calculated_power.append(psd[idx_line_noise])
        self.assertTrue(np.all(calculated_power == self.expected_power_transform))

    def test_fit_transform(self):
        jd = JD()
        jd.fit(baseline_data=self.data, bias_filtered_data=self.filtered_data)
        result = jd.transform(self.data)
        result_fit_transform = jd.fit_transform(baseline_data=self.data,
                                                bias_filtered_data=self.filtered_data)
        self.assertTrue(np.all(result == result_fit_transform))

    def test_project_out(self):
        jd = JD()
        jd.fit(baseline_data=self.data,
               bias_filtered_data=self.filtered_data)

        power_ratio_threshold = 0.01
        clean_data = jd.project_out(self.data, power_ratio_threshold=power_ratio_threshold)
        self.assertEqual(clean_data.shape, self.data.shape)

        calculated_power = []
        for i, dat in enumerate(clean_data):
            freqs, psd = welch_psd(dat, fs=self.sampling_rate, freq_res=1.0)
            idx_line_noise = np.where(freqs == self.freq_noise.magnitude)[0][0]
            calculated_power.append(psd[idx_line_noise])
        self.assertTrue(np.all(calculated_power == self.expected_power_after_cleaning))


        with self.assertRaises(ValueError):
            jd.project_out(self.data[1:])
        with self.assertRaises(ValueError):
            jd.project_out(self.data, components_to_discard=0)
        with self.assertRaises(ValueError):
            jd.project_out(self.data, components_to_discard=list(range(jd.n_components+1)))
        with self.assertRaises(ValueError):
            jd.project_out(self.data, components_to_discard=[jd.n_components+1])

        with self.assertRaises(ValueError):
            jd.project_out(self.data, components_to_discard=[0,1],
                           power_ratio_threshold=0.1)

    def test_project_in(self):
        jd = JD()
        jd.fit(baseline_data=self.data,
               bias_filtered_data=self.filtered_data)

        power_ratio_threshold = 0.01
        line_noise = jd.project_in(self.data, power_ratio_threshold=power_ratio_threshold)
        self.assertEqual(line_noise.shape, self.data.shape)

        calculated_power = []
        for i, dat in enumerate(line_noise):
            freqs, psd = welch_psd(dat, fs=self.sampling_rate, freq_res=1.0)
            idx_line_noise = np.where(freqs == self.freq_noise.magnitude)[0][0]
            calculated_power.append(psd[idx_line_noise])
        self.assertTrue(np.all(calculated_power == self.expected_power_line_noise))

        with self.assertRaises(ValueError):
            jd.project_in(self.data[1:])
        with self.assertRaises(ValueError):
            jd.project_in(self.data, components_to_keep=0)
        with self.assertRaises(ValueError):
            jd.project_in(self.data, components_to_keep=list(range(jd.n_components+1)))
        with self.assertRaises(ValueError):
            jd.project_in(self.data, components_to_keep=[jd.n_components+1])

        with self.assertRaises(ValueError):
            jd.project_in(self.data, components_to_keep=[0,1],
                           power_ratio_threshold=0.1)


def suite():
    suite = unittest.makeSuite(JDTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    unittest.main()
