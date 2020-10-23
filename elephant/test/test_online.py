import unittest

import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal

from elephant.online import MeanOnline, VarianceOnline
from elephant.statistics import mean_firing_rate, cv, isi


class TestMeanOnline(unittest.TestCase):
    def test_floats(self):
        np.random.seed(0)
        arr = np.random.rand(100)
        online = MeanOnline()
        for val in arr:
            online.update(val)
        self.assertIsNone(online.units)
        self.assertIsInstance(online.get_mean(), float)
        self.assertAlmostEqual(online.get_mean(), arr.mean())

    def test_numpy_array(self):
        np.random.seed(1)
        arr = np.random.rand(10, 100)
        online = MeanOnline()
        for arr_vec in arr:
            online.update(arr_vec)
        self.assertIsInstance(online.get_mean(), np.ndarray)
        self.assertIsNone(online.units)
        self.assertEqual(online.get_mean().shape, (arr.shape[1],))
        assert_array_almost_equal(online.get_mean(), arr.mean(axis=0))

    def test_quantity_scalar(self):
        np.random.seed(2)
        arr = np.random.rand(100) * pq.Hz
        online = MeanOnline()
        for val in arr:
            online.update(val)
        self.assertEqual(online.units, arr.units)
        self.assertAlmostEqual(online.get_mean(), arr.mean())

    def test_quantities_vector(self):
        np.random.seed(3)
        arr = np.random.rand(10, 100) * pq.ms
        online = MeanOnline()
        for arr_vec in arr:
            online.update(arr_vec)
        self.assertEqual(online.units, arr.units)
        self.assertEqual(online.get_mean().shape, (arr.shape[1],))
        assert_array_almost_equal(online.get_mean(), arr.mean(axis=0))

    def test_reset(self):
        target_value = 2.5
        online = MeanOnline()
        online.update(target_value)
        self.assertEqual(online.get_mean(), target_value)
        online.reset()
        self.assertIsNone(online.mean)
        self.assertIsNone(online.units)
        self.assertEqual(online.count, 0)

    def test_mean_firing_rate(self):
        np.random.seed(4)
        spiketrain = np.random.rand(10000).cumsum()
        rate_target = mean_firing_rate(spiketrain)
        online = MeanOnline()
        n_batches = 10
        t_start = None
        for st_window in np.array_split(spiketrain, n_batches):
            rate_batch = mean_firing_rate(st_window, t_start=t_start)
            online.update(rate_batch)
            t_start = st_window[-1]
        self.assertAlmostEqual(online.get_mean(), rate_target, places=3)


class TestVarianceOnline(unittest.TestCase):
    def test_floats(self):
        np.random.seed(0)
        arr = np.random.rand(100)
        online = VarianceOnline()
        for val in arr:
            online.update(val)
        self.assertIsNone(online.units)
        self.assertIsInstance(online.get_mean(), float)
        self.assertAlmostEqual(online.get_mean(), arr.mean())
        for unbiased in (False, True):
            mean, std = online.get_mean_std(unbiased=unbiased)
            self.assertAlmostEqual(mean, arr.mean())
            self.assertAlmostEqual(std, arr.std(ddof=unbiased))

    def test_numpy_array(self):
        np.random.seed(1)
        arr = np.random.rand(10, 100)
        online = VarianceOnline()
        for arr_vec in arr:
            online.update(arr_vec)
        self.assertIsNone(online.units)
        self.assertIsInstance(online.get_mean(), np.ndarray)
        self.assertEqual(online.get_mean().shape, (arr.shape[1],))
        assert_array_almost_equal(online.get_mean(), arr.mean(axis=0))
        for unbiased in (False, True):
            mean, std = online.get_mean_std(unbiased=unbiased)
            assert_array_almost_equal(mean, arr.mean(axis=0))
            assert_array_almost_equal(std, arr.std(axis=0, ddof=unbiased))

    def test_quantity_scalar(self):
        np.random.seed(2)
        arr = np.random.rand(100) * pq.Hz
        online = VarianceOnline()
        for val in arr:
            online.update(val)
        self.assertEqual(online.units, arr.units)
        self.assertAlmostEqual(online.get_mean(), arr.mean())
        for unbiased in (False, True):
            mean, std = online.get_mean_std(unbiased=unbiased)
            self.assertAlmostEqual(mean, arr.mean())
            self.assertAlmostEqual(std, arr.std(ddof=unbiased))

    def test_quantities_vector(self):
        np.random.seed(3)
        arr = np.random.rand(10, 100) * pq.ms
        online = VarianceOnline()
        for arr_vec in arr:
            online.update(arr_vec)
        self.assertEqual(online.units, arr.units)
        self.assertEqual(online.get_mean().shape, (arr.shape[1],))
        assert_array_almost_equal(online.get_mean(), arr.mean(axis=0))
        for unbiased in (False, True):
            mean, std = online.get_mean_std(unbiased=unbiased)
            assert_array_almost_equal(mean, arr.mean(axis=0))
            assert_array_almost_equal(std, arr.std(axis=0, ddof=unbiased))

    def test_reset(self):
        target_value = 2.5
        online = VarianceOnline()
        online.update(target_value)
        self.assertEqual(online.get_mean(), target_value)
        online.reset()
        self.assertIsNone(online.mean)
        self.assertIsNone(online.units)
        self.assertEqual(online.count, 0)
        self.assertEqual(online.variance_sum, 0.)

    def test_cv(self):
        np.random.seed(4)
        spiketrain = np.random.rand(10000).cumsum()
        isi_all = isi(spiketrain)
        cv_target = cv(isi_all)
        online = VarianceOnline(batch_mode=True)
        n_batches = 10
        for st_window in np.array_split(spiketrain, n_batches):
            isi_batch = isi(st_window)
            online.update(isi_batch)
        isi_mean, isi_std = online.get_mean_std(unbiased=False)
        cv_online = isi_std / isi_mean
        self.assertAlmostEqual(cv_online, cv_target, places=3)


if __name__ == '__main__':
    unittest.main()
