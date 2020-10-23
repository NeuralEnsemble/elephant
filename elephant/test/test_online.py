import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal

import quantities as pq
from elephant.online import MeanOnline, VarianceOnline


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
        online = MeanOnline(val=target_value)
        self.assertEqual(online.get_mean(), target_value)
        online.reset()
        self.assertIsNone(online.mean)
        self.assertIsNone(online.units)
        self.assertEqual(online.count, 0)


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
        online = VarianceOnline(val=target_value)
        self.assertEqual(online.get_mean(), target_value)
        online.reset()
        self.assertIsNone(online.mean)
        self.assertIsNone(online.units)
        self.assertEqual(online.count, 0)
        self.assertEqual(online.variance_sum, 0.)


if __name__ == '__main__':
    unittest.main()
