import unittest

import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal

from elephant import statistics

from elephant.online import MeanOnline, VarianceOnline, CovarianceOnline, \
    PearsonCorrelationCoefficientOnline, InterSpikeIntervalOnline

from elephant.spike_train_generation import StationaryPoissonProcess
from elephant.spike_train_synchrony import spike_contrast


class TestSlidingWindowGeneric(unittest.TestCase):
    def test_fanofactor(self):
        """
        This test computes the Fano factor in a sliding window fashion
        without the use of MeanOnline or VarianceOnline.
        """
        np.random.seed(0)
        t_stop = 10 * pq.s
        spiketrains = [StationaryPoissonProcess(
            rate=20 * pq.Hz, t_stop=t_stop).generate_spiketrain()
                       for _ in range(10)]
        fanofactor_target = statistics.fanofactor(spiketrains)
        spike_counts = np.zeros(len(spiketrains))
        checkpoints = np.linspace(0 * pq.s, t_stop, num=10)
        for t_start, t_stop in zip(checkpoints[:-1], checkpoints[1:]):
            sts_chunk = [st.time_slice(t_start=t_start, t_stop=t_stop)
                         for st in spiketrains]
            spike_counts += list(map(len, sts_chunk))
        fanofactor = spike_counts.var() / spike_counts.mean()
        self.assertAlmostEqual(fanofactor, fanofactor_target)


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
        rate_target = statistics.mean_firing_rate(spiketrain)
        online = MeanOnline()
        n_batches = 10
        t_start = None
        for spiketrain_chunk in np.array_split(spiketrain, n_batches):
            rate_batch = statistics.mean_firing_rate(spiketrain_chunk,
                                                     t_start=t_start)
            online.update(rate_batch)
            t_start = spiketrain_chunk[-1]
        self.assertAlmostEqual(online.get_mean(), rate_target, places=3)

    def test_cv2(self):
        np.random.seed(5)
        spiketrain = np.random.rand(10000).cumsum()
        cv2_target = statistics.cv2(statistics.isi(spiketrain))
        online = MeanOnline()
        n_batches = 10
        for spiketrain_chunk in np.array_split(spiketrain, n_batches):
            isi_batch = statistics.isi(spiketrain_chunk)
            cv2_batch = statistics.cv2(isi_batch)
            online.update(cv2_batch)
        self.assertAlmostEqual(online.get_mean(), cv2_target, places=3)

    def test_lv(self):
        np.random.seed(6)
        spiketrain = np.random.rand(10000).cumsum()
        lv_target = statistics.lv(statistics.isi(spiketrain))
        online = MeanOnline()
        n_batches = 10
        for spiketrain_chunk in np.array_split(spiketrain, n_batches):
            isi_batch = statistics.isi(spiketrain_chunk)
            lv_batch = statistics.lv(isi_batch)
            online.update(lv_batch)
        self.assertAlmostEqual(online.get_mean(), lv_target, places=3)

    def test_lvr(self):
        np.random.seed(6)
        spiketrain = np.random.rand(10000).cumsum()
        lvr_target = statistics.lvr(statistics.isi(spiketrain))
        online = MeanOnline()
        n_batches = 10
        for spiketrain_chunk in np.array_split(spiketrain, n_batches):
            isi_batch = statistics.isi(spiketrain_chunk)
            lvr_batch = statistics.lvr(isi_batch)
            online.update(lvr_batch)
        self.assertAlmostEqual(online.get_mean(), lvr_target, places=1)

    def test_spike_contrast(self):
        np.random.seed(1)
        t_stop = 100 * pq.s
        spiketrains = [StationaryPoissonProcess(
            rate=20 * pq.Hz, t_stop=t_stop).generate_spiketrain()
                       for _ in range(10)]
        synchrony_target = spike_contrast(spiketrains)
        checkpoints = np.linspace(0 * pq.s, t_stop, num=10)
        online = MeanOnline()
        for t_start, t_stop in zip(checkpoints[:-1], checkpoints[1:]):
            synchrony_batch = spike_contrast(spiketrains,
                                             t_start=t_start,
                                             t_stop=t_stop)
            online.update(synchrony_batch)
        synchrony = online.get_mean()
        self.assertAlmostEqual(synchrony, synchrony_target, places=1)


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
        isi_all = statistics.isi(spiketrain)
        cv_target = statistics.cv(isi_all)
        online = VarianceOnline(batch_mode=True)
        n_batches = 10
        for spiketrain_chunk in np.array_split(spiketrain, n_batches):
            isi_batch = statistics.isi(spiketrain_chunk)
            online.update(isi_batch)
        isi_mean, isi_std = online.get_mean_std(unbiased=False)
        cv_online = isi_std / isi_mean
        self.assertAlmostEqual(cv_online, cv_target, places=3)


class TestInterSpikeIntervalOnline(unittest.TestCase):
    def test_single_floats(self):
        np.random.seed(0)
        arr = np.sort(np.random.rand(100))
        online_isi = InterSpikeIntervalOnline()
        for val in arr:
            online_isi.update(val)
        self.assertIsNone(online_isi.units)
        standard_isi_histo, _ = np.histogram(statistics.isi(arr),
                                             bins=online_isi.bin_edges)
        np.testing.assert_allclose(online_isi.get_isi(), standard_isi_histo,
                                   rtol=1e-15, atol=1e-15)

    def test_numpy_array(self):
        np.random.seed(1)
        arr = np.sort(np.random.rand(10 * 100)).reshape(10, 100)
        online_isi = InterSpikeIntervalOnline(batch_mode=True)
        for arr_vec in arr:
            online_isi.update(arr_vec)
        self.assertIsNone(online_isi.units)
        standard_isi_histo, _ = np.histogram(
            statistics.isi(arr.reshape(1, 10*100)), bins=online_isi.bin_edges)
        np.testing.assert_allclose(online_isi.get_isi(), standard_isi_histo,
                                   rtol=1e-15, atol=1e-15)

    def test_quantity_scalar(self):
        np.random.seed(2)
        arr = np.sort(np.random.rand(100)) * pq.s
        online_isi = InterSpikeIntervalOnline()
        for val in arr:
            online_isi.update(val)
        self.assertEqual(online_isi.units, arr.units)
        standard_isi_histo, _ = np.histogram(statistics.isi(arr),
                                             bins=online_isi.bin_edges)
        np.testing.assert_allclose(online_isi.get_isi().magnitude,
                                   standard_isi_histo, rtol=1e-15, atol=1e-15)

    def test_quantities_vector(self):
        np.random.seed(3)
        arr = np.sort(np.random.rand(10 * 100)).reshape(10, 100) * pq.s
        online_isi = InterSpikeIntervalOnline(batch_mode=True)
        for arr_vec in arr:
            online_isi.update(arr_vec)
        self.assertEqual(online_isi.units, arr.units)
        standard_isi_histo, _ = np.histogram(
            statistics.isi(arr.reshape(1, 10 * 100)),
            bins=online_isi.bin_edges)
        np.testing.assert_allclose(online_isi.get_isi().magnitude,
                                   standard_isi_histo, rtol=1e-15, atol=1e-15)

    def test_reset(self):
        np.random.seed(4)
        arr = np.sort(np.random.rand(100)) * pq.s
        online_isi = InterSpikeIntervalOnline()
        for val in arr:
            online_isi.update(val)
        self.assertEqual(online_isi.units, arr.units)
        standard_isi_histo, _ = np.histogram(statistics.isi(arr),
                                             bins=online_isi.bin_edges)
        np.testing.assert_allclose(online_isi.get_isi().magnitude,
                                   standard_isi_histo, rtol=1e-15, atol=1e-15)
        online_isi.reset()
        self.assertIsNone(online_isi.units)
        self.assertIsNone(online_isi.last_spike_time)
        np.testing.assert_allclose(online_isi.current_isi_histogram,
                                   np.zeros(shape=online_isi.num_bins))


class TestCovarianceOnline(unittest.TestCase):
    def test_simple_small_sets_XY_unbatched(self):
        X = np.array([1, 2, 3, 2, 3])
        Y = np.array([4, 4, 3, 3, 4])
        online_cov = CovarianceOnline(batch_mode=False)
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i])
        self.assertIsNone(online_cov.units)
        for unbiased in (False, True):  # -0.12 (biased) / -0.15 (unbiased)
            self.assertAlmostEqual(online_cov.get_cov(unbiased=unbiased),
                                   np.cov([X, Y], bias=not unbiased)[0][1])

    def test_simple_small_sets_XY_batched(self):
        X = np.array([[1, 2, 3, 2, 3], [1, 2, 3, 2, 3], [1, 2, 3, 2, 3]])
        Y = np.array([[4, 4, 3, 3, 4], [4, 4, 3, 3, 4], [4, 4, 3, 3, 4]])
        online_cov = CovarianceOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i])
        self.assertIsNone(online_cov.units)
        for unbiased in (False, True):  # -0.12 (biased) / -0.15 (unbiased)
            self.assertAlmostEqual(online_cov.get_cov(unbiased=unbiased),
                                   np.cov([X.reshape(15), Y.reshape(15)],
                                          bias=not unbiased)[0][1])

    def test_floats(self):
        np.random.seed(0)
        X = np.random.rand(100)
        Y = np.random.rand(100)
        online_cov = CovarianceOnline()
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i])
        self.assertIsNone(online_cov.units)
        self.assertIsInstance(online_cov.get_cov(), float)
        for unbiased in (False, True):
            self.assertAlmostEqual(online_cov.get_cov(unbiased=unbiased),
                                   np.cov([X, Y], bias=not unbiased)[0][1])

    def test_numpy_array(self):
        np.random.seed(1)
        X = np.random.rand(10, 100)
        Y = np.random.rand(10, 100)
        online_cov = CovarianceOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i])
        self.assertIsNone(online_cov.units)
        self.assertIsInstance(online_cov.get_cov(), float)
        for unbiased in (False, True):
            self.assertAlmostEqual(online_cov.get_cov(unbiased=unbiased),
                                   np.cov(X.reshape(10*100), Y.reshape(10*100),
                                          bias=not unbiased)[0][1], places=5)

    def test_quantity_scaler(self):
        np.random.seed(2)
        X = np.random.rand(100) * pq.Hz
        Y = np.random.rand(100) * pq.Hz
        online_cov = CovarianceOnline()
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i] * X.units)
        self.assertEqual(online_cov.units, X.units)
        self.assertIsInstance(online_cov.get_cov(), float)
        for unbiased in (False, True):
            self.assertAlmostEqual(
                online_cov.get_cov(unbiased=unbiased),
                np.cov([X, Y], bias=not unbiased)[0][1])

    def test_quantities_vector(self):
        np.random.seed(3)
        X = np.random.rand(10, 100) * pq.ms
        Y = np.random.rand(10, 100) * pq.ms
        online_cov = CovarianceOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i] * X.units)
        self.assertEqual(online_cov.units, X.units)
        self.assertIsInstance(online_cov.get_cov(), float)
        for unbiased in (False, True):
            self.assertAlmostEqual(
                online_cov.get_cov(unbiased=unbiased),
                np.cov(X.reshape(10*100), Y.reshape(10*100),
                       bias=not unbiased)[0][1], places=4)

    def test_reset(self):
        X = np.array([1, 2, 3, 2, 3])
        Y = np.array([4, 4, 3, 3, 4])
        online_cov = CovarianceOnline()
        for x_i, y_i in zip(X, Y):
            online_cov.update([x_i, y_i])
        self.assertEqual(online_cov.get_cov(), -0.12)
        online_cov.reset()
        self.assertIsNone(online_cov.var_x.mean)
        self.assertIsNone(online_cov.var_y.mean)
        self.assertIsNone(online_cov.units)
        self.assertEqual(online_cov.count, 0)
        self.assertEqual(online_cov.covariance_sum, 0.)

    def test_units(self):
        X = [[1, 2, 3, 2, 3], [1, 2, 3, 2, 3]]
        Y = [[4, 4, 3, 3, 4], [4, 4, 3, 3, 4]]
        online_cov = CovarianceOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            if online_cov.count == 5:
                np.testing.assert_raises(ValueError, online_cov.update,
                                         [x_i, y_i]*pq.ms)
            else:
                online_cov.update([x_i, y_i]*pq.s)


class TestPearsonCorrelationCoefficientOnline(unittest.TestCase):
    def test_simple_small_sets_XY_unbatched(self):
        X = np.array([1, 2, 3, 2, 3])
        Y = np.array([4, 4, 3, 3, 4])
        online_pcc = PearsonCorrelationCoefficientOnline(batch_mode=False)
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i])
        self.assertIsNone(online_pcc.units)
        self.assertAlmostEqual(online_pcc.get_pcc(), np.corrcoef([X, Y])[0][1])

    def test_simple_small_sets_XY_batched(self):
        X = np.array([[1, 2, 3, 2, 3], [1, 2, 3, 2, 3], [1, 2, 3, 2, 3]])
        Y = np.array([[4, 4, 3, 3, 4], [4, 4, 3, 3, 4], [4, 4, 3, 3, 4]])
        online_pcc = PearsonCorrelationCoefficientOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i])
        self.assertIsNone(online_pcc.units)
        self.assertAlmostEqual(online_pcc.get_pcc(),
                               np.corrcoef([X.reshape(15),
                                            Y.reshape(15)])[0][1])

    def test_floats(self):
        np.random.seed(0)
        X = np.random.rand(100)
        Y = np.random.rand(100)
        online_pcc = PearsonCorrelationCoefficientOnline()
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i])
        self.assertIsNone(online_pcc.units)
        self.assertIsInstance(online_pcc.get_pcc(), float)
        self.assertAlmostEqual(online_pcc.get_pcc(), np.corrcoef([X, Y])[0][1])

    def test_numpy_array(self):
        np.random.seed(1)
        X = np.random.rand(10, 100)
        Y = np.random.rand(10, 100)
        online_pcc = PearsonCorrelationCoefficientOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i])
        self.assertIsNone(online_pcc.units)
        self.assertIsInstance(online_pcc.get_pcc(), float)
        self.assertAlmostEqual(online_pcc.get_pcc(),
                               np.corrcoef(
                                   [X.reshape(10*100),
                                    Y.reshape(10*100)])[0][1], places=3)

    def test_quantity_scaler(self):
        np.random.seed(2)
        X = np.random.rand(100) * pq.Hz
        Y = np.random.rand(100) * pq.Hz
        online_pcc = PearsonCorrelationCoefficientOnline()
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i] * X.units)
        self.assertEqual(online_pcc.units, X.units)
        self.assertIsInstance(online_pcc.get_pcc(), float)
        self.assertAlmostEqual(online_pcc.get_pcc(), np.corrcoef([X, Y])[0][1])

    def test_quantities_vector(self):
        np.random.seed(3)
        X = np.random.rand(10, 100) * pq.ms
        Y = np.random.rand(10, 100) * pq.ms
        online_pcc = PearsonCorrelationCoefficientOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i] * X.units)
        self.assertEqual(online_pcc.units, X.units)
        self.assertIsInstance(online_pcc.get_pcc(), float)
        self.assertAlmostEqual(
            online_pcc.get_pcc(),
            np.corrcoef(X.reshape(10*100), Y.reshape(10*100))[0][1], places=2)

    def test_reset(self):
        X = np.array([1, 2, 3, 2, 3])
        Y = X
        online_pcc = PearsonCorrelationCoefficientOnline()
        for x_i, y_i in zip(X, Y):
            online_pcc.update([x_i, y_i])
        self.assertEqual(online_pcc.get_pcc(), 1)
        online_pcc.reset()
        self.assertIsNone(online_pcc.covariance_xy.var_x.mean)
        self.assertEqual(online_pcc.covariance_xy.var_x.variance_sum, 0.)
        self.assertEqual(online_pcc.covariance_xy.var_x.count, 0)
        self.assertIsNone(online_pcc.covariance_xy.var_x.units)
        self.assertIsNone(online_pcc.covariance_xy.var_y.mean)
        self.assertEqual(online_pcc.covariance_xy.var_y.variance_sum, 0.)
        self.assertEqual(online_pcc.covariance_xy.var_y.count, 0)
        self.assertIsNone(online_pcc.covariance_xy.var_y.units)
        self.assertIsNone(online_pcc.covariance_xy.units)
        self.assertEqual(online_pcc.covariance_xy.covariance_sum, 0.)
        self.assertEqual(online_pcc.covariance_xy.count, 0)
        self.assertIsNone(online_pcc.units)
        self.assertEqual(online_pcc.count, 0)
        self.assertEqual(online_pcc.R_xy, 0.)

    def test_units(self):
        X = [[1, 2, 3, 2, 3], [1, 2, 3, 2, 3]]
        Y = [[4, 4, 3, 3, 4], [4, 4, 3, 3, 4]]
        online_pcc = PearsonCorrelationCoefficientOnline(batch_mode=True)
        for x_i, y_i in zip(X, Y):
            if online_pcc.count == 5:
                np.testing.assert_raises(ValueError, online_pcc.update,
                                         [x_i, y_i]*pq.ms)
            else:
                online_pcc.update([x_i, y_i]*pq.s)


if __name__ == '__main__':
    unittest.main()
