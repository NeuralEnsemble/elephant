from copy import deepcopy

import numpy as np
import quantities as pq

from elephant.statistics import isi

msg_same_units = "Each batch must have the same units."


class MeanOnline(object):
    def __init__(self, batch_mode=False):
        self.mean = None
        self.count = 0
        self.units = None
        self.batch_mode = batch_mode

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.batch_mode:
            batch_size = new_val.shape[0]
            new_val_sum = new_val.sum(axis=0)
        else:
            batch_size = 1
            new_val_sum = new_val
        self.count += batch_size
        if self.mean is None:
            self.mean = deepcopy(new_val_sum / batch_size)
            self.units = units
        else:
            if units != self.units:
                raise ValueError(msg_same_units)
            self.mean += (new_val_sum - self.mean * batch_size) / self.count

    def as_units(self, val):
        if self.units is None:
            return val
        return pq.Quantity(val, units=self.units, copy=False)

    def get_mean(self):
        return self.as_units(deepcopy(self.mean))

    def reset(self):
        self.mean = None
        self.count = 0
        self.units = None


class VarianceOnline(MeanOnline):
    def __init__(self, batch_mode=False):
        super(VarianceOnline, self).__init__(batch_mode=batch_mode)
        self.variance_sum = 0.

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.mean is None:
            self.mean = 0.
            self.variance_sum = 0.
            self.units = units
        elif units != self.units:
            raise ValueError(msg_same_units)
        delta_var = new_val - self.mean
        if self.batch_mode:
            batch_size = new_val.shape[0]
            self.count += batch_size
            delta_mean = new_val.sum(axis=0) - self.mean * batch_size
            self.mean += delta_mean / self.count
            delta_var *= new_val - self.mean
            delta_var = delta_var.sum(axis=0)
        else:
            self.count += 1
            self.mean += delta_var / self.count
            delta_var *= new_val - self.mean
        self.variance_sum += delta_var

    def get_mean_std(self, unbiased=False):
        if self.mean is None:
            return None, None
        if self.count > 1:
            count = self.count - 1 if unbiased else self.count
            std = np.sqrt(self.variance_sum / count)
        else:
            # with 1 update biased & unbiased sample variance is zero
            std = 0.
        mean = self.as_units(deepcopy(self.mean))
        std = self.as_units(std)
        return mean, std

    def reset(self):
        super(VarianceOnline, self).reset()
        self.variance_sum = 0.


class InterSpikeIntervalOnline(object):
    def __init__(self, bin_size=0.0005, max_isi_value=1, batch_mode=False):
        self.max_isi_value = max_isi_value  # in sec
        self.last_spike_time = None
        self.bin_size = bin_size  # in sec
        self.num_bins = int(self.max_isi_value / self.bin_size)
        self.bin_edges = np.linspace(start=0, stop=self.max_isi_value,
                                     num=self.num_bins + 1)
        self.current_isi_histogram = np.zeros(shape=self.num_bins)
        self.bach_mode = batch_mode
        self.units = None

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.last_spike_time is None:  # for first batch
            if self.bach_mode:
                new_isi = isi(new_val)
                self.last_spike_time = new_val[-1]
            else:
                new_isi = np.array([])
                self.last_spike_time = new_val
            self.units = units
        else:  # for second to last batch
            if units != self.units:
                raise ValueError(msg_same_units)
            if self.bach_mode:
                new_isi = isi(np.append(self.last_spike_time, new_val))
                self.last_spike_time = new_val[-1]
            else:
                new_isi = np.array([new_val - self.last_spike_time])
                self.last_spike_time = new_val
        isi_hist, _ = np.histogram(new_isi, bins=self.bin_edges)
        self.current_isi_histogram += isi_hist

    def as_units(self, val):
        if self.units is None:
            return val
        return pq.Quantity(val, units=self.units, copy=False)

    def get_isi(self):
        return self.as_units(deepcopy(self.current_isi_histogram))

    def reset(self):
        self.last_spike_time = None
        self.units = None
        self.current_isi_histogram = np.zeros(shape=self.num_bins)


class CovarianceOnline(object):
    def __init__(self, batch_mode=False):
        self.batch_mode = batch_mode
        self.var_x = VarianceOnline(batch_mode=batch_mode)
        self.var_y = VarianceOnline(batch_mode=batch_mode)
        self.units = None
        self.covariance_sum = 0.
        self.count = 0

    def update(self, new_val_pair):
        units = None
        if isinstance(new_val_pair, pq.Quantity):
            units = new_val_pair.units
            new_val_pair = new_val_pair.magnitude
        if self.count == 0:
            self.var_x.mean = 0.
            self.var_y.mean = 0.
            self.covariance_sum = 0.
            self.units = units
        elif units != self.units:
            raise ValueError(msg_same_units)
        if self.batch_mode:
            self.var_x.update(new_val_pair[0])
            self.var_y.update(new_val_pair[1])
            delta_var_x = new_val_pair[0] - self.var_x.mean
            delta_var_y = new_val_pair[1] - self.var_y.mean
            delta_covar = delta_var_x * delta_var_y
            batch_size = len(new_val_pair[0])
            self.count += batch_size
            delta_covar = delta_covar.sum(axis=0)
            self.covariance_sum += delta_covar
        else:
            delta_var_x = new_val_pair[0] - self.var_x.mean
            delta_var_y = new_val_pair[1] - self.var_y.mean
            delta_covar = delta_var_x * delta_var_y
            self.var_x.update(new_val_pair[0])
            self.var_y.update(new_val_pair[1])
            self.count += 1
            self.covariance_sum += \
                ((self.count - 1) / self.count) * delta_covar

    def get_cov(self, unbiased=False):
        if self.var_x.mean is None and self.var_y.mean is None:
            return None
        if self.count > 1:
            count = self.count - 1 if unbiased else self.count
            cov = self.covariance_sum / count
        else:
            cov = 0.
        return cov

    def reset(self):
        self.var_x.reset()
        self.var_y.reset()
        self.units = None
        self.covariance_sum = 0.
        self.count = 0


class PearsonCorrelationCoefficientOnline(object):
    def __init__(self, batch_mode=False):
        self.batch_mode = batch_mode
        self.covariance_xy = CovarianceOnline(batch_mode=batch_mode)
        self.units = None
        self.R_xy = 0.
        self.count = 0

    def update(self, new_val_pair):
        units = None
        if isinstance(new_val_pair, pq.Quantity):
            units = new_val_pair.units
            new_val_pair = new_val_pair.magnitude
        if self.count == 0:
            self.covariance_xy.var_y.mean = 0.
            self.covariance_xy.var_y.mean = 0.
            self.units = units
        elif units != self.units:
            raise ValueError(msg_same_units)
        self.covariance_xy.update(new_val_pair)
        if self.batch_mode:
            batch_size = len(new_val_pair[0])
            self.count += batch_size
        else:
            self.count += 1
        if self.count > 1:
            self.R_xy = np.divide(
                self.covariance_xy.covariance_sum,
                (np.sqrt(self.covariance_xy.var_x.variance_sum *
                 self.covariance_xy.var_y.variance_sum)))

    def get_pcc(self):
        if self.count == 0:
            return None
        elif self.count == 1:
            return 0.
        else:
            return self.R_xy

    def reset(self):
        self.count = 0
        self.units = None
        self.R_xy = 0.
        self.covariance_xy.reset()
