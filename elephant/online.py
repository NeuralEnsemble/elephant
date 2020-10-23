from copy import deepcopy

import numpy as np
import quantities as pq


class MeanOnline(object):
    def __init__(self, val=None):
        self.mean = None
        self.count = 0
        self.units = None
        if val is not None:
            self.update(new_val=val)

    def update(self, new_val):
        self.count += 1
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.mean is None:
            self.mean = deepcopy(new_val)
            self.units = units
        else:
            if units != self.units:
                raise ValueError("Each batch must have the same units.")
            self.mean += (new_val - self.mean) / self.count

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
    def __init__(self, val=None):
        self.variance_sum = 0.
        super(VarianceOnline, self).__init__(val)

    def update(self, new_val):
        self.count += 1
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.mean is None:
            self.mean = deepcopy(new_val)
            self.variance_sum = 0.
            self.units = units
        else:
            if units != self.units:
                raise ValueError("Each batch must have the same units.")
            delta = new_val - self.mean
            self.mean += delta / self.count
            delta2 = new_val - self.mean
            self.variance_sum += delta * delta2

    def get_mean_std(self, unbiased=True):
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
