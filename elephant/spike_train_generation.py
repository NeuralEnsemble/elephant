# -*- coding: utf-8 -*-
"""
Functions to generate/extract spike trains from analog signals, or to generate
random spike trains.

Extract spike times from time series
***************************************
.. autosummary::
    :toctree: _toctree/spike_train_generation

    spike_extraction
    threshold_detection
    peak_detection


Random spike train processes
****************************
.. autosummary::
    :toctree: _toctree/spike_train_generation

    StationaryPoissonProcess
    StationaryGammaProcess
    StationaryLogNormalProcess
    StationaryInverseGaussianProcess
    NonStationaryPoissonProcess
    NonStationaryGammaProcess


Coincident spike times generation
*********************************
.. autosummary::
    :toctree: _toctree/spike_train_generation

    single_interaction_process
    compound_poisson_process

Some functions are based on the NeuroTools stgen module, which was mostly
written by Eilif Muller, or from the NeuroTools signals.analogs module.


References
----------

.. bibliography:: ../bib/elephant.bib
   :labelprefix: gen
   :keyprefix: generation-
   :style: unsrt


:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
from typing import List, Union, Optional

import neo
import numpy as np
import quantities as pq
from scipy import stats
from scipy import integrate
from scipy.optimize import root_scalar
from scipy.special import gammainc, gammaincc

from elephant.spike_train_surrogates import dither_spike_train
from elephant.utils import deprecated_alias

__all__ = [
    "spike_extraction",
    "threshold_detection",
    "peak_detection",
    "StationaryPoissonProcess",
    "StationaryGammaProcess",
    "StationaryLogNormalProcess",
    "StationaryInverseGaussianProcess",
    "NonStationaryPoissonProcess",
    "NonStationaryGammaProcess",
    "homogeneous_poisson_process",
    "inhomogeneous_poisson_process",
    "homogeneous_gamma_process",
    "inhomogeneous_gamma_process",
    "single_interaction_process",
    "compound_poisson_process"
]


@deprecated_alias(extr_interval='interval')
def spike_extraction(signal, threshold=0.0 * pq.mV, sign='above',
                     time_stamps=None, interval=(-2 * pq.ms, 4 * pq.ms)):
    """
    Return the peak times for all events that cross threshold and the
    waveforms. Usually used for extracting spikes from a membrane
    potential to calculate waveform properties.

    Parameters
    ----------
    signal : neo.AnalogSignal
        An analog input signal.
    threshold : pq.Quantity, optional
        Contains a value that must be reached for an event to be detected.
        Default: 0.0 * pq.mV
    sign : {'above', 'below'}, optional
        Determines whether to count threshold crossings that cross above or
        below the threshold.
        Default: 'above'
    time_stamps : pq.Quantity, optional
        If `spike_train` is a `pq.Quantity` array, `time_stamps` provides the
        time stamps around which the waveform is extracted. If it is None, the
        function `peak_detection` is used to calculate the time_stamps
        from signal.
        Default: None
    interval : tuple of pq.Quantity
        Specifies the time interval around the `time_stamps` where the waveform
        is extracted.
        Default: (-2 * pq.ms, 4 * pq.ms)

    Returns
    -------
    result_st : neo.SpikeTrain
        Contains the time_stamps of each of the spikes and the waveforms in
        `result_st.waveforms`.

    See Also
    --------
    elephant.spike_train_generation.peak_detection
    """
    # Get spike time_stamps
    if time_stamps is None:
        time_stamps = peak_detection(signal, threshold, sign=sign)
    elif hasattr(time_stamps, 'times'):
        time_stamps = time_stamps.times
    elif isinstance(time_stamps, pq.Quantity):
        raise TypeError("time_stamps must be None, a pq.Quantity array or" +
                        " expose the.times interface")

    if len(time_stamps) == 0:
        return neo.SpikeTrain(time_stamps, units=signal.times.units,
                              t_start=signal.t_start, t_stop=signal.t_stop,
                              waveforms=np.array([]),
                              sampling_rate=signal.sampling_rate)

    # Unpack the extraction interval from tuple or array
    extr_left, extr_right = interval
    if extr_left > extr_right:
        raise ValueError("interval[0] must be < interval[1]")

    if any(np.diff(time_stamps) < interval[1]):
        warnings.warn("Waveforms overlap.", UserWarning)

    data_left = (extr_left * signal.sampling_rate).simplified.magnitude

    data_right = (extr_right * signal.sampling_rate).simplified.magnitude

    data_stamps = (((time_stamps - signal.t_start) *
                    signal.sampling_rate).simplified).magnitude

    data_stamps = data_stamps.astype(int)

    borders_left = data_stamps + data_left

    borders_right = data_stamps + data_right

    borders = np.dstack((borders_left, borders_right)).flatten()

    waveforms = np.array(
        np.split(np.array(signal), borders.astype(int))[1::2]) * signal.units

    # len(np.shape(waveforms)) == 1 if waveforms do not have the same width.
    # this can occur when extraction interval indexes beyond the signal.
    # Workaround: delete spikes shorter than the maximum length with
    if len(np.shape(waveforms)) == 1:
        max_len = max(len(waveform) for waveform in waveforms)
        to_delete = np.array([idx for idx, x in enumerate(waveforms)
                              if len(x) < max_len])
        waveforms = np.delete(waveforms, to_delete, axis=0)
        warnings.warn("Waveforms " +
                      ("{:d}, " * len(to_delete)).format(*to_delete) +
                      "exceeded signal and had to be deleted. " +
                      "Change 'interval' to keep.")

    waveforms = waveforms[:, np.newaxis, :]

    return neo.SpikeTrain(time_stamps, units=signal.times.units,
                          t_start=signal.t_start, t_stop=signal.t_stop,
                          sampling_rate=signal.sampling_rate,
                          waveforms=waveforms,
                          left_sweep=extr_left)


def threshold_detection(signal, threshold=0.0 * pq.mV, sign='above'):
    """
    Returns the times when the analog signal crosses a threshold.
    Usually used for extracting spike times from a membrane potential.

    Parameters
    ----------
    signal : neo.AnalogSignal
        An analog input signal.
    threshold : pq.Quantity, optional
        Contains a value that must be reached for an event to be detected.
        Default: 0.0 * pq.mV
    sign : {'above', 'below'}, optional
        Determines whether to count threshold crossings that cross above or
        below the threshold.
        Default: 'above'

    Returns
    -------
    result_st : neo.SpikeTrain
        Contains the spike times of each of the events (spikes) extracted from
        the signal.
    """

    if not isinstance(threshold, pq.Quantity):
        raise ValueError('threshold must be a pq.Quantity')

    if sign not in ('above', 'below'):
        raise ValueError("sign should be 'above' or 'below'")

    if sign == 'above':
        cutout = np.where(signal > threshold)[0]
    else:
        # sign == 'below'
        cutout = np.where(signal < threshold)[0]

    if len(cutout) == 0:
        events_base = np.zeros(0)
    else:
        take = np.where(np.diff(cutout) > 1)[0] + 1
        take = np.append(0, take)

        time = signal.times
        events = time[cutout][take]

        events_base = events.magnitude
        if events_base is None:
            # This occurs in some Python 3 builds due to some
            # bug in quantities.
            events_base = np.array(
                [event.magnitude for event in events])  # Workaround

    result_st = neo.SpikeTrain(events_base, units=signal.times.units,
                               t_start=signal.t_start, t_stop=signal.t_stop)
    return result_st


@deprecated_alias(format='as_array')
def peak_detection(signal, threshold=0.0 * pq.mV, sign='above',
                   as_array=False):
    """
    Return the peak times for all events that cross threshold.
    Usually used for extracting spike times from a membrane potential.
    Similar to spike_train_generation.threshold_detection.

    Parameters
    ----------
    signal : neo.AnalogSignal
        An analog input signal.
    threshold : pq.Quantity, optional
        Contains a value that must be reached for an event to be detected.
        Default: 0.*pq.mV
    sign : {'above', 'below'}, optional
        Determines whether to count threshold crossings that cross above or
        below the threshold.
        Default: 'above'
    as_array : bool, optional
        If True, a NumPy array of the resulting peak times is returned instead
        of a (default) `neo.SpikeTrain` object.
        Default: False
    format : {None, 'raw'}, optional
        .. deprecated:: 0.8.0
        Whether to return as SpikeTrain (None) or as a plain array of times
        ('raw').
        Deprecated. Use `as_array=False` for None format and `as_array=True`
        otherwise.
        Default: None

    Returns
    -------
    result_st : neo.SpikeTrain
        Contains the spike times of each of the events (spikes) extracted from
        the signal.
    """
    if not isinstance(threshold, pq.Quantity):
        raise ValueError("threshold must be a pq.Quantity")

    if sign not in ('above', 'below'):
        raise ValueError("sign should be 'above' or 'below'")

    if as_array in (None, 'raw'):
        warnings.warn("'format' is deprecated; use as_array=True",
                      DeprecationWarning)
        as_array = bool(as_array)

    if sign == 'above':
        cutout = np.where(signal > threshold)[0]
        peak_func = np.argmax
    else:
        # sign == 'below'
        cutout = np.where(signal < threshold)[0]
        peak_func = np.argmin

    if len(cutout) == 0:
        events_base = np.zeros(0)
    else:
        # Select thr crossings lasting at least 2 dtps, np.diff(cutout) > 2
        # This avoids empty slices
        border_start = np.where(np.diff(cutout) > 1)[0]
        border_end = border_start + 1
        borders = sorted(np.r_[0, border_start, border_end, len(cutout) - 1])
        true_borders = cutout[borders]
        right_borders = true_borders[1::2] + 1
        true_borders = np.sort(np.append(true_borders[0::2], right_borders))

        # Workaround for bug that occurs when signal goes below thr for 1 dtp,
        # Workaround eliminates empty slices from np. split
        backward_mask = np.absolute(np.ediff1d(true_borders, to_begin=1)) > 0
        forward_mask = np.absolute(np.ediff1d(true_borders[::-1],
                                              to_begin=1)[::-1]) > 0
        true_borders = true_borders[backward_mask * forward_mask]
        split_signal = np.split(np.array(signal), true_borders)[1::2]

        maxima_idc_split = np.array([peak_func(x) for x in split_signal])

        max_idc = maxima_idc_split + true_borders[0::2]

        events = signal.times[max_idc]
        events_base = events.magnitude

        if events_base is None:
            # This occurs in some Python 3 builds due to some
            # bug in quantities.
            events_base = np.array(
                [event.magnitude for event in events])  # Workaround

    result_st = neo.SpikeTrain(events_base, units=signal.times.units,
                               t_start=signal.t_start,
                               t_stop=signal.t_stop)
    if as_array:
        result_st = result_st.magnitude

    return result_st


class AbstractPointProcess:
    """
    Abstract point process to subclass from.

    Parameters
    ----------
    t_start : pq.Quantity, optional
        The start of the spike train.
        Default: 0.*pq.s
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1.*pq.s
    """
    def __init__(
            self,
            t_stop: pq.Quantity = 1.*pq.s,
            t_start: pq.Quantity = 0.*pq.s
    ):
        if not (isinstance(t_start, pq.Quantity) and
                isinstance(t_stop, pq.Quantity)):
            raise ValueError("t_start and t_stop must be of type pq.Quantity")
        if t_stop <= t_start:
            raise ValueError('t_start must be smaller than t_stop.')

        self.units = t_stop.units
        self._t_stop = t_stop.item()
        self._t_start = t_start.rescale(self.units).item()

    @property
    def t_start(self):
        """
        t_start quantity; there are no spike times below this value.
        """
        return self._t_start * self.units

    @property
    def t_stop(self):
        """
        t_stop quantity; there are no spike times above this value.
        """
        return self._t_stop * self.units

    def _generate_spiketrain_as_array(self) -> np.ndarray:
        raise NotImplementedError

    def generate_spiketrain(
            self, as_array: bool = False
    ) -> Union[neo.SpikeTrain, np.ndarray]:
        """
        Generates a single spike train.

        Parameters
        ----------
        as_array : bool, optional
            If True, a NumPy array of sorted spikes is returned,
            rather than a `neo.SpikeTrain` object.
            Default: False

        Returns
        -------
        spiketrain : neo.SpikeTrain or np.ndarray
            The generated spike train in the specified format.
        """
        spikes = self._generate_spiketrain_as_array()
        if as_array:
            return spikes
        # else:
        return neo.SpikeTrain(
            spikes,
            t_start=self.t_start, t_stop=self.t_stop, units=self.units)

    def generate_n_spiketrains(
            self,
            n_spiketrains: int,
            as_array: bool = False
    ) -> Union[List[neo.SpikeTrain], List[np.ndarray]]:
        """
        Generates a list of spike trains.

        Parameters
        ----------
        n_spiketrains : int
            The number of spike trains to generate.
        as_array : bool, optional
            If True, a NumPy array of sorted spikes is returned,
            rather than a `neo.SpikeTrain` object.
            Default: False

        Returns
        -------
        list_of_spiketrain : list of neo.SpikeTrain or list of np.ndarray
            A list generated spike trains in the specified format.
        """
        return [self.generate_spiketrain(as_array=as_array)
                for _ in range(n_spiketrains)]


class RenewalProcess(AbstractPointProcess):
    """
    Abstract renewal process to subclass from.

    Parameters
    ----------
    rate : pq.Quantity
        The constant firing rate.
    t_start : pq.Quantity, optional
        The start of the spike train.
        Default: 0.*pq.s
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1.*pq.s
    equilibrium : bool, optional
        Generate an equilibrium or an ordinary renewal process.
        Default: True
    """
    isi_generator: stats.rv_continuous

    def __init__(
            self,
            rate: pq.Quantity,
            t_start: pq.Quantity = 0.*pq.s,
            t_stop: pq.Quantity = 1.*pq.s,
            equilibrium: bool = True
    ):
        super().__init__(t_start=t_start, t_stop=t_stop)
        if not isinstance(rate, pq.Quantity):
            raise ValueError("rate must be of type pq.Quantity")
        self.rate = rate.rescale(1./self.units).item()

        self.equilibrium = equilibrium

        self.n_expected_spikes = int(np.ceil(
            ((self._t_stop - self._t_start) * self.rate)))

        if self.n_expected_spikes < 0:
            raise ValueError(
                f"Expected no. of spikes: {self.n_expected_spikes} < 0. "
                f"The firing rate ({self.rate/self.units}) "
                f"cannot be negative.")

    def _cdf_first_spike_equilibrium(self, time):
        """
        Integral over the p.d.f. of the first spike which is:
        p(t) = rate * survival-function(t) * Heaviside(t).
        See Bouss (2020).

        The parameter time is a magnitude of a time value given in seconds.
        """
        return self.rate * integrate.quad(self.isi_generator.sf, 0., time)[0]

    def _get_first_spike_equilibrium(self):
        """
        Return a numerically drawn sample of the p.d.f of the first spike.

        By solving:
        x = integral(c.d.f(t) from 0 to t),
        where x is drawn from a uniform distribution.
        """
        random_uniform = np.random.random()
        equation_solver = root_scalar

        def function_to_solve(time):
            """
            # integral(c.d.f(t) from 0 to t) - random-number-x)
            """
            return self._cdf_first_spike_equilibrium(time) - random_uniform

        def derivative_of_function_to_solve(time):
            """
            derivative of the c.d.f, which is rate times
            the survival function
            """
            return self.rate * self.isi_generator.sf(time)

        # Initial guess is solution for Poisson process
        initial_guess = -np.log(1.-random_uniform)/self.rate
        duration = self._t_stop-self._t_start
        limits_for_first_spike = (0., duration)

        # test if solution for first spike is inside the boundaries. If not
        # return t_stop of the spike train.
        if self._cdf_first_spike_equilibrium(duration) <= random_uniform:
            return self._t_stop

        non_shifted_position_of_first_spike = equation_solver(
                function_to_solve,
                x0=initial_guess,
                bracket=limits_for_first_spike,
                fprime=derivative_of_function_to_solve
            ).root

        return non_shifted_position_of_first_spike + self._t_start

    def _generate_spiketrain_as_array(self) -> np.ndarray:
        if self.n_expected_spikes == 0:
            return np.array([])

        if self.equilibrium:  # equilibrium renewal process
            # First spike of equilibrium renewal process drawn according to
            # Bouss (2020), Master's Thesis
            first_spike = self._get_first_spike_equilibrium()
        else:  # ordinary renewal process
            first_spike = self.isi_generator.rvs() + self._t_start

        spikes = np.array([first_spike])

        # 3 STDs corresponds to 99.7%
        n_spikes_three_stds = int(np.ceil(
            self.n_expected_spikes + 3 * np.sqrt(self.n_expected_spikes)))

        # Continue until whole time range is covered
        while spikes[-1] < self._t_stop:
            isi = self.isi_generator.rvs(size=n_spikes_three_stds)

            t_last_spike = spikes[-1]
            spikes = np.r_[spikes, t_last_spike + np.cumsum(isi)]

        index_last_spike = spikes.searchsorted(self._t_stop)
        spikes = spikes[:index_last_spike]

        return spikes

    @property
    def expected_cv(self):
        """
        The expected coefficient of variation given the ISI distribution.
        """
        return self.isi_generator.std()/self.isi_generator.mean()


class StationaryPoissonProcess(RenewalProcess):
    """
    Generates spike trains whose spikes are realizations of a stationary
    Poisson process with the given rate, starting at time `t_start` and
    stopping at time `t_stop`. Optionally, a absolute refractory period /
    dead time can be specified.

    Parameters
    ----------
    rate : pq.Quantity
        The constant firing rate.
    t_start : pq.Quantity, optional
        The start of the spike train.
        Default: 0.*pq.s
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1.*pq.s
    refractory_period : pq.Quantity, optional
        The time period after one spike in which no other spike is emitted.
        This can be called an absolute refractory period or a dead time as
        used in :cite:`generation-Deger12_443`.
        Default : None
    equilibrium : bool, optional
        Generate an equilibrium or an ordinary renewal process.
        Default: True

    Raises
    ------
    ValueError
        If one of `rate`, `t_start` and `t_stop` is not of type `pq.Quantity`.

        If `refractory_period` is not of type `pq.Quantity` nor None.

        If the period between two successive spikes (`1 / rate`) is smaller
        or equal than the `refractory_period`.

    Examples
    --------
    >>> import quantities as pq
    >>> spiketrain = StationaryPoissonProcess(rate=50.*pq.Hz, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms).generate_spiketrain()
    >>> spiketrain_array = StationaryPoissonProcess(
    ...     rate=20*pq.Hz, t_start=5000*pq.ms, t_stop=10000*pq.ms
    ...     ).generate_spiketrain(as_array=True)
    >>> spiketrain = StationaryPoissonProcess(
    ...     rate=50*pq.Hz,
    ...     t_start=0*pq.ms, t_stop=1000*pq.ms,
    ...     refractory_period = 3*pq.ms).generate_spiketrain()
    """
    def __init__(
            self,
            rate: pq.Quantity,
            t_stop: pq.Quantity = 1.*pq.s,
            t_start: pq.Quantity = 0.*pq.s,
            refractory_period: Optional[pq.Quantity] = None,
            equilibrium: bool = True
    ):
        super().__init__(
            rate=rate, t_start=t_start, t_stop=t_stop, equilibrium=equilibrium)

        if refractory_period is not None:
            if not isinstance(refractory_period, pq.Quantity):
                raise ValueError(
                    "refractory_period must be of type pq.Quantity")
            self.refractory_period = refractory_period.rescale(
                self.units).item()

            if self.rate * self.refractory_period >= 1.:
                raise ValueError(
                    "Period between two successive spikes must be larger "
                    "than the refractory period. Decrease either the "
                    "firing rate or the refractory period.")
        else:
            self.refractory_period = refractory_period

        if self.n_expected_spikes > 0 and refractory_period is None:
            self.isi_generator = stats.expon(scale=1./self.rate)

        elif self.n_expected_spikes > 0 and refractory_period is not None:
            self.effective_rate = self.rate / \
                                  (1. - self.rate * self.refractory_period)
            self.isi_generator = stats.expon(
                scale=1. / self.effective_rate, loc=self.refractory_period)

    def _get_first_spike_equilibrium(self):
        if self.refractory_period is None:
            return self.isi_generator.rvs() + self._t_start

        # the case with dead time
        random_uniform = np.random.random()
        if random_uniform <= self.rate * self.refractory_period:
            return random_uniform / self.rate + self._t_start
        # random_uniform > self.rate * self.refractory_period
        return (np.log(1. - self.rate * self.refractory_period)
                - np.log(1. - random_uniform)
                ) / self.effective_rate + self.refractory_period

    @property
    def expected_cv(self):
        """
        The expected coefficient of variation given the ISI distribution.
        """
        if self.refractory_period is None:
            return 1.

        # the case with dead time
        return 1. - self.rate * self.refractory_period


class StationaryGammaProcess(RenewalProcess):
    """
    Generates spike trains whose spikes are realizations of a stationary
    Gamma process with the given rate and `shape_factor`
    starting at time `t_start` and stopping at time `t_stop`.

    Parameters
    ----------
    rate : pq.Quantity
        The constant firing rate.
    shape_factor : float
        The shape parameter of the gamma distribution.
    t_start : pq.Quantity, optional
        The start of the spike train.
        Default: 0.*pq.s
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1.*pq.s
    equilibrium : bool, optional
        Generate an equilibrium or an ordinary renewal process.
        Default: True

    Raises
    ------
    ValueError
        If one of `rate`, `t_start` and `t_stop` is not of type `pq.Quantity`.

    Examples
    --------
    >>> import quantities as pq
    >>> spiketrain = StationaryGammaProcess(
    ...     rate=50*pq.Hz, shape_factor=2.0, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms).generate_spiketrain()
    >>> spiketrain_array = StationaryGammaProcess(
    ...     rate=20*pq.Hz, shape_factor=5.0, t_start=5000*pq.ms,
    ...     t_stop=10000*pq.ms).generate_spiketrain(as_array=True)
    """
    def __init__(
            self,
            rate: pq.Quantity,
            shape_factor: float,
            t_start: pq.Quantity = 0.*pq.s,
            t_stop: pq.Quantity = 1.*pq.s,
            equilibrium: bool = True
    ):
        super().__init__(
            rate=rate, t_start=t_start, t_stop=t_stop, equilibrium=equilibrium)
        if self.n_expected_spikes > 0:
            self.shape_factor = shape_factor
            self.isi_generator = stats.gamma(
                a=shape_factor, scale=1./(shape_factor * self.rate))

    def _cdf_first_spike_equilibrium(self, time):
        """
        The parameter time is a magnitude of a time value given in seconds.
        """
        if time < 0.:
            return 0.
        return self.rate * time * \
            gammaincc(self.shape_factor,
                      self.shape_factor*self.rate*time)\
            + gammainc(self.shape_factor+1.,
                       self.shape_factor*self.rate*time)

    @property
    def expected_cv(self):
        """
        The expected coefficient of variation given the ISI distribution.
        """
        return 1./np.sqrt(self.shape_factor)


class StationaryLogNormalProcess(RenewalProcess):
    """
    Generates spike trains whose spikes are realizations of a stationary
    LogNormal process with the given rate and `sigma`
    starting at time `t_start` and stopping at time `t_stop`.

    Parameters
    ----------
    rate : pq.Quantity
        The constant firing rate.
    sigma : float
        The sigma/ s parameter of the Log-Normal distribution.
    t_start : pq.Quantity, optional
        The start of the spike train.
        Default: 0.*pq.s
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1.*pq.s
    equilibrium : bool, optional
        Generate an equilibrium or an ordinary renewal process.
        Default: True

    Raises
    ------
    ValueError
        If one of `rate`, `t_start` and `t_stop` is not of type `pq.Quantity`.

    Examples
    --------
    >>> import quantities as pq
    >>> spiketrain = StationaryLogNormalProcess(
    ...     rate=50*pq.Hz, sigma=2.0, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms).generate_spiketrain()
    >>> spiketrain_array = StationaryLogNormalProcess(
    ...     rate=20*pq.Hz, sigma=5.0, t_start=5000*pq.ms,
    ...     t_stop=10000*pq.ms).generate_spiketrain(as_array=True)
    """
    def __init__(
            self,
            rate: pq.Quantity,
            sigma: float,
            t_start: pq.Quantity = 0.*pq.s,
            t_stop: pq.Quantity = 1.*pq.s,
            equilibrium: bool = True
    ):
        super().__init__(
            rate=rate, t_start=t_start, t_stop=t_stop, equilibrium=equilibrium)
        self.sigma = sigma
        if self.n_expected_spikes > 0:
            self.isi_generator = stats.lognorm(
                s=self.sigma, scale=np.exp(self.mu))

    @property
    def mu(self):
        """
        The parameter mu of the log-normal distribution.
        """
        return -np.log(self.rate) - self.sigma**2/2

    @property
    def expected_cv(self):
        """
        The expected coefficient of variation given the ISI distribution.
        """
        return np.sqrt(np.exp(self.sigma**2) - 1)


class StationaryInverseGaussianProcess(RenewalProcess):
    """
    Generates spike trains whose spikes are realizations of a stationary
    Gamma process with the given rate and `cv`
    starting at time `t_start` and stopping at time `t_stop`.

    Raises
    ------
    ValueError
        If one of `rate`, `t_start` and `t_stop` is not of type `pq.Quantity`.

    Parameters
    ----------
    rate : pq.Quantity
        The constant firing rate.
    cv : float
        The expected coefficient of variation.
    t_start : pq.Quantity, optional
        The start of the spike train.
        Default: 0.*pq.s
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1.*pq.s
    equilibrium : bool, optional
        Generate an equilibrium or an ordinary renewal process.
        Default: True

    Examples
    --------
    >>> import quantities as pq
    >>> spiketrain = StationaryInverseGaussianProcess(
    ...     rate=50*pq.Hz, cv=2.0, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms).generate_spiketrain()
    >>> spiketrain_array = StationaryInverseGaussianProcess(
    ...     rate=20*pq.Hz, cv=5.0, t_start=5000*pq.ms,
    ...     t_stop=10000*pq.ms).generate_spiketrain(as_array=True)
    """
    def __init__(
            self,
            rate: pq.Quantity,
            cv: float,
            t_start: pq.Quantity = 0.*pq.s,
            t_stop: pq.Quantity = 1.*pq.s,
            equilibrium: bool = True
    ):
        super().__init__(
            rate=rate, t_start=t_start, t_stop=t_stop, equilibrium=equilibrium)
        self._cv = cv
        if self.n_expected_spikes > 0:
            self.isi_generator = stats.invgauss(
                mu=cv**2, scale=1./(self.rate*cv**2))

    @property
    def expected_cv(self):
        """
        The expected coefficient of variation given the ISI distribution.
        """
        return self._cv


class RateModulatedProcess(AbstractPointProcess):
    """
    Abstract rate-modulated process to subclass from.

    Parameters
    ----------
    rate_signal : neo.AnalogSignal
        A `neo.AnalogSignal` representing the rate profile evolving over
        time. Its values have all to be `>=0`. The generated spike trains
        will have `t_start = rate.t_start` and `t_stop = rate.t_stop`

    Raises
    ------
    ValueError
        If `rate_signal` is not a neo AnalogSignal
        If `rate_signal` contains a negative value.
        If `rate_signal` is empty.
    """
    process_operational_time: RenewalProcess

    def __init__(self, rate_signal: neo.AnalogSignal):

        if not isinstance(rate_signal, neo.AnalogSignal):
            raise ValueError(
                f'rate_signal should be of type neo.AnalogSignal.'
                f' Currently it is of type: {type(rate_signal)}')
        if len(rate_signal) == 0:
            raise ValueError('rate_signal can not be empty.')
        if any(rate_signal < 0):
            raise ValueError(
                'All elements of rate_signal should be positive.')

        super().__init__(
            t_start=rate_signal.t_start, t_stop=rate_signal.t_stop)

        self.rate_signal = rate_signal

        self.mean_rate = np.mean(rate_signal.rescale(1./self.units).magnitude)

        if self.mean_rate == 0.:
            # if the firing rate is zero, the init functions stops here, since
            # the other parameters are then not needed.
            return None

        self.sampling_period = \
            self.rate_signal.sampling_period.rescale(self.units).magnitude
        # Operational time corresponds to the integral of the firing rate
        # over time, here normalized by the average firing rate
        operational_time = np.cumsum(
            rate_signal.rescale(1./self.units).magnitude)
        operational_time *= (self.sampling_period / self.mean_rate)
        operational_time = np.hstack((0., operational_time))
        self.operational_time = operational_time + self._t_start

        # The time points at which the firing rates are given
        self.real_time = np.hstack(
            (rate_signal.times.rescale(self.units).magnitude,
             self._t_stop))

    def _generate_spiketrain_as_array(self) -> np.ndarray:
        spiketrain_operational_time = \
            self.process_operational_time._generate_spiketrain_as_array()
        if len(spiketrain_operational_time) == 0:
            return spiketrain_operational_time
        # indices where between which points in operational time the spikes lie
        indices = np.searchsorted(self.operational_time,
                                  spiketrain_operational_time)

        # In real time the spikes are first aligned
        # to the left border of the bin.
        # Note that indices are greater than 0 because 'operational_time' was
        # padded with zeros.
        spiketrain = self.real_time[indices - 1]
        # the relative position of the spikes in the operational time bins
        positions_in_bins = \
            (spiketrain_operational_time
             - self.operational_time[indices - 1]) / \
            (self.operational_time[indices]
             - self.operational_time[indices - 1])

        # add the positions in the bin times the sampling period in real time
        spiketrain += self.sampling_period * positions_in_bins
        return spiketrain


class NonStationaryPoissonProcess(RateModulatedProcess):
    """
    Generates spike trains whose spikes are realizations of a non-stationary
    Poisson process with the given `rate-signal`. Optionally, you can specify a
    dead time.

    Parameters
    ----------
    rate_signal : neo.AnalogSignal
        A `neo.AnalogSignal` representing the rate profile evolving over
        time.Its values have all to be `>=0`. The generated spike trains
        will have `t_start = rate.t_start` and `t_stop = rate.t_stop`
    refractory_period : pq.Quantity, optional
        The time period after one spike in which no other spike is emitted.
        This can be called an absolute refractory period or a dead time.
        Default : None

    Raises
    ------
    ValueError
        If `rate_signal` is not a neo AnalogSignal
        If `rate_signal` contains a negative value.
        If `rate_signal` is empty.
        If `refractory_period` is not of type `pq.Quantity` nor None.
    """
    def __init__(self, rate_signal: neo.AnalogSignal,
                 refractory_period: Optional[pq.Quantity] = None):

        if refractory_period is not None:
            if not isinstance(refractory_period, pq.Quantity):
                raise ValueError(
                    "refractory_period must be of type pq.Quantity")
            rate_signal = \
                rate_signal / (1. - rate_signal.simplified.magnitude
                               * refractory_period.simplified.item())

        super().__init__(rate_signal=rate_signal)
        self.process_operational_time = StationaryPoissonProcess(
            rate=self.mean_rate * 1./self.units,
            t_start=self.t_start,
            t_stop=self.t_stop)

        self.refractory_period = refractory_period
        if self.refractory_period is not None:
            self.refractory_period = self.refractory_period.rescale(
                self.units).item()

    def _generate_spiketrain_as_array(self) -> np.ndarray:
        if self.refractory_period is None:
            return super()._generate_spiketrain_as_array()

        spiketrain = super()._generate_spiketrain_as_array()
        thinned_spiketrain = []

        previous_spike = self._t_start - self.refractory_period

        for spike in spiketrain:
            if spike > previous_spike + self.refractory_period:
                thinned_spiketrain.append(spike)
                previous_spike = spike
        return np.array(thinned_spiketrain)


class NonStationaryGammaProcess(RateModulatedProcess):
    """
    Generates spike trains whose spikes are realizations of a non-stationary
    Gamma process with the given `rate-signal`.

    Parameters
    ----------
    rate_signal : neo.AnalogSignal
        A `neo.AnalogSignal` representing the rate profile evolving over
        time.Its values have all to be `>=0`. The generated spike trains
        will have `t_start = rate.t_start` and `t_stop = rate.t_stop`
    shape_factor : float
        The shape parameter of the gamma distribution.

    Raises
    ------
    ValueError
        If `rate_signal` is not a neo AnalogSignal
        If `rate_signal` contains a negative value.
        If `rate_signal` is empty.
    """
    def __init__(self, rate_signal: neo.AnalogSignal, shape_factor: float):
        super().__init__(rate_signal=rate_signal)
        self.process_operational_time = StationaryGammaProcess(
            rate=self.mean_rate * 1./self.units,
            shape_factor=shape_factor,
            t_start=self.t_start,
            t_stop=self.t_stop)


def homogeneous_poisson_process(rate, t_start=0.0 * pq.ms,
                                t_stop=1000.0 * pq.ms, as_array=False,
                                refractory_period=None):
    """
    Returns a spike train whose spikes are a realization of a Poisson process
    with the given rate, starting at time `t_start` and stopping time `t_stop`.

    All numerical values should be given as Quantities, e.g. `100*pq.Hz`.

    Parameters
    ----------
    rate : pq.Quantity
        The rate of the discharge.
    t_start : pq.Quantity, optional
        The beginning of the spike train.
        Default: 0 * pq.ms
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1000 * pq.ms
    as_array : bool, optional
        If True, a NumPy array of sorted spikes is returned,
        rather than a `neo.SpikeTrain` object.
        Default: False
    refractory_period : pq.Quantity or None, optional
        `pq.Quantity` scalar with dimension time. The time period after one
        spike no other spike is emitted.
        Default: None

    Returns
    -------
    spiketrain : neo.SpikeTrain or np.ndarray
        Homogeneous Poisson process realization, stored in `neo.SpikeTrain`
        if `as_array` is False (default) and `np.ndarray` otherwise.

    Raises
    ------
    ValueError
        If one of `rate`, `t_start` and `t_stop` is not of type `pq.Quantity`.

        If `refractory_period` is not None or not of type `pq.Quantity`.

        If `refractory_period` is not None and the period between two
        successive spikes (`1 / rate`) is smaller than the `refractory_period`.

    Examples
    --------
    >>> import quantities as pq
    >>> spikes = homogeneous_poisson_process(50*pq.Hz, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms)
    >>> spikes = homogeneous_poisson_process(
    ...     20*pq.Hz, t_start=5000*pq.ms, t_stop=10000*pq.ms, as_array=True)
    >>> spikes = homogeneous_poisson_process(50*pq.Hz, t_start=0*pq.ms,
    ...     t_stop=1000*pq.ms, refractory_period = 3*pq.ms)

    """
    warnings.warn(
        "'homogeneous_poisson_process' is deprecated;"
        " use 'StationaryPoissonProcess'.", DeprecationWarning)
    process = StationaryPoissonProcess(
        rate=rate, t_start=t_start, t_stop=t_stop,
        refractory_period=refractory_period, equilibrium=False)
    return process.generate_spiketrain(as_array=as_array)


def inhomogeneous_poisson_process(rate, as_array=False,
                                  refractory_period=None):
    """
    Returns a spike train whose spikes are a realization of an inhomogeneous
    Poisson process with the given rate profile.

    Parameters
    ----------
    rate : neo.AnalogSignal
        A `neo.AnalogSignal` representing the rate profile evolving over time.
        Its values have all to be `>=0`. The output spiketrain will have
        `t_start = rate.t_start` and `t_stop = rate.t_stop`
    as_array : bool, optional
        If True, a NumPy array of sorted spikes is returned,
        rather than a SpikeTrain object.
        Default: False
    refractory_period : pq.Quantity or None, optional
        `pq.Quantity` scalar with dimension time. The time period after one
        spike no other spike is emitted.
        Default: None

    Returns
    -------
    spiketrain : neo.SpikeTrain or np.ndarray
        Inhomogeneous Poisson process realization, of type `neo.SpikeTrain`
        if `as_array` is False (default) and `np.ndarray` otherwise.

    Raises
    ------
    ValueError
        If `rate` contains a negative value.

        If `refractory_period` is not None or not of type `pq.Quantity`.

        If `refractory_period` is not None and the period between two
        successive spikes (`1 / rate`) is smaller than the `refractory_period`.

    """
    warnings.warn(
        "'inhomogeneous_poisson_process' is deprecated;"
        " use 'NonStationaryPoissonProcess'.",
        DeprecationWarning)
    process = NonStationaryPoissonProcess(
        rate_signal=rate,
        refractory_period=refractory_period)
    return process.generate_spiketrain(
            as_array=as_array)


def homogeneous_gamma_process(a, b, t_start=0.0 * pq.ms, t_stop=1000.0 * pq.ms,
                              as_array=False):
    """
    Returns a spike train whose spikes are a realization of a gamma process
    with the given parameters, starting at time `t_start` and stopping time
    `t_stop` (average rate will be `b/a`). All numerical values should be
    given as Quantities, e.g. `100*pq.Hz`.

    Parameters
    ----------
    a : int or float
        The shape parameter of the gamma distribution.
    b : pq.Quantity
        The rate parameter of the gamma distribution.
    t_start : pq.Quantity, optional
        The beginning of the spike train.
        Default: 0 * pq.ms
    t_stop : pq.Quantity, optional
        The end of the spike train.
        Default: 1000 * pq.ms
    as_array : bool, optional
        If True, a NumPy array of sorted spikes is returned, rather than a
        `neo.SpikeTrain` object.
        Default: False

    Returns
    -------
    spiketrain : neo.SpikeTrain or np.ndarray
        Homogeneous Gamma process realization, stored in `neo.SpikeTrain`
        if `as_array` is False (default) and `np.ndarray` otherwise.

    Raises
    ------
    ValueError
        If `t_start` and `t_stop` are not of type `pq.Quantity`.

    Examples
    --------
    >>> import quantities as pq
    >>> spikes = homogeneous_gamma_process(2.0, 50*pq.Hz, 0*pq.ms,
    ...                                       1000*pq.ms)
    >>> spikes = homogeneous_gamma_process(
    ...        5.0, 20*pq.Hz, 5000*pq.ms, 10000*pq.ms, as_array=True)

    """
    warnings.warn(
        "'homogeneous_gamma_process' is deprecated;"
        " use 'StationaryGammaProcess'.",
        DeprecationWarning)
    process = StationaryGammaProcess(
        rate=b / a, shape_factor=a, t_stop=t_stop, t_start=t_start,
        equilibrium=False)
    return process.generate_spiketrain(as_array=as_array)


def inhomogeneous_gamma_process(rate, shape_factor, as_array=False):
    """
    Returns a spike train whose spikes are a realization of an inhomogeneous
    Gamma process with the given rate profile and the given shape factor
    :cite:`generation-Nawrot2008_374`.

    Parameters
    ----------
    rate : neo.AnalogSignal
        A `neo.AnalogSignal` representing the rate profile evolving over time.
        Its values have all to be `>=0`. The output spiketrain will have
        `t_start = rate.t_start` and `t_stop = rate.t_stop`
    shape_factor : float
        The shape factor of the Gamma process
    as_array : bool, optional
        If True, a NumPy array of sorted spikes is returned,
        rather than a SpikeTrain object.
        Default: False

    Returns
    -------
    spiketrain : neo.SpikeTrain or np.ndarray
        Inhomogeneous Poisson process realization, of type `neo.SpikeTrain`
        if `as_array` is False (default) and `np.ndarray` otherwise.

    Raises
    ------
    ValueError
        If `rate` is not a neo AnalogSignal
        If `rate` contains a negative value.

    """
    warnings.warn(
        "'inhomogeneous_gamma_process' is deprecated;"
        " use 'nonStationaryGammaProcess'.",
        DeprecationWarning)
    process = NonStationaryGammaProcess(
        rate_signal=rate, shape_factor=shape_factor)
    return process.generate_spiketrain(as_array=as_array)


@deprecated_alias(n='n_spiketrains')
def _n_poisson(rate, t_stop, t_start=0.0 * pq.ms, n_spiketrains=1):
    """
    Generates one or more independent Poisson spike trains.

    Parameters
    ----------
    rate : pq.Quantity scalar or pq.Quantity array
        Expected firing rate (frequency) of each output SpikeTrain.
        Can be one of:
        *  a single pq.Quantity value: expected firing rate of each output
           SpikeTrain
        *  a pq.Quantity array: rate[i] is the expected firing rate of the i-th
           output SpikeTrain
    t_stop : pq.Quantity
        Single common stop time of each output SpikeTrain. Must be > t_start.
    t_start : pq.Quantity, optional
        Single common start time of each output SpikeTrain. Must be < t_stop.
        Default: 0 * pq.ms
    n_spiketrains : int, optional
        If rate is a single pq.Quantity value, n specifies the number of
        SpikeTrains to be generated. If rate is an array, n is ignored and the
        number of SpikeTrains is equal to len(rate).
        Default: 1


    Returns
    -------
    list of neo.SpikeTrain
        Each SpikeTrain contains one of the independent Poisson spike trains,
        either n SpikeTrains of the same rate, or len(rate) SpikeTrains with
        varying rates according to the rate parameter. The time unit of the
        SpikeTrains is given by t_stop.
    """
    # Check that the provided input is Hertz
    if not isinstance(rate, pq.Quantity):
        raise ValueError('rate must be a pq.Quantity')

    # Set number n of output spike trains (specified or set to len(rate))
    if not (isinstance(n_spiketrains, int) and n_spiketrains > 0):
        raise ValueError(
            f'n_spiketrains (={n_spiketrains}) must be a positive integer')

    # one rate for all spike trains
    if rate.ndim == 0:
        return StationaryPoissonProcess(
            rate=rate, t_start=t_start, t_stop=t_stop
        ).generate_n_spiketrains(n_spiketrains)

    # different rate for each spike train
    return [StationaryPoissonProcess(
        rate=single_rate, t_start=t_start, t_stop=t_stop).generate_spiketrain()
            for single_rate in rate]


@deprecated_alias(rate_c='coincidence_rate', n='n_spiketrains',
                  return_coinc='return_coincidences')
def single_interaction_process(
        rate, coincidence_rate, t_stop, n_spiketrains=2, jitter=0 * pq.ms,
        coincidences='deterministic', t_start=0 * pq.ms, min_delay=0 * pq.ms,
        return_coincidences=False):
    """
    Generates a multidimensional Poisson SIP (single interaction process)
    plus independent Poisson processes :cite:`generation-Kuhn2003_67`.

    A Poisson SIP consists of Poisson time series which are independent
    except for simultaneous events in all of them. This routine generates
    a SIP plus additional parallel independent Poisson processes.

    Parameters
    ----------
    t_stop : pq.Quantity
        Total time of the simulated processes. The events are drawn between
        0 and `t_stop`.
    rate : pq.Quantity
        Overall mean rate of the time series to be generated (coincidence
        rate `coincidence_rate` is subtracted to determine the background
        rate). Can be:
        * a float, representing the overall mean rate of each process. If
          so, it must be higher than `coincidence_rate`.
        * an iterable of floats (one float per process), each float
          representing the overall mean rate of a process. If so, all the
          entries must be larger than `coincidence_rate`.
    coincidence_rate : pq.Quantity
        Coincidence rate (rate of coincidences for the n-dimensional SIP).
        The SIP spike trains will have coincident events with rate
        `coincidence_rate` plus independent 'background' events with rate
        `rate-rate_coincidence`.
    n_spiketrains : int, optional
        If `rate` is a single pq.Quantity value, `n_spiketrains` specifies the
        number of SpikeTrains to be generated. If rate is an array,
        `n_spiketrains` is ignored and the number of SpikeTrains is equal to
        `len(rate)`.
        Default: 2
    jitter : pq.Quantity, optional
        Jitter for the coincident events. If `jitter == 0`, the events of all
        n correlated processes are exactly coincident. Otherwise, they are
        jittered around a common time randomly, up to +/- `jitter`.
        Default: 0 * pq.ms
    coincidences : {'deterministic', 'stochastic'}, optional
        Whether the total number of injected coincidences must be determin-
        istic (i.e. rate_coincidence is the actual rate with which coincidences
        are generated) or stochastic (i.e. rate_coincidence is the mean rate of
        coincidences):
          * 'deterministic': deterministic rate

          * 'stochastic': stochastic rate
        Default: 'deterministic'
    t_start : pq.Quantity, optional
        Starting time of the series. If specified, it must be lower than
        `t_stop`.
        Default: 0 * pq.ms
    min_delay : pq.Quantity, optional
        Minimum delay between consecutive coincidence times.
        Default: 0 * pq.ms
    return_coincidences : bool, optional
        Whether to return the coincidence times for the SIP process
        Default: False

    Returns
    -------
    output : list
        Realization of a SIP consisting of `n_spiketrains` Poisson processes
        characterized by synchronous events (with the given jitter).
        If `return_coinc` is `True`, the coincidence times are returned as a
        second output argument. They also have an associated time unit (same
        as `t_stop`).

    Examples
    --------
    >>> import quantities as pq
    >>> import elephant.spike_train_generation as stg
    >>> sip, coinc = stg.single_interaction_process(
    ... rate=20*pq.Hz, coincidence_rate=4.*pq.Hz,
    ... t_stop=1*pq.s, n_spiketrains=10, return_coincidences = True)
    """

    # Check if n is a positive integer
    if not (isinstance(n_spiketrains, int) and n_spiketrains > 0):
        raise ValueError(
            f'n_spiketrains (={n_spiketrains}) must be a positive integer')
    if coincidences not in ('deterministic', 'stochastic'):
        raise ValueError(
            "coincidences must be 'deterministic' or 'stochastic'")

    # Assign time unit to jitter, or check that its existing unit is a time
    # unit
    jitter = abs(jitter)

    # Define the array of rates from input argument rate. Check that its length
    # matches with n
    if rate.ndim == 0:
        if rate < 0 * pq.Hz:
            raise ValueError(
                f'rate (={rate}) must be non-negative.')
        rates_b = np.repeat(rate, n_spiketrains)
    else:
        rates_b = rate.flatten()
        if not all(rates_b >= 0.*pq.Hz):
            raise ValueError('*rate* must have non-negative elements')

    # Check: rate>=rate_coincidence
    if np.any(rates_b < coincidence_rate):
        raise ValueError(
            'all elements of *rate* must be >= *rate_coincidence*')

    # Check min_delay < 1./rate_coincidence
    if not (coincidence_rate == 0 * pq.Hz
            or min_delay < 1. / coincidence_rate):
        raise ValueError(
            "'*min_delay* (%s) must be lower than 1/*rate_coincidence* (%s)." %
            (str(min_delay), str((1. / coincidence_rate).rescale(
                min_delay.units))))

    # Generate the n Poisson processes there are the basis for the SIP
    # (coincidences still lacking)
    embedded_poisson_trains = _n_poisson(
        rate=rates_b - coincidence_rate, t_stop=t_stop, t_start=t_start)
    # Convert the trains from neo SpikeTrain objects to simpler pq.Quantity
    # objects
    embedded_poisson_trains = [
        emb.view(pq.Quantity) for emb in embedded_poisson_trains]

    # Generate the array of times for coincident events in SIP, not closer than
    # min_delay. The array is generated as a pq.Quantity.
    if coincidences == 'deterministic':
        # P. Bouss: we want the closest approximation to the average
        # coincidence count.
        n_coincidences = (t_stop - t_start) * coincidence_rate
        # Conversion to integer necessary for python 2
        n_coincidences = int(round(n_coincidences.simplified.item()))
        while True:
            coinc_times = t_start + \
                np.sort(np.random.random(n_coincidences)) * (
                    t_stop - t_start)
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
    else:  # coincidences == 'stochastic'
        poisson_process = StationaryPoissonProcess(
            rate=coincidence_rate, t_stop=t_stop, t_start=t_start)
        while True:
            coinc_times = poisson_process.generate_spiketrain()
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
        coinc_times = coinc_times.simplified
        units = coinc_times.units
        # Set the coincidence times to T-jitter if larger. This ensures that
        # the last jittered spike time is <T
        effective_t_stop = t_stop - jitter
        coinc_times = np.minimum(coinc_times.magnitude,
                                 effective_t_stop.simplified.magnitude)
        coinc_times = coinc_times * units

    # Replicate coinc_times n times, and jitter each event in each array by
    # +/- jitter (within (t_start, t_stop))
    embedded_coinc = coinc_times + \
        np.random.random(
            (len(rates_b), len(coinc_times))) * 2 * jitter - jitter
    embedded_coinc = embedded_coinc + \
        (t_start - embedded_coinc) * (embedded_coinc < t_start) - \
        (t_stop - embedded_coinc) * (embedded_coinc > t_stop)

    # Inject coincident events into the n SIP processes generated above, and
    # merge with the n independent processes
    sip_process = [
        np.sort(np.concatenate((
            embedded_poisson_trains[m].rescale(t_stop.units),
            embedded_coinc[m].rescale(t_stop.units))) * t_stop.units)
        for m in range(len(rates_b))]

    # Convert back sip_process and coinc_times from pq.Quantity objects to
    # neo.SpikeTrain objects
    sip_process = [
        neo.SpikeTrain(t, t_start=t_start, t_stop=t_stop).rescale(t_stop.units)
        for t in sip_process]
    coinc_times = [
        neo.SpikeTrain(t, t_start=t_start, t_stop=t_stop).rescale(t_stop.units)
        for t in embedded_coinc]

    # Return the processes in the specified output_format
    if not return_coincidences:
        output = sip_process
    else:
        output = sip_process, coinc_times

    return output


def _pool_two_spiketrains(spiketrain_1, spiketrain_2, extremes='inner'):
    """
    Pool the spikes of two spike trains a and b into a unique spike train.

    Parameters
    ----------
    spiketrain_1, spiketrain_2 : neo.SpikeTrain
        Spiketrains to be pooled.
    extremes : {'inner', 'outer'}, optional
        Only spikes of a and b in the specified extremes are considered.
        * 'inner': pool all spikes from max(a.tstart_ b.t_start) to
           min(a.t_stop, b.t_stop)
        * 'outer': pool all spikes from min(a.tstart_ b.t_start) to
           max(a.t_stop, b.t_stop)
        Default: 'inner'

    Returns
    -------
    neo.SpikeTrain
        containing all spikes of the two input spiketrains falling in the
        specified extremes
    """

    unit = spiketrain_1.units
    spiketrain_2 = spiketrain_2.rescale(unit)
    times_1_dimless = spiketrain_1.magnitude
    times_2_dimless = spiketrain_2.rescale(unit).magnitude
    times = np.sort(np.concatenate((times_1_dimless, times_2_dimless)))

    if extremes == 'outer':
        t_start = min(spiketrain_1.t_start, spiketrain_2.t_start)
        t_stop = max(spiketrain_1.t_stop, spiketrain_2.t_stop)
    elif extremes == 'inner':
        t_start = max(spiketrain_1.t_start, spiketrain_2.t_start)
        t_stop = min(spiketrain_1.t_stop, spiketrain_2.t_stop)
        times = times[times > t_start.magnitude]
        times = times[times < t_stop.magnitude]
    else:
        raise ValueError(
            'extremes (%s) can only be "inner" or "outer"' % extremes)

    return neo.SpikeTrain(times=times, units=unit, t_start=t_start,
                          t_stop=t_stop)


def _sample_int_from_pdf(probability_density, n_samples):
    """
    Draw n independent samples from the set {0,1,...,L}, where L=len(a)-1,
    according to the probability distribution a.
    a[j] is the probability to sample j, for each j from 0 to L.


    Parameters
    ----------
    probability_density : np.ndarray
        Probability vector (i..e array of sum 1) that at each entry j carries
        the probability to sample j (j=0,1,...,len(a)-1).
    n_samples : int
        Number of samples generated with the function

    Returns
    -------
    np.ndarray
        An array of n samples taking values between `0` and `n=len(a)-1`.
    """

    cumulative_distribution = np.cumsum(probability_density)
    random_uniforms = np.random.uniform(0, 1, size=n_samples)
    random_uniforms = np.repeat(np.expand_dims(random_uniforms, axis=1),
                                repeats=len(probability_density),
                                axis=1)
    return (cumulative_distribution < random_uniforms).sum(axis=1)


def _mother_proc_cpp_stat(
        amplitude_distribution, t_stop, rate, t_start=0 * pq.ms):
    """
    Generate the hidden ("mother") Poisson process for a Compound Poisson
    Process (CPP).


    Parameters
    ----------
    amplitude_distribution : np.ndarray
        CPP's amplitude distribution :math:`A`. `A[j]` represents the
        probability of a synchronous event of size `j` among the generated
        spike trains. The sum over all entries of :math:`A` must be equal to
        one.
    t_stop : pq.Quantity
        The stopping time of the mother process
    rate : pq.Quantity
        Homogeneous rate of the n spike trains that will be generated by the
        CPP function
    t_start : pq.Quantity, optional
        The starting time of the mother process
        Default: 0 pq.ms

    Returns
    -------
    Poisson spike train representing the mother process generating the CPP
    """
    n_spiketrains = len(amplitude_distribution) - 1
    # expected amplitude
    exp_amplitude = np.dot(
        amplitude_distribution, np.arange(n_spiketrains + 1))
    # expected rate of the mother process
    exp_mother_rate = (n_spiketrains * rate) / exp_amplitude
    return StationaryPoissonProcess(
        rate=exp_mother_rate, t_stop=t_stop, t_start=t_start
    ).generate_spiketrain()


def _cpp_hom_stat(amplitude_distribution, t_stop, rate, t_start=0 * pq.ms):
    """
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    amplitude_distribution : np.ndarray
        CPP's amplitude distribution :math:`A`. `A[j]` represents the
        probability of a synchronous event of size `j` among the generated
        spike trains. The sum over all entries of :math:`A` must be equal to
        one.
    t_stop : pq.Quantity
        The end time of the output spike trains
    rate : pq.Quantity
        Average rate of each spike train generated
    t_start : pq.Quantity, optional
        The start time of the output spike trains
        Default: 0 pq.ms

    Returns
    -------
    list of neo.SpikeTrain
        with n elements, having average firing rate r and correlated such to
        form a CPP with amplitude distribution a
    """

    # Generate mother process and associated spike labels
    mother = _mother_proc_cpp_stat(
        amplitude_distribution=amplitude_distribution,
        t_stop=t_stop, rate=rate, t_start=t_start)
    labels = _sample_int_from_pdf(amplitude_distribution, len(mother))
    n_spiketrains = len(amplitude_distribution) - 1
    # Number of trains in output

    spiketrains = [[]] * n_spiketrains
    try:  # Faster but more memory-consuming approach
        n_mother_trains = len(mother)  # number of spikes in the mother process
        spike_matrix = np.zeros((n_spiketrains, n_mother_trains), dtype=bool)
        # for each spike, take its label
        for spike_id, label in enumerate(labels):
            # choose label random trains
            train_ids = np.random.choice(n_spiketrains, label, replace=False)
            # and set the spike matrix for that train
            for train_id in train_ids:
                spike_matrix[train_id, spike_id] = True  # and spike to True

        for train_id, row in enumerate(spike_matrix):
            spiketrains[train_id] = mother[row].view(pq.Quantity)

    except MemoryError:  # Slower (~2x) but less memory-consuming approach
        print('memory case')
        for mother_spiketrain, label in zip(mother, labels):
            train_ids = np.random.choice(n_spiketrains, label)
            for train_id in train_ids:
                spiketrains[train_id].append(mother_spiketrain)

    return [neo.SpikeTrain(times=spiketrain, t_start=t_start, t_stop=t_stop)
            for spiketrain in spiketrains]


def _cpp_het_stat(amplitude_distribution, t_stop, rates, t_start=0.*pq.ms):
    """
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    amplitude_distribution : np.ndarray
        CPP's amplitude distribution :math:`A`. `A[j]` represents the
        probability of a synchronous event of size `j` among the generated
        spike trains. The sum over all entries of :math:`A` must be equal to
        one.
    t_stop : pq.Quantity
        The end time of the output spike trains
    rates : pq.Quantity
        Array of firing rates of each spike train generated with
    t_start : pq.Quantity, optional
        The start time of the output spike trains
        Default: 0 pq.ms

    Returns
    -------
    list of neo.SpikeTrain
        List of neo.SpikeTrains with different firing rates, forming
        a CPP with amplitude distribution `A`.
    """

    # Computation of Parameters of the two CPPs that will be merged
    # (uncorrelated with heterog. rates + correlated with homog. rates)
    n_spiketrains = len(rates)  # number of output spike trains
    # amplitude expectation
    expected_amplitude = np.dot(
        amplitude_distribution, np.arange(n_spiketrains + 1))
    r_sum = np.sum(rates)  # sum of all output firing rates
    r_min = np.min(rates)  # minimum of the firing rates

    # rate of the uncorrelated CPP
    r_uncorrelated = r_sum - n_spiketrains * r_min
    # rate of the correlated CPP
    r_correlated = r_sum / expected_amplitude - r_uncorrelated
    # rate of the hidden mother process
    r_mother = r_uncorrelated + r_correlated

    # Check the analytical constraint for the amplitude distribution
    if amplitude_distribution[1] < (r_uncorrelated / r_mother).rescale(
            pq.dimensionless).magnitude:
        raise ValueError('A[1] too small / A[i], i>1 too high')

    # Compute the amplitude distribution of the correlated CPP, and generate it
    amplitude_distribution = \
        amplitude_distribution * (r_mother / r_correlated).magnitude
    amplitude_distribution[1] = \
        amplitude_distribution[1] - r_uncorrelated / r_correlated
    compound_poisson_spiketrains = _cpp_hom_stat(
        amplitude_distribution, t_stop, r_min, t_start)

    # Generate the independent heterogeneous Poisson processes
    poisson_spiketrains = \
        [StationaryPoissonProcess(
            rate=rate - r_min, t_start=t_start, t_stop=t_stop
                                  ).generate_spiketrain()
         for rate in rates]

    # Pool the correlated CPP and the corresponding Poisson processes
    return [_pool_two_spiketrains(compound_poisson_spiketrain,
                                  poisson_spiketrain)
            for compound_poisson_spiketrain, poisson_spiketrain
            in zip(compound_poisson_spiketrains, poisson_spiketrains)]


@deprecated_alias(A='amplitude_distribution')
def compound_poisson_process(
        rate, amplitude_distribution, t_stop, shift=None, t_start=0 * pq.ms):
    """
    Generate a Compound Poisson Process (CPP; see
    :cite:`generation-Staude2010_327`) with a given `amplitude_distribution`
    :math:`A` and stationary marginal rates `rate`.

    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of
    `len(A)-1` spike trains with a correlation structure determined by the
    amplitude distribution :math:`A`: A[j] is the probability that a spike
    occurs synchronously in any `j` spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to `j` of the output spike trains with
    probability `A[j]`.

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    rate : pq.Quantity
        Average rate of each spike train generated. Can be:
          - a single value, all spike trains will have same rate rate
          - an array of values (of length `len(A)-1`), each indicating the
            firing rate of one process in output
    amplitude_distribution : np.ndarray or list
        CPP's amplitude distribution :math:`A`. `A[j]` represents the
        probability of a synchronous event of size `j` among the generated
        spike trains. The sum over all entries of :math:`A` must be equal to
        one.
    t_stop : pq.Quantity
        The end time of the output spike trains.
    shift : pq.Quantity, optional
        If `None`, the injected synchrony is exact.
        If shift is a `pq.Quantity`, all the spike trains are shifted
        independently by a random amount in the interval `[-shift, +shift]`.
        Default: None
    t_start : pq.Quantity, optional
        The `t_start` time of the output spike trains.
        Default: 0 pq.ms

    Returns
    -------
    list of neo.SpikeTrain
        A list of `len(A) - 1` neo.SpikeTrains with specified firing rates
        forming the CPP with amplitude distribution :math:`A`.
    """
    if not isinstance(amplitude_distribution, np.ndarray):
        amplitude_distribution = np.array(amplitude_distribution)
    # Check A is a probability distribution (it sums to 1 and is positive)
    if abs(sum(amplitude_distribution) - 1) > np.finfo('float').eps:
        raise ValueError(
            f"'amplitude_distribution' must be a probability vector: "
            f"sum(A) = {sum(amplitude_distribution)} != 1")
    if np.any(amplitude_distribution < 0):
        raise ValueError("'amplitude_distribution' must be a probability "
                         "vector with positive entries")
    # Check that the rate is not an empty pq.Quantity
    if rate.ndim == 1 and len(rate) == 0:
        raise ValueError('Rate is an empty pq.Quantity array')
    # Return empty spike trains for specific parameters
    if amplitude_distribution[0] == 1 or np.sum(np.abs(rate.magnitude)) == 0:
        return [neo.SpikeTrain([] * t_stop.units,
                               t_stop=t_stop,
                               t_start=t_start)] * (
                len(amplitude_distribution) - 1)

    # Homogeneous rates
    if rate.ndim == 0:
        compound_poisson_spiketrains = _cpp_hom_stat(
            amplitude_distribution=amplitude_distribution,
            t_stop=t_stop, rate=rate,
            t_start=t_start)
    # Heterogeneous rates
    else:
        compound_poisson_spiketrains = _cpp_het_stat(
            amplitude_distribution=amplitude_distribution,
            t_stop=t_stop, rates=rate,
            t_start=t_start)

    if shift is not None:
        # Dither the output spiketrains
        compound_poisson_spiketrains = \
            [dither_spike_train(spiketrain, shift=shift, edges=True)[0]
             for spiketrain in compound_poisson_spiketrains]

    return compound_poisson_spiketrains


# Alias for :func:`compound_poisson_process`
cpp = compound_poisson_process
