# -*- coding: utf-8 -*-
"""
Identification of spectral properties in analog signals (e.g., the power
spectrum).

:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import neo
import warnings
import numpy as np
import quantities as pq
import scipy.signal

from elephant.utils import deprecated_alias

__all__ = [
    "welch_psd",
    "welch_coherence"
]


@deprecated_alias(num_seg='n_segments', len_seg='len_segment',
                  freq_res='frequency_resolution')
def welch_psd(signal, n_segments=8, len_segment=None,
              frequency_resolution=None, overlap=0.5, fs=1.0, window='hanning',
              nfft=None, detrend='constant', return_onesided=True,
              scaling='density', axis=-1):
    """
    Estimates power spectrum density (PSD) of a given `neo.AnalogSignal`
    using Welch's method.

    The PSD is obtained through the following steps:

    1. Cut the given data into several overlapping segments. The degree of
       overlap can be specified by parameter `overlap` (default is 0.5,
       i.e. segments are overlapped by the half of their length).
       The number and the length of the segments are determined according
       to the parameters `n_segments`, `len_segment` or `frequency_resolution`.
       By default, the data is cut into 8 segments;

    2. Apply a window function to each segment. Hanning window is used by
       default. This can be changed by giving a window function or an
       array as parameter `window` (see Notes [2]);

    3. Compute the periodogram of each segment;

    4. Average the obtained periodograms to yield PSD estimate.

    Parameters
    ----------
    signal : neo.AnalogSignal or pq.Quantity or np.ndarray
        Time series data, of which PSD is estimated. When `signal` is
        `pq.Quantity` or `np.ndarray`, sampling frequency should be given
        through the keyword argument `fs`. Otherwise, the default value is
        used (`fs` = 1.0).
    n_segments : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_segment` or `frequency_resolution` is
        given.
        Default: 8.
    len_segment : int, optional
        Length of segments. This parameter is ignored if `frequency_resolution`
        is given. If None, it will be determined from other parameters.
        Default: None.
    frequency_resolution : pq.Quantity or float, optional
        Desired frequency resolution of the obtained PSD estimate in terms of
        the interval between adjacent frequency bins. When given as a `float`,
        it is taken as frequency in Hz.
        If None, it will be determined from other parameters.
        Default: None.
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped).
    fs : pq.Quantity or float, optional
        Specifies the sampling frequency of the input time series. When the
        input is given as a `neo.AnalogSignal`, the sampling frequency is
        taken from its attribute and this parameter is ignored.
        Default: 1.0.
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        See Notes [2].
        Default: 'hanning'.
    nfft : int, optional
        Length of the FFT used.
        See Notes [2].
        Default: None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment.
        See Notes [2].
        Default: 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data.
        If False return a two-sided spectrum.
        See Notes [2].
        Default: True.
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz. If 'spectrum', computes the power spectrum where Pxx has
        units of V**2, if `signal` is measured in V and `fs` is measured in
        Hz.
        See Notes [2].
        Default: 'density'.
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [2].
        Default: last axis (-1).

    Returns
    -------
    freqs : pq.Quantity or np.ndarray
        Frequencies associated with the power estimates in `psd`.
        `freqs` is always a vector irrespective of the shape of the input
        data in `signal`.
        If `signal` is `neo.AnalogSignal` or `pq.Quantity`, a `pq.Quantity`
        array is returned.
        Otherwise, a `np.ndarray` containing frequency in Hz is returned.
    psd : pq.Quantity or np.ndarray
        PSD estimates of the time series in `signal`.
        If `signal` is `neo.AnalogSignal`, a `pq.Quantity` array is returned.
        Otherwise, the return is a `np.ndarray`.

    Raises
    ------
    ValueError
        If `overlap` is not in the interval `[0, 1)`.

        If `frequency_resolution` is not positive.

        If `frequency_resolution` is too high for the given data size.

        If `frequency_resolution` is None and `len_segment` is not a positive
        number.

        If `frequency_resolution` is None and `len_segment` is greater than the
        length of data at `axis`.

        If both `frequency_resolution` and `len_segment` are None and
        `n_segments` is not a positive number.

        If both `frequency_resolution` and `len_segment` are None and
        `n_segments` is greater than the length of data at `axis`.

    Notes
    -----
    1. The computation steps used in this function are implemented in
       `scipy.signal` module, and this function is a wrapper which provides
       a proper set of parameters to `scipy.signal.welch` function.
    2. The parameters `window`, `nfft`, `detrend`, `return_onesided`,
       `scaling`, and `axis` are directly passed to the `scipy.signal.welch`
       function. See the respective descriptions in the docstring of
       `scipy.signal.welch` for usage.
    3. When only `n_segments` is given, parameter `nperseg` of
       `scipy.signal.welch` function is determined according to the expression

       `signal.shape[axis] / (n_segments - overlap * (n_segments - 1))`

       converted to integer.

    See Also
    --------
    scipy.signal.welch
    welch_cohere

    """

    # initialize a parameter dict (to be given to scipy.signal.welch()) with
    # the parameters directly passed on to scipy.signal.welch()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'return_onesided': return_onesided,
              'scaling': scaling, 'axis': axis}

    # add the input data to params. When the input is AnalogSignal, the
    # data is added after rolling the axis for time index to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))
    params['x'] = data

    # if the data is given as AnalogSignal, use its attribute to specify
    # the sampling frequency
    if hasattr(signal, 'sampling_rate'):
        params['fs'] = signal.sampling_rate.rescale('Hz').magnitude
    else:
        params['fs'] = fs

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if frequency_resolution is not None:
        if frequency_resolution <= 0:
            raise ValueError("frequency_resolution must be positive")
        if isinstance(frequency_resolution, pq.quantity.Quantity):
            dF = frequency_resolution.rescale('Hz').magnitude
        else:
            dF = frequency_resolution
        nperseg = int(params['fs'] / dF)
        if nperseg > data.shape[axis]:
            raise ValueError("frequency_resolution is too high for the given "
                             "data size")
    elif len_segment is not None:
        if len_segment <= 0:
            raise ValueError("len_seg must be a positive number")
        elif data.shape[axis] < len_segment:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_segment
    else:
        if n_segments <= 0:
            raise ValueError("n_segments must be a positive number")
        elif data.shape[axis] < n_segments:
            raise ValueError("n_segments must be smaller than the data length")
        # when only *n_segments* is given, *nperseg* is determined by solving
        # the following equation:
        #  n_segments * nperseg - (n_segments-1) * overlap * nperseg =
        #     data.shape[-1]
        #  --------------------   ===============================  ^^^^^^^^^^^
        # summed segment lengths        total overlap              data length
        nperseg = int(data.shape[axis] / (n_segments - overlap * (
            n_segments - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, psd = scipy.signal.welch(**params)

    # attach proper units to return values
    if isinstance(signal, pq.quantity.Quantity):
        if 'scaling' in params and params['scaling'] == 'spectrum':
            psd = psd * signal.units * signal.units
        else:
            psd = psd * signal.units * signal.units / pq.Hz
        freqs = freqs * pq.Hz

    return freqs, psd


def multitaper_psd(signal, fs=1, nw=4, num_tapers=None,
                   peak_resolution=None):
    """
    Estimates power spectrum density (PSD) of a given 'neo.AnalogSignal'
    using Multitaper method

    The PSD is obtained through the following steps:

    1. Calculate 'num_tapers' approximately independent estimates of the
       spectrum by multiplying the signal with the discrete prolate spheroidal
       functions (also known as Slepian function) and calculate the PSD of the
       products

    2. Average the approximately independent estimates to decrease overall
       variance of the estimates

    Parameters
    ----------
    signal : neo.AnalogSignal
        Time series data of which PSD is estimated. When `signal` is np.ndarray
        sampling frequency should be given through keyword argument `fs`.
    fs : float, optional
        Specifies the sampling frequency of the input time series
        Default: 1.0
    nw : float, optional
        Time bandwidth product
        Default: 4.0
    num_tapers : int, optional
        Number of tapers used in 1. to obtain estimate of PSD. By default
        [2*nw] - 1 is chosen.
        Default: None
    peak_resolution : float, optional
        Desired frequency resolution of the obtained PSD estimate. When given
        as a `float`, it is taken as frequency in Hz.
        If None, it will be determined from other parameters.
        Default: None.

    Returns
    -------
    freqs : np.ndarray
        Frequencies associated with power estimate in `psd`
    psd : np.ndarray
        PSD estimate of the time series in `signal`
    """

    # When the input is AnalogSignal, the data is added after rolling the axis
    # for time index to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))

    # If the data is given as AnalogSignal, use its attribute to specify the
    # sampling frequency
    if hasattr(signal, 'sampling_rate'):
        fs = signal.sampling_rate.rescale('Hz').magnitude

    # If fs and frequency resolution is pq Quantity get magnitude
    if isinstance(fs, pq.quantity.Quantity):
        fs = fs.rescale('Hz').magnitude
    if isinstance(peak_resolution, pq.quantity.Quantity):
        peak_resolution = peak_resolution.rescale('Hz').magnitude

    # Number of data points in time series
    if data.ndim == 1:
        length_signal = np.shape(data)[0]
    else:
        length_signal = np.shape(data)[1]

    # Determine time-halfbandwidth product from given parameters
    if peak_resolution is not None:
        if peak_resolution <= 0:
            raise ValueError("peak_resolution must be positive")
        else:
            nw = length_signal / fs * peak_resolution / 2

    if num_tapers is None:
        num_tapers = np.floor(2*nw).astype(int) - 1
    else:
        if not isinstance(num_tapers, int):
            raise TypeError("num_tapers must be integer")
        elif num_tapers <= 0:
            raise ValueError("num_tapers must be positive")

    print(f'Number of tapers: {num_tapers}')

    # Generate frequencies
    freqs = np.fft.rfftfreq(length_signal, d=1/fs)

    # Get slepian functions
    slepain_fcts = scipy.signal.windows.dpss(M=length_signal,
                                             NW=nw,
                                             Kmax=num_tapers,
                                             sym='False')

    # Calculate approximately independent spectrum estimates
    if data.ndim == 1:
        tapered_signal = data * slepain_fcts
    else:
        tapered_signal = data[:, np.newaxis] * slepain_fcts

    # Determine Fourier transform of tapered signal
    spectrum_estimates = np.abs(np.fft.rfft(tapered_signal, axis=-1))**2
    spectrum_estimates[..., 1:] *= 2

    # Average Fourier transform windowed signal
    psd = np.mean(spectrum_estimates, axis=-2) / fs

    # Attach proper units to return values
    if isinstance(signal, pq.quantity.Quantity):
        psd = psd * signal.units * signal.units / pq.Hz
        freqs = freqs * pq.Hz

    return freqs, psd


def multitaper_cross_spectrum(signals, fs=1, nw=4,
                              num_tapers=None, peak_resolution=None):
    """
    Estimates power spectrum density (PSD) of a given 'neo.AnalogSignal'
    using Multitaper method

    The PSD is obtained through the following steps:

    1. Calculate 'num_tapers' approximately independent estimates of the
       spectrum by multiplying the signal with the discrete prolate spheroidal
       functions (also known as Slepian function) and calculate the PSD of the
       products

    2. Average the approximately independent estimates to decrease overall
       variance of the estimates

    Parameters
    ----------
    signal : neo.AnalogSignal
        Time series data of which PSD is estimated. When `signal` is np.ndarray
        sampling frequency should be given through keyword argument `fs`.
    fs : float, optional
        Specifies the sampling frequency of the input time series
        Default: 1.0
    nw : float, optional
        Time bandwidth product
        Default: 4.0
    num_tapers : int, optional
        Number of tapers used in 1. to obtain estimate of PSD. By default
        [2*nw] - 1 is chosen.
        Default: None
    peak_resolution : float, optional
        Desired frequency resolution of the obtained PSD estimate. When given
        as a `float`, it is taken as frequency in Hz.
        If None, it will be determined from other parameters.
        Default: None.

    Returns
    -------
    freqs : np.ndarray
        Frequencies associated with power estimate in `psd`
    psd : np.ndarray
        PSD estimate of the time series in `signal`
    """

    # When the input is AnalogSignal, the data is added after rolling the axis
    # for time index to the last
    data = np.asarray(signals)
    if isinstance(signals, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))

    # If the data is given as AnalogSignal, use its attribute to specify the
    # sampling frequency
    if hasattr(signals, 'sampling_rate'):
        fs = signals.sampling_rate.rescale('Hz').magnitude

    # If fs and frequency resolution is pq Quantity get magnitude
    if isinstance(fs, pq.quantity.Quantity):
        fs = fs.rescale('Hz').magnitude
    if isinstance(peak_resolution, pq.quantity.Quantity):
        peak_resolution = peak_resolution.rescale('Hz').magnitude

    # Number of data points in time series
    length_signal = np.shape(data)[0]

    # Determine time-halfbandwidth product from given parameters
    if peak_resolution is not None:
        if peak_resolution <= 0:
            raise ValueError("peak_resolution must be positive")
        else:
            nw = length_signal / fs * peak_resolution / 2

    if num_tapers is None:
        num_tapers = np.floor(2*nw).astype(int) - 1
    else:
        if not isinstance(num_tapers, int):
            raise TypeError("num_tapers must be integer")
        elif num_tapers <= 0:
            raise ValueError("num_tapers must be positive")

    print(f'Number of tapers: {num_tapers}')

    # Generate frequencies
    freqs = np.fft.fftfreq(length_signal, d=1/fs)

    # Get slepian functions
    slepian_fcts = scipy.signal.windows.dpss(M=length_signal,
                                             NW=nw,
                                             Kmax=num_tapers,
                                             sym='False')

    # Calculate approximately independent spectrum estimates
    tapered_signal = signals.T * slepian_fcts[:, np.newaxis]
    tapered_signal = tapered_signal.T  # Time, dimension, taper

    # Determine Fourier transform of tapered signal
    spectrum_estimates = np.fft.fft(tapered_signal, axis=0)

    # temp = np.multiply.outer(spectrum_estimates, np.conjugate(spectrum_estimates))
    temp = spectrum_estimates[:, np.newaxis, :, :] * \
           np.conjugate(spectrum_estimates[:, :, np.newaxis, :])


    # Average Fourier transform windowed signal
    amp_cross_spec = np.mean(temp, axis=-1) / fs

    phase_cross_spec = np.angle(np.mean(temp, axis=-1))

    # Attach proper units to return values
    if isinstance(signals, pq.quantity.Quantity):
        cross_spec = amp_cross_spec * signals.units * signals.units / pq.Hz
        freqs = freqs * pq.Hz

    return freqs, phase_cross_spec, amp_cross_spec


def multitaper_coherence(signals, fs=1, nw=4, num_tapers=None,
                         peak_resolution=None):

    freqs, _, Pxy = multitaper_cross_spectrum(signals,fs, nw, num_tapers,
                                          peak_resolution)

    coherency = np.abs(Pxy[:512, 0, 1]) ** 2 / \
                (Pxy[:512, 0, 0] * Pxy[:512, 1, 1])

    phase_lag = np.angle(2*Pxy[:512, 0, 1])


    return freqs, coherency, phase_lag


@deprecated_alias(x='signal_i', y='signal_j', num_seg='n_segments',
                  len_seg='len_segment', freq_res='frequency_resolution')
def welch_coherence(signal_i, signal_j, n_segments=8, len_segment=None,
                    frequency_resolution=None, overlap=0.5, fs=1.0,
                    window='hanning', nfft=None, detrend='constant',
                    scaling='density', axis=-1):
    r"""
    Estimates coherence between a given pair of analog signals.

    The estimation is performed with Welch's method: the given pair of data
    are cut into short segments, cross-spectra are calculated for each pair of
    segments, and the cross-spectra are averaged and normalized by respective
    auto-spectra.

    By default, the data are cut into 8 segments with 50% overlap between
    neighboring segments. These numbers can be changed through respective
    parameters.

    Parameters
    ----------
    signal_i : neo.AnalogSignal or pq.Quantity or np.ndarray
        First time series data of the pair between which coherence is
        computed.
    signal_j : neo.AnalogSignal or pq.Quantity or np.ndarray
        Second time series data of the pair between which coherence is
        computed.
        The shapes and the sampling frequencies of `signal_i` and `signal_j`
        must be identical. When `signal_i` and `signal_j` are not
        `neo.AnalogSignal`, sampling frequency should be specified through the
        keyword argument `fs`. Otherwise, the default value is used
        (`fs` = 1.0).
    n_segments : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_seg` or `frequency_resolution` is given.
        Default: 8.
    len_segment : int, optional
        Length of segments. This parameter is ignored if `frequency_resolution`
        is given. If None, it is determined from other parameters.
        Default: None.
    frequency_resolution : pq.Quantity or float, optional
        Desired frequency resolution of the obtained coherence estimate in
        terms of the interval between adjacent frequency bins. When given as a
        `float`, it is taken as frequency in Hz.
        If None, it is determined from other parameters.
        Default: None.
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped).
    fs : pq.Quantity or float, optional
        Specifies the sampling frequency of the input time series. When the
        input time series are given as `neo.AnalogSignal`, the sampling
        frequency is taken from their attribute and this parameter is ignored.
        Default: 1.0.
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        See Notes [1].
        Default: 'hanning'.
    nfft : int, optional
        Length of the FFT used.
        See Notes [1].
        Default: None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment.
        See Notes [1].
        Default: 'constant'.
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz. If 'spectrum', computes the power spectrum where Pxx has
        units of V**2, if `signal` is measured in V and `fs` is measured in
        Hz.
        See Notes [1].
        Default: 'density'.
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [1].
        Default: last axis (-1).

    Returns
    -------
    freqs : pq.Quantity or np.ndarray
        Frequencies associated with the estimates of coherency and phase lag.
        `freqs` is always a vector irrespective of the shape of the input
        data. If `signal_i` and `signal_j` are `neo.AnalogSignal` or
        `pq.Quantity`, a `pq.Quantity` array is returned. Otherwise, a
        `np.ndarray` containing frequency in Hz is returned.
    coherency : np.ndarray
        Estimate of coherency between the input time series. For each
        frequency, coherency takes a value between 0 and 1, with 0 or 1
        representing no or perfect coherence, respectively.
        When the input arrays `signal_i` and `signal_j` are multi-dimensional,
        `coherency` is of the same shape as the inputs, and the frequency is
        indexed depending on the type of the input. If the input is
        `neo.AnalogSignal`, the first axis indexes frequency. Otherwise,
        frequency is indexed by the last axis.
    phase_lag : pq.Quantity or np.ndarray
        Estimate of phase lag in radian between the input time series. For
        each frequency, phase lag takes a value between :math:`-\pi` and
        :math:`\pi`, with positive values meaning phase precession of
        `signal_i` ahead of `signal_j`, and vice versa. If `signal_i` and
        `signal_j` are `neo.AnalogSignal` or `pq.Quantity`, a `pq.Quantity`
        array is returned. Otherwise, a `np.ndarray` containing phase lag in
        radian is returned. The axis for frequency index is determined in the
        same way as for `coherency`.

    Raises
    ------
    ValueError
        Same as in :func:`welch_psd`.

    Notes
    -----
    1. The computation steps used in this function are implemented in
       `scipy.signal` module, and this function is a wrapper which provides
       a proper set of parameters to `scipy.signal.welch` function.
    2. The parameters `window`, `nfft`, `detrend`, `return_onesided`,
       `scaling`, and `axis` are directly passed to the `scipy.signal.welch`
       function. See the respective descriptions in the docstring of
       `scipy.signal.welch` for usage.
    3. When only `n_segments` is given, parameter `nperseg` of
       `scipy.signal.welch` function is determined according to the expression

       `signal.shape[axis] / (n_segments - overlap * (n_segments - 1))`

       converted to integer.

    See Also
    --------
    welch_psd

    """

    # TODO: code duplication with welch_psd()

    # initialize a parameter dict for scipy.signal.csd()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'scaling': scaling, 'axis': axis}

    # When the input is AnalogSignal, the axis for time index is rolled to
    # the last
    xdata = np.asarray(signal_i)
    ydata = np.asarray(signal_j)
    if isinstance(signal_i, neo.AnalogSignal):
        xdata = np.rollaxis(xdata, 0, len(xdata.shape))
        ydata = np.rollaxis(ydata, 0, len(ydata.shape))

    # if the data is given as AnalogSignal, use its attribute to specify
    # the sampling frequency
    if hasattr(signal_i, 'sampling_rate'):
        params['fs'] = signal_i.sampling_rate.rescale('Hz').magnitude
    else:
        params['fs'] = fs

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if frequency_resolution is not None:
        if isinstance(frequency_resolution, pq.quantity.Quantity):
            dF = frequency_resolution.rescale('Hz').magnitude
        else:
            dF = frequency_resolution
        nperseg = int(params['fs'] / dF)
        if nperseg > xdata.shape[axis]:
            raise ValueError("frequency_resolution is too high for the given"
                             "data size")
    elif len_segment is not None:
        if len_segment <= 0:
            raise ValueError("len_seg must be a positive number")
        elif xdata.shape[axis] < len_segment:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_segment
    else:
        if n_segments <= 0:
            raise ValueError("n_segments must be a positive number")
        elif xdata.shape[axis] < n_segments:
            raise ValueError("n_segments must be smaller than the data length")
        # when only *n_segments* is given, *nperseg* is determined by solving
        # the following equation:
        #  n_segments * nperseg - (n_segments-1) * overlap * nperseg =
        #      data.shape[-1]
        #  -------------------    ===============================  ^^^^^^^^^^^
        # summed segment lengths        total overlap              data length
        nperseg = int(xdata.shape[axis] / (n_segments - overlap * (
            n_segments - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, Pxx = scipy.signal.welch(xdata, **params)
    _, Pyy = scipy.signal.welch(ydata, **params)
    _, Pxy = scipy.signal.csd(xdata, ydata, **params)

    coherency = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    phase_lag = np.angle(Pxy)

    # attach proper units to return values
    if isinstance(signal_i, pq.quantity.Quantity):
        freqs = freqs * pq.Hz
        phase_lag = phase_lag * pq.rad

    # When the input is AnalogSignal, the axis for frequency index is
    # rolled to the first to comply with the Neo convention about time axis
    if isinstance(signal_i, neo.AnalogSignal):
        coherency = np.rollaxis(coherency, -1)
        phase_lag = np.rollaxis(phase_lag, -1)

    return freqs, coherency, phase_lag


def welch_cohere(*args, **kwargs):
    warnings.warn("'welch_cohere' is deprecated; use 'welch_coherence'",
                  DeprecationWarning)


# if __name__ == "__main__":
signals = np.random.normal(0, 1, size=(1000, 2))
freq_m, psd_1 = multitaper_psd(signals[:, 0], fs=1000, nw=4)
freq_m, psd_2 = multitaper_psd(signals[:, 1], fs=1000, nw=4)
freq, phase, amp = multitaper_cross_spectrum(signals, fs=1000, nw=4)


print(freq_m)


def _generate_ground_truth(length_2d=30000):
    order = 2
    signal = np.zeros((2, length_2d + order))

    weights_1 = np.array([[0.9, 0], [0.9, -0.8]])
    weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]])

    weights = np.stack((weights_1, weights_2))

    noise_covariance = np.array([[1., 0.0], [0.0, 1.]])

    for i in range(length_2d):
        for lag in range(order):
            signal[:, i + order] += np.dot(weights[lag],
                                           signal[:, i + 1 - lag])
        rnd_var = np.random.multivariate_normal([0, 0],
                                                noise_covariance)
        signal[:, i + order] += rnd_var

    signal = signal[:, 2:]

    # Return signals as Nx2
    return signal.T


test_data = _generate_ground_truth(length_2d=2**10)

fx, Pxx = welch_psd(test_data[:, 0])
fy, Pyy = welch_psd(test_data[:, 1])

fc, Coh, _ = welch_coherence(test_data[:, 0], test_data[:, 1], frequency_resolution=0.005)

fm, Pxxm = multitaper_psd(test_data.T)

import matplotlib.pyplot as plt

plt.figure()
plt.semilogy(fx, Pxx, label="Pxx")
plt.semilogy(fy, Pyy, label="Pyy")
plt.semilogy(fm, Pxxm[0], label="Pxx Multitaper")
plt.semilogy(fm, Pxxm[1], label="Pyy Multitaper")
plt.legend()
plt.show()




fcs, _, Pcs = multitaper_cross_spectrum(test_data, num_tapers=20)

plt.figure()
plt.semilogy(fm, Pxxm[0], 'k', label="Pxx Multitaper")
plt.semilogy(fm, Pxxm[1], 'g', label="Pyy Multitaper")
plt.semilogy(fcs[:512], 2*Pcs[:512, 0, 0],'r:', label="Pxx Multitaper")
plt.semilogy(fcs[:512], 2*Pcs[:512, 1, 1], 'b:', label="Pyy Multitaper")
plt.legend()
plt.show()


plt.figure()
plt.plot(fc, Coh, label="Welch Coh")
plt.plot(fcs[:512], np.abs(Pcs[:512, 0, 1])**2 / (Pcs[:512, 0, 0] * Pcs[:512, 1, 1]), 'b:', label="Multitaper Coh")
plt.legend()
plt.show()