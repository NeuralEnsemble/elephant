# -*- coding: utf-8 -*-
"""
Identification of spectral properties in analog signals (e.g., the power
spectrum).

:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings

import numpy as np
import scipy.signal
import scipy.fftpack as fftpack
import scipy.signal.signaltools as signaltools
from scipy.signal.windows import get_window
from six import string_types
import quantities as pq
import neo


def _welch(x, y, fs=1.0, window='hanning', nperseg=256, noverlap=None,
           nfft=None, detrend='constant', scaling='density', axis=-1):
    """
    A helper function to estimate cross spectral density using Welch's method.

    Welch's method [1]_ computes an estimate of the cross spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the cross-periodograms.

    Parameters
    ----------
    x : list or tuple or np.ndarray
        Time series of measurement values for the first signal.
        It will be converted to `np.ndarray`.
    y : list or tuple or np.ndarray
        Time series of measurement values for the second signal.
        It will be converted to `np.ndarray`.
    fs : float, optional
        Sampling frequency of the `x` and `y` time series in units of Hz.
        Default: 1.0.
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        If `window` is a string, see `scipy.signal.get_window` for a list of
        windows and required parameters.
        If `window` is tuple or `np.ndarray`, it will be used directly as the
        window, and its length will be used for `nperseg`. If not string,
        it will be treated as `np.ndarray`, and the result must be a vector
        with length less than or equal to the length of the last axes of `x`
        and `y`.
        Default: 'hanning'.
    nperseg : int, optional
        Length of each segment.
        Default: 256.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, `noverlap` will be set to `nperseg`/2.
        Default: None.
    nfft : int, optional
        Length of the FFT used, if a zero-padded FFT is desired.
        If None, the FFT length is `nperseg`.
        Default: None.
    detrend : str or function, optional
        Specifies how to detrend each segment.
        If `detrend` is a string, it is passed as the `type` argument to
        `scipy.signal.detrend`.
        If it is a function, it takes a segment as input and returns a
        detrended segment.
        Default: 'constant'.
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz if `x` is measured in V.
        If 'spectrum', computes the power spectrum where Pxx has units of V**2
        if `x` is measured in V.
        Default: 'density'.
    axis : int, optional
        Axis along which the periodogram is computed.
        Default: last axis (-1).

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies.
    Pxy : np.ndarray
        Cross spectral density or cross spectrum of `x` and `y`.

    Raises
    ------
    ValueError
        If `x` and `y` do not have the same shape.

        If `window` is tuple or `np.ndarray` and has more than 1 dimension.

        If length of `window` is greater than the length of the `axis`.

        If `scaling` is neither 'density' nor 'spectrum'.

        If `noverlap` is greater than or equal to `nperseg`.

        If `nfft` is less than `nperseg`.

    Warns
    -----
    UserWarning
        If `nperseg` is greater than the shape of the axis used (`axis`),
        `nperseg` will be changed to the shape of `axis`.

    Notes
    -----
    1. This function is a slightly modified version of `scipy.signal.welch`
       function, with modifications based on
       `matplotlib.mlab._spectral_helper`.
    2. An appropriate amount of overlap will depend on the choice of window
       and on your requirements. For the default 'hanning' window, an overlap
       of 50% is a reasonable trade off between accurately estimating
       the signal power, while not over counting any of the data. Narrower
       windows may require a larger overlap.
    3. If `noverlap` is 0, this method is equivalent to Bartlett's method
       [2]_.

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust., vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
    # TODO: This function should be replaced by `scipy.signal.csd()`, which
    # will appear in SciPy 0.16.0.

    # The checks for if y is x are so that we can use the same function to
    # obtain both power spectrum and cross spectrum without doing extra
    # calculations.
    same_data = y is x
    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if same_data:
        y = x
    else:
        if x.shape != y.shape:
            raise ValueError("x and y must be of the same shape.")
        y = np.asarray(y)

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape)

    if axis != -1:
        x = np.rollaxis(x, axis, len(x.shape))
        if not same_data:
            y = np.rollaxis(y, axis, len(y.shape))

    if x.shape[-1] < nperseg:
        warnings.warn('nperseg = %d, is greater than x.shape[%d] = %d, using '
                      'nperseg = x.shape[%d]'
                      % (nperseg, axis, x.shape[axis], axis))
        nperseg = x.shape[-1]

    if isinstance(window, string_types) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] > x.shape[-1]:
            raise ValueError('window is longer than x.')
        nperseg = win.shape[0]

    if scaling == 'density':
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if noverlap is None:
        noverlap = nperseg // 2
    elif noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')

    if not hasattr(detrend, '__call__'):
        detrend_func = lambda seg: signaltools.detrend(seg, type=detrend)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(seg):
            seg = np.rollaxis(seg, -1, axis)
            seg = detrend(seg)
            return np.rollaxis(seg, axis, len(seg.shape))
    else:
        detrend_func = detrend

    step = nperseg - noverlap
    indices = np.arange(0, x.shape[-1] - nperseg + 1, step)

    for k, ind in enumerate(indices):
        x_dt = detrend_func(x[..., ind:ind + nperseg])
        xft = fftpack.fft(x_dt * win, nfft)
        if same_data:
            yft = xft
        else:
            y_dt = detrend_func(y[..., ind:ind + nperseg])
            yft = fftpack.fft(y_dt * win, nfft)
        if k == 0:
            Pxy = (xft * yft.conj())
        else:
            Pxy *= k / (k + 1.0)
            Pxy += (xft * yft.conj()) / (k + 1.0)
    Pxy *= scale
    f = fftpack.fftfreq(nfft, 1.0 / fs)

    if axis != -1:
        Pxy = np.rollaxis(Pxy, -1, axis)

    return f, Pxy


def welch_psd(signal, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
              fs=1.0, window='hanning', nfft=None, detrend='constant',
              return_onesided=True, scaling='density', axis=-1):
    """
    Estimates power spectrum density (PSD) of a given `neo.AnalogSignal`
    using Welch's method.

    The PSD is obtained through the following steps:

    1. Cut the given data into several overlapping segments. The degree of
       overlap can be specified by parameter `overlap` (default is 0.5,
       i.e. segments are overlapped by the half of their length).
       The number and the length of the segments are determined according
       to the parameters `num_seg`, `len_seg` or `freq_res`. By default, the
       data is cut into 8 segments;

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
    num_seg : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_seg` or `freq_res` is given.
        Default: 8.
    len_seg : int, optional
        Length of segments. This parameter is ignored if `freq_res` is given.
        If None, it will be determined from other parameters.
        Default: None.
    freq_res : pq.Quantity or float, optional
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
        If `overlap` is not in the interval [0, 1).

        If `freq_res` is not positive.

        If `freq_res` is too high for the given data size.

        If `freq_res` is None and `len_seg` is not a positive number.

        If `freq_res` is None and `len_seg` is greater than the length of data
        on `axis`.

        If both `freq_res` and `len_seg` are None and `num_seg` is not a
        positive number.

        If both `freq_res` and `len_seg` are None and `num_seg` is greater
        than the length of data on `axis`.

    Notes
    -----
    1. The computation steps used in this function are implemented in
       `scipy.signal` module, and this function is a wrapper which provides
       a proper set of parameters to `scipy.signal.welch` function.
    2. The parameters `window`, `nfft`, `detrend`, `return_onesided`,
       `scaling`, and `axis` are directly passed to the `scipy.signal.welch`
       function. See the respective descriptions in the docstring of
       `scipy.signal.welch` for usage.
    3. When only `num_seg` is given, parameter `nperseg` of
       `scipy.signal.welch` function is determined according to the expression

       `signal.shape[axis]` / (`num_seg` - `overlap` * (`num_seg` - 1))

       converted to integer.

    See Also
    --------
    scipy.signal.welch

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
    if freq_res is not None:
        if freq_res <= 0:
            raise ValueError("freq_res must be positive")
        dF = freq_res.rescale('Hz').magnitude \
            if isinstance(freq_res, pq.quantity.Quantity) else freq_res
        nperseg = int(params['fs'] / dF)
        if nperseg > data.shape[axis]:
            raise ValueError("freq_res is too high for the given data size")
    elif len_seg is not None:
        if len_seg <= 0:
            raise ValueError("len_seg must be a positive number")
        elif data.shape[axis] < len_seg:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_seg
    else:
        if num_seg <= 0:
            raise ValueError("num_seg must be a positive number")
        elif data.shape[axis] < num_seg:
            raise ValueError("num_seg must be smaller than the data length")
        # when only *num_seg* is given, *nperseg* is determined by solving the
        # following equation:
        #  num_seg * nperseg - (num_seg-1) * overlap * nperseg = data.shape[-1]
        #  -----------------   ===============================   ^^^^^^^^^^^
        # summed segment lengths        total overlap            data length
        nperseg = int(data.shape[axis] / (num_seg - overlap * (num_seg - 1)))
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


def welch_cohere(x, y, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
                 fs=1.0, window='hanning', nfft=None, detrend='constant',
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
    x : neo.AnalogSignal or pq.Quantity or np.ndarray
        First time series data of the pair between which coherence is
        computed.
    y : neo.AnalogSignal or pq.Quantity or np.ndarray
        Second time series data of the pair between which coherence is
        computed.
        The shapes and the sampling frequencies of `x` and `y` must be
        identical. When `x` and `y` are not `neo.AnalogSignal`, sampling
        frequency should be specified through the keyword argument `fs`.
        Otherwise, the default value is used (`fs` = 1.0).
    num_seg : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_seg` or `freq_res` is given.
        Default: 8.
    len_seg : int, optional
        Length of segments. This parameter is ignored if `freq_res` is given.
        If None, it is determined from other parameters.
        Default: None.
    freq_res : pq.Quantity or float, optional
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
        data. If `x` and `y` are `neo.AnalogSignal` or `pq.Quantity`, a
        `pq.Quantity` array is returned. Otherwise, a `np.ndarray` containing
        frequency in Hz is returned.
    coherency : np.ndarray
        Estimate of coherency between the input time series. For each
        frequency, coherency takes a value between 0 and 1, with 0 or 1
        representing no or perfect coherence, respectively.
        When the input arrays `x` and `y` are multi-dimensional, `coherency`
        is of the same shape as the inputs, and the frequency is indexed
        depending on the type of the input. If the input is
        `neo.AnalogSignal`, the first axis indexes frequency. Otherwise,
        frequency is indexed by the last axis.
    phase_lag : pq.Quantity or np.ndarray
        Estimate of phase lag in radian between the input time series. For
        each frequency, phase lag takes a value between :math:`-\pi` and
        :math:`\pi`, with positive values meaning phase precession of `x`
        ahead of `y`, and vice versa. If `x` and `y` are `neo.AnalogSignal` or
        `pq.Quantity`, a `pq.Quantity` array is returned. Otherwise, a
        `np.ndarray` containing phase lag in radian is returned.
        The axis for frequency index is determined in the same way as for
        `coherency`.

    Raises
    ------
    ValueError
        If `overlap` is not in the interval [0, 1).

        If `freq_res` is not positive.

        If `freq_res` is too high for the given data size.

        If `freq_res` is None and `len_seg` is not a positive number.

        If `freq_res` is None and `len_seg` is greater than the length of data
        on `axis`.

        If both `freq_res` and `len_seg` are None and `num_seg` is not a
        positive number.

        If both `freq_res` and `len_seg` are None and `num_seg` is greater
        than the length of data on `axis`.

    Notes
    -----
    1. The parameters `window`, `nfft`, `detrend`, `scaling`, and `axis` are
       directly passed to the helper function `_welch`. See the
       respective descriptions in the docstring of `_welch` for usage.
    2. When only `num_seg` is given, parameter `nperseg` for `_welch` function
       is determined according to the expression

       `x.shape[axis]` / (`num_seg` - `overlap` * (`num_seg` - 1))

       converted to integer.

    See Also
    --------
    spectral._welch

    """

    # initialize a parameter dict (to be given to _welch()) with
    # the parameters directly passed on to _welch()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'scaling': scaling, 'axis': axis}

    # When the input is AnalogSignal, the axis for time index is rolled to
    # the last
    xdata = np.asarray(x)
    ydata = np.asarray(y)
    if isinstance(x, neo.AnalogSignal):
        xdata = np.rollaxis(xdata, 0, len(xdata.shape))
        ydata = np.rollaxis(ydata, 0, len(ydata.shape))

    # if the data is given as AnalogSignal, use its attribute to specify
    # the sampling frequency
    if hasattr(x, 'sampling_rate'):
        params['fs'] = x.sampling_rate.rescale('Hz').magnitude
    else:
        params['fs'] = fs

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if freq_res is not None:
        if freq_res <= 0:
            raise ValueError("freq_res must be positive")
        dF = freq_res.rescale('Hz').magnitude \
            if isinstance(freq_res, pq.quantity.Quantity) else freq_res
        nperseg = int(params['fs'] / dF)
        if nperseg > xdata.shape[axis]:
            raise ValueError("freq_res is too high for the given data size")
    elif len_seg is not None:
        if len_seg <= 0:
            raise ValueError("len_seg must be a positive number")
        elif xdata.shape[axis] < len_seg:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_seg
    else:
        if num_seg <= 0:
            raise ValueError("num_seg must be a positive number")
        elif xdata.shape[axis] < num_seg:
            raise ValueError("num_seg must be smaller than the data length")
        # when only *num_seg* is given, *nperseg* is determined by solving the
        # following equation:
        #  num_seg * nperseg - (num_seg-1) * overlap * nperseg = data.shape[-1]
        #  -----------------   ===============================   ^^^^^^^^^^^
        # summed segment lengths        total overlap            data length
        nperseg = int(xdata.shape[axis] / (num_seg - overlap * (num_seg - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, Pxy = _welch(xdata, ydata, **params)
    freqs, Pxx = _welch(xdata, xdata, **params)
    freqs, Pyy = _welch(ydata, ydata, **params)
    coherency = np.abs(Pxy)**2 / (np.abs(Pxx) * np.abs(Pyy))
    phase_lag = np.angle(Pxy)

    # attach proper units to return values
    if isinstance(x, pq.quantity.Quantity):
        freqs = freqs * pq.Hz
        phase_lag = phase_lag * pq.rad

    # When the input is AnalogSignal, the axis for frequency index is
    # rolled to the first to comply with the Neo convention about time axis
    if isinstance(x, neo.AnalogSignal):
        coherency = np.rollaxis(coherency, -1)
        phase_lag = np.rollaxis(phase_lag, -1)

    return freqs, coherency, phase_lag
