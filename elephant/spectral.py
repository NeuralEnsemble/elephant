# -*- coding: utf-8 -*-
"""
Identification of spectral properties in analog signals (e.g., the power spectrum)

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
import scipy.signal
import quantities as pq
import neo


def welch_psd(signal, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
              fs=1.0, window='hanning', nfft=None, detrend='constant',
              return_onesided=True, scaling='density', axis=-1):
    """
    Estimates power spectrum density (PSD) of a given AnalogSignal using
    Welch's method, which works in the following steps:
        1. cut the given data into several overlapping segments. The degree of
            overlap can be specified by parameter *overlap* (default is 0.5,
            i.e. segments are overlapped by the half of their length).
            The number and the length of the segments are determined according
            to parameter *num_seg*, *len_seg* or *freq_res*. By default, the
            data is cut into 8 segments.
        2. apply a window function to each segment. Hanning window is used by
            default. This can be changed by giving a window function or an
            array as parameter *window* (for details, see the docstring of
            `scipy.signal.welch()`)
        3. compute the periodogram of each segment
        4. average the obtained periodograms to yield PSD estimate
    These steps are implemented in `scipy.signal`, and this function is a
    wrapper which provides a proper set of parameters to
    `scipy.signal.welch()`. Some parameters for scipy.signal.welch(), such as
    `nfft`, `detrend`, `window`, `return_onesided` and `scaling`, also works
    for this function.

    Parameters
    ----------
    signal: Neo AnalogSignalArray or Quantity array or Numpy ndarray
        Time series data, of which PSD is estimated. When a Quantity array or
        Numpy ndarray is given, sampling frequency should be given through the
        keyword argument `fs`, otherwise the default value (`fs=1.0`) is used.
    num_seg: int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if *len_seg* or *freq_res* is given. Default is 8.
    len_seg: int, optional
        Length of segments. This parameter is ignored if *freq_res* is given.
        Default is None (determined from other parameters).
    freq_res: Quantity or float, optional
        Desired frequency resolution of the obtained PSD estimate in terms of
        the interval between adjacent frequency bins. When given as a float, it
        is taken as frequency in Hz. Default is None (determined from other
        parameters).
    overlap: float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap). Default is 0.5 (half-overlapped).
    fs: Quantity array or float, optional
        Specifies the sampling frequency of the input time series. When the
        input is given as an AnalogSignalArray, the sampling frequency is taken
        from its attribute and this parameter is ignored. Default is 1.0.
    window, nfft, detrend, return_onesided, scaling, axis: optional
        These arguments are directly passed on to scipy.signal.welch(). See the
        respective descriptions in the docstring of `scipy.signal.welch()` for
        usage.

    Returns
    -------
    freqs: Quantity array or Numpy ndarray
        Frequencies associated with the power estimates in `psd`. `freqs` is
        always a 1-dimensional array irrespective of the shape of the input
        data. Quantity array is returned if `signal` is AnalogSignalArray or
        Quantity array. Otherwise Numpy ndarray containing frequency in Hz is
        returned.
    psd: Quantity array or Numpy ndarray
        PSD estimates of the time series in `signal`. Quantity array is
        returned if `data` is AnalogSignalArray or Quantity array. Otherwise
        Numpy ndarray is returned.
    """

    # initialize a parameter dict (to be given to scipy.signal.welch()) with
    # the parameters directly passed on to scipy.signal.welch()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'return_onesided': return_onesided,
              'scaling': scaling, 'axis': axis}

    # add the input data to params. When the input is AnalogSignalArray, the
    # data is added after rolling the axis for time index to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignalArray):
        data = np.rollaxis(data, 0, len(data.shape))
    params['x'] = data

    # if the data is given as AnalogSignalArray, use its attribute to specify
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
        if 'scaling' in params and params['scaling'] is 'spectrum':
            psd = psd * signal.units * signal.units
        else:
            psd = psd * signal.units * signal.units / pq.Hz
        freqs = freqs * pq.Hz

    return freqs, psd
