# -*- coding: utf-8 -*-
"""
docstring goes here.
:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
import scipy.signal
import quantities as pq
import neo


def welchpsd(data, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
             **kargs):
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
            array as parameter *window* (for details, see the documentation of
            scipy.signal.welch())
        3. compute the periodogram of each segment
        4. average the obtained periodograms to yield PSD estimate
    These steps are implemented in scipy.signal, and this function is a wrapper
    which provides a proper set of parameters to scipy.signal.welch(). Some
    parameters for scipy.signal.welch(), such as *nfft*, *detrend*, *window*,
    *return_onesided* and *scaling*, also works for this function.

    Parameters
    ----------
    data: AnalogSignal or AnalogSignalArray
        time series data, of which PSD is estimated. When an AnalogSignalArray
        is given, the function computes PSD estimates of single AnalogSignals
        in the array
    num_seg: int
        number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if *len_seg* or *freq_res* is given. Default is 8
    len_seg: int
        length of segments. This parameter is ignored if *freq_res* is given.
        Default is None
    freq_res: Quantity or float
        desired frequency resolution of the obtained PSD estimate. Default is
        None
    overlap: float
        overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap). Default is 0.5 (half overlapped)

    Returns
    -------
    psd: Quantity array
        estimated PSD. When an AnalogSignalArray is given as input, *psd* is an
        array of the same shape as the input, which contains PSD estimates of
        single AnalogSignals in the array
    freqs Quantity array
        frequencies associated with the power estimates in *psd*. *freqs* is
        always a 1-dimensional array irrespective of whether the input is given
        as AnalogSignal or AnalogSignalArray
    """

    # initialize a parameter dict (to be given to scipy.signal.welch()) with
    # the data array
    params = {'x': np.asarray(data)}

    # if parameters supported by scipy.signal.welch() are given in *kargs*, add
    # them to the dict. Some of the parameter values may be overridden
    # afterwards
    for key, val in kargs.items():
        if key in ('fs', 'nperseg', 'detrend', 'window', 'noverlap', 'nfft',
                   'return_onesided', 'scaling'):
            params[key] = val
        else:
            raise ValueError("Unsupported keyword argument '{}'".format(key))

    # specify sampling frequency if the data is given as AnalogSignal(Array)
    if isinstance(data, neo.AnalogSignal) or \
            isinstance(data, neo.AnalogSignalArray):
        params['fs'] = data.sampling_rate.rescale('Hz').magnitude

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if freq_res is not None:
        if freq_res <= 0:
            raise ValueError("freq_res must be positive")
        Fs = data.sampling_rate.rescale('Hz').magnitude
        dF = freq_res.rescale('Hz').magnitude \
            if isinstance(freq_res, pq.quantity.Quantity) else freq_res
        nperseg = int(Fs / dF)
        if nperseg > data.shape[-1]:
            raise ValueError("freq_res is too high for the given data size")
    elif len_seg is not None:
        if len_seg <= 0:
            raise ValueError("len_seg must be a positive number")
        elif data.shape[-1] < len_seg:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_seg
    else:
        if num_seg <= 0:
            raise ValueError("num_seg must be a positive number")
        elif data.shape[-1] < num_seg:
            raise ValueError("num_seg must be smaller than the data length")
        # when only *num_seg* is given, *nperseg* is determined by solving the
        # following equation:
        #  num_seg * nperseg - (num_seg-1) * overlap * nperseg = data.shape[-1]
        #  -----------------   ===============================   ^^^^^^^^^^^
        # summed segment lengths        total overlap            data length
        nperseg = int(data.shape[-1] / (num_seg - overlap * (num_seg - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, psd = scipy.signal.welch(**params)

    # attach proper units to return values
    if isinstance(data, pq.quantity.Quantity):
        if 'scaling' in params and params['scaling'] is 'spectrum':
            psd = psd * data.units * data.units
        else:
            psd = psd * data.units * data.units / pq.Hz
        freqs = freqs * pq.Hz

    return psd, freqs