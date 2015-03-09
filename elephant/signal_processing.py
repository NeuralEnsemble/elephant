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


def butter(signal, highpass_freq=None, lowpass_freq=None, order=4,
           filter_function='filtfilt', fs=1.0, axis=-1):
    """
    Butterworth filtering function for neo.AnalogSignalArray. Filter type is
    determined according to how values of `highpass_freq` and `lowpass_freq`
    are given (see Parameters section for details).

    Parameters
    ----------
    signal : AnalogSignalArray or Quantity array or NumPy ndarray
        Time series data to be filtered. When given as Quantity array or NumPy
        ndarray, the sampling frequency should be given through the keyword
        argument `fs`.
    highpass_freq, lowpass_freq : Quantity or float
        High-pass and low-pass cut-off frequencies, respectively. When given as
        float, the given value is taken as frequency in Hz.
        Filter type is determined depending on values of these arguments:
            highpass_freq only (lowpass_freq = None):    highpass filter
            lowpass_freq only (highpass_freq = None):    lowpass filter
            highpass_freq < lowpass_freq:    bandpass filter
            highpass_freq > lowpass_freq:    bandstop filter
    order : int
        Order of Butterworth filter. Default is 4.
    filter_function: string
        Filtering function to be used. Either 'filtfilt'
        (`scipy.signal.filtfilt()`) or 'lfilter' (`scipy.signal.lfilter()`). In
        most applications 'filtfilt' should be used, because it doesn't bring
        about phase shift due to filtering. Default is 'filtfilt'.
    fs : Quantity or float
        The sampling frequency of the input time series. When given as float,
        its value is taken as frequency in Hz. When the input is given as neo
        AnalogSignalArray, its attribute is used to specify the sampling
        frequency and this parameter is ignored. Default is 1.0.
    axis : int
        Axis along which filter is applied. Default is -1.

    Returns
    -------
    filtered_signal : AnalogSignalArray or Quantity array or NumPy ndarray
        Filtered input data. The shape and type is identical to those of the
        input.
    """

    def _design_butterworth_filter(Fs, hpfreq=None, lpfreq=None, order=4):
        # set parameters for filter design
        Fn = Fs / 2.
        # - filter type is determined according to the values of cut-off
        # frequencies
        if lpfreq and hpfreq:
            if hpfreq < lpfreq:
                Wn = (hpfreq / Fn, lpfreq / Fn)
                btype = 'bandpass'
            else:
                Wn = (lpfreq / Fn, hpfreq / Fn)
                btype = 'bandstop'
        elif lpfreq:
            Wn = lpfreq / Fn
            btype = 'lowpass'
        elif hpfreq:
            Wn = hpfreq / Fn
            btype = 'highpass'
        else:
            raise ValueError(
                "Either highpass_freq or lowpath_freq must be given"
            )

        # return filter coefficients
        return scipy.signal.butter(order, Wn, btype=btype)

    # design filter
    Fs = signal.sampling_rate.rescale(pq.Hz).magnitude \
        if hasattr(signal, 'sampling_rate') else fs
    Fh = highpass_freq.rescale(pq.Hz).magnitude \
        if isinstance(highpass_freq, pq.quantity.Quantity) else highpass_freq
    Fl = lowpass_freq.rescale(pq.Hz).magnitude \
        if isinstance(lowpass_freq, pq.quantity.Quantity) else lowpass_freq
    b, a = _design_butterworth_filter(Fs, Fh, Fl, order)

    # When the input is AnalogSignalArray, the axis for time index (i.e. the
    # first axis) needs to be rolled to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignalArray):
        data = np.rollaxis(data, 0, len(data.shape))

    # apply filter
    if filter_function is 'lfilter':
        filtered_data = scipy.signal.lfilter(b, a, data, axis=axis)
    elif filter_function is 'filtfilt':
        filtered_data = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        raise ValueError(
            "filter_func must to be either 'filtfilt' or 'lfilter'"
        )

    if isinstance(signal, neo.AnalogSignalArray):
        return signal.duplicate_with_new_array(filtered_data.T)
    elif isinstance(signal, pq.quantity.Quantity):
        return filtered_data * signal.units
    else:
        return filtered_data
