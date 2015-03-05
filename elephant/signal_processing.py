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


def butter(anasig, highpassfreq=None, lowpassfreq=None, order=4,
           filtfunc='filtfilt'):
    """
    Butterworth filtering function for AnalogSignal objects. Filter type is
    determined according to how values of highpassfreq and lowpassfreq are
    given (see Parameters for details).

    Parameters
    ----------
    anasig: AnalogSignal
        time series data to be filtered
    highpassfreq, lowpassfreq: Quantity or None
        high-pass and low-pass cut-off frequencies, respectively.
        Filter type is determined depending on values of these arguments:
            highpassfreq only (lowpassfreq = None):    highpass filter
            lowpassfreq only (highpassfreq = None):    lowpass filter
            highpassfreq < lowpassfreq:    bandpass filter
            highpassfreq > lowpassfreq:    bandstop filter
    order: int
        Order of Butterworth filter. Default is 4.
    filtfunc: string
        Filtering function to be used. Either 'filtfilt'
        (scipy.signal.filtfilt) or 'lfilter' (scipy.signal.lfilter). In most
        applications 'filtfilt' should be used, because it doesn't bring about
        phase shift. Default is 'filtfilt'.

    Returns
    -------
    anasig_out: AnalogSignal
        Filtered signal
    """

    def apply_filter(signal, Fs, highpassfreq=None, lowpassfreq=None, order=4,
                     filtfunc='filtfilt'):
        # set the function for filtering
        if filtfunc is 'lfilter':
            ffunc = scipy.signal.lfilter
        elif filtfunc is 'filtfilt':
            ffunc = scipy.signal.filtfilt
        else:
            raise ValueError(
                "filtfunc must to be either 'filtfilt' or 'lfilter'"
            )

        # set parameters for filter design
        Fn = Fs / 2.
        # - filter type is determined according to the values of cut-off
        # frequencies
        if lowpassfreq and highpassfreq:
            if highpassfreq < lowpassfreq:
                Wn = (highpassfreq / Fn, lowpassfreq / Fn)
                btype = 'bandpass'
            else:
                Wn = (lowpassfreq / Fn, highpassfreq / Fn)
                btype = 'bandstop'
        elif lowpassfreq:
            Wn = lowpassfreq / Fn
            btype = 'lowpass'
        elif highpassfreq:
            Wn = highpassfreq / Fn
            btype = 'highpass'
        else:
            raise ValueError(
                "Either highpassfreq or lowpathfreq must be given"
            )

        # filter design
        b, a = scipy.signal.butter(order, Wn, btype=btype)

        return ffunc(b, a, signal)

    data = anasig.magnitude
    Fs = anasig.sampling_rate.rescale(pq.Hz).magnitude
    Fh = highpassfreq.rescale(pq.Hz).magnitude \
        if isinstance(highpassfreq, pq.quantity.Quantity) else highpassfreq
    Fl = lowpassfreq.rescale(pq.Hz).magnitude \
        if isinstance(lowpassfreq, pq.quantity.Quantity) else lowpassfreq
    filtered_data = apply_filter(data, Fs, Fh, Fl, order, filtfunc)

    return anasig.duplicate_with_new_array(filtered_data)


