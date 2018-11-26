# -*- coding: utf-8 -*-
'''
Basic processing procedures for analog signals (e.g., performing a z-score of a
signal, or filtering a signal).

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
'''

from __future__ import division, print_function
import numpy as np
import scipy.signal
import quantities as pq
import neo
import numpy.matlib as npm


def zscore(signal, inplace=True):
    '''
    Apply a z-score operation to one or several AnalogSignal objects.

    The z-score operation subtracts the mean :math:`\\mu` of the signal, and
    divides by its standard deviation :math:`\\sigma`:

    .. math::
         Z(x(t))= \\frac{x(t)-\\mu}{\\sigma}

    If an AnalogSignal containing multiple signals is provided, the
    z-transform is always calculated for each signal individually.

    If a list of AnalogSignal objects is supplied, the mean and standard
    deviation are calculated across all objects of the list. Thus, all list
    elements are z-transformed by the same values of :math:`\\mu` and
    :math:`\\sigma`. For AnalogSignals, each signal of the array is
    treated separately across list elements. Therefore, the number of signals
    must be identical for each AnalogSignal of the list.

    Parameters
    ----------
    signal : neo.AnalogSignal or list of neo.AnalogSignal
        Signals for which to calculate the z-score.
    inplace : bool
        If True, the contents of the input signal(s) is replaced by the
        z-transformed signal. Otherwise, a copy of the original
        AnalogSignal(s) is returned. Default: True

    Returns
    -------
    neo.AnalogSignal or list of neo.AnalogSignal
        The output format matches the input format: for each supplied
        AnalogSignal object a corresponding object is returned containing
        the z-transformed signal with the unit dimensionless.

    Use Case
    --------
    You may supply a list of AnalogSignal objects, where each object in
    the list contains the data of one trial of the experiment, and each signal
    of the AnalogSignal corresponds to the recordings from one specific
    electrode in a particular trial. In this scenario, you will z-transform the
    signal of each electrode separately, but transform all trials of a given
    electrode in the same way.

    Examples
    --------
    >>> a = neo.AnalogSignal(
    ...       np.array([1, 2, 3, 4, 5, 6]).reshape(-1,1)*mV,
    ...       t_start=0*s, sampling_rate=1000*Hz)

    >>> b = neo.AnalogSignal(
    ...       np.transpose([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]])*mV,
    ...       t_start=0*s, sampling_rate=1000*Hz)

    >>> c = neo.AnalogSignal(
    ...       np.transpose([[21, 22, 23, 24, 25, 26], [31, 32, 33, 34, 35, 36]])*mV,
    ...       t_start=0*s, sampling_rate=1000*Hz)

    >>> print zscore(a)
    [[-1.46385011]
     [-0.87831007]
     [-0.29277002]
     [ 0.29277002]
     [ 0.87831007]
     [ 1.46385011]] dimensionless

    >>> print zscore(b)
    [[-1.46385011 -1.46385011]
     [-0.87831007 -0.87831007]
     [-0.29277002 -0.29277002]
     [ 0.29277002  0.29277002]
     [ 0.87831007  0.87831007]
     [ 1.46385011  1.46385011]] dimensionless

    >>> print zscore([b,c])
    [<AnalogSignal(array([[-1.11669108, -1.08361877],
       [-1.0672076 , -1.04878252],
       [-1.01772411, -1.01394628],
       [-0.96824063, -0.97911003],
       [-0.91875714, -0.94427378],
       [-0.86927366, -0.90943753]]) * dimensionless, [0.0 s, 0.006 s],
       sampling rate: 1000.0 Hz)>,
       <AnalogSignal(array([[ 0.78170952,  0.84779261],
       [ 0.86621866,  0.90728682],
       [ 0.9507278 ,  0.96678104],
       [ 1.03523694,  1.02627526],
       [ 1.11974608,  1.08576948],
       [ 1.20425521,  1.1452637 ]]) * dimensionless, [0.0 s, 0.006 s],
       sampling rate: 1000.0 Hz)>]
    '''
    # Transform input to a list
    if type(signal) is not list:
        signal = [signal]

    # Calculate mean and standard deviation
    m = np.mean(np.concatenate(signal), axis=0)
    s = np.std(np.concatenate(signal), axis=0)

    if not inplace:
        # Create new signal instance
        result = []
        for sig in signal:
            sig_dimless = sig.duplicate_with_new_data(
                (sig.magnitude - m.magnitude) / s.magnitude) / sig.units
            result.append(sig_dimless)
    else:
        result = []
        # Overwrite signal
        for sig in signal:
            sig[:] = pq.Quantity(
                (sig.magnitude - m.magnitude) / s.magnitude,
                units=sig.units)
            sig_dimless = sig / sig.units
            result.append(sig_dimless)
    # Return single object, or list of objects
    if len(result) == 1:
        return result[0]
    else:
        return result


def cross_correlation_function(signal, ch_pairs, env=False, nlags=None):

    """
    Computes unbiased estimator of the cross-correlation function.

    Calculates the unbiased estimator of the cross-correlation function [1]_
    
    .. math::
             R(\\tau) = \\frac{1}{N-|k|} R'(\\tau) \\ ,
    
    where :math:`R'(\\tau) = \\left<x(t)y(t+\\tau)\\right>` in a pairwise 
    manner, i.e. `signal[ch_pairs[0,0]]` vs `signal2[ch_pairs[0,1]]`,
    `signal[ch_pairs[1,0]]` vs `signal2[ch_pairs[1,1]]`, and so on. The
    cross-correlation function is obtained by `scipy.signal.fftconvolve`.
    Time series in signal are zscored beforehand. Alternatively returns the
    Hilbert envelope of :math:`R(\\tau)`, which is useful to determine the 
    correlation length of oscillatory signals.

    Parameters
    -----------
    signal : neo.AnalogSignal (`nt` x `nch`)
        Signal with nt number of samples that contains nch LFP channels
    ch_pairs : list (or array with shape `(n,2)`)
        list with n channel pairs for which to compute cross-correlation,
        each element of list must contain 2 channel indices
    env : bool
        Return Hilbert envelope of cross-correlation function
        Default: False
    nlags : int
        Defines number of lags for cross-correlation function. Float will be
        rounded to nearest integer. Number of samples of output is `2*nlags+1`.
        If None, number of samples of output is equal to number of samples of
        input signal, namely `nt`
        Default: None

    Returns
    -------
    cross_corr : neo.AnalogSgnal (`2*nlag+1` x `n`)
        Pairwise cross-correlation functions for channel pairs given by
        `ch_pairs`. If `env=True`, the output is the Hilbert envelope of the
        pairwise cross-correlation function. This is helpful to compute the
        correlation length for oscillating cross-correlation functions

    Raises
    ------
    ValueError
        If the input signal is not a neo.AnalogSignal.
    ValueError
        If `ch_pairs` is not a list of channel pair indices with shape `(n,2)`.
    KeyError
        If keyword `env` is not a boolean.
    KeyError
        If `nlags` is not an integer or float larger than 0.

    Examples
    --------
        >>> dt = 0.02
        >>> N = 2018
        >>> f = 0.5
        >>> t = np.arange(N)*dt
        >>> x = np.zeros((N,2))
        >>> x[:,0] = 0.2 * np.sin(2.*np.pi*f*t)
        >>> x[:,1] = 5.3 * np.cos(2.*np.pi*f*t)
        >>> # Generate neo.AnalogSignals from x
        >>> signal = neo.AnalogSignal(x, units='mV', t_start=0.*pq.ms,
        >>>     sampling_rate=1/dt*pq.Hz, dtype=float)
        >>> rho = elephant.signal_processing.cross_correlation_function(
        >>>     signal, [0,1], nlags=150)
        >>> env = elephant.signal_processing.cross_correlation_function(
        >>>     signal, [0,1], nlags=150, env=True)
        >>> plt.plot(rho.times, rho)
        >>> plt.plot(env.times, env) # should be equal to one
        >>> plt.show()

    References
    ----------
    .. [1] Hall & River (2009) "Spectral Analysis of Signals, Spectral Element
       Method in Structural Dynamics", Eq. 2.2.3
    """

    # Make ch_pairs a 2D array
    pairs = np.array(ch_pairs)
    if pairs.ndim == 1:
        pairs = pairs[:, np.newaxis]

    # Check input
    if not isinstance(signal, neo.AnalogSignal):
        raise ValueError('Input signal is not a neo.AnalogSignal!')
    if np.shape(pairs)[1] != 2:
        pairs = pairs.T
    if np.shape(pairs)[1] != 2:
        raise ValueError('ch_pairs is not a list of channel pair indices.'\
                         'Cannot define pairs for cross-correlation.')
    if not isinstance(env, bool):
        raise KeyError('env is not a boolean!')
    if nlags is not None:
        if not isinstance(nlags, (int, float)):
            raise KeyError('nlags must be an integer or float larger than 0!')
        if nlags <= 0:
            raise KeyError('nlags must be an integer or float larger than 0!')

    # z-score analog signal and store channel time series in different arrays
    # Cross-correlation will be calculated between xsig and ysig
    xsig = np.array([zscore(signal).magnitude[:, pair[0]] \
        for pair in pairs]).T
    ysig = np.array([zscore(signal).magnitude[:, pair[1]] \
        for pair in pairs]).T

    # Define vector of lags tau
    nt, nch = np.shape(xsig)
    tau = (np.arange(nt) - nt//2)

    # Calculate cross-correlation by taking Fourier transform of signal,
    # multiply in Fourier space, and transform back. Correct for bias due
    # to zero-padding
    xcorr = np.zeros((nt, nch))
    for i in range(nch):
        xcorr[:, i] = scipy.signal.fftconvolve(xsig[:, i], ysig[::-1, i],
                                               mode='same')
    xcorr = xcorr / npm.repmat((nt-abs(tau)), nch, 1).T

    # Calculate envelope of cross-correlation function with Hilbert transform.
    # This is useful for transient oscillatory signals.
    if env:
        for i in range(nch):
            xcorr[:, i] = np.abs(scipy.signal.hilbert(xcorr[:, i]))

    # Cut off lags outside desired range
    if nlags is not None:
        nlags = int(np.round(nlags))
        tau0 = int(np.argwhere(tau == 0))
        xcorr = xcorr[tau0-nlags:tau0+nlags+1, :]

    # Return neo.AnalogSignal
    cross_corr = neo.AnalogSignal(xcorr,
                                  units='',
                                  t_start=np.min(tau)*signal.sampling_period,
                                  t_stop=np.max(tau)*signal.sampling_period,
                                  sampling_rate=signal.sampling_rate,
                                  dtype=float)
    return cross_corr


def butter(signal, highpass_freq=None, lowpass_freq=None, order=4,
           filter_function='filtfilt', fs=1.0, axis=-1):
    """
    Butterworth filtering function for neo.AnalogSignal. Filter type is
    determined according to how values of `highpass_freq` and `lowpass_freq`
    are given (see Parameters section for details).

    Parameters
    ----------
    signal : AnalogSignal or Quantity array or NumPy ndarray
        Time series data to be filtered. When given as Quantity array or NumPy
        ndarray, the sampling frequency should be given through the keyword
        argument `fs`.
    highpass_freq, lowpass_freq : Quantity or float
        High-pass and low-pass cut-off frequencies, respectively. When given as
        float, the given value is taken as frequency in Hz.
        Filter type is determined depending on values of these arguments:
            * highpass_freq only (lowpass_freq = None):    highpass filter
            * lowpass_freq only (highpass_freq = None):    lowpass filter
            * highpass_freq < lowpass_freq:    bandpass filter
            * highpass_freq > lowpass_freq:    bandstop filter
    order : int
        Order of Butterworth filter. Default is 4.
    filter_function : string
        Filtering function to be used. Either 'filtfilt'
        (`scipy.signal.filtfilt()`) or 'lfilter' (`scipy.signal.lfilter()`). In
        most applications 'filtfilt' should be used, because it doesn't bring
        about phase shift due to filtering. Default is 'filtfilt'.
    fs : Quantity or float
        The sampling frequency of the input time series. When given as float,
        its value is taken as frequency in Hz. When the input is given as neo
        AnalogSignal, its attribute is used to specify the sampling
        frequency and this parameter is ignored. Default is 1.0.
    axis : int
        Axis along which filter is applied. Default is -1.

    Returns
    -------
    filtered_signal : AnalogSignal or Quantity array or NumPy ndarray
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
                "Either highpass_freq or lowpass_freq must be given"
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

    # When the input is AnalogSignal, the axis for time index (i.e. the
    # first axis) needs to be rolled to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignal):
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

    if isinstance(signal, neo.AnalogSignal):
        return signal.duplicate_with_new_data(np.rollaxis(filtered_data, -1, 0))
    elif isinstance(signal, pq.quantity.Quantity):
        return filtered_data * signal.units
    else:
        return filtered_data


def wavelet_transform(signal, freq, nco=6.0, fs=1.0, zero_padding=True):
    """
    Compute the wavelet transform of a given signal with Morlet mother wavelet.
    The parametrization of the wavelet is based on [1].

    Parameters
    ----------
    signal : neo.AnalogSignal or array_like
        Time series data to be wavelet-transformed. When multi-dimensional
        array_like is given, the time axis must be the last dimension of
        the array_like.
    freq : float or list of floats
        Center frequency of the Morlet wavelet in Hz. Multiple center
        frequencies can be given as a list, in which case the function
        computes the wavelet transforms for all the given frequencies at once.
    nco : float (optional)
        Size of the mother wavelet (approximate number of oscillation cycles
        within a wavelet; related to the wavelet number w as w ~ 2 pi nco / 6),
        as defined in [1]. A larger nco value leads to a higher frequency
        resolution and a lower temporal resolution, and vice versa. Typically
        used values are in a range of 3 - 8, but one should be cautious when
        using a value smaller than ~ 6, in which case the admissibility of the
        wavelet is not ensured (cf. [2]). Default value is 6.0.
    fs : float (optional)
        Sampling rate of the input data in Hz. When `signal` is given as an
        AnalogSignal, the sampling frequency is taken from its attribute and
        this parameter is ignored. Default value is 1.0.
    zero_padding : bool (optional)
        Specifies whether the data length is extended to the least power of
        2 greater than the original length, by padding zeros to the tail, for
        speeding up the computation. In the case of True, the extended part is
        cut out from the final result before returned, so that the output
        has the same length as the input. Default is True.

    Returns
    -------
    signal_wt: complex array
        Wavelet transform of the input data. When `freq` was given as a list,
        the way how the wavelet transforms for different frequencies are
        returned depends on the input type. When the input was an AnalogSignal
        of shape (Nt, Nch), where Nt and Nch are the numbers of time points and
        channels, respectively, the returned array has a shape (Nt, Nch, Nf),
        where Nf = `len(freq)`, such that the last dimension indexes the
        frequencies. When the input was an array_like of shape
        (a, b, ..., c, Nt), the returned array has a shape
        (a, b, ..., c, Nf, Nt), such that the second last dimension indexes the
        frequencies.
        To summarize, `signal_wt.ndim` = `signal.ndim` + 1, with the additional
        dimension in the last axis (for AnalogSignal input) or the second last
        axis (for array_like input) indexing the frequencies.

    Raises
    ------
    ValueError
        If `freq` (or one of the values in `freq` when it is a list) is greater
        than the half of `fs`, or `nco` is not positive.

    References
    ----------
    1. Le van Quyen et al. J Neurosci Meth 111:83-98 (2001)
    2. Farge, Annu Rev Fluid Mech 24:395-458 (1992)
    """
    def _morlet_wavelet_ft(freq, nco, fs, n):
        # Generate the Fourier transform of Morlet wavelet as defined
        # in Le van Quyen et al. J Neurosci Meth 111:83-98 (2001).
        sigma = nco / (6. * freq)
        freqs = np.fft.fftfreq(n, 1.0 / fs)
        heaviside = np.array(freqs > 0., dtype=np.float)
        ft_real = np.sqrt(2 * np.pi * freq) * sigma * np.exp(
            -2 * (np.pi * sigma * (freqs - freq)) ** 2) * heaviside * fs
        ft_imag = np.zeros_like(ft_real)
        return ft_real + 1.0j * ft_imag

    data = np.asarray(signal)
    # When the input is AnalogSignal, the axis for time index (i.e. the
    # first axis) needs to be rolled to the last
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, data.ndim)

    # When the input is AnalogSignal, use its attribute to specify the
    # sampling frequency
    if hasattr(signal, 'sampling_rate'):
        fs = signal.sampling_rate
    if isinstance(fs, pq.quantity.Quantity):
        fs = fs.rescale('Hz').magnitude

    if isinstance(freq, (list, tuple, np.ndarray)):
        freqs = np.asarray(freq)
    else:
        freqs = np.array([freq,])
    if isinstance(freqs[0], pq.quantity.Quantity):
        freqs = [f.rescale('Hz').magnitude for f in freqs]

    # check whether the given central frequencies are less than the
    # Nyquist frequency of the signal
    if np.any(freqs >= fs / 2):
        raise ValueError("`freq` must be less than the half of " +
                         "the sampling rate `fs` = {} Hz".format(fs))

    # check if nco is positive
    if nco <= 0:
        raise ValueError("`nco` must be positive")

    n_orig = data.shape[-1]
    if zero_padding:
        n = 2 ** (int(np.log2(n_orig)) + 1)
    else:
        n = n_orig

    # generate Morlet wavelets (in the frequency domain)
    wavelet_fts = np.empty([len(freqs), n], dtype=np.complex)
    for i, f in enumerate(freqs):
        wavelet_fts[i] = _morlet_wavelet_ft(f, nco, fs, n)

    # perform wavelet transform by convoluting the signal with the wavelets
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    data = np.expand_dims(data, data.ndim-1)
    data = np.fft.ifft(np.fft.fft(data, n) * wavelet_fts)
    signal_wt = data[..., 0:n_orig]

    # reshape the result array according to the input
    if isinstance(signal, neo.AnalogSignal):
        signal_wt = np.rollaxis(signal_wt, -1)
        if not isinstance(freq, (list, tuple, np.ndarray)):
            signal_wt = signal_wt[..., 0]
    else:
        if signal.ndim == 1:
            signal_wt = signal_wt[0]
        if not isinstance(freq, (list, tuple, np.ndarray)):
            signal_wt = signal_wt[..., 0, :]

    return signal_wt


def hilbert(signal, N='nextpow'):
    '''
    Apply a Hilbert transform to an AnalogSignal object in order to obtain its
    (complex) analytic signal.

    The time series of the instantaneous angle and amplitude can be obtained as
    the angle (np.angle) and absolute value (np.abs) of the complex analytic
    signal, respectively.

    By default, the function will zero-pad the signal to a length corresponding
    to the next higher power of 2. This will provide higher computational
    efficiency at the expense of memory. In addition, this circumvents a
    situation where for some specific choices of the length of the input,
    scipy.signal.hilbert() will not terminate.

    Parameters
    -----------
    signal : neo.AnalogSignal
        Signal(s) to transform
    N : string or int
        Defines whether the signal is zero-padded.
            'none': no padding
            'nextpow':  zero-pad to the next length that is a power of 2
            int: directly specify the length to zero-pad to (indicates the
                number of Fourier components, see parameter N of
                scipy.signal.hilbert()).
        Default: 'nextpow'.

    Returns
    -------
    neo.AnalogSignal
        Contains the complex analytic signal(s) corresponding to the input
        signals. The unit of the analytic signal is dimensionless.

    Example
    -------
    Create a sine signal at 5 Hz with increasing amplitude and calculate the
    instantaneous phases

    >>> t = np.arange(0, 5000) * ms
    >>> f = 5. * Hz
    >>> a = neo.AnalogSignal(
    ...       np.array(
    ...           (1 + t.magnitude / t[-1].magnitude) * np.sin(
    ...               2. * np.pi * f * t.rescale(s))).reshape((-1,1))*mV,
    ...       t_start=0*s, sampling_rate=1000*Hz)

    >>> analytic_signal = hilbert(a, N='nextpow')
    >>> angles = np.angle(analytic_signal)
    >>> amplitudes = np.abs(analytic_signal)
    >>> print angles
            [[-1.57079633]
             [-1.51334228]
             [-1.46047675]
             ...,
             [-1.73112977]
             [-1.68211683]
             [-1.62879501]]
    >>> plt.plot(t,angles)
    '''
    # Length of input signals
    n_org = signal.shape[0]

    # Right-pad signal to desired length using the signal itself
    if type(N) == int:
        # User defined padding
        n = N
    elif N == 'nextpow':
        # To speed up calculation of the Hilbert transform, make sure we change
        # the signal to be of a length that is a power of two. Failure to do so
        # results in computations of certain signal lengths to not finish (or
        # finish in absurd time). This might be a bug in scipy (0.16), e.g.,
        # the following code will not terminate for this value of k:
        #
        # import numpy
        # import scipy.signal
        # k=679346
        # t = np.arange(0, k) / 1000.
        # a = (1 + t / t[-1]) * np.sin(2 * np.pi * 5 * t)
        # analytic_signal = scipy.signal.hilbert(a)
        #
        # For this reason, nextpow is the default setting for now.

        n = 2 ** (int(np.log2(n_org - 1)) + 1)
    elif N == 'none':
        # No padding
        n = n_org
    else:
        raise ValueError("'{}' is an unknown N.".format(N))

    output = signal.duplicate_with_new_data(
        scipy.signal.hilbert(signal.magnitude, N=n, axis=0)[:n_org])
    return output / output.units
