# -*- coding: utf-8 -*-
"""
Basic processing procedures for time series (e.g., performing a z-score of a
signal, or filtering a signal).

.. autosummary::
    :toctree: _toctree/signal_processing

    zscore
    cross_correlation_function
    butter
    wavelet_transform
    hilbert
    rauc
    derivative

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import neo
import numpy as np
import quantities as pq
import scipy.signal

from elephant.utils import deprecated_alias, check_same_units

__all__ = [
    "zscore",
    "cross_correlation_function",
    "butter",
    "wavelet_transform",
    "hilbert",
    "rauc",
    "derivative"
]


def zscore(signal, inplace=True):
    r"""
    Apply a z-score operation to one or several `neo.AnalogSignal` objects.

    The z-score operation subtracts the mean :math:`\mu` of the signal, and
    divides by its standard deviation :math:`\sigma`:

    .. math::
         Z(x(t)) = \frac{x(t)-\mu}{\sigma}

    If a `neo.AnalogSignal` object containing multiple signals is provided,
    the z-transform is always calculated for each signal individually.

    If a list of `neo.AnalogSignal` objects is supplied, the mean and standard
    deviation are calculated across all objects of the list. Thus, all list
    elements are z-transformed by the same values of :math:`\\mu` and
    :math:`\sigma`. For a `neo.AnalogSignal` that contains multiple signals,
    each signal of the array is treated separately across list elements.
    Therefore, the number of signals must be identical for each
    `neo.AnalogSignal` object of the list.

    Parameters
    ----------
    signal : neo.AnalogSignal or list of neo.AnalogSignal
        Signals for which to calculate the z-score.
    inplace : bool, optional
        If True, the contents of the input `signal` is replaced by the
        z-transformed signal, if possible, i.e when the signal type is float.
        If False, a copy of the original `signal` is returned.
        Default: True

    Returns
    -------
    signal_ztransofrmed : neo.AnalogSignal or list of neo.AnalogSignal
        The output format matches the input format: for each input
        `neo.AnalogSignal`, a corresponding `neo.AnalogSignal` is returned,
        containing the z-transformed signal with dimensionless unit.

    Notes
    -----
    You may supply a list of `neo.AnalogSignal` objects, where each object in
    the list contains the data of one trial of the experiment, and each signal
    of the `neo.AnalogSignal` corresponds to the recordings from one specific
    electrode in a particular trial. In this scenario, you will z-transform
    the signal of each electrode separately, but transform all trials of a
    given electrode in the same way.

    Examples
    --------
    Z-transform a single `neo.AnalogSignal`, containing only a single signal.

    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.signal_processing import zscore
    ...
    >>> a = neo.AnalogSignal(
    ...       np.array([1, 2, 3, 4, 5, 6]).reshape(-1,1) * pq.mV,
    ...       t_start=0*pq.s, sampling_rate=1000*pq.Hz)
    >>> zscore(a).as_quantity()
    [[-1.46385011]
     [-0.87831007]
     [-0.29277002]
     [ 0.29277002]
     [ 0.87831007]
     [ 1.46385011]] dimensionless

    Z-transform a single `neo.AnalogSignal` containing multiple signals.

    >>> b = neo.AnalogSignal(
    ...       np.transpose([[1, 2, 3, 4, 5, 6],
    ...                     [11, 12, 13, 14, 15, 16]]) * pq.mV,
    ...       t_start=0*pq.s, sampling_rate=1000*pq.Hz)
    >>> zscore(b).as_quantity()
    [[-1.46385011 -1.46385011]
     [-0.87831007 -0.87831007]
     [-0.29277002 -0.29277002]
     [ 0.29277002  0.29277002]
     [ 0.87831007  0.87831007]
     [ 1.46385011  1.46385011]] dimensionless

    Z-transform a list of `neo.AnalogSignal`, each one containing more than
    one signal:

    >>> c = neo.AnalogSignal(
    ...       np.transpose([[21, 22, 23, 24, 25, 26],
    ...                     [31, 32, 33, 34, 35, 36]]) * pq.mV,
    ...       t_start=0*pq.s, sampling_rate=1000*pq.Hz)
    >>> zscore([b, c])
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

    """
    # Transform input to a list
    if isinstance(signal, neo.AnalogSignal):
        signal = [signal]
    check_same_units(signal, object_type=neo.AnalogSignal)

    # Calculate mean and standard deviation
    signal_stacked = np.vstack(signal).magnitude
    mean = signal_stacked.mean(axis=0)
    std = signal_stacked.std(axis=0)

    signal_ztransofrmed = []
    for sig in signal:
        sig_normalized = sig.magnitude.astype(mean.dtype, copy=not inplace)
        sig_normalized -= mean
        # items where std is zero are already zero
        np.divide(sig_normalized, std, out=sig_normalized, where=std != 0)
        sig_dimless = neo.AnalogSignal(signal=sig_normalized,
                                       units=pq.dimensionless,
                                       dtype=sig_normalized.dtype,
                                       copy=False,
                                       t_start=sig.t_start,
                                       sampling_rate=sig.sampling_rate,
                                       name=sig.name,
                                       file_origin=sig.file_origin,
                                       description=sig.description,
                                       array_annotations=sig.array_annotations,
                                       **sig.annotations)
        signal_ztransofrmed.append(sig_dimless)

    # Return single object, or list of objects
    if len(signal_ztransofrmed) == 1:
        signal_ztransofrmed = signal_ztransofrmed[0]
    return signal_ztransofrmed


@deprecated_alias(ch_pairs='channel_pairs', nlags='n_lags',
                  env='hilbert_envelope')
def cross_correlation_function(signal, channel_pairs, hilbert_envelope=False,
                               n_lags=None, scaleopt='unbiased'):
    r"""
    Computes an estimator of the cross-correlation function
    :cite:`signal-Stoica2005`.

    .. math::

             R(\tau) = \frac{1}{N-|k|} R'(\tau) \\

    where :math:`R'(\tau) = \left<x(t)y(t+\tau)\right>` in a pairwise
    manner, i.e.:

    `signal[channel_pairs[0,0]]` vs `signal[channel_pairs[0,1]]`,

    `signal[channel_pairs[1,0]]` vs `signal[channel_pairs[1,1]]`,

    and so on.

    The input time series are z-scored beforehand. `scaleopt` controls the
    choice of :math:`R_{xy}(\tau)` normalizer. Alternatively, returns the
    Hilbert envelope of :math:`R_{xy}(\tau)`, which is useful to determine the
    correlation length of oscillatory signals.

    Parameters
    ----------
    signal : (nt, nch) neo.AnalogSignal
        Signal with `nt` number of samples that contains `nch` LFP channels.
    channel_pairs : list or (n, 2) np.ndarray
        List with `n` channel pairs for which to compute cross-correlation.
        Each element of the list must contain 2 channel indices.
        If `np.ndarray`, the second axis must have dimension 2.
    hilbert_envelope : bool, optional
        If True, returns the Hilbert envelope of cross-correlation function
        result.
        Default: False
    n_lags : int, optional
        Defines the number of lags for cross-correlation function. If a `float`
        is passed, it will be rounded to the nearest integer. Number of
        samples of output is `2*n_lags+1`.
        If None, the number of samples of the output is equal to the number of
        samples of the input signal (namely `nt`).
        Default: None
    scaleopt : {'none', 'biased', 'unbiased', 'normalized', 'coeff'}, optional
        Normalization option, equivalent to matlab `xcorr(..., scaleopt)`.
        Specified as one of the following.

        * 'none': raw, unscaled cross-correlation

        .. math::
            R_{xy}(\tau)

        * 'biased': biased estimate of the cross-correlation:

        .. math::
            R_{xy,biased}(\tau) = \frac{1}{N} R_{xy}(\tau)

        * 'unbiased': unbiased estimate of the cross-correlation:

        .. math::
            R_{xy,unbiased}(\tau) = \frac{1}{N-\tau} R_{xy}(\tau)

        * 'normalized' or 'coeff': normalizes the sequence so that the
          autocorrelations at zero lag equal 1:

        .. math::
            R_{xy,coeff}(\tau) = \frac{1}{\sqrt{R_{xx}(0) R_{yy}(0)}}
                                 R_{xy}(\tau)

        Default: 'unbiased'

    Returns
    -------
    cross_corr : neo.AnalogSignal
        Shape: `[2*n_lags+1, n]`
        Pairwise cross-correlation functions for channel pairs given by
        `channel_pairs`. If `hilbert_envelope` is True, the output is the
        Hilbert envelope of the pairwise cross-correlation function. This is
        helpful to compute the correlation length for oscillating
        cross-correlation functions.

    Raises
    ------
    ValueError
        If input `signal` is not a `neo.AnalogSignal`.

        If `channel_pairs` is not a list of channel pair indices with shape
        `(n,2)`.

        If `hilbert_envelope` is not a boolean.

        If `n_lags` is not a positive integer.

        If `scaleopt` is not one of the predefined above keywords.

    Examples
    --------
    >>> import neo
    >>> import quantities as pq
    >>> import matplotlib.pyplot as plt
    >>> from elephant.signal_processing import cross_correlation_function
    >>> dt = 0.02
    >>> N = 2018
    >>> f = 0.5
    >>> t = np.arange(N)*dt
    >>> x = np.zeros((N,2))
    >>> x[:,0] = 0.2 * np.sin(2.*np.pi*f*t)
    >>> x[:,1] = 5.3 * np.cos(2.*np.pi*f*t)

    Generate neo.AnalogSignals from x and find cross-correlation

    >>> signal = neo.AnalogSignal(x, units='mV', t_start=0.*pq.ms,
    >>>     sampling_rate=1/dt*pq.Hz, dtype=float)
    >>> rho = cross_correlation_function(signal, [0,1], n_lags=150)
    >>> env = cross_correlation_function(signal, [0,1], n_lags=150,
    ...     hilbert_envelope=True)
    ...
    >>> plt.plot(rho.times, rho)
    >>> plt.plot(env.times, env) # should be equal to one
    >>> plt.show()

    """

    # Make channel_pairs a 2D array
    pairs = np.asarray(channel_pairs)
    if pairs.ndim == 1:
        pairs = np.expand_dims(pairs, axis=0)

    # Check input
    if not isinstance(signal, neo.AnalogSignal):
        raise ValueError('Input signal must be of type neo.AnalogSignal')
    if pairs.shape[1] != 2:
        raise ValueError("'channel_pairs' is not a list of channel pair "
                         "indices. Cannot define pairs for cross-correlation.")
    if not isinstance(hilbert_envelope, bool):
        raise ValueError("'hilbert_envelope' must be a boolean value")
    if n_lags is not None:
        if not isinstance(n_lags, int) or n_lags <= 0:
            raise ValueError('n_lags must be a non-negative integer')

    # z-score analog signal and store channel time series in different arrays
    # Cross-correlation will be calculated between xsig and ysig
    z_transformed = signal.magnitude - signal.magnitude.mean(axis=0)
    z_transformed = np.divide(z_transformed, signal.magnitude.std(axis=0),
                              out=z_transformed,
                              where=z_transformed != 0)
    # transpose (nch, xy, nt) -> (xy, nt, nch)
    xsig, ysig = np.transpose(z_transformed.T[pairs], (1, 2, 0))

    # Define vector of lags tau
    nt, nch = xsig.shape
    tau = np.arange(nt) - nt // 2

    # Calculate cross-correlation by taking Fourier transform of signal,
    # multiply in Fourier space, and transform back. Correct for bias due
    # to zero-padding
    xcorr = scipy.signal.fftconvolve(xsig, ysig[::-1], mode='same', axes=0)
    if scaleopt == 'biased':
        xcorr /= nt
    elif scaleopt == 'unbiased':
        normalizer = np.expand_dims(nt - np.abs(tau), axis=1)
        xcorr /= normalizer
    elif scaleopt in ('normalized', 'coeff'):
        normalizer = np.sqrt((xsig ** 2).sum(axis=0) * (ysig ** 2).sum(axis=0))
        xcorr /= normalizer
    elif scaleopt != 'none':
        raise ValueError("Invalid scaleopt mode: '{}'".format(scaleopt))

    # Calculate envelope of cross-correlation function with Hilbert transform.
    # This is useful for transient oscillatory signals.
    if hilbert_envelope:
        xcorr = np.abs(scipy.signal.hilbert(xcorr, axis=0))

    # Cut off lags outside the desired range
    if n_lags is not None:
        tau0 = np.argwhere(tau == 0).item()
        xcorr = xcorr[tau0 - n_lags: tau0 + n_lags + 1, :]

    # Return neo.AnalogSignal
    cross_corr = neo.AnalogSignal(xcorr,
                                  units='',
                                  t_start=tau[0] * signal.sampling_period,
                                  t_stop=tau[-1] * signal.sampling_period,
                                  sampling_rate=signal.sampling_rate,
                                  dtype=float)
    return cross_corr


@deprecated_alias(highpass_freq='highpass_frequency',
                  lowpass_freq='lowpass_frequency',
                  fs='sampling_frequency')
def butter(signal, highpass_frequency=None, lowpass_frequency=None, order=4,
           filter_function='filtfilt', sampling_frequency=1.0, axis=-1):
    """
    Butterworth filtering function for `neo.AnalogSignal`.

    Filter type is determined according to how values of `highpass_frequency`
    and `lowpass_frequency` are given (see "Parameters" section for details).

    Parameters
    ----------
    signal : neo.AnalogSignal or pq.Quantity or np.ndarray
        Time series data to be filtered.
        If `pq.Quantity` or `np.ndarray`, the sampling frequency should be
        given through the keyword argument `fs`.
    highpass_frequency : pq.Quantity of float, optional
        High-pass cut-off frequency. If `float`, the given value is taken as
        frequency in Hz.
        Default: None
    lowpass_frequency : pq.Quantity or float, optional
        Low-pass cut-off frequency. If `float`, the given value is taken as
        frequency in Hz.
        Filter type is determined depending on the values of
        `lowpass_frequency` and `highpass_frequency`:

        * `highpass_frequency` only (`lowpass_frequency` is None):
        highpass filter

        * `lowpass_frequency` only (`highpass_frequency` is None):
        lowpass filter

        * `highpass_frequency` < `lowpass_frequency`: bandpass filter

        * `highpass_frequency` > `lowpass_frequency`: bandstop filter

        Default: None
    order : int, optional
        Order of the Butterworth filter.
        Default: 4
    filter_function : {'filtfilt', 'lfilter', 'sosfiltfilt'}, optional
        Filtering function to be used. Available filters:

        * 'filtfilt': `scipy.signal.filtfilt`;

        * 'lfilter': `scipy.signal.lfilter`;

        * 'sosfiltfilt': `scipy.signal.sosfiltfilt`.

        In most applications 'filtfilt' should be used, because it doesn't
        bring about phase shift due to filtering. For numerically stable
        filtering, in particular higher order filters, use 'sosfiltfilt'
        (see https://github.com/NeuralEnsemble/elephant/issues/220).
        Default: 'filtfilt'
    sampling_frequency : pq.Quantity or float, optional
        The sampling frequency of the input time series. When given as
        `float`, its value is taken as frequency in Hz. When `signal` is given
        as `neo.AnalogSignal`, its attribute is used to specify the sampling
        frequency and this parameter is ignored.
        Default: 1.0
    axis : int, optional
        Axis along which filter is applied.
        Default: last axis (-1)

    Returns
    -------
    filtered_signal : neo.AnalogSignal or pq.Quantity or np.ndarray
        Filtered input data. The shape and type is identical to those of the
        input `signal`.

    Raises
    ------
    ValueError
        If `filter_function` is not one of 'lfilter', 'filtfilt',
        or 'sosfiltfilt'.

        If both `highpass_frequency` and `lowpass_frequency` are None.

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.signal_processing import butter
    >>> noise = neo.AnalogSignal(np.random.normal(size=5000),
    ...     sampling_rate=1000 * pq.Hz, units='mV')
    >>> filtered_noise = butter(noise, highpass_frequency=250.0 * pq.Hz)
    >>> filtered_noise
    AnalogSignal with 1 channels of length 5000; units mV; datatype float64
    sampling rate: 1000.0 Hz
    time: 0.0 s to 5.0 s

    Let's check that the normal noise power spectrum at zero frequency is close
    to zero.

    >>> from elephant.spectral import welch_psd
    >>> freq, psd = welch_psd(filtered_noise, fs=1000.0)
    >>> psd.shape
    (1, 556)
    >>> freq[0], psd[0, 0]
    (array(0.) * Hz, array(7.21464674e-08) * mV**2/Hz)

    """
    available_filters = 'lfilter', 'filtfilt', 'sosfiltfilt'
    if filter_function not in available_filters:
        raise ValueError("Invalid `filter_function`: {filter_function}. "
                         "Available filters: {available_filters}".format(
                             filter_function=filter_function,
                             available_filters=available_filters))
    # design filter
    if hasattr(signal, 'sampling_rate'):
        sampling_frequency = signal.sampling_rate.rescale(pq.Hz).magnitude
    if isinstance(highpass_frequency, pq.quantity.Quantity):
        highpass_frequency = highpass_frequency.rescale(pq.Hz).magnitude
    if isinstance(lowpass_frequency, pq.quantity.Quantity):
        lowpass_frequency = lowpass_frequency.rescale(pq.Hz).magnitude
    Fn = sampling_frequency / 2.
    # filter type is determined according to the values of cut-off
    # frequencies
    if lowpass_frequency and highpass_frequency:
        if highpass_frequency < lowpass_frequency:
            Wn = (highpass_frequency / Fn, lowpass_frequency / Fn)
            btype = 'bandpass'
        else:
            Wn = (lowpass_frequency / Fn, highpass_frequency / Fn)
            btype = 'bandstop'
    elif lowpass_frequency:
        Wn = lowpass_frequency / Fn
        btype = 'lowpass'
    elif highpass_frequency:
        Wn = highpass_frequency / Fn
        btype = 'highpass'
    else:
        raise ValueError(
            "Either highpass_frequency or lowpass_frequency must be given"
        )
    if filter_function == 'sosfiltfilt':
        output = 'sos'
    else:
        output = 'ba'
    designed_filter = scipy.signal.butter(order, Wn, btype=btype,
                                          output=output)

    # When the input is AnalogSignal, the axis for time index (i.e. the
    # first axis) needs to be rolled to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))

    # apply filter
    if filter_function == 'lfilter':
        b, a = designed_filter
        filtered_data = scipy.signal.lfilter(b=b, a=a, x=data, axis=axis)
    elif filter_function == 'filtfilt':
        b, a = designed_filter
        filtered_data = scipy.signal.filtfilt(b=b, a=a, x=data, axis=axis)
    else:
        filtered_data = scipy.signal.sosfiltfilt(sos=designed_filter,
                                                 x=data, axis=axis)

    if isinstance(signal, neo.AnalogSignal):
        filtered_data = np.rollaxis(filtered_data, -1, 0)
        signal_out = signal.duplicate_with_new_data(filtered_data)
        # todo use flag once is fixed
        #      https://github.com/NeuralEnsemble/python-neo/issues/752
        signal_out.array_annotate(**signal.array_annotations)
        return signal_out
    elif isinstance(signal, pq.quantity.Quantity):
        return filtered_data * signal.units
    else:
        return filtered_data


@deprecated_alias(nco='n_cycles', freq='frequency', fs='sampling_frequency')
def wavelet_transform(signal, frequency, n_cycles=6.0, sampling_frequency=1.0,
                      zero_padding=True):
    r"""
    Compute the wavelet transform of a given signal with Morlet mother
    wavelet. The parametrization of the wavelet is based on
    :cite:`signal-Le2001_83`.

    Parameters
    ----------
    signal : (Nt, Nch) neo.AnalogSignal or np.ndarray or list
        Time series data to be wavelet-transformed. When multi-dimensional
        `np.ndarray` or list is given, the time axis must be the last
        dimension. If `neo.AnalogSignal`, `Nt` is the number of time points
        and `Nch` is the number of channels.
    frequency : float or list of float
        Center frequency of the Morlet wavelet in Hz. Multiple center
        frequencies can be given as a list, in which case the function
        computes the wavelet transforms for all the given frequencies at once.
    n_cycles : float, optional
        Size of the mother wavelet (approximate number of oscillation cycles
        within a wavelet). Corresponds to :math:`nco` in
        :cite:`signal-Le2001_83`. A larger `n_cycles` value leads to a higher
        frequency resolution and a lower temporal resolution, and vice versa.
        Typically used values are in a range of 3–8, but one should be cautious
        when using a value smaller than ~ 6, in which case the admissibility of
        the wavelet is not ensured :cite:`signal-Farge1992_395`.
        Default: 6.0
    sampling_frequency : float, optional
        Sampling rate of the input data in Hz.
        When `signal` is given as a `neo.AnalogSignal`, the sampling frequency
        is taken from its attribute and this parameter is ignored.
        Default: 1.0
    zero_padding : bool, optional
        Specifies whether the data length is extended to the least power of
        2 greater than the original length, by padding zeros to the tail, for
        speeding up the computation.
        If True, the extended part is cut out from the final result before
        returned, so that the output has the same length as the input.
        Default: True

    Returns
    -------
    signal_wt : np.ndarray
        Wavelet transform of the input data. When `frequency` was given as a
        list, the way how the wavelet transforms for different frequencies are
        returned depends on the input type:

        * when the input was a `neo.AnalogSignal`, the returned array has
          shape (`Nt`, `Nch`, `Nf`), where `Nf` = `len(freq)`, such that the
          last dimension indexes the frequencies;

        * when the input was a `np.ndarray` or list of shape
          (`a`, `b`, ..., `c`, `Nt`), the returned array has a shape
          (`a`, `b`, ..., `c`, `Nf`, `Nt`), such that the second last
          dimension indexes the frequencies.

        To summarize, `signal_wt.ndim` = `signal.ndim` + 1, with the
        additional dimension in the last axis (for `neo.AnalogSignal` input)
        or the second last axis (`np.ndarray` or list input) indexing the
        frequencies.

    Raises
    ------
    ValueError
        If `frequency` (or one of the values in `frequency` when it is a list)
        is greater than the half of `sampling_frequency`.

        If `n_cycles` is not positive.

    Notes
    -----
    `n_cycles` is related to the wavelet number :math:`w` as
    :math:`w \sim 2 \pi \frac{n_{\text{cycles}}}{6}` as defined in
    :cite:`signal-Le2001_83`.

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.signal_processing import wavelet_transform
    >>> noise = neo.AnalogSignal(np.random.normal(size=7),
    ...     sampling_rate=11 * pq.Hz, units='mV')

    The wavelet frequency must be less than the half of the sampling rate;
    picking at 5 Hz.

    >>> wavelet_transform(noise, frequency=5)
    array([[-1.00890049+3.003473j  ],
       [-1.43664254-2.8389273j ],
       [ 3.02499511+0.96534578j],
       [-2.79543976+1.4581079j ],
       [ 0.94387304-2.98159518j],
       [ 1.41476471+2.77389985j],
       [-2.95996766-0.9872236j ]])

    """
    def _morlet_wavelet_ft(freq, n_cycles, fs, n):
        # Generate the Fourier transform of Morlet wavelet as defined
        # in Le van Quyen et al. J Neurosci Meth 111:83-98 (2001).
        sigma = n_cycles / (6. * freq)
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
        sampling_frequency = signal.sampling_rate
    if isinstance(sampling_frequency, pq.quantity.Quantity):
        sampling_frequency = sampling_frequency.rescale('Hz').magnitude

    if isinstance(frequency, (list, tuple, np.ndarray)):
        freqs = np.asarray(frequency)
    else:
        freqs = np.array([frequency, ])
    if isinstance(freqs[0], pq.quantity.Quantity):
        freqs = [f.rescale('Hz').magnitude for f in freqs]

    # check whether the given central frequencies are less than the
    # Nyquist frequency of the signal
    if np.any(freqs >= sampling_frequency / 2):
        raise ValueError("'frequency' elements must be less than the half of "
                         "the 'sampling_frequency' ({}) Hz"
                         .format(sampling_frequency))

    # check if n_cycles is positive
    if n_cycles <= 0:
        raise ValueError("`n_cycles` must be positive")

    n_orig = data.shape[-1]
    if zero_padding:
        n = 2 ** (int(np.log2(n_orig)) + 1)
    else:
        n = n_orig

    # generate Morlet wavelets (in the frequency domain)
    wavelet_fts = np.empty([len(freqs), n], dtype=np.complex)
    for i, f in enumerate(freqs):
        wavelet_fts[i] = _morlet_wavelet_ft(f, n_cycles, sampling_frequency, n)

    # perform wavelet transform by convoluting the signal with the wavelets
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    data = np.expand_dims(data, data.ndim - 1)
    data = np.fft.ifft(np.fft.fft(data, n) * wavelet_fts)
    signal_wt = data[..., 0:n_orig]

    # reshape the result array according to the input
    if isinstance(signal, neo.AnalogSignal):
        signal_wt = np.rollaxis(signal_wt, -1)
        if not isinstance(frequency, (list, tuple, np.ndarray)):
            signal_wt = signal_wt[..., 0]
    else:
        if signal.ndim == 1:
            signal_wt = signal_wt[0]
        if not isinstance(frequency, (list, tuple, np.ndarray)):
            signal_wt = signal_wt[..., 0, :]

    return signal_wt


@deprecated_alias(N='padding')
def hilbert(signal, padding='nextpow'):
    """
    Apply a Hilbert transform to a `neo.AnalogSignal` object in order to
    obtain its (complex) analytic signal.

    The time series of the instantaneous angle and amplitude can be obtained
    as the angle (`np.angle` function) and absolute value (`np.abs` function)
    of the complex analytic signal, respectively.

    By default, the function will zero-pad the signal to a length
    corresponding to the next higher power of 2. This will provide higher
    computational efficiency at the expense of memory. In addition, this
    circumvents a situation where, for some specific choices of the length of
    the input, `scipy.signal.hilbert` function will not terminate.

    Parameters
    ----------
    signal : neo.AnalogSignal
        Signal(s) to transform.
    padding : int, {'none', 'nextpow'}, or None, optional
        Defines whether the signal is zero-padded.
        The `padding` argument corresponds to `N` in
        `scipy.signal.hilbert(signal, N=padding)` function.
        If 'none' or None, no padding.
        If 'nextpow', zero-pad to the next length that is a power of 2.
        If it is an `int`, directly specify the length to zero-pad to
        (indicates the number of Fourier components).
        Default: 'nextpow'

    Returns
    -------
    neo.AnalogSignal
        Contains the complex analytic signal(s) corresponding to the input
        `signal`. The unit of the returned `neo.AnalogSignal` is
        dimensionless.

    Raises
    ------
    ValueError:
        If `padding` is not an integer or neither 'nextpow' nor 'none' (None).

    Examples
    --------
    Create a sine signal at 5 Hz with increasing amplitude and calculate the
    instantaneous phases:

    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> import matplotlib.pyplot as plt
    >>> from elephant.signal_processing import hilbert
    >>> t = np.arange(0, 5000) * pq.ms
    >>> f = 5. * pq.Hz
    >>> a = neo.AnalogSignal(
    ...       np.array(
    ...           (1 + t.magnitude / t[-1].magnitude) * np.sin(
    ...               2. * np.pi * f * t.rescale(pq.s))).reshape(
    ...                   (-1,1)) * pq.mV,
    ...       t_start=0*pq.s,
    ...       sampling_rate=1000*pq.Hz)
    ...
    >>> analytic_signal = hilbert(a, padding='nextpow')
    >>> angles = np.angle(analytic_signal)
    >>> amplitudes = np.abs(analytic_signal)
    >>> print(angles)
    [[-1.57079633]
     [-1.51334228]
     [-1.46047675]
     ...,
     [-1.73112977]
     [-1.68211683]
     [-1.62879501]]
    >>> plt.plot(t, angles)

    """
    # Length of input signals
    n_org = signal.shape[0]

    # Right-pad signal to desired length using the signal itself
    if isinstance(padding, int):
        # User defined padding
        n = padding
    elif padding == 'nextpow':
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
    elif padding == 'none' or padding is None:
        # No padding
        n = n_org
    else:
        raise ValueError("Invalid padding '{}'.".format(padding))

    output = signal.duplicate_with_new_data(
        scipy.signal.hilbert(signal.magnitude, N=n, axis=0)[:n_org])
    # todo use flag once is fixed
    #      https://github.com/NeuralEnsemble/python-neo/issues/752
    output.array_annotate(**signal.array_annotations)
    return output / output.units


def rauc(signal, baseline=None, bin_duration=None, t_start=None, t_stop=None):
    """
    Calculate the rectified area under the curve (RAUC) for a
    `neo.AnalogSignal`.

    The signal is optionally divided into bins with duration `bin_duration`,
    and the rectified signal (absolute value) is integrated within each bin to
    find the area under the curve. The mean or median of the signal or an
    arbitrary baseline may optionally be subtracted before rectification.

    Parameters
    ----------
    signal : neo.AnalogSignal
        The signal to integrate. If `signal` contains more than one channel,
        each is integrated separately.
    baseline : pq.Quantity or {'mean', 'median'}, optional
        A factor to subtract from the signal before rectification.
        If 'mean', the mean value of the entire `signal` is subtracted on a
        channel-by-channel basis.
        If 'median', the median value of the entire `signal` is subtracted on
        a channel-by-channel basis.
        Default: None
    bin_duration : pq.Quantity, optional
        The length of time that each integration should span.
        If None, there will be only one bin spanning the entire signal
        duration.
        If `bin_duration` does not divide evenly into the signal duration, the
        end of the signal is padded with zeros to accomodate the final,
        overextending bin.
        Default: None
    t_start : pq.Quantity, optional
        Time to start the algorithm.
        If None, starts at the beginning of `signal`.
        Default: None
    t_stop : pq.Quantity, optional
        Time to end the algorithm.
        If None, ends at the last time of `signal`.
        The signal is cropped using `signal.time_slice(t_start, t_stop)` after
        baseline removal. Useful if you want the RAUC for a short section of
        the signal but want the mean or median calculation (`baseline`='mean'
        or `baseline`='median') to use the entire signal for better baseline
        estimation.
        Default: None

    Returns
    -------
    pq.Quantity or neo.AnalogSignal
        If the number of bins is 1, the returned object is a scalar or
        vector `pq.Quantity` containing a single RAUC value for each channel.
        Otherwise, the returned object is a `neo.AnalogSignal` containing the
        RAUC(s) for each bin stored as a sample, with times corresponding to
        the center of each bin. The output signal will have the same number
        of channels as the input signal.

    Raises
    ------
    ValueError
        If `signal` is not `neo.AnalogSignal`.

        If `bin_duration` is not None or `pq.Quantity`.

        If `baseline` is not None, 'mean', 'median', or `pq.Quantity`.

    See Also
    --------
    neo.AnalogSignal.time_slice : how `t_start` and `t_stop` are used

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.signal_processing import rauc
    >>> signal = neo.AnalogSignal(np.arange(10), sampling_rate=20 * pq.Hz,
    ...     units='mV')
    >>> rauc(signal)
    array(2.025) * mV/Hz

    """

    if not isinstance(signal, neo.AnalogSignal):
        raise ValueError('Input signal is not a neo.AnalogSignal!')

    if baseline is None:
        pass
    elif baseline == 'mean':
        # subtract mean from each channel
        signal = signal - signal.mean(axis=0)
    elif baseline == 'median':
        # subtract median from each channel
        signal = signal - np.median(signal.as_quantity(), axis=0)
    elif isinstance(baseline, pq.Quantity):
        # subtract arbitrary baseline
        signal = signal - baseline
    else:
        raise ValueError("baseline must be either None, 'mean', 'median', or "
                         "a Quantity. Got {}".format(baseline))

    # slice the signal after subtracting baseline
    signal = signal.time_slice(t_start, t_stop)

    if bin_duration is not None:
        # from bin duration, determine samples per bin and number of bins
        if isinstance(bin_duration, pq.Quantity):
            samples_per_bin = int(
                np.round(
                    bin_duration.rescale('s') /
                    signal.sampling_period.rescale('s')))
            n_bins = int(np.ceil(signal.shape[0] / samples_per_bin))
        else:
            raise ValueError("bin_duration must be a Quantity. Got {}".format(
                bin_duration))
    else:
        # all samples in one bin
        samples_per_bin = signal.shape[0]
        n_bins = 1

    # store the actual bin duration
    bin_duration = samples_per_bin * signal.sampling_period

    # reshape into equal size bins, padding the end with zeros if necessary
    n_channels = signal.shape[1]
    sig_binned = signal.as_quantity().copy()
    sig_binned.resize(n_bins * samples_per_bin, n_channels, refcheck=False)
    sig_binned = sig_binned.reshape(n_bins, samples_per_bin, n_channels)

    # rectify and integrate over each bin
    rauc = np.trapz(np.abs(sig_binned), dx=signal.sampling_period, axis=1)

    if n_bins == 1:
        # return a single value for each channel
        return rauc.squeeze()

    else:
        # return an AnalogSignal with times corresponding to center of each bin
        t_start = signal.t_start.rescale(bin_duration.units) + bin_duration / 2
        rauc_sig = neo.AnalogSignal(rauc, t_start=t_start,
                                    sampling_period=bin_duration)
        return rauc_sig


def derivative(signal):
    """
    Calculate the derivative of a `neo.AnalogSignal`.

    Parameters
    ----------
    signal : neo.AnalogSignal
        The signal to differentiate. If `signal` contains more than one
        channel, each is differentiated separately.

    Returns
    -------
    derivative_sig : neo.AnalogSignal
        The returned object is a `neo.AnalogSignal` containing the differences
        between each successive sample value of the input signal divided by
        the sampling period. Times are centered between the successive samples
        of the input. The output signal will have the same number of channels
        as the input signal.

    Raises
    ------
    TypeError
        If `signal` is not a `neo.AnalogSignal`.

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.signal_processing import derivative
    >>> signal = neo.AnalogSignal([0, 3, 4, 11, -1], sampling_rate=1 * pq.Hz,
    ...     units='mV')
    >>> print(derivative(signal))
    [[  3.]
     [  1.]
     [  7.]
     [-12.]] mV*Hz
    """

    if not isinstance(signal, neo.AnalogSignal):
        raise TypeError('Input signal is not a neo.AnalogSignal!')

    derivative_sig = neo.AnalogSignal(
        np.diff(signal.as_quantity(), axis=0) / signal.sampling_period,
        t_start=signal.t_start + signal.sampling_period / 2,
        sampling_period=signal.sampling_period)

    return derivative_sig
