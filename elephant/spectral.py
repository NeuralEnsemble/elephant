# -*- coding: utf-8 -*-
"""
Identification of spectral properties in analog signals (e.g., the power
spectrum).

.. autosummary::
    :toctree: _toctree/spectral

    welch_psd
    welch_coherence

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import warnings

import neo
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
              frequency_resolution=None, overlap=0.5, fs=1.0, window='hann',
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
        Default: 8
    len_segment : int, optional
        Length of segments. This parameter is ignored if `frequency_resolution`
        is given. If None, it will be determined from other parameters.
        Default: None
    frequency_resolution : pq.Quantity or float, optional
        Desired frequency resolution of the obtained PSD estimate in terms of
        the interval between adjacent frequency bins. When given as a `float`,
        it is taken as frequency in Hz.
        If None, it will be determined from other parameters.
        Default: None
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped)
    fs : pq.Quantity or float, optional
        Specifies the sampling frequency of the input time series. When the
        input is given as a `neo.AnalogSignal`, the sampling frequency is
        taken from its attribute and this parameter is ignored.
        Default: 1.0
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        See Notes [2].
        Default: 'hann'
    nfft : int, optional
        Length of the FFT used.
        See Notes [2].
        Default: None
    detrend : str or function or False, optional
        Specifies how to detrend each segment.
        See Notes [2].
        Default: 'constant'
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data.
        If False return a two-sided spectrum.
        See Notes [2].
        Default: True
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz. If 'spectrum', computes the power spectrum where Pxx has
        units of V**2, if `signal` is measured in V and `fs` is measured in
        Hz.
        See Notes [2].
        Default: 'density'
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [2].
        Default: last axis (-1)

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

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.spectral import welch_psd
    >>> signal = neo.AnalogSignal(np.cos(np.linspace(0, 2 * np.pi, num=100)),
    ...     sampling_rate=20 * pq.Hz, units='mV')

    Sampling frequency will be taken as `signal.sampling_rate`.

    >>> freq, psd = welch_psd(signal)
    >>> freq
    array([ 0.        ,  0.90909091,  1.81818182,  2.72727273,  3.63636364,
            4.54545455,  5.45454545,  6.36363636,  7.27272727,  8.18181818,
            9.09090909, 10.        ]) * Hz

    >>> psd # noqa
    array([[1.09566410e-03, 2.33607943e-02, 1.35436832e-03, 6.74408723e-05,
            1.00810196e-05, 2.40079315e-06, 7.35821437e-07, 2.58361700e-07,
            9.44183422e-08, 3.14573483e-08, 6.82050475e-09, 1.18183354e-10]]) * mV**2/Hz


    """
    # 'hanning' window was removed with release of scipy 1.9.0, it was
    # deprecated since 1.1.0.
    if window == 'hanning':
        warnings.warn("'hanning' is deprecated and was removed from scipy "
                      "with release 1.9.0. Please use 'hann' instead",
                      DeprecationWarning)
        window = 'hann'
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


def multitaper_psd(signal, n_segments=1, len_segment=None,
                   frequency_resolution=None, overlap=0.5, fs=1,
                   nw=4, num_tapers=None, peak_resolution=None, axis=-1):
    """
    Estimates power spectrum density (PSD) of a given 'neo.AnalogSignal'
    using Multitaper method

    The PSD is obtained through the following steps:

    1. Cut the given data into several overlapping segments. The degree of
       overlap can be specified by parameter `overlap` (default is 0.5,
       i.e. segments are overlapped by the half of their length).
       The number and the length of the segments are determined according
       to the parameters `n_segments`, `len_segment` or `frequency_resolution`.
       By default, the data is cut into 8 segments;

    2. Calculate 'num_tapers' approximately independent estimates of the
       spectrum by multiplying the signal with the discrete prolate spheroidal
       functions (also known as Slepian function) and calculate the PSD of each
       tapered segment

    3. Average the approximately independent estimates of each segment to
       decrease overall variance of the estimates

    4. Average the obtained estimates for each segment

    Parameters
    ----------
    signal : neo.AnalogSignal
        Time series data of which PSD is estimated. When `signal` is np.ndarray
        sampling frequency should be given through keyword argument `fs`.
        Signal should be passed as (n_channels, n_samples)
    fs : float, optional
        Specifies the sampling frequency of the input time series
        Default: 1.0.
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
    nw : float, optional
        Time bandwidth product
        Default: 4.0.
    num_tapers : int, optional
        Number of tapers used in 1. to obtain estimate of PSD. By default,
        [2*nw] - 1 is chosen.
        Default: None.
    peak_resolution : pq.Quantity float, optional
        Quantity in Hz determining the number of tapers used for analysis.
        Fine peak resolution --> low numerical value --> low number of tapers
        High peak resolution --> high numerical value --> high number of tapers
        When given as a `float`, it is taken as frequency in Hz.
        Default: None.
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [2].
        Default: last axis (-1).

    Notes
    -----
    1. There is a parameter hierarchy regarding n_segments and len_segment. The
       former parameter is ignored if the latter one is passed.

    2. There is a parameter hierarchy regarding nw, num_tapers and
       peak_resolution. If peak_resolution is provided, it determines both nw
       and the num_tapers. Specifying num_tapers has an effect only if
       peak_resolution is not provided.

    Returns
    -------
    freqs : np.ndarray
        Frequencies associated with power estimate in `psd`
    psd : np.ndarray
        PSD estimate of the time series in `signal`

    Raises
    ------
    ValueError
        If `peak_resolution` is None and `num_tapers` is not a positive number.

        If `frequency_resolution` is too high for the given data size.

        If `frequency_resolution` is None and `len_segment` is not a positive
        number.

        If `frequency_resolution` is None and `len_segment` is greater than the
        length of data at `axis`.

        If both `frequency_resolution` and `len_segment` are None and
        `n_segments` is not a positive number.

        If both `frequency_resolution` and `len_segment` are None and
        `n_segments` is greater than the length of data at `axis`.

    TypeError
        If `peak_resolution` is None and `num_tapers` is not an int.
    """

    # When the input is AnalogSignal, the data is added after rolling the axis
    # for time index to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))

    # Number of data points in time series
    if data.ndim == 1:
        length_signal = np.shape(data)[0]
    else:
        length_signal = np.shape(data)[1]

    # If the data is given as AnalogSignal, use its attribute to specify the
    # sampling frequency
    if hasattr(signal, 'sampling_rate'):
        fs = signal.sampling_rate.rescale('Hz').magnitude

    # If fs and peak resolution is pq.Quantity, get magnitude
    if isinstance(fs, pq.quantity.Quantity):
        fs = fs.rescale('Hz').magnitude

    # Determine length per segment - n_per_seg
    if frequency_resolution is not None:
        if frequency_resolution <= 0:
            raise ValueError("frequency_resolution must be positive")
        if isinstance(frequency_resolution, pq.quantity.Quantity):
            dF = frequency_resolution.rescale('Hz').magnitude
        else:
            dF = frequency_resolution
        n_per_seg = int(fs / dF)
        if n_per_seg > data.shape[axis]:
            raise ValueError("frequency_resolution is too high for the given "
                             "data size")
    elif len_segment is not None:
        if len_segment <= 0:
            raise ValueError("len_seg must be a positive number")
        elif data.shape[axis] < len_segment:
            raise ValueError("len_seg must be shorter than the data length")
        n_per_seg = len_segment
    else:
        if n_segments <= 0:
            raise ValueError("n_segments must be a positive number")
        elif data.shape[axis] < n_segments:
            raise ValueError("n_segments must be smaller than the data length")
        # when only *n_segments* is given, *n_per_seg* is determined by solving
        # the following equation:
        #  n_segments * n_per_seg - (n_segments-1) * overlap * n_per_seg =
        #     data.shape[-1]
        #  --------------------   ===============================  ^^^^^^^^^^^
        # summed segment lengths        total overlap              data length
        n_per_seg = int(data.shape[axis] /
                        (n_segments - overlap * (n_segments - 1)))

    n_overlap = int(n_per_seg * overlap)
    n_segments = int((length_signal - n_overlap) / (n_per_seg - n_overlap))

    if isinstance(peak_resolution, pq.quantity.Quantity):
        peak_resolution = peak_resolution.rescale('Hz').magnitude

    # Determine time-halfbandwidth product from given parameters
    if peak_resolution is not None:
        if peak_resolution <= 0:
            raise ValueError("peak_resolution must be positive")
        nw = n_per_seg / fs * peak_resolution / 2
        num_tapers = int(np.floor(2*nw) - 1)

    if num_tapers is None:
        num_tapers = int(np.floor(2*nw) - 1)
    else:
        if not isinstance(num_tapers, int):
            raise TypeError("num_tapers must be integer")
        if num_tapers <= 0:
            raise ValueError("num_tapers must be positive")

    # Generate frequencies of PSD estimate
    freqs = np.fft.rfftfreq(n_per_seg, d=1/fs)

    # Zero-pad signal to fit segment length
    remainder = length_signal % n_per_seg

    if data.ndim == 1:
        data = np.pad(data, pad_width=(0, remainder),
                      mode='constant', constant_values=0)
        # Generate array for storing PSD estimates of segments
        psd_estimates = np.zeros((n_segments, len(freqs)))
    else:
        data = np.pad(data, [(0, 0), (0, remainder)],
                      mode='constant', constant_values=0)
        # Generate array for storing PSD estimates of segments
        psd_estimates = np.zeros((n_segments, data.shape[0], len(freqs)))

    # Determine the number of samples given overlap
    n_overlap_step = n_per_seg - n_overlap

    for i in range(n_segments):
        # Get slepian functions (sym=False used for spectral analysis)
        slepian_fcts = scipy.signal.windows.dpss(M=n_per_seg,
                                                 NW=nw,
                                                 Kmax=num_tapers,
                                                 sym=False)

        # Calculate approximately independent spectrum estimates
        if data.ndim == 1:
            tapered_signal = (data[i * n_overlap_step:
                                   i * n_overlap_step + n_per_seg]
                              * slepian_fcts)
        else:
            # Use broadcasting to match dim for point-wise multiplication
            tapered_signal = (data[:,
                                   np.newaxis,
                                   i * n_overlap_step:
                                   i * n_overlap_step + n_per_seg]
                              * slepian_fcts)

        # Determine Fourier transform of tapered signal
        spectrum_estimates = np.abs(np.fft.rfft(tapered_signal, axis=-1))**2
        spectrum_estimates[..., 1:] *= 2

        # Average Fourier transform windowed signal
        psd_segment = np.mean(spectrum_estimates, axis=-2) / fs

        psd_estimates[i] = psd_segment

    psd = np.mean(np.asarray(psd_estimates), axis=0)

    # Attach proper units to return values
    if isinstance(signal, pq.quantity.Quantity):
        psd = psd * signal.units * signal.units / pq.Hz
        freqs = freqs * pq.Hz

    return freqs, psd


@deprecated_alias(x='signal_i', y='signal_j', num_seg='n_segments',
                  len_seg='len_segment', freq_res='frequency_resolution')
def welch_coherence(signal_i, signal_j, n_segments=8, len_segment=None,
                    frequency_resolution=None, overlap=0.5, fs=1.0,
                    window='hann', nfft=None, detrend='constant',
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
        Default: 8
    len_segment : int, optional
        Length of segments. This parameter is ignored if `frequency_resolution`
        is given. If None, it is determined from other parameters.
        Default: None
    frequency_resolution : pq.Quantity or float, optional
        Desired frequency resolution of the obtained coherence estimate in
        terms of the interval between adjacent frequency bins. When given as a
        `float`, it is taken as frequency in Hz.
        If None, it is determined from other parameters.
        Default: None
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped)
    fs : pq.Quantity or float, optional
        Specifies the sampling frequency of the input time series. When the
        input time series are given as `neo.AnalogSignal`, the sampling
        frequency is taken from their attribute and this parameter is ignored.
        Default: 1.0
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        See Notes [1].
        Default: 'hann'
    nfft : int, optional
        Length of the FFT used.
        See Notes [1].
        Default: None
    detrend : str or function or False, optional
        Specifies how to detrend each segment.
        See Notes [1].
        Default: 'constant'
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz. If 'spectrum', computes the power spectrum where Pxx has
        units of V**2, if `signal` is measured in V and `fs` is measured in
        Hz.
        See Notes [1].
        Default: 'density'
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [1].
        Default: last axis (-1)

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
        When the input arrays `signal_i` and `signal_j` are multidimensional,
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

    Examples
    --------
    >>> import neo
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.spectral import welch_coherence
    >>> signal = neo.AnalogSignal(np.cos(np.linspace(0, 2 * np.pi, num=100)),
    ...     sampling_rate=20 * pq.Hz, units='mV')

    Sampling frequency will be taken as `signal.sampling_rate`.

    >>> freq, coherency, phase_lag = welch_coherence(signal, signal)
    >>> freq
    array([ 0.        ,  0.90909091,  1.81818182,  2.72727273,  3.63636364,
            4.54545455,  5.45454545,  6.36363636,  7.27272727,  8.18181818,
            9.09090909, 10.        ]) * Hz


    >>> coherency.flatten()
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> phase_lag.flatten()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) * rad

    """

    # TODO: code duplication with welch_psd()
    # 'hanning' window was removed with release of scipy 1.9.0, it was
    # deprecated since 1.1.0.
    if window == 'hanning':
        warnings.warn("'hanning' is deprecated and was removed from scipy "
                      "with release 1.9.0. Please use 'hann' instead",
                      DeprecationWarning)
        window = 'hann'

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
        if xdata.shape[axis] < len_segment:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_segment
    else:
        if n_segments <= 0:
            raise ValueError("n_segments must be a positive number")
        if xdata.shape[axis] < n_segments:
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
    return welch_coherence(*args, **kwargs)
