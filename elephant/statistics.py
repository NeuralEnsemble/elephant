# -*- coding: utf-8 -*-
"""
Statistical measures of spike trains (e.g., Fano factor) and functions to estimate firing rates.

:copyright: Copyright 2014-2016 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import numpy as np
import quantities as pq
import scipy.stats
import scipy.signal
import neo
from neo.core import SpikeTrain
import elephant.conversion as conv
import elephant.kernels as kernels
import warnings
# warnings.simplefilter('always', DeprecationWarning)


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the same as spiketrain.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    axis : int, optional
           The axis along which the difference is taken.
           Default is the last axis.

    Returns
    -------

    NumPy array or quantities array.

    """
    if axis is None:
        axis = -1
    if isinstance(spiketrain, neo.SpikeTrain):
        intervals = np.diff(
            np.sort(spiketrain.times.view(pq.Quantity)), axis=axis)
    else:
        intervals = np.diff(np.sort(spiketrain), axis=axis)
    return intervals


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the inverse of the spiketrain.

    The interval over which the firing rate is calculated can be optionally
    controlled with `t_start` and `t_stop`

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    t_start : float or Quantity scalar, optional
              The start time to use for the interval.
              If not specified, retrieved from the``t_start`
              attribute of `spiketrain`.  If that is not present, default to
              `0`.  Any value from `spiketrain` below this value is ignored.
    t_stop : float or Quantity scalar, optional
             The stop time to use for the time points.
             If not specified, retrieved from the `t_stop`
             attribute of `spiketrain`.  If that is not present, default to
             the maximum value of `spiketrain`.  Any value from
             `spiketrain` above this value is ignored.
    axis : int, optional
           The axis over which to do the calculation.
           Default is `None`, do the calculation over the flattened array.

    Returns
    -------

    float, quantities scalar, NumPy array or quantities array.

    Notes
    -----

    If `spiketrain` is a Quantity or Neo SpikeTrain and `t_start` or `t_stop`
    are not, `t_start` and `t_stop` are assumed to have the same units as
    `spiketrain`.

    Raises
    ------

    TypeError
        If `spiketrain` is a NumPy array and `t_start` or `t_stop`
        is a quantity scalar.

    """
    if t_start is None:
        t_start = getattr(spiketrain, 't_start', 0)

    found_t_start = False
    if t_stop is None:
        if hasattr(spiketrain, 't_stop'):
            t_stop = spiketrain.t_stop
        else:
            t_stop = np.max(spiketrain, axis=axis)
            found_t_start = True

    # figure out what units, if any, we are dealing with
    if hasattr(spiketrain, 'units'):
        units = spiketrain.units
    else:
        units = None

    # convert everything to the same units
    if hasattr(t_start, 'units'):
        if units is None:
            raise TypeError('t_start cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_start = t_start.rescale(units)
    elif units is not None:
        t_start = pq.Quantity(t_start, units=units)
    if hasattr(t_stop, 'units'):
        if units is None:
            raise TypeError('t_stop cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_stop = t_stop.rescale(units)
    elif units is not None:
        t_stop = pq.Quantity(t_stop, units=units)

    if not axis or not found_t_start:
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop),
                      axis=axis) / (t_stop - t_start)
    else:
        # this is needed to handle broadcasting between spiketrain and t_stop
        t_stop_test = np.expand_dims(t_stop, axis)
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop_test),
                      axis=axis) / (t_stop - t_start)


# we make `cv` an alias for scipy.stats.variation for the convenience
# of former NeuroTools users
cv = scipy.stats.variation


def fanofactor(spiketrains):
    """
    Evaluates the empirical Fano factor F of the spike counts of
    a list of `neo.core.SpikeTrain` objects.

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], F is defined as:

                        F := var(v)/mean(v).

    The Fano factor is typically computed for spike trains representing the
    activity of the same neuron over different trials. The higher F, the larger
    the cross-trial non-stationarity. In theory for a time-stationary Poisson
    process, F=1.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain objects, quantity arrays, numpy arrays or lists
        Spike trains for which to compute the Fano factor of spike counts.

    Returns
    -------
    fano : float or nan
        The Fano factor of the spike counts of the input spike trains. If an
        empty list is specified, or if all spike trains are empty, F:=nan.
    """
    # Build array of spike counts (one per spike train)
    spike_counts = np.array([len(t) for t in spiketrains])

    # Compute FF
    if all([count == 0 for count in spike_counts]):
        fano = np.nan
    else:
        fano = spike_counts.var() / spike_counts.mean()
    return fano


def lv(v):
    """
    Calculate the measure of local variation LV for
    a sequence of time intervals between events.

    Given a vector v containing a sequence of intervals, the LV is
    defined as:

    .math $$ LV := \\frac{1}{N}\\sum_{i=1}^{N-1}

                   \\frac{3(isi_i-isi_{i+1})^2}
                          {(isi_i+isi_{i+1})^2} $$

    The LV is typically computed as a substitute for the classical
    coefficient of variation for sequences of events which include
    some (relatively slow) rate fluctuation.  As with the CV, LV=1 for
    a sequence of intervals generated by a Poisson process.

    Parameters
    ----------

    v : quantity array, numpy array or list
        Vector of consecutive time intervals

    Returns
    -------
    lvar : float
       The LV of the inter-spike interval of the input sequence.

    Raises
    ------
    AttributeError :
       If an empty list is specified, or if the sequence has less
       than two entries, an AttributeError will be raised.
    ValueError :
        Only vector inputs are supported.  If a matrix is passed to the
        function a ValueError will be raised.


    References
    ----------
    ..[1] Shinomoto, S., Shima, K., & Tanji, J. (2003). Differences in spiking
    patterns among cortical neurons. Neural Computation, 15, 2823–2842.


    """
    # convert to array, cast to float
    v = np.asarray(v)

    # ensure we have enough entries
    if v.size < 2:
        raise AttributeError("Input size is too small. Please provide "
                             "an input with more than 1 entry.")

    # calculate LV and return result
    # raise error if input is multi-dimensional
    return 3. * np.mean(np.power(np.diff(v) / (v[:-1] + v[1:]), 2))


def cv2(v):
    """
    Calculate the measure of CV2 for a sequence of time intervals between 
    events.

    Given a vector v containing a sequence of intervals, the CV2 is
    defined as:

    .math $$ CV2 := \\frac{1}{N}\\sum_{i=1}^{N-1}

                   \\frac{2|isi_{i+1}-isi_i|}
                          {|isi_{i+1}+isi_i|} $$

    The CV2 is typically computed as a substitute for the classical
    coefficient of variation (CV) for sequences of events which include
    some (relatively slow) rate fluctuation.  As with the CV, CV2=1 for
    a sequence of intervals generated by a Poisson process.

    Parameters
    ----------

    v : quantity array, numpy array or list
        Vector of consecutive time intervals

    Returns
    -------
    cv2 : float
       The CV2 of the inter-spike interval of the input sequence.

    Raises
    ------
    AttributeError :
       If an empty list is specified, or if the sequence has less
       than two entries, an AttributeError will be raised.
    AttributeError :
        Only vector inputs are supported.  If a matrix is passed to the
        function an AttributeError will be raised.

    References
    ----------
    ..[1] Holt, G. R., Softky, W. R., Koch, C., & Douglas, R. J. (1996). 
    Comparison of discharge variability in vitro and in vivo in cat visual 
    cortex neurons. Journal of neurophysiology, 75(5), 1806-1814.
    """
    # convert to array, cast to float
    v = np.asarray(v)

    # ensure the input ia a vector
    if len(v.shape) > 1:
        raise AttributeError("Input shape is larger than 1. Please provide "
                             "a vector in input.")

    # ensure we have enough entries
    if v.size < 2:
        raise AttributeError("Input size is too small. Please provide "
                             "an input with more than 1 entry.")

    # calculate CV2 and return result
    return 2. * np.mean(np.absolute(np.diff(v)) / (v[:-1] + v[1:]))


# sigma2kw and kw2sigma only needed for oldfct_instantaneous_rate!
# to finally be taken out of Elephant

def sigma2kw(form): # pragma: no cover
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
    if form.upper() == 'BOX':
        coeff = 2.0 * np.sqrt(3)
    elif form.upper() == 'TRI':
        coeff = 2.0 * np.sqrt(6)
    elif form.upper() == 'EPA':
        coeff = 2.0 * np.sqrt(5)
    elif form.upper() == 'GAU':
        coeff = 2.0 * 2.7  # > 99% of distribution weight
    elif form.upper() == 'ALP':
        coeff = 5.0
    elif form.upper() == 'EXP':
        coeff = 5.0

    return coeff


def kw2sigma(form): # pragma: no cover
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
    return 1/sigma2kw(form)


# to finally be taken out of Elephant
def make_kernel(form, sigma, sampling_period, direction=1): # pragma: no cover
    """
    Creates kernel functions for convolution.

    Constructs a numeric linear convolution kernel of basic shape to be used
    for data smoothing (linear low pass filtering) and firing rate estimation
    from single trial or trial-averaged spike trains.

    Exponential and alpha kernels may also be used to represent postynaptic
    currents / potentials in a linear (current-based) model.

    Parameters
    ----------
    form : {'BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'}
        Kernel form. Currently implemented forms are BOX (boxcar),
        TRI (triangle), GAU (gaussian), EPA (epanechnikov), EXP (exponential),
        ALP (alpha function). EXP and ALP are asymmetric kernel forms and
        assume optional parameter `direction`.
    sigma : Quantity
        Standard deviation of the distribution associated with kernel shape.
        This parameter defines the time resolution of the kernel estimate
        and makes different kernels comparable (cf. [1] for symmetric kernels).
        This is used here as an alternative definition to the cut-off
        frequency of the associated linear filter.
    sampling_period : float
        Temporal resolution of input and output.
    direction : {-1, 1}
        Asymmetric kernels have two possible directions.
        The values are -1 or 1, default is 1. The
        definition here is that for direction = 1 the
        kernel represents the impulse response function
        of the linear filter. Default value is 1.

    Returns
    -------
    kernel : numpy.ndarray
        Array of kernel. The length of this array is always an odd
        number to represent symmetric kernels such that the center bin
        coincides with the median of the numeric array, i.e for a
        triangle, the maximum will be at the center bin with equal
        number of bins to the right and to the left.
    norm : float
        For rate estimates. The kernel vector is normalized such that
        the sum of all entries equals unity sum(kernel)=1. When
        estimating rate functions from discrete spike data (0/1) the
        additional parameter `norm` allows for the normalization to
        rate in spikes per second.

        For example:
        ``rate = norm * scipy.signal.lfilter(kernel, 1, spike_data)``
    m_idx : int
        Index of the numerically determined median (center of gravity)
        of the kernel function.

    Examples
    --------
    To obtain single trial rate function of trial one should use::

        r = norm * scipy.signal.fftconvolve(sua, kernel)

    To obtain trial-averaged spike train one should use::

        r_avg = norm * scipy.signal.fftconvolve(sua, np.mean(X,1))

    where `X` is an array of shape `(l,n)`, `n` is the number of trials and
    `l` is the length of each trial.

    See also
    --------
    elephant.statistics.instantaneous_rate

    References
    ----------

    .. [1] Meier R, Egert U, Aertsen A, Nawrot MP, "FIND - a unified framework
       for neural data analysis"; Neural Netw. 2008 Oct; 21(8):1085-93.

    .. [2] Nawrot M, Aertsen A, Rotter S, "Single-trial estimation of neuronal
       firing rates - from single neuron spike trains to population activity";
       J. Neurosci Meth 94: 81-92; 1999.

    """
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
    forms_abbreviated = np.array(['BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'])
    forms_verbose = np.array(['boxcar', 'triangle', 'gaussian', 'epanechnikov',
                     'exponential', 'alpha'])
    if form in forms_verbose:
        form = forms_abbreviated[forms_verbose == form][0]

    assert form.upper() in ('BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'), \
        "form must be one of either 'BOX','TRI','GAU','EPA','EXP' or 'ALP'!"

    assert direction in (1, -1), "direction must be either 1 or -1"

    # conversion to SI units (s)
    if sigma < 0:
        raise ValueError('sigma must be positive!')

    SI_sigma = sigma.rescale('s').magnitude
    SI_time_stamp_resolution = sampling_period.rescale('s').magnitude

    norm = 1. / SI_time_stamp_resolution

    if form.upper() == 'BOX':
        w = 2.0 * SI_sigma * np.sqrt(3)
        # always odd number of bins
        width = 2 * np.floor(w / 2.0 / SI_time_stamp_resolution) + 1
        height = 1. / width
        kernel = np.ones((1, width)) * height  # area = 1

    elif form.upper() == 'TRI':
        w = 2 * SI_sigma * np.sqrt(6)
        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)
        trileft = np.arange(1, halfwidth + 2)
        triright = np.arange(halfwidth, 0, -1)  # odd number of bins
        triangle = np.append(trileft, triright)
        kernel = triangle / triangle.sum()  # area = 1

    elif form.upper() == 'EPA':
        w = 2.0 * SI_sigma * np.sqrt(5)
        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)
        base = np.arange(-halfwidth, halfwidth + 1)
        parabula = base**2
        epanech = parabula.max() - parabula  # inverse parabula
        kernel = epanech / epanech.sum()  # area = 1

    elif form.upper() == 'GAU':
        w = 2.0 * SI_sigma * 2.7  # > 99% of distribution weight
        halfwidth = np.floor(w / 2.0 / SI_time_stamp_resolution)  # always odd
        base = np.arange(-halfwidth, halfwidth + 1) * SI_time_stamp_resolution
        g = np.exp(
            -(base**2) / 2.0 / SI_sigma**2) / SI_sigma / np.sqrt(2.0 * np.pi)
        kernel = g / g.sum()  # area = 1

    elif form.upper() == 'ALP':
        w = 5.0 * SI_sigma
        alpha = np.arange(
            1, (
                2.0 * np.floor(w / SI_time_stamp_resolution / 2.0) + 1) +
            1) * SI_time_stamp_resolution
        alpha = (2.0 / SI_sigma**2) * alpha * np.exp(
            -alpha * np.sqrt(2) / SI_sigma)
        kernel = alpha / alpha.sum()  # normalization
        if direction == -1:
            kernel = np.flipud(kernel)

    elif form.upper() == 'EXP':
        w = 5.0 * SI_sigma
        expo = np.arange(
            1, (
                2.0 * np.floor(w / SI_time_stamp_resolution / 2.0) + 1) +
            1) * SI_time_stamp_resolution
        expo = np.exp(-expo / SI_sigma)
        kernel = expo / expo.sum()
        if direction == -1:
            kernel = np.flipud(kernel)

    kernel = kernel.ravel()
    m_idx = np.nonzero(kernel.cumsum() >= 0.5)[0].min()

    return kernel, norm, m_idx


# to finally be taken out of Elephant
def oldfct_instantaneous_rate(spiketrain, sampling_period, form,
                       sigma='auto', t_start=None, t_stop=None,
                       acausal=True, trim=False): # pragma: no cover
    """
    Estimate instantaneous firing rate by kernel convolution.

    Parameters
    -----------
    spiketrain: 'neo.SpikeTrain'
        Neo object that contains spike times, the unit of the time stamps
        and t_start and t_stop of the spike train.
    sampling_period : Quantity
        time stamp resolution of the spike times. the same resolution will
        be assumed for the kernel
    form : {'BOX', 'TRI', 'GAU', 'EPA', 'EXP', 'ALP'}
        Kernel form. Currently implemented forms are BOX (boxcar),
        TRI (triangle), GAU (gaussian), EPA (epanechnikov), EXP (exponential),
        ALP (alpha function). EXP and ALP are asymmetric kernel forms and
        assume optional parameter `direction`.
    sigma : string or Quantity
        Standard deviation of the distribution associated with kernel shape.
        This parameter defines the time resolution of the kernel estimate
        and makes different kernels comparable (cf. [1] for symmetric kernels).
        This is used here as an alternative definition to the cut-off
        frequency of the associated linear filter.
        Default value is 'auto'. In this case, the optimized kernel width for
        the rate estimation is calculated according to [1]. Note that the
        automatized calculation of the kernel width ONLY works for gaussian
        kernel shapes!
    t_start : Quantity (Optional)
        start time of the interval used to compute the firing rate, if None
        assumed equal to spiketrain.t_start
        Default:None
    t_stop : Qunatity
        End time of the interval used to compute the firing rate (included).
        If none assumed equal to spiketrain.t_stop
        Default:None
    acausal : bool
        if True, acausal filtering is used, i.e., the gravity center of the
        filter function is aligned with the spike to convolve
        Default:None
    m_idx : int
        index of the value in the kernel function vector that corresponds
        to its gravity center. this parameter is not mandatory for
        symmetrical kernels but it is required when asymmetrical kernels
        are to be aligned at their gravity center with the event times if None
        is assumed to be the median value of the kernel support
        Default : None
    trim : bool
        if True, only the 'valid' region of the convolved
        signal are returned, i.e., the points where there
        isn't complete overlap between kernel and spike train
        are discarded
        NOTE: if True and an asymmetrical kernel is provided
        the output will not be aligned with [t_start, t_stop]

    Returns
    -------
    rate : neo.AnalogSignal
        Contains the rate estimation in unit hertz (Hz).
        Has a property 'rate.times' which contains the time axis of the rate
        estimate. The unit of this property is the same as the resolution that
        is given as an argument to the function.

    Raises
    ------
    TypeError:
        If argument value for the parameter `sigma` is not a quantity object
        or string 'auto'.

    See also
    --------
    elephant.statistics.make_kernel

    References
    ----------
    ..[1] H. Shimazaki, S. Shinomoto, J Comput Neurosci (2010) 29:171–182.
    """
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
    if sigma == 'auto':
        form = 'GAU'
        unit = spiketrain.units
        kernel_width = sskernel(spiketrain.magnitude, tin=None,
                                bootstrap=True)['optw']
        sigma = kw2sigma(form) * kernel_width * unit
    elif not isinstance(sigma, pq.Quantity):
        raise TypeError('sigma must be either a quantities object or "auto".'
                        ' Found: %s, value %s' % (type(sigma), str(sigma)))

    kernel, norm, m_idx = make_kernel(form=form, sigma=sigma,
                                      sampling_period=sampling_period)
    units = pq.CompoundUnit(
        "%s*s" % str(sampling_period.rescale('s').magnitude))
    spiketrain = spiketrain.rescale(units)
    if t_start is None:
        t_start = spiketrain.t_start
    else:
        t_start = t_start.rescale(spiketrain.units)

    if t_stop is None:
        t_stop = spiketrain.t_stop
    else:
        t_stop = t_stop.rescale(spiketrain.units)

    time_vector = np.zeros(int((t_stop - t_start)) + 1)

    spikes_slice = spiketrain.time_slice(t_start, t_stop) \
        if len(spiketrain) else np.array([])

    for spike in spikes_slice:
        index = int((spike - t_start))
        time_vector[index] += 1

    r = norm * scipy.signal.fftconvolve(time_vector, kernel, 'full')
    if np.any(r < 0):
        warnings.warn('Instantaneous firing rate approximation contains '
                      'negative values, possibly caused due to machine '
                      'precision errors')

    if acausal:
        if not trim:
            r = r[m_idx:-(kernel.size - m_idx)]

        elif trim:
            r = r[2 * m_idx:-2 * (kernel.size - m_idx)]
            t_start = t_start + m_idx * spiketrain.units
            t_stop = t_stop - ((kernel.size) - m_idx) * spiketrain.units

    else:
        if not trim:
            r = r[m_idx:-(kernel.size - m_idx)]

        elif trim:
            r = r[2 * m_idx:-2 * (kernel.size - m_idx)]
            t_start = t_start + m_idx * spiketrain.units
            t_stop = t_stop - ((kernel.size) - m_idx) * spiketrain.units

    rate = neo.AnalogSignal(signal=r.reshape(r.size, 1),
                                 sampling_period=sampling_period,
                                 units=pq.Hz, t_start=t_start)

    return rate, sigma


def instantaneous_rate(spiketrain, sampling_period, kernel='auto',
                       cutoff=5.0, t_start=None, t_stop=None, trim=False):

    """
    Estimates instantaneous firing rate by kernel convolution.

    Parameters
    -----------
    spiketrain : 'neo.SpikeTrain'
        Neo object that contains spike times, the unit of the time stamps
        and t_start and t_stop of the spike train.
    sampling_period : Time Quantity
        Time stamp resolution of the spike times. The same resolution will
        be assumed for the kernel
    kernel : string 'auto' or callable object of :class:`Kernel` from module
        'kernels.py'. Currently implemented kernel forms are rectangular,
        triangular, epanechnikovlike, gaussian, laplacian, exponential,
        and alpha function.
        Example: kernel = kernels.RectangularKernel(sigma=10*ms, invert=False)
        The kernel is used for convolution with the spike train and its
        standard deviation determines the time resolution of the instantaneous
        rate estimation.
        Default: 'auto'. In this case, the optimized kernel width for the 
        rate estimation is calculated according to [1] and with this width
        a gaussian kernel is constructed. Automatized calculation of the 
        kernel width is not available for other than gaussian kernel shapes.
    cutoff : float
        This factor determines the cutoff of the probability distribution of
        the kernel, i.e., the considered width of the kernel in terms of 
        multiples of the standard deviation sigma.
        Default: 5.0
    t_start : Time Quantity (optional)
        Start time of the interval used to compute the firing rate. If None
        assumed equal to spiketrain.t_start
        Default: None
    t_stop : Time Quantity (optional)
        End time of the interval used to compute the firing rate (included).
        If None assumed equal to spiketrain.t_stop
        Default: None
    trim : bool
        if False, the output of the Fast Fourier Transformation being a longer
        vector than the input vector by the size of the kernel is reduced back
        to the original size of the considered time interval of the spiketrain
        using the median of the kernel.
        if True, only the region of the convolved signal is returned, where
        there is complete overlap between kernel and spike train. This is
        achieved by reducing the length of the output of the Fast Fourier
        Transformation by a total of two times the size of the kernel, and
        t_start and t_stop are adjusted.
        Default: False

    Returns
    -------
    rate : neo.AnalogSignal
        Contains the rate estimation in unit hertz (Hz).
        Has a property 'rate.times' which contains the time axis of the rate
        estimate. The unit of this property is the same as the resolution that
        is given via the argument 'sampling_period' to the function.

    Raises
    ------
    TypeError:
        If `spiketrain` is not an instance of :class:`SpikeTrain` of Neo.
        If `sampling_period` is not a time quantity.
        If `kernel` is neither instance of :class:`Kernel` or string 'auto'.
        If `cutoff` is neither float nor int.
        If `t_start` and `t_stop` are neither None nor a time quantity.
        If `trim` is not bool.

    ValueError:
        If `sampling_period` is smaller than zero.

    Example
    --------
    kernel = kernels.AlphaKernel(sigma = 0.05*s, invert = True)
    rate = instantaneous_rate(spiketrain, sampling_period = 2*ms, kernel)

    References
    ----------
    ..[1] H. Shimazaki, S. Shinomoto, J Comput Neurosci (2010) 29:171–182.

    """
    # Checks of input variables:
    if not isinstance(spiketrain, SpikeTrain):
        raise TypeError(
            "spiketrain must be instance of :class:`SpikeTrain` of Neo!\n"
            "    Found: %s, value %s" % (type(spiketrain), str(spiketrain)))

    if not (isinstance(sampling_period, pq.Quantity) and
            sampling_period.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality):
        raise TypeError(
            "The sampling period must be a time quantity!\n"
            "    Found: %s, value %s" % (type(sampling_period), str(sampling_period)))

    if sampling_period.magnitude < 0:
        raise ValueError("The sampling period must be larger than zero.")

    if kernel == 'auto':
        kernel_width = sskernel(spiketrain.magnitude, tin=None,
                                bootstrap=True)['optw']
        unit = spiketrain.units
        sigma = 1/(2.0 * 2.7) * kernel_width * unit
        # factor 2.0 connects kernel width with its half width,
        # factor 2.7 connects half width of Gaussian distribution with
        #             99% probability mass with its standard deviation.
        kernel = kernels.GaussianKernel(sigma)
    elif not isinstance(kernel, kernels.Kernel):
        raise TypeError(
            "kernel must be either instance of :class:`Kernel` "
            "or the string 'auto'!\n"
            "    Found: %s, value %s" % (type(kernel), str(kernel)))

    if not (isinstance(cutoff, float) or isinstance(cutoff, int)):
        raise TypeError("cutoff must be float or integer!")

    if not (t_start is None or (isinstance(t_start, pq.Quantity) and
            t_start.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality)):
        raise TypeError("t_start must be a time quantity!")

    if not (t_stop is None or (isinstance(t_stop, pq.Quantity) and
            t_stop.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality)):
        raise TypeError("t_stop must be a time quantity!")

    if not (isinstance(trim, bool)):
        raise TypeError("trim must be bool!")

    # main function:
    units = pq.CompoundUnit("%s*s" % str(sampling_period.rescale('s').magnitude))
    spiketrain = spiketrain.rescale(units)
    if t_start is None:
        t_start = spiketrain.t_start
    else:
        t_start = t_start.rescale(spiketrain.units)

    if t_stop is None:
        t_stop = spiketrain.t_stop
    else:
        t_stop = t_stop.rescale(spiketrain.units)

    time_vector = np.zeros(int((t_stop - t_start)) + 1)

    spikes_slice = spiketrain.time_slice(t_start, t_stop) \
        if len(spiketrain) else np.array([])

    for spike in spikes_slice:
        index = int((spike - t_start))
        time_vector[index] += 1

    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")

    t_arr = np.arange(-cutoff * kernel.sigma.rescale(units).magnitude,
                      cutoff * kernel.sigma.rescale(units).magnitude +
                      sampling_period.rescale(units).magnitude,
                      sampling_period.rescale(units).magnitude) * units

    r = scipy.signal.fftconvolve(time_vector,
                                 kernel(t_arr).rescale(pq.Hz).magnitude, 'full')
    if np.any(r < 0):
        warnings.warn("Instantaneous firing rate approximation contains "
                      "negative values, possibly caused due to machine "
                      "precision errors.")

    if not trim:
        r = r[kernel.median_index(t_arr):-(kernel(t_arr).size -
                                           kernel.median_index(t_arr))]
    elif trim:
        r = r[2 * kernel.median_index(t_arr):-2 * (kernel(t_arr).size -
                                                   kernel.median_index(t_arr))]
        t_start += kernel.median_index(t_arr) * spiketrain.units
        t_stop -= (kernel(t_arr).size -
                   kernel.median_index(t_arr)) * spiketrain.units

    rate = neo.AnalogSignal(signal=r.reshape(r.size, 1),
                            sampling_period=sampling_period,
                            units=pq.Hz, t_start=t_start, t_stop=t_stop)

    return rate


def time_histogram(spiketrains, binsize, t_start=None, t_stop=None,
                   output='counts', binary=False):
    """
    Time Histogram of a list of :attr:`neo.SpikeTrain` objects.

    Parameters
    ----------
    spiketrains : List of neo.SpikeTrain objects
        Spiketrains with a common time axis (same `t_start` and `t_stop`)
    binsize : quantities.Quantity
        Width of the histogram's time bins.
    t_start, t_stop : Quantity (optional)
        Start and stop time of the histogram. Only events in the input
        `spiketrains` falling between `t_start` and `t_stop` (both included)
        are considered in the histogram. If `t_start` and/or `t_stop` are not
        specified, the maximum `t_start` of all :attr:spiketrains is used as
        `t_start`, and the minimum `t_stop` is used as `t_stop`.
        Default: t_start = t_stop = None
    output : str (optional)
        Normalization of the histogram. Can be one of:
        * `counts`'`: spike counts at each bin (as integer numbers)
        * `mean`: mean spike counts per spike train
        * `rate`: mean spike rate per spike train. Like 'mean', but the
          counts are additionally normalized by the bin width.
    binary : bool (optional)
        If **True**, indicates whether all spiketrain objects should first
        binned to a binary representation (using the `BinnedSpikeTrain` class
        in the `conversion` module) and the calculation of the histogram is
        based on this representation.
        Note that the output is not binary, but a histogram of the converted,
        binary representation.
        Default: False

    Returns
    -------
    time_hist : neo.AnalogSignal
        A neo.AnalogSignal object containing the histogram values.
        `AnalogSignal[j]` is the histogram computed between
        `t_start + j * binsize` and `t_start + (j + 1) * binsize`.

    See also
    --------
    elephant.conversion.BinnedSpikeTrain
    """
    min_tstop = 0
    if t_start is None:
        # Find the internal range for t_start, where all spike trains are
        # defined; cut all spike trains taking that time range only
        max_tstart, min_tstop = conv._get_start_stop_from_input(spiketrains)
        t_start = max_tstart
        if not all([max_tstart == t.t_start for t in spiketrains]):
            warnings.warn(
                "Spiketrains have different t_start values -- "
                "using maximum t_start as t_start.")

    if t_stop is None:
        # Find the internal range for t_stop
        if min_tstop:
            t_stop = min_tstop
            if not all([min_tstop == t.t_stop for t in spiketrains]):
                warnings.warn(
                    "Spiketrains have different t_stop values -- "
                    "using minimum t_stop as t_stop.")
        else:
            min_tstop = conv._get_start_stop_from_input(spiketrains)[1]
            t_stop = min_tstop
            if not all([min_tstop == t.t_stop for t in spiketrains]):
                warnings.warn(
                    "Spiketrains have different t_stop values -- "
                    "using minimum t_stop as t_stop.")

    sts_cut = [st.time_slice(t_start=t_start, t_stop=t_stop) for st in
               spiketrains]

    # Bin the spike trains and sum across columns
    bs = conv.BinnedSpikeTrain(sts_cut, t_start=t_start, t_stop=t_stop,
                               binsize=binsize)

    if binary:
        bin_hist = bs.to_sparse_bool_array().sum(axis=0)
    else:
        bin_hist = bs.to_sparse_array().sum(axis=0)
    # Flatten array
    bin_hist = np.ravel(bin_hist)
    # Renormalise the histogram
    if output == 'counts':
        # Raw
        bin_hist = bin_hist * pq.dimensionless
    elif output == 'mean':
        # Divide by number of input spike trains
        bin_hist = bin_hist * 1. / len(spiketrains) * pq.dimensionless
    elif output == 'rate':
        # Divide by number of input spike trains and bin width
        bin_hist = bin_hist * 1. / len(spiketrains) / binsize
    else:
        raise ValueError('Parameter output is not valid.')

    return neo.AnalogSignal(signal=bin_hist.reshape(bin_hist.size, 1),
                                 sampling_period=binsize, units=bin_hist.units,
                                 t_start=t_start)


def complexity_pdf(spiketrains, binsize):
    """
    Complexity Distribution [1] of a list of :attr:`neo.SpikeTrain` objects.

    Probability density computed from the complexity histogram which is the
    histogram of the entries of the population histogram of clipped (binary)
    spike trains computed with a bin width of binsize.
    It provides for each complexity (== number of active neurons per bin) the
    number of occurrences. The normalization of that histogram to 1 is the
    probability density.

    Parameters
    ----------
    spiketrains : List of neo.SpikeTrain objects
    Spiketrains with a common time axis (same `t_start` and `t_stop`)
    binsize : quantities.Quantity
    Width of the histogram's time bins.

    Returns
    -------
    time_hist : neo.AnalogSignal
    A neo.AnalogSignal object containing the histogram values.
    `AnalogSignal[j]` is the histogram computed between .

    See also
    --------
    elephant.conversion.BinnedSpikeTrain

    References
    ----------
    [1]Gruen, S., Abeles, M., & Diesmann, M. (2008). Impact of higher-order
    correlations on coincidence distributions of massively parallel data.
    In Dynamic Brain-from Neural Spikes to Behaviors (pp. 96-114).
    Springer Berlin Heidelberg.

    """
    # Computing the population histogram with parameter binary=True to clip the
    # spike trains before summing
    pophist = time_histogram(spiketrains, binsize, binary=True)

    # Computing the histogram of the entries of pophist (=Complexity histogram)
    complexity_hist = np.histogram(
        pophist.magnitude, bins=range(0, len(spiketrains) + 2))[0]

    # Normalization of the Complexity Histogram to 1 (probabilty distribution)
    complexity_hist = complexity_hist / complexity_hist.sum()
    # Convert the Complexity pdf to an neo.AnalogSignal
    complexity_distribution = neo.AnalogSignal(
        np.array(complexity_hist).reshape(len(complexity_hist), 1) *
        pq.dimensionless, t_start=0 * pq.dimensionless,
        sampling_period=1 * pq.dimensionless)

    return complexity_distribution


"""Kernel Bandwidth Optimization.

Python implementation by Subhasis Ray.

Original matlab code (sskernel.m) here:
http://2000.jukuin.keio.ac.jp/shimazaki/res/kernel.html

This was translated into Python by Subhasis Ray, NCBS. Tue Jun 10
23:01:43 IST 2014

"""


def nextpow2(x):
    """ Return the smallest integral power of 2 that >= x """
    n = 2
    while n < x:
        n = 2 * n
    return n


def fftkernel(x, w):
    """

    y = fftkernel(x,w)

    Function `fftkernel' applies the Gauss kernel smoother to an input
    signal using FFT algorithm.

    Input argument
    x:    Sample signal vector.
    w: 	Kernel bandwidth (the standard deviation) in unit of
    the sampling resolution of x.

    Output argument
    y: 	Smoothed signal.

    MAY 5/23, 2012 Author Hideaki Shimazaki
    RIKEN Brain Science Insitute
    http://2000.jukuin.keio.ac.jp/shimazaki

    Ported to Python: Subhasis Ray, NCBS. Tue Jun 10 10:42:38 IST 2014

    """
    L = len(x)
    Lmax = L + 3 * w
    n = nextpow2(Lmax)
    X = np.fft.fft(x, n)
    f = np.arange(0, n, 1.0) / n
    f = np.concatenate((-f[:int(n / 2)], f[int(n / 2):0:-1]))
    K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)
    y = np.fft.ifft(X * K, n)
    y = y[:L].copy()
    return y


def logexp(x):
    if x < 1e2:
        y = np.log(1 + np.exp(x))
    else:
        y = x
    return y


def ilogexp(x):
    if x < 1e2:
        y = np.log(np.exp(x) - 1)
    else:
        y = x
    return y


def cost_function(x, N, w, dt):
    """

    The cost function
    Cn(w) = sum_{i,j} int k(x - x_i) k(x - x_j) dx - 2 sum_{i~=j} k(x_i - x_j)

     """
    yh = np.abs(fftkernel(x, w / dt))  # density
    # formula for density
    C = np.sum(yh ** 2) * dt - 2 * np.sum(yh * x) * \
               dt + 2 / np.sqrt(2 * np.pi) / w / N
    C = C * N * N
    # formula for rate
    # C = dt*sum( yh.^2 - 2*yh.*y_hist + 2/sqrt(2*pi)/w*y_hist )
    return C, yh


def sskernel(spiketimes, tin=None, w=None, bootstrap=False):
    """

    Calculates optimal fixed kernel bandwidth.

    spiketimes: sequence of spike times (sorted to be ascending).

    tin: (optional) time points at which the kernel bandwidth is to be estimated.

    w: (optional) vector of kernel bandwidths. If specified, optimal
    bandwidth is selected from this.

    bootstrap (optional): whether to calculate the 95% confidence
    interval. (default False)

    Returns

    A dictionary containing the following key value pairs:

    'y': estimated density,
    't': points at which estimation was computed,
    'optw': optimal kernel bandwidth,
    'w': kernel bandwidths examined,
    'C': cost functions of w,
    'confb95': (lower bootstrap confidence level, upper bootstrap confidence level),
    'yb': bootstrap samples.


    Ref: Shimazaki, Hideaki, and Shigeru Shinomoto. 2010. Kernel
    Bandwidth Optimization in Spike Rate Estimation. Journal of
    Computational Neuroscience 29 (1-2):
    171-82. doi:10.1007/s10827-009-0180-4.

    """

    if tin is None:
        time = np.max(spiketimes) - np.min(spiketimes)
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        tin = np.linspace(np.min(spiketimes),
                          np.max(spiketimes),
                          min(int(time / dt + 0.5), 1000))  # The 1000 seems somewhat arbitrary
        t = tin
    else:
        time = np.max(tin) - np.min(tin)
        spiketimes = spiketimes[(spiketimes >= np.min(tin)) &
                                (spiketimes <= np.max(tin))].copy()
        isi = np.diff(spiketimes)
        isi = isi[isi > 0].copy()
        dt = np.min(isi)
        if dt > np.min(np.diff(tin)):
            t = np.linspace(np.min(tin), np.max(tin),
                            min(int(time / dt + 0.5), 1000))
        else:
            t = tin
    dt = np.min(np.diff(tin))
    yhist, bins = np.histogram(spiketimes, np.r_[t - dt / 2, t[-1] + dt / 2])
    N = np.sum(yhist)
    yhist = yhist / (N * dt)  # density
    optw = None
    y = None
    if w is not None:
        C = np.zeros(len(w))
        Cmin = np.inf
        for k, w_ in enumerate(w):
            C[k], yh = cost_function(yhist, N, w_, dt)
            if C[k] < Cmin:
                Cmin = C[k]
                optw = w_
                y = yh
    else:
        # Golden section search on a log-exp scale
        wmin = 2 * dt
        wmax = max(spiketimes) - min(spiketimes)
        imax = 20  # max iterations
        w = np.zeros(imax)
        C = np.zeros(imax)
        tolerance = 1e-5
        phi = 0.5 * (np.sqrt(5) + 1)  # The Golden ratio
        a = ilogexp(wmin)
        b = ilogexp(wmax)
        c1 = (phi - 1) * a + (2 - phi) * b
        c2 = (2 - phi) * a + (phi - 1) * b
        f1, y1 = cost_function(yhist, N, logexp(c1), dt)
        f2, y2 = cost_function(yhist, N, logexp(c2), dt)
        k = 0
        while (np.abs(b - a) > (tolerance * (np.abs(c1) + np.abs(c2))))\
              and (k < imax):
            if f1 < f2:
                b = c2
                c2 = c1
                c1 = (phi - 1) * a + (2 - phi) * b
                f2 = f1
                f1, y1 = cost_function(yhist, N, logexp(c1), dt)
                w[k] = logexp(c1)
                C[k] = f1
                optw = logexp(c1)
                y = y1 / (np.sum(y1 * dt))
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, y2 = cost_function(yhist, N, logexp(c2), dt)
                w[k] = logexp(c2)
                C[k] = f2
                optw = logexp(c2)
                y = y2 / np.sum(y2 * dt)
            k = k + 1
    # Bootstrap confidence intervals
    confb95 = None
    yb = None
    if bootstrap:
        nbs = 1000
        yb = np.zeros((nbs, len(tin)))
        for ii in range(nbs):
            idx = np.floor(np.random.rand(N) * N).astype(int)
            xb = spiketimes[idx]
            y_histb, bins = np.histogram(
                xb, np.r_[t - dt / 2, t[-1] + dt / 2]) / dt / N
            yb_buf = fftkernel(y_histb, optw / dt).real
            yb_buf = yb_buf / np.sum(yb_buf * dt)
            yb[ii, :] = np.interp(tin, t, yb_buf)
        ybsort = np.sort(yb, axis=0)
        y95b = ybsort[np.floor(0.05 * nbs).astype(int), :]
        y95u = ybsort[np.floor(0.95 * nbs).astype(int), :]
        confb95 = (y95b, y95u)
    ret = np.interp(tin, t, y)
    return {'y': ret,
            't': tin,
            'optw': optw,
            'w': w,
            'C': C,
            'confb95': confb95,
            'yb': yb}
