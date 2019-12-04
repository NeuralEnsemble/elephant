# -*- coding: utf-8 -*-
"""
Module to generate surrogates of a spike train by randomising its spike times
in different ways (see [1]). Different methods destroy different features of
the original data:

* randomise_spikes:
    randomly reposition all spikes inside the time interval (t_start, t_stop).
    Keeps spike count, generates Poisson spike trains with time-stationary
    firing rate
* dither_spikes:
    dither each spike time around original position by a random amount;
    keeps spike count and firing rates computed on a slow temporal scale;
    destroys ISIs, making them more exponentially distributed
* dither_spike_train:
    dither the whole input spike train (i.e. all spikes equally) by a random
    amount; keeps spike count, ISIs, and firing rates computed on a slow
    temporal scale
* jitter_spikes:
    discretise the full time interval (t_start, t_stop) into time segments
    and locally randomise the spike times (see randomise_spikes) inside each
    segment. Keeps spike count inside each segment and creates locally Poisson
    spike trains with locally time-stationary rates
* shuffle_isis:
    shuffle the inter-spike intervals (ISIs) of the spike train randomly,
    keeping the first spike time fixed and generating the others from the
    new sequence of ISIs. Keeps spike count and ISIs, flattens the firing rate
    profile
* joint_isi_dithering:
    calculate the Joint-ISI distribution and moves spike according to the
    probability distribution, that results from a fixed sum of ISI_before
    and the ISI_afterwards. For further details see [1].

[1] Louis et al (2010) Surrogate Spike Train Generation Through Dithering in
    Operational Time. Front Comput Neurosci. 2010; 4: 127.

Original implementation by: Emiliano Torre [e.torre@fz-juelich.de]
:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
import quantities as pq
import neo
from scipy.ndimage import gaussian_filter
import math

try:
    import elephant.statistics as es

    isi = es.isi
except ImportError:
    from .statistics import isi  # Convenience when in elephant working dir.


def dither_spikes(spiketrain, dither, n=1, decimals=None, edges=True):
    """
    Generates surrogates of a spike train by spike dithering.

    The surrogates are obtained by uniformly dithering times around the
    original position. The dithering is performed independently for each
    surrogate.

    The surrogates retain the :attr:`t_start` and :attr:`t_stop` of the
    original `SpikeTrain` object. Spikes moved beyond this range are lost or
    moved to the range's ends, depending on the parameter edge.


    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates
    dither : quantities.Quantity
        Amount of dithering. A spike at time t is placed randomly within
        ]t-dither, t+dither[.
    n : int (optional)
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        Number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None
    edges : bool (optional)
        For surrogate spikes falling outside the range
        `[spiketrain.t_start, spiketrain.t_stop)`, whether to drop them out
        (for edges = True) or set that to the range's closest end
        (for edges = False).
        Default: True

    Returns
    -------
    list of neo.SpikeTrain
      A list of `neo.SpikeTrain`, each obtained from :attr:`spiketrain` by
      randomly dithering its spikes. The range of the surrogate spike trains
      is the same as :attr:`spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>> print dither_spikes(st, dither = 20*pq.ms)   # doctest: +SKIP
    [<SpikeTrain(array([  96.53801903,  248.57047376,  601.48865767,
     815.67209811]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print dither_spikes(st, dither = 20*pq.ms, n=2)   # doctest: +SKIP
    [<SpikeTrain(array([ 104.24942044,  246.0317873 ,  584.55938657,
        818.84446913]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 111.36693058,  235.15750163,  618.87388515,
        786.1807108 ]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print dither_spikes(st, dither = 20*pq.ms,
                            decimals=0)   # doctest: +SKIP
    [<SpikeTrain(array([  81.,  242.,  595.,  799.]) * ms,
        [0.0 ms, 1000.0 ms])>]
    """

    # Transform spiketrain into a Quantity object (needed for matrix algebra)
    data = spiketrain.view(pq.Quantity)

    # Main: generate the surrogates
    surr = data.reshape((1, len(data))) + 2 * dither * np.random.random_sample(
        (n, len(data))) - dither
    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        surr = surr.round(decimals)

    if edges is False:
        # Move all spikes outside [spiketrain.t_start, spiketrain.t_stop] to
        # the range's ends
        surr = np.minimum(np.maximum(surr.simplified.magnitude,
                                     spiketrain.t_start.simplified.magnitude),
                          spiketrain.t_stop.simplified.magnitude) * pq.s
    else:
        # Leave out all spikes outside [spiketrain.t_start, spiketrain.t_stop]
        tstart, tstop = spiketrain.t_start.simplified.magnitude, \
                        spiketrain.t_stop.simplified.magnitude
        surr = [np.sort(s[np.all([s >= tstart, s < tstop], axis=0)]) * pq.s
                for s in surr.simplified.magnitude]

    # Return the surrogates as SpikeTrains
    return [neo.SpikeTrain(s,
                           t_start=spiketrain.t_start,
                           t_stop=spiketrain.t_stop).rescale(spiketrain.units)
            for s in surr]


def randomise_spikes(spiketrain, n=1, decimals=None):
    """
    Generates surrogates of a spike trains by spike time randomisation.

    The surrogates are obtained by keeping the spike count of the original
    `SpikeTrain` object, but placing them randomly into the interval
    `[spiketrain.t_start, spiketrain.t_stop]`.
    This generates independent Poisson neo.SpikeTrain objects (exponentially
    distributed inter-spike intervals) while keeping the spike count as in
    :attr:`spiketrain`.

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates
    n : int (optional)
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        Number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None

    Returns
    -------
    list of neo.SpikeTrain object(s)
      A list of `neo.SpikeTrain` objects, each obtained from :attr:`spiketrain`
      by randomly dithering its spikes. The range of the surrogate spike trains
      is the same as :attr:`spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>> print randomise_spikes(st)   # doctest: +SKIP
        [<SpikeTrain(array([ 131.23574603,  262.05062963,  549.84371387,
                            940.80503832]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print randomise_spikes(st, n=2)   # doctest: +SKIP
        [<SpikeTrain(array([  84.53274955,  431.54011743,  733.09605806,
              852.32426583]) * ms, [0.0 ms, 1000.0 ms])>,
         <SpikeTrain(array([ 197.74596726,  528.93517359,  567.44599968,
              775.97843799]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print randomise_spikes(st, decimals=0)   # doctest: +SKIP
        [<SpikeTrain(array([  29.,  667.,  720.,  774.]) * ms,
              [0.0 ms, 1000.0 ms])>]
    """

    # Create surrogate spike trains as rows of a Quantity array
    sts = ((spiketrain.t_stop - spiketrain.t_start) *
           np.random.random(size=(n, len(spiketrain))) +
           spiketrain.t_start).rescale(spiketrain.units)

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        sts = sts.round(decimals)

    # Convert the Quantity array to a list of SpikeTrains, and return them
    return [neo.SpikeTrain(np.sort(st), t_start=spiketrain.t_start,
                           t_stop=spiketrain.t_stop)
            for st in sts]


def shuffle_isis(spiketrain, n=1, decimals=None):
    """
    Generates surrogates of a neo.SpikeTrain object by inter-spike-interval
    (ISI) shuffling.

    The surrogates are obtained by randomly sorting the ISIs of the given input
    :attr:`spiketrain`. This generates independent `SpikeTrain` object(s) with
    same ISI distribution and spike count as in :attr:`spiketrain`, while
    destroying temporal dependencies and firing rate profile.

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates
    n : int (optional)
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        Number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None

    Returns
    -------
    list of SpikeTrain
      A list of spike trains, each obtained from `spiketrain` by random ISI
      shuffling. The range of the surrogate `neo.SpikeTrain` objects is the
      same as :attr:`spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>> print shuffle_isis(st)   # doctest: +SKIP
        [<SpikeTrain(array([ 200.,  350.,  700.,  800.]) * ms,
                 [0.0 ms, 1000.0 ms])>]
    >>> print shuffle_isis(st, n=2)   # doctest: +SKIP
        [<SpikeTrain(array([ 100.,  300.,  450.,  800.]) * ms,
              [0.0 ms, 1000.0 ms])>,
         <SpikeTrain(array([ 200.,  350.,  700.,  800.]) * ms,
              [0.0 ms, 1000.0 ms])>]

    """

    if len(spiketrain) > 0:
        isi0 = spiketrain[0] - spiketrain.t_start
        ISIs = np.hstack([isi0, isi(spiketrain)])

        # Round the ISIs to decimal position, if requested
        if decimals is not None:
            ISIs = ISIs.round(decimals)

        # Create list of surrogate spike trains by random ISI permutation
        sts = []
        for i in range(n):
            surr_times = np.cumsum(np.random.permutation(ISIs)) * \
                         spiketrain.units + spiketrain.t_start
            sts.append(neo.SpikeTrain(
                surr_times, t_start=spiketrain.t_start,
                t_stop=spiketrain.t_stop))

    else:
        sts = []
        empty_train = neo.SpikeTrain([] * spiketrain.units,
                                     t_start=spiketrain.t_start,
                                     t_stop=spiketrain.t_stop)
        for i in range(n):
            sts.append(empty_train)

    return sts


def dither_spike_train(spiketrain, shift, n=1, decimals=None, edges=True):
    """
    Generates surrogates of a neo.SpikeTrain by spike train shifting.

    The surrogates are obtained by shifting the whole spike train by a
    random amount (independent for each surrogate). Thus, ISIs and temporal
    correlations within the spike train are kept. For small shifts, the
    firing rate profile is also kept with reasonable accuracy.

    The surrogates retain the :attr:`t_start` and :attr:`t_stop` of the
    :attr:`spiketrain`. Spikes moved beyond this range are lost or moved to
    the range's ends, depending on the parameter edge.

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates
    shift : quantities.Quantity
        Amount of shift. spiketrain is shifted by a random amount uniformly
        drawn from the range ]-shift, +shift[.
    n : int (optional)
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None (optional)
        Number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None
    edges : bool
        For surrogate spikes falling outside the range `[spiketrain.t_start,
        spiketrain.t_stop)`, whether to drop them out (for edges = True) or set
        that to the range's closest end (for edges = False).
        Default: True

    Returns
    -------
    list of SpikeTrain
      A list of spike trains, each obtained from spiketrain by randomly
      dithering its spikes. The range of the surrogate spike trains is the
      same as :attr:`spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([100, 250, 600, 800]*pq.ms, t_stop=1*pq.s)
    >>>
    >>> print dither_spike_train(st, shift = 20*pq.ms)   # doctest: +SKIP
    [<SpikeTrain(array([  96.53801903,  248.57047376,  601.48865767,
     815.67209811]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print dither_spike_train(st, shift = 20*pq.ms, n=2)   # doctest: +SKIP
    [<SpikeTrain(array([  92.89084054,  242.89084054,  592.89084054,
        792.89084054]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([  84.61079043,  234.61079043,  584.61079043,
        784.61079043]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print dither_spike_train(st, shift = 20*pq.ms,
                                 decimals=0)   # doctest: +SKIP
    [<SpikeTrain(array([  82.,  232.,  582.,  782.]) * ms,
        [0.0 ms, 1000.0 ms])>]
    """

    # Transform spiketrain into a Quantity object (needed for matrix algebra)
    data = spiketrain.view(pq.Quantity)

    # Main: generate the surrogates by spike train shifting
    surr = data.reshape(
        (1, len(data))) + 2 * shift * np.random.random_sample((n, 1)) - shift

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        surr = surr.round(decimals)

    if edges is False:
        # Move all spikes outside [spiketrain.t_start, spiketrain.t_stop] to
        # the range's ends
        surr = np.minimum(np.maximum(surr.simplified.magnitude,
                                     spiketrain.t_start.simplified.magnitude),
                          spiketrain.t_stop.simplified.magnitude) * pq.s
    else:
        # Leave out all spikes outside [spiketrain.t_start, spiketrain.t_stop]
        tstart, tstop = spiketrain.t_start.simplified.magnitude, \
                        spiketrain.t_stop.simplified.magnitude
        surr = [np.sort(s[np.all([s >= tstart, s < tstop], axis=0)]) * pq.s
                for s in surr.simplified.magnitude]

    # Return the surrogates as SpikeTrains
    return [neo.SpikeTrain(s, t_start=spiketrain.t_start,
                           t_stop=spiketrain.t_stop).rescale(spiketrain.units)
            for s in surr]


def jitter_spikes(spiketrain, binsize, n=1):
    """
    Generates surrogates of a :attr:`spiketrain` by spike jittering.

    The surrogates are obtained by defining adjacent time bins spanning the
    :attr:`spiketrain` range, and random re-positioning (independently for each
    surrogate) each spike in the time bin it falls into.

    The surrogates retain the :attr:`t_start and :attr:`t_stop` of the
    :attr:`spike train`. Note that within each time bin the surrogate
    `neo.SpikeTrain` objects are locally Poissonian (the inter-spike-interval
    are exponentially distributed).

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates
    binsize : quantities.Quantity
        Size of the time bins within which to randomise the spike times.
        Note: the last bin arrives until `spiketrain.t_stop` and might have
        width different from `binsize`.
    n : int (optional)
        Number of surrogates to be generated.
        Default: 1

    Returns
    -------
    list of SpikeTrain
      A list of spike trains, each obtained from `spiketrain` by randomly
      replacing its spikes within bins of user-defined width. The range of the
      surrogate spike trains is the same as `spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    >>>
    >>> st = neo.SpikeTrain([80, 150, 320, 480]*pq.ms, t_stop=1*pq.s)
    >>> print jitter_spikes(st, binsize=100*pq.ms)   # doctest: +SKIP
    [<SpikeTrain(array([  98.82898293,  178.45805954,  346.93993867,
        461.34268507]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print jitter_spikes(st, binsize=100*pq.ms, n=2)   # doctest: +SKIP
    [<SpikeTrain(array([  97.15720041,  199.06945744,  397.51928207,
        402.40065162]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([  80.74513157,  173.69371317,  338.05860962,
        495.48869981]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print jitter_spikes(st, binsize=100*pq.ms)   # doctest: +SKIP
    [<SpikeTrain(array([  4.55064897e-01,   1.31927046e+02,   3.57846265e+02,
         4.69370604e+02]) * ms, [0.0 ms, 1000.0 ms])>]
    """
    # Define standard time unit; all time Quantities are converted to
    # scalars after being rescaled to this unit, to use the power of numpy
    std_unit = binsize.units

    # Compute bin edges for the jittering procedure
    # !: the last bin arrives until spiketrain.t_stop and might have
    # size != binsize
    start_dl = spiketrain.t_start.rescale(std_unit).magnitude
    stop_dl = spiketrain.t_stop.rescale(std_unit).magnitude

    bin_edges = start_dl + np.arange(start_dl, stop_dl, binsize.magnitude)
    bin_edges = np.hstack([bin_edges, stop_dl])

    # Create n surrogates with spikes randomly placed in the interval (0,1)
    surr_poiss01 = np.random.random_sample((n, len(spiketrain)))

    # Compute the bin id of each spike
    bin_ids = np.array(
        (spiketrain.view(pq.Quantity) /
         binsize).rescale(pq.dimensionless).magnitude, dtype=int)

    # Compute the size of each time bin (as a numpy array)
    bin_sizes_dl = np.diff(bin_edges)

    # For each spike compute its offset (the left end of the bin it falls
    # into) and the size of the bin it falls into
    offsets = start_dl + np.array([bin_edges[bin_id] for bin_id in bin_ids])
    dilats = np.array([bin_sizes_dl[bin_id] for bin_id in bin_ids])

    # Compute each surrogate by dilating and shifting each spike s in the
    # poisson 0-1 spike trains to dilat * s + offset. Attach time unit again
    surr = np.sort(surr_poiss01 * dilats + offsets, axis=1) * std_unit

    return [neo.SpikeTrain(s, t_start=spiketrain.t_start,
                           t_stop=spiketrain.t_stop).rescale(spiketrain.units)
            for s in surr]


class JointISI(object):
    """
    The class :class:`JointISI` is implemented for Joint-ISI dithering
    as a continuation of the ideas of Louis et al. (2010) and Gerstein (2004).

    When creating an class instance all necessary preprocessing steps are done,
    to use the method dithering().

    Attributes
    ----------
    spiketrain: neo.SpikeTrain
        For this spiketrain the surrogates will be created

    dither: pq.Quantity
        This quantity describes the maximum displacement of a spike, when
        method is 'window'. It is also used for the uniform dithering for
        the spikes, which are outside the regime in the Joint-ISI
        histogram, where Joint-ISI dithering is applicable.
        Default: 15.*pq.ms
    truncation_limit: pq.Quantity
        The Joint-ISI distribution is as such defined on a range for ISI_i and
        ISI_(i+1) from 0 to inf. Since this is computationally not feasible,
        the Joint-ISI distribution is truncated for high ISI. The Joint-ISI
        histogram is calculated for ISI_i, ISI_(i+1) from 0 to
        truncation_limit.
        Default: 100*pq.ms
    num_bins: int
        The size of the joint-ISI-distribution will be num_bins*num_bins/2.
        Default: 100
    sigma: pq.Quantity
        The standard deviation of the Gaussian kernel, with which
        the data is convoluted.
        Default: 2.*pq.ms
    alternate: boolean
        If alternate == True: then first all even and then all odd spikes are
        dithered. Else: in ascending order from the first to the last spike,
        all spikes are moved.
        Default: True.
    use_sqrt: boolean
        if use_sqrt == True a sqrt is applied to the joint-ISI histogram,
        following Gerstein et al. 2004
        Default: False
    method: string
        if 'window': the spike movement is limited to the parameter dither.
        if 'fast': the spike can move in all the range between the previous
            spike and the subsequent spike. This is computationally much faster
            and thus is called 'fast'.
        Default: 'fast'
    cutoff: boolean
        if True then the Filtering of the Joint-ISI histogram is
        limited to the lower side by the minimal ISI.
        This can be necessary, if in the data there is a certain refractory
        period, which would be destroyed by the convolution with the
        2d-Gaussian function.
        Default: True
    refr_period:
        Since this dither-method should conserve the refractory period,
        It is internally calculated as the minimum of the value given here
        and the least ISI in the spiketrain.
        Default: 4.*pq.ms
    Methods
    ----------
    dithering()
        Returns a list of dithered spiketrains.
    """

    # min_spikes is the number of spikes is lower than this number,
    # when the dithering method is called, the spiketrain is just given back.
    MIN_SPIKES = 3

    def __init__(self,
                 spiketrain,
                 dither=15. * pq.ms,
                 truncation_limit=100. * pq.ms,
                 num_bins=100,
                 sigma=2. * pq.ms,
                 alternate=True,
                 use_sqrt=False,
                 method='fast',
                 cutoff=True,
                 refr_period=4. * pq.ms
                 ):
        """
        Parameters
        ----------
        spiketrain: neo.SpikeTrain
            For this spiketrain the surrogates will be created
        dither: pq.Quantity
            This quantity describes the maximum displacement of a spike, when
            method is 'window'. It is also used for the uniform dithering for
            the spikes, which are outside the regime in the Joint-ISI
            histogram, where Joint-ISI dithering is applicable.
            Default: 15.*pq.ms
        truncation_limit: pq.Quantity
            The Joint-ISI distribution is as such defined on a range for ISI_i
            and ISI_(i+1) from 0 to inf. Since this is computationally not
            feasible, the Joint-ISI distribution is truncated for high ISI.
            The Joint-ISI histogram is calculated for ISI_i, ISI_(i+1) from 0
            to truncation_limit.
            Default: 100*pq.ms
        num_bins: int
            The size of the joint-ISI-distribution will be num_bins*num_bins/2.
            Default: 100
        sigma: pq.Quantity
            The standard deviation of the Gaussian kernel, with which
            the data is convoluted.
            Default: 2.*pq.ms
        alternate: boolean
            If alternate == True: then first all even and then all odd spikes
            are dithered. Else: in ascending order from the first to the last
            spike, all spikes are moved.
            Default: True.
        use_sqrt: boolean
            if use_sqrt == True a sqrt is applied to the joint-ISI histogram,
            following Gerstein et al. 2004
            Default: False
        method: string
            if 'window': the spike movement is limited to the parameter dither.
            if 'fast': the spike can move in all the range between the previous
                spike and the subsequent spike. This is computationally much
                faster and thus is called 'fast'.
            Default: 'fast'
        cutoff: boolean
            if True then the Filtering of the Joint-ISI histogram is
            limited to the lower side by the minimal ISI.
            This can be necessary, if in the data there is a certain refractory
            period, which would be destroyed by the convolution with the
            2d-Gaussian function.
            Default: True
        refr_period:
            Since this dither-method should conserve the refractory period,
            It is internally calculated as the minimum of the value given here
            and the least ISI in the spiketrain.
            Default: 4.*pq.ms
        """
        self.spiketrain = spiketrain
        self.truncation_limit = self.get_magnitude(truncation_limit)
        self.num_bins = num_bins

        self.dither = self.get_magnitude(dither)

        self.sigma = self.get_magnitude(sigma)
        self.alternate = alternate

        if method not in ['fast', 'window']:
            error_message = (
                'method can only be either \'fast\' or \'window\','
                ' but not \'{0}\' .'.format(method))
            raise ValueError(error_message)
        self.method = method
        
        refr_period = self.get_magnitude(refr_period)
        if not self.too_less_spikes:
            minimal_isi = np.min(self.isi)
            refr_period = min(refr_period, minimal_isi)
        self.refr_period = refr_period
        
        self.cutoff = cutoff
        self.use_sqrt = use_sqrt
        self._jisih_cumulatives = None

        self.max_change_index = self.isi_to_index(self.dither)
        self.max_change_isi = self.index_to_isi(self.max_change_index)

    def get_magnitude(self, x):
        """
        if x is pq.Quantity: returns the magnitude rescaled to units of the
            spiketrain
        else: returns x
        Parameters
        ----------
        x: Union[pq.Quantity, float, int]

        Returns
        -------
        Union[float, int]
        """
        if isinstance(x, pq.Quantity):
            return x.rescale(self.unit).magnitude
        return x

    @property
    def too_less_spikes(self):
        """
        This is a check if the spiketrain has enough spikes to evaluate the
        joint-ISI histogram. With a default value of 4. There need to be at
        least 2 spikes with a previous and a subsequent spike.

        Returns
        -------
        bool
        """
        return len(self.spiketrain) < self.MIN_SPIKES

    @property
    def unit(self):
        return self.spiketrain.units

    @property
    def isi(self):
        if self.too_less_spikes:
            return None
        return isi(self.spiketrain.magnitude)

    @property
    def bin_width(self):
        return self.truncation_limit / self.num_bins

    def isi_to_index(self, inter_spike_interval):
        """
        A function that gives for each ISI the corresponding index in the
        Joint-ISI distribution.

        Parameters
        ----------
        inter_spike_interval: np.ndarray or float
            An array of ISIs or a single ISI.

        Returns
        -------
        np.ndarray of int or int:
            For each ISI the corresponding index.
        """
        return np.floor(inter_spike_interval / self.bin_width).astype(int)

    def index_to_isi(self, isi_index):
        """
        Gives an array, taking an element with an index of the Joint-ISI
        distribution gives back the corresponding ISI.
        And this is effectively the middle of the ISI bin.

        Parameters
        ----------
        isi_index: np.ndarray of int or int
            Index corresponding to an ISI.

        Returns
        -------
        np.ndarray of float or float:
            For each index the corresponding ISI.

        """
        return (isi_index + 0.5) * self.bin_width

    @property
    def joint_isi_histogram(self):
        """
        This function calculates the joint-ISI histogram.
        """
        if self.too_less_spikes:
            return None
        isis = self.isi
        joint_isi_histogram = np.histogram2d(
            isis[:-1], isis[1:],
            bins=[self.num_bins, self.num_bins],
            range=[[0., self.truncation_limit],
                   [0., self.truncation_limit]])[0]

        if self.use_sqrt:
            joint_isi_histogram = np.sqrt(joint_isi_histogram)

        if self.sigma:
            if self.cutoff:
                start_index = self.isi_to_index(self.refr_period)
                joint_isi_histogram[
                    start_index:, start_index:] = gaussian_filter(
                    joint_isi_histogram[start_index:, start_index:],
                    self.sigma / self.bin_width)

                joint_isi_histogram[:start_index, :] = np.zeros_like(
                    joint_isi_histogram[:start_index, :])
                joint_isi_histogram[:, :start_index] = np.zeros_like(
                    joint_isi_histogram[:, :start_index])

            else:
                joint_isi_histogram = gaussian_filter(joint_isi_histogram,
                                                      self.sigma /
                                                      self.bin_width)
        return joint_isi_histogram

    @staticmethod
    def normalize_cumulative_distribution(array):
        """
        This function normalizes parts of a cumulative distribution function,
        to be a cumulative distribution function again.

        Parameters
        ----------
        array: np.ndarray
            A monotonously increasing array, as a part of an unnormalized
             cumulative distribution function.
        Returns
        -------
        np.ndarray
            Monotonously increasing array, starting at 0 going to 1.
        """

        if array[-1] - array[0] > 0.:
            return (array - array[0]) / (array[-1] - array[0])
        return np.zeros_like(array)

    def dithering(self, n_surr=1):
        """
        Implementation of Joint-ISI-dithering for spiketrains that pass the
        threshold of the dense rate, if not a uniform dithered spiketrain is
        given back. The implementation continued the ideas of Louis et al.
        (2010) and Gerstein (2004).

        Parameters
        ----------
        n_surr: int
            The number of dithered spiketrains to be returned.
            Default: 1

        Returns
        ----------
        dithered_sts: list
            list of spiketrains, that are dithered versions of the given
            spiketrain
        """
        if self.too_less_spikes:
            return [self.spiketrain] * n_surr

        # Checks, whether the preprocessing is already done.
        if self._jisih_cumulatives is not None:
            self._determine_cumulative_functions()

        dithered_sts = []
        isi_to_dither = self.isi
        for surr_number in range(n_surr):
            dithered_isi = self._get_dithered_isi(isi_to_dither)

            dithered_st = self.spiketrain[0].magnitude + np.hstack(
                (np.array(0.), np.cumsum(dithered_isi)))
            dithered_st = neo.SpikeTrain(dithered_st * self.unit,
                                         t_start=self.spiketrain.t_start,
                                         t_stop=self.spiketrain.t_stop)
            dithered_sts.append(dithered_st)
        return dithered_sts

    def _determine_cumulative_functions(self):
        rotated_jisih = np.rot90(self.joint_isi_histogram)

        if self.method == 'fast':
            self._jisih_cumulatives = [self.normalize_cumulative_distribution(
                np.cumsum(np.diagonal(rotated_jisih,
                                      -self.num_bins + double_index + 1)))
                for double_index in range(self.num_bins)]
            return

        self._jisih_cumulatives = self._window_cumulatives(rotated_jisih)

    def _window_diagonal_cumulatives(self, flipped_jisih):
        jisih_diag_cums = np.zeros((self.num_bins,
                                    self.num_bins
                                    + 2 * self.max_change_index))

        for double_index in range(self.num_bins):
            cum_diag = np.cumsum(np.diagonal(flipped_jisih,
                                             - self.num_bins
                                             + double_index + 1))
            jisih_diag_cums[double_index,
                            self.max_change_index:
                            double_index
                            + self.max_change_index + 1
                            ] = cum_diag

            cum_bound = np.repeat(jisih_diag_cums[double_index,
                                                  double_index +
                                                  self.max_change_index],
                                  self.max_change_index)

            jisih_diag_cums[double_index,
                            double_index + self.max_change_index + 1:
                            double_index
                            + 2 * self.max_change_index + 1
                            ] = cum_bound
        return jisih_diag_cums

    def _window_cumulatives(self, flipped_jisih):

        jisih_diag_cums = self._window_diagonal_cumulatives(flipped_jisih)
        jisih_cumulatives = np.zeros(
            (self.num_bins, self.num_bins,
             2 * self.max_change_index + 1))
        for back_index in range(self.num_bins):
            for for_index in range(self.num_bins - back_index):
                double_index = for_index + back_index
                cum_slice = jisih_diag_cums[double_index,
                                            back_index:
                                            back_index +
                                            2 * self.max_change_index + 1]
                normalized_cum = self.normalize_cumulative_distribution(
                    cum_slice)
                jisih_cumulatives[back_index][for_index] = normalized_cum
        return jisih_cumulatives

    def _get_dithered_isi(self, isi_to_dither):
        dithered_isi = isi_to_dither
        # if alternate is true, a sampling_rhythm of 2 means that we have two
        # partitions of spikes, the "odd" and the "even" spikes and first
        # dither the "even" ones and then the "odd" ones.
        # if alternate is false, we just go dither from the first to the last
        # spike, which corresponds to a sampling_rhythm of 1.
        sampling_rhythm = self.alternate + 1
        number_of_isis = len(dithered_isi)
        random_list = np.random.random(number_of_isis)
        if self.method == 'fast':
            update_dithered_isi = self._update_dithered_isi_fast
        else:
            update_dithered_isi = self._update_dithered_isi_window

        for start in range(sampling_rhythm):
            dithered_isi_indices = self.isi_to_index(dithered_isi)
            for i in range(start, number_of_isis - 1,
                           sampling_rhythm):
                step = update_dithered_isi(
                    dithered_isi,
                    dithered_isi_indices,
                    random_list[i],
                    i)
                dithered_isi[i] += step
                dithered_isi[i + 1] -= step

        return dithered_isi

    def _update_dithered_isi_fast(self,
                                  dithered_isi,
                                  dithered_isi_indices,
                                  random_number,
                                  i):
        back_index = dithered_isi_indices[i]
        for_index = dithered_isi_indices[i + 1]
        double_index = back_index + for_index
        if double_index < self.num_bins:
            if self._jisih_cumulatives[double_index][-1]:
                cond = (self._jisih_cumulatives[double_index]
                        > random_number)
                new_index = np.where(
                    cond,
                    self._jisih_cumulatives[double_index],
                    np.inf).argmin()
                step = (self.index_to_isi(new_index)
                        - self.index_to_isi(back_index))
                return step
        return self._uniform_dither_not_jisi_movable_spikes(
            dithered_isi[i],
            dithered_isi[i + 1],
            random_number)

    def _update_dithered_isi_window(self,
                                    dithered_isi,
                                    dithered_isi_indices,
                                    random_number,
                                    i):
        back_index = dithered_isi_indices[i]
        for_index = dithered_isi_indices[i + 1]
        if back_index + for_index < self.num_bins:
            cum_dist_func = self._jisih_cumulatives[
                back_index][for_index]
            if cum_dist_func[-1]:
                cond = cum_dist_func > random_number
                new_index = np.where(
                    cond,
                    cum_dist_func,
                    np.inf).argmin()
                step = (self.index_to_isi(new_index)
                        - self.max_change_isi)
                return step
        return self._uniform_dither_not_jisi_movable_spikes(
            dithered_isi[i],
            dithered_isi[i + 1],
            random_number)

    def _uniform_dither_not_jisi_movable_spikes(self,
                                                previous_isi,
                                                subsequent_isi,
                                                random_number):
        left_dither = min(previous_isi - self.refr_period, self.dither)
        right_dither = min(subsequent_isi - self.refr_period, self.dither)
        step = random_number * (right_dither + left_dither) - left_dither
        return step


def surrogates(
        spiketrain, n=1, surr_method='dither_spike_train', dt=None,
        decimals=None, edges=True):
    """
    Generates surrogates of a :attr:`spiketrain` by a desired generation
    method.

    This routine is a wrapper for the other surrogate generators in the
    module.

    The surrogates retain the :attr:`t_start` and :attr:`t_stop` of the
    original :attr:`spiketrain`.


    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates
    n : int, optional
        Number of surrogates to be generated.
        Default: 1
    surr_method : str, optional
        The method to use to generate surrogate spike trains. Can be one of:
        * 'dither_spike_train': see surrogates.dither_spike_train() [dt needed]
        * 'dither_spikes': see surrogates.dither_spikes() [dt needed]
        * 'jitter_spikes': see surrogates.jitter_spikes() [dt needed]
        * 'randomise_spikes': see surrogates.randomise_spikes()
        * 'shuffle_isis': see surrogates.shuffle_isis()
        * 'joint_isi_dithering': see surrogates.joint_isi_dithering()
        Default: 'dither_spike_train'
    dt : quantities.Quantity, optional
        For methods shifting spike times randomly around their original time
        (spike dithering, train shifting) or replacing them randomly within a
        certain window (spike jittering), dt represents the size of that
        shift / window. For other methods, dt is ignored.
        Default: None
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None
    edges : bool
        For surrogate spikes falling outside the range `[spiketrain.t_start,
        spiketrain.t_stop)`, whether to drop them out (for edges = True) or set
        that to the range's closest end (for edges = False).
        Default: True

    Returns
    -------
    list of neo.SpikeTrain objects
      A list of spike trains, each obtained from `spiketrain` by randomly
      dithering its spikes. The range of the surrogate `neo.SpikeTrain`
      object(s) is the same as `spiketrain`.
    """

    # Define the surrogate function to use, depending on the specified method
    surrogate_types = {
        'dither_spike_train': dither_spike_train,
        'dither_spikes': dither_spikes,
        'jitter_spikes': jitter_spikes,
        'randomise_spikes': randomise_spikes,
        'shuffle_isis': shuffle_isis,
        'joint_isi_dithering': None}

    if surr_method not in surrogate_types.keys():
        raise ValueError('specified surr_method (=%s) not valid' % surr_method)

    if surr_method in ['dither_spike_train', 'dither_spikes', 'jitter_spikes']:
        return surrogate_types[surr_method](
            spiketrain, dt, n=n, decimals=decimals, edges=edges)
    elif surr_method in ['randomise_spikes', 'shuffle_isis']:
        return surrogate_types[surr_method](
            spiketrain, n=n, decimals=decimals)
    elif surr_method == 'joint_isi_dithering':
        return JointISI(spiketrain).dithering(n)
