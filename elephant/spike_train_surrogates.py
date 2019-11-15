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
    probability distribution, that results from a fiexed sum of ISI_before
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
            surr_times = np.cumsum(np.random.permutation(ISIs)) *\
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
    surr = data.reshape((1, len(data))) + 2 * shift * \
        np.random.random_sample((n, 1)) - shift

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
    `neo.SpikeTrain` objects are locally poissonian (the inter-spike-interval
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

    # Compute each surrogate by dilatating and shifting each spike s in the
    # poisson 0-1 spike trains to dilat * s + offset. Attach time unit again
    surr = np.sort(surr_poiss01 * dilats + offsets, axis=1) * std_unit

    return [neo.SpikeTrain(s, t_start=spiketrain.t_start,
                           t_stop=spiketrain.t_stop).rescale(spiketrain.units)
            for s in surr]


def joint_isi_dithering(spiketrain,
                        n=1,
                        dither=15.*pq.ms,
                        unit=pq.s,
                        window_length=120.*pq.ms,
                        num_bins=120,
                        sigma=1.*pq.ms,
                        isi_median_threshold=30*pq.ms,
                        alternate=True,
                        show_plot=False,
                        print_mode=False,
                        use_sqrt=False,
                        method='fast',
                        cutoff=True,
                        min_spikes=10
                        ):
    """
    Implementation of Joint-ISI-dithering for spiketrains that pass the
    threshold of the dense rate, if not a uniform dithered spiketrain is
    given back. The implementation continued the ideas of Louis et al.
    (2010) and Gerstein (2004).

    If you want to further analyse the results or if you want to speed up the
    runtime saving the preprocessing results, it is useful to work with the
    class :class:`Joint_ISI_Space` of the module: joint_isi_dithering_class.

    To make the dithering-procedure stable, the spiketrain needs to pass two
    thresholds, that the Joint-ISI dithering is applied, if not uniform
    dithering is used.

    Attributes
    ----------
    spiketrain: neo.SpikeTrain
                For this spiketrain the surrogates will be created

    n: int, optional
        Number of surrogates to be created.
        Default: 1
    dither: pq.Quantity
        The range of the dithering for the uniform dithering,
        which is also used for the method 'window'.
        Default: 15.*pq.ms
    unit: pq.unit
        The unit of the spiketrain in the output.
        Default: pq.s
    window_length: pq.Quantity
        The extent in which the joint-ISI-distribution is calculated.
        Default: 120*pq.ms
    num_bins: int
        The size of the joint-ISI-ditribution will be num_bins*num_bins.
        Default: 120
    sigma: pq.Quantity
        The standard deviation of the Gaussian kernel, with which
        the data is convoluted.
        Default: 0.001*pq.s
    isi_median_threshold: pq.Quantity
        Only if the median of the ISI distribution is smaller than
        isi_median_threshold the Joint-ISI dithering is applied, if not the
        uniform dithering is used.
        Default: 30*pq.ms
    alternate: boolean
        If alternate == True: then first all even and then all odd spikes are
        dithered. Else: in acending order from the first to the last spike, all
        spikes are moved.
        Default: True.
    show_plot: boolean
        if show_plot == True the joint-ISI distribution will be plotted
        Default: False
    print_mode: boolean
        If True, also the way of how the dithered spikes are evaluated
        is returned so 'uniform' for uniform and dithering and 'jisid' for
        joint-ISI-dithering
        Default: False
    use_sqrt: boolean
        if use_sqrt == True a sqrt is applied to the joint-ISI histogram,
        following Gerstein et al. 2004
        Default: False
    method: string
        if 'fast' the entire diagonals of the joint-ISI histograms are
        used if 'window' only the values of the diagonals are used, whose
        distance is lower than dither
        Default: 'fast'
    cutoff: boolean
        if True than the Filtering of the Joint-ISI histogram is
        limited to the lower side by the minimal ISI.
        This can be necessary, if in the data there is a certain dead time,
        which would be destroyed by the convolution with the 2d-Gaussian
        function.
        Default: True
    min_spikes: int
        if the number of spikes is lower than this number, the spiketrain
        is directly passed to the uniform dithering.
        Default: 10


    Returns
    ----------
    dithered_sts: list
        list of spiketrains, that are dithered versions of the given
        spiketrain
    if print_mode == True
    mode: string
        Indicates, which method was used to dither the spikes.
        'jisid' if joint-ISI was used,
        'uniform' if the ISI median was too low and uniform dithering was
        used.
    """
    return JointISISpace(spiketrain,
                         n_surr=n,
                         dither=dither,
                         unit=unit,
                         window_length=window_length,
                         num_bins=num_bins,
                         sigma=sigma,
                         isi_median_threshold=isi_median_threshold,
                         alternate=alternate,
                         show_plot=show_plot,
                         print_mode=print_mode,
                         use_sqrt=use_sqrt,
                         method=method,
                         cutoff=cutoff,
                         min_spikes=min_spikes
                         ).dithering()


class JointISISpace:
    """
    The class :class:`Joint_ISI_Space` is implemented for Joint-ISI dithering
    as a continuation of the ideas of Louis et al. (2010) and Gerstein (2004).

    When creating an class instance all necessary preprocessing steps are done,
    to use the method dithering().

    To make the dithering-procedure stable, the spiketrain needs to pass two
    thresholds, that the Joint-ISI dithering is applied, if not uniform
    dithering is used.

    Attributes
    ----------
    st: neo.SpikeTrain
        For this spiketrain the surrogates will be created

    n_surr: int, optional
        Number of surrogates to be created.
        Default: 1
    dither: pq.Quantity
        The range of the dithering for the uniform dithering,
        which is also used for the method 'window'.
        Default: 15.*pq.ms
    window_length: pq.Quantity
        The Joint-ISI distribution is as such defined on a range for ISI_i and
        ISI_(i+1) from 0 to inf. Since this is computationally not feasible,
        the Joint-ISI distribution is truncated for high ISI. The Joint-ISI
        histogram is calculated for ISI_i, ISI_(i+1) from 0 to window_length.
        Default: 120*pq.ms
    num_bins: int
        The size of the joint-ISI-distribution will be num_bins*num_bins.
        Default: 120
    sigma: pq.Quantity
        The standard deviation of the Gaussian kernel, with which
        the data is convoluted.
        Default: 0.001*pq.s
    isi_median_threshold: pq.Quantity
        Only if the median of the ISI distribution is smaller than
        isi_median_threshold the Joint-ISI dithering is applied, if not the
        uniform dithering is used.
        Default: 30*pq.ms
    alternate: boolean
        If alternate == True: then first all even and then all odd spikes are
        dithered. Else: in ascending order from the first to the last spike,
        all spikes are moved.
        Default: True.
    print_mode: boolean
        If True, also the way of how the dithered spikes are evaluated
        is returned so 'uniform' for uniform and dithering and 'jisid' for
        joint-ISI-dithering
        Default: False
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
    min_spikes: int
        if the number of spikes is lower than this number, the spiketrain
        is directly passed to the uniform dithering.
        Default: 10

    Methods
    ----------
    preprocessing()
        The preprocessing function is called in the initialization process.
        Outside of it is only necessary if the attributes of the
        :class:`Joint_ISI_Space` were changed after the initialization, than it
        prepares the class again to create dithered spiketrains.
    dithering()
        Returns a list of dithered spiketrains and if print_mode it returns
        also a string 'uniform' or 'jisid' indicating the way, how the dithered
        spiketrains were obtained.
    """

    def __init__(self,
                 st,
                 n_surr=1,
                 dither=15. * pq.ms,
                 window_length=120. * pq.ms,
                 num_bins=120,
                 sigma=1. * pq.ms,
                 isi_median_threshold=30 * pq.ms,
                 alternate=True,
                 print_mode=False,
                 use_sqrt=False,
                 method='fast',
                 cutoff=True,
                 min_spikes=10
                 ):
        self.st = st

        self.n_surr = n_surr

        self.dither = dither
        self.window_length = window_length
        self.sigma = sigma
        self.isi_median_threshold = isi_median_threshold

        self.num_bins = num_bins

        self.alternate = alternate
        self.print_mode = print_mode
        self.use_sqrt = use_sqrt
        self.method = method
        self.cutoff = cutoff
        self.min_spikes = min_spikes

        self.preprocessing()

    def preprocessing(self):
        """
        To perform the Joint-ISI dithering a preprocessing procedure for each
        spiketrain is necessary. This is part of the initializer (__init___).
        If after calling the class for the first time, a parameter is changed,
        the preprocessing needs to be done again.

        First, two checks are done. If they are not passed, self.method is
        set to 'uniform'. The first one asks for the number of spikes.
        The second compares the median of the ISI-distribution against a
        threshold.

        If the method is not 'uniform' the cumulative distribution functions
        for the Joint-ISI dither process are evaluated.

        If method is 'fast':
        For each slice of the joint-ISI
        distribution (parallel to the anti-diagonal) a cumulative distribution
        function is calculated.

        If method is 'window':
        For each point in the joint-ISI distribution a on the line parallel to
        the anti-diagonal all points up to the dither-parameter are included,
        to calculate the cumulative distribution function.

        The function has no output, but stores its result inside the class.
        """
        if len(self.st) < self.min_spikes:
            self.method = 'uniform'
            return None

        self._unit = self.st.units

        self._isi = isi(self.st.rescale(self._unit).magnitude)
        isi_median = np.median(self._isi)

        if isi_median > self.isi_median_threshold.rescale(
                self._unit).magnitude:
            self.method = 'uniform'
            return None

        if isinstance(self.dither, pq.Quantity):
            self.dither = self.dither.rescale(self._unit).magnitude
        if isinstance(self.window_length, pq.Quantity):
            self.window_length = self.window_length.rescale(
                self._unit).magnitude
        if isinstance(self.sigma, pq.Quantity):
            self.sigma = self.sigma.rescale(self._unit).magnitude

        self._sampling_rhythm = self.alternate + 1

        self._bin_width = self.window_length / self.num_bins

        def isi_to_index(isi):
            return np.rint(isi / self._bin_width - 0.5).astype(int)

        self._isi_to_index = isi_to_index

        self._number_of_isis = len(self._isi)
        self._first_spike = self.st[0].rescale(self._unit).magnitude
        self._t_stop = self.st.t_stop.rescale(self._unit).magnitude

        self._get_joint_isi_histogram()

        # Gives an array, taking an element with an index of the Joint-ISI
        # distribution gives back the corresponding ISI.
        self._indices_to_isi = (np.arange(self.num_bins)
                                + 0.5) * self._bin_width

        flipped_jisih = np.flip(self.jisih.T, 0)

        def normalize(v):
            if v[-1] - v[0] > 0.:
                return (v - v[0]) / (v[-1] - v[0])
            return np.zeros_like(v)

        self._normalize = normalize

        if self.method == 'fast':
            self._jisih_cumulatives = [normalize(
                np.cumsum(np.diagonal(flipped_jisih,
                                      -self.num_bins + double_index + 1)))
                for double_index in range(self.num_bins)]
            return None

        if self.method == 'window':
            self._jisih_cumulatives = self._window_cumulatives(flipped_jisih)
            return None

        error_message = ('method must can only be \'uniform\' or \'fast\' '
                         'or \'window\', but not \'' + self.method + '\' .')
        raise ValueError(error_message)

    def dithering(self):
        """
        Implementation of Joint-ISI-dithering for spiketrains that pass the
        threshold of the dense rate, if not a uniform dithered spiketrain is
        given back. The implementation continued the ideas of Louis et al.
        (2010) and Gerstein (2004).

        Returns
        ----------
        dithered_sts: list
            list of spiketrains, that are dithered versions of the given
            spiketrain
        if print_mode == True
        mode: string
            Indicates, which method was used to dither the spikes.
            'jisid' if joint-ISI was used,
            'uniform' if the ISI median was too low and uniform dithering was
            used.
        """
        if self.method == 'uniform':
            if self.print_mode:
                return dither_spikes(
                    self.st, self.dither,
                    n=self.n_surr), 'uniform'
            return dither_spikes(
                self.st, self.dither,
                n=self.n_surr)

        if self.method == 'fast' or self.method == 'window':
            if self.print_mode:
                return self._dithering_process(), 'jisid'
            return self._dithering_process()

        error_message = ('method must can only be \'uniform\' or \'fast\' '
                         'or \'window\', but not \'' + self.method + '\' .')
        raise ValueError(error_message)

    def _get_joint_isi_histogram(self):
        """
        This function calculates the joint-ISI histogram.
        """
        jisih = np.histogram2d(self._isi[:-1], self._isi[1:],
                               bins=[self.num_bins, self.num_bins],
                               range=[[0., self.window_length],
                                      [0., self.window_length]])[0]

        if self.use_sqrt:
            jisih = np.sqrt(jisih)

        if self.cutoff:
            minimal_isi = np.min(self._isi)
            start_index = self._isi_to_index(minimal_isi)
            jisih[start_index:, start_index:] = gaussian_filter(
                jisih[start_index:, start_index:],
                self.sigma / self._bin_width)

            jisih[:start_index + 1, :] = np.zeros_like(
                jisih[:start_index + 1, :])
            jisih[:, :start_index + 1] = np.zeros_like(
                jisih[:, :start_index + 1])

        else:
            jisih = gaussian_filter(jisih, self.sigma / self._bin_width)
        self.jisih = jisih
        return None

    def _window_diagonal_cumulatives(self, flipped_jisih):
        self.max_change_index = self._isi_to_index(self.dither)
        self.max_change_isi = self._indices_to_isi[self.max_change_index]

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
                            + self.max_change_index + 1] = cum_diag

            cum_bound = np.repeat(jisih_diag_cums[double_index,
                                                  double_index +
                                                  self.max_change_index],
                                  self.max_change_index)

            jisih_diag_cums[double_index,
                            double_index + self.max_change_index + 1:
                            double_index
                            + 2 * self.max_change_index + 1] = cum_bound
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
                normalized_cum = self._normalize(cum_slice)
                jisih_cumulatives[back_index][for_index] = normalized_cum
        return jisih_cumulatives

    def _dithering_process(self):
        """
        Dithering process for the Joint-ISI dithering.

        Returns
        --------
        dithered_sts
            list of neo.SpikeTrain: A list of len n_surr,
            each entry is one dithered spiketrain.
        """

        dithered_sts = []
        for surr_number in range(self.n_surr):
            dithered_isi = self._get_dithered_isi()

            dithered_st = self._first_spike + np.hstack(
                (np.array(0.), np.cumsum(dithered_isi)))
            dithered_st = neo.SpikeTrain(dithered_st * self._unit,
                                         t_stop=self._t_stop)
            dithered_sts.append(dithered_st)
        return dithered_sts

    def _get_dithered_isi(self):
        dithered_isi = self._isi
        random_list = np.random.random(self._number_of_isis)
        if self.method == 'fast':
            for start in range(self._sampling_rhythm):
                dithered_isi_indices = self._isi_to_index(dithered_isi)
                for i in range(start, self._number_of_isis - 1,
                               self._sampling_rhythm):
                    self._update_dithered_isi_fast(dithered_isi,
                                                   dithered_isi_indices,
                                                   random_list[i],
                                                   i)
        else:
            for start in range(self._sampling_rhythm):
                dithered_isi_indices = self._isi_to_index(dithered_isi)
                for i in range(start, self._number_of_isis - 1,
                               self._sampling_rhythm):
                    self._update_dithered_isi_window(dithered_isi,
                                                     dithered_isi_indices,
                                                     random_list[i],
                                                     i)
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
                step = (self._indices_to_isi[new_index]
                        - self._indices_to_isi[back_index])
                dithered_isi[i] += step
                dithered_isi[i + 1] -= step
        return None

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
                step = (self._indices_to_isi[new_index]
                        - self.max_change_isi)
                dithered_isi[i] += step
                dithered_isi[i + 1] -= step
        return None


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
        'joint_isi_dithering': joint_isi_dithering}

    if surr_method not in surrogate_types.keys():
        raise ValueError('specified surr_method (=%s) not valid' % surr_method)

    if surr_method in ['dither_spike_train', 'dither_spikes', 'jitter_spikes']:
        return surrogate_types[surr_method](
            spiketrain, dt, n=n, decimals=decimals, edges=edges)
    elif surr_method in ['randomise_spikes', 'shuffle_isis']:
        return surrogate_types[surr_method](
            spiketrain, n=n, decimals=decimals)
    elif surr_method == 'joint_isi_dithering':
        return surrogate_types[surr_method](
            spiketrain, n=n)
