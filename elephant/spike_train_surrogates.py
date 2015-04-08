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

[1] Louis et al (2010) Surrogate Spike Train Generation Through Dithering in
    Operational Time. Front Comput Neurosci. 2010; 4: 127.

:original implementation by: Emiliano Torre [e.torre@fz-juelich.de]
:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
import quantities as pq
import neo
import elephant.statistics as es


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
    >>> print dither_spikes(st, dither = 20*pq.ms, decimals=0)   # doctest: +SKIP
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
        surr = np.minimum(np.maximum(surr.base,
            (spiketrain.t_start / spiketrain.units).base),
            (spiketrain.t_stop / spiketrain.units).base) * spiketrain.units
    else:
        # Leave out all spikes outside [spiketrain.t_start, spiketrain.t_stop]
        tstart, tstop = (spiketrain.t_start / spiketrain.units).base, \
                        (spiketrain.t_stop / spiketrain.units).base
        surr = [s[np.all([s >= tstart, s < tstop], axis=0)] * spiketrain.units
                for s in surr.base]

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
    return [neo.SpikeTrain(np.sort(st), t_start=spiketrain.t_start, t_stop=spiketrain.t_stop)
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
        ISIs = np.hstack([isi0, es.isi(spiketrain)])

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
        empty_train = neo.SpikeTrain([]*spiketrain.units,
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
    >>> print dither_spike_train(st, shift = 20*pq.ms, decimals=0)   # doctest: +SKIP
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
        surr = np.minimum(np.maximum(surr.base,
            (spiketrain.t_start / spiketrain.units).base),
            (spiketrain.t_stop / spiketrain.units).base) * spiketrain.units
    else:
        # Leave out all spikes outside [spiketrain.t_start, spiketrain.t_stop]
        tstart, tstop = (spiketrain.t_start / spiketrain.units).base,\
                        (spiketrain.t_stop / spiketrain.units).base
        surr = [s[np.all([s >= tstart, s < tstop], axis=0)] * spiketrain.units
                for s in surr.base]

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


def surrogates(
        spiketrain, n=1, surr_method='dither_spike_train', dt=None, decimals=None,
        edges=True):
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
        'shuffle_isis': shuffle_isis}

    if surr_method not in surrogate_types.keys():
        raise ValueError('specified surr_method (=%s) not valid' % surr_method)

    if surr_method in ['dither_spike_train', 'dither_spikes', 'jitter_spikes']:
        return surrogate_types[surr_method](
            spiketrain, dt, n=n, decimals=decimals, edges=edges)
    elif surr_method in ['randomise_spikes', 'shuffle_isis']:
        return surrogate_types[surr_method](
            spiketrain, n=n, decimals=decimals)