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

References
----------
[1] Louis et al (2010). Surrogate Spike Train Generation Through Dithering in
    Operational Time. Front Comput Neurosci. 2010; 4: 127.
[2] Gerstein, G. L. (2004). Searching for significance in spatio-temporal
    firing patterns. Acta Neurobiologiae Experimentalis, 64(2), 203-208.

Original implementation by: Emiliano Torre [e.torre@fz-juelich.de]
:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import random
import warnings

import neo
import numpy as np
import quantities as pq
from scipy.ndimage import gaussian_filter

from elephant.statistics import isi
from elephant.utils import deprecated_alias

# List of all available surrogate methods
SURR_METHODS = ['dither_spike_train', 'dither_spikes', 'jitter_spikes',
                'randomise_spikes', 'shuffle_isis', 'joint_isi_dithering',
                'dither_spikes_with_refractory_period']


def _dither_spikes_with_refractory_period(spiketrain, dither, n_surrogates,
                                          refractory_period):
    units = spiketrain.units
    t_start = spiketrain.t_start.rescale(units).magnitude
    t_stop = spiketrain.t_stop.rescale(units).magnitude

    dither = dither.rescale(units).magnitude
    refractory_period = refractory_period.rescale(units).magnitude
    # The initially guesses refractory period is compared to the minimal ISI.
    # The smaller value is taken as the refractory to calculate with.
    refractory_period = np.min(np.diff(spiketrain.magnitude),
                               initial=refractory_period)

    dithered_spiketrains = []
    for _ in range(n_surrogates):
        dithered_st = np.copy(spiketrain.magnitude)
        random_ordered_ids = np.arange(len(spiketrain))
        np.random.shuffle(random_ordered_ids)

        for random_id in random_ordered_ids:
            spike = dithered_st[random_id]
            prev_spike = dithered_st[random_id - 1] \
                if random_id > 0 \
                else t_start - refractory_period
            # subtract refractory period so that the first spike can move up
            # to t_start
            next_spike = dithered_st[random_id + 1] \
                if random_id < len(spiketrain) - 1 \
                else t_stop + refractory_period
            # add refractory period so that the last spike can move up
            # to t_stop

            # Dither range in the direction to the previous spike
            prev_dither = min(dither, spike - prev_spike - refractory_period)
            # Dither range in the direction to the next spike
            next_dither = min(dither, next_spike - spike - refractory_period)

            dt = (prev_dither + next_dither) * random.random() - prev_dither
            dithered_st[random_id] += dt

        dithered_spiketrains.append(dithered_st)

    dithered_spiketrains = np.array(dithered_spiketrains) * units

    return dithered_spiketrains


@deprecated_alias(n='n_surrogates')
def dither_spikes(spiketrain, dither, n_surrogates=1, decimals=None,
                  edges=True, refractory_period=None):
    """
    Generates surrogates of a spike train by spike dithering.

    The surrogates are obtained by uniformly dithering times around the
    original position. The dithering is performed independently for each
    surrogate.

    The surrogates retain the `spiketrain.t_start` and `spiketrain.t_stop`.
    Spikes moved beyond this range are lost or moved to the range's ends,
    depending on the parameter `edges`.

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates.
    dither : pq.Quantity
        Amount of dithering. A spike at time `t` is placed randomly within
        `]t-dither, t+dither[`.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1.
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates at a
        millisecond level.
        If None, machine precision is used.
        Default: None.
    edges : bool, optional
        For surrogate spikes falling outside the range
        `[spiketrain.t_start, spiketrain.t_stop)`, whether to drop them out
        (for `edges = True`) or set them to the range's closest end
        (for `edges = False`).
        Default: True.
    refractory_period : pq.Quantity or None, optional
        The dither range of each spike is adjusted such that the spike can not
        fall into the `refractory_period` of the previous or next spike.
        To account this, the refractory period is estimated as the smallest ISI
        of the spike train. The given argument `refractory_period` here is thus
        an initial estimation.
        Note, that with this option a spike cannot "jump" over the previous or
        next spike as it is normally possible.
        If set to `None`, no refractoriness is in dithering.
        Default: None

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        randomly dithering its spikes. The range of the surrogate spike trains
        is the same as of `spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    ...
    >>> st = neo.SpikeTrain([100, 250, 600, 800] * pq.ms, t_stop=1 * pq.s)
    >>> print(dither_spikes(st, dither = 20 * pq.ms))  # doctest: +SKIP
    [<SpikeTrain(array([  96.53801903,  248.57047376,  601.48865767,
     815.67209811]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(dither_spikes(st, dither = 20 * pq.ms, n_surrogates=2))
    [<SpikeTrain(array([ 104.24942044,  246.0317873 ,  584.55938657,
        818.84446913]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 111.36693058,  235.15750163,  618.87388515,
        786.1807108 ]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(dither_spikes(st, dither = 20 * pq.ms,
                            decimals=0))  # doctest: +SKIP
    [<SpikeTrain(array([  81.,  242.,  595.,  799.]) * ms,
        [0.0 ms, 1000.0 ms])>]

    """
    if len(spiketrain) == 0:
        # return the empty spiketrain n times
        return [spiketrain.copy() for _ in range(n_surrogates)]

    units = spiketrain.units
    t_start = spiketrain.t_start.rescale(units).magnitude
    t_stop = spiketrain.t_stop.rescale(units).magnitude

    if refractory_period is None or refractory_period == 0:
        # Main: generate the surrogates
        dither = dither.rescale(units).magnitude
        dithered_spiketrains = \
            spiketrain.magnitude.reshape((1, len(spiketrain))) \
            + 2 * dither * np.random.random_sample(
                (n_surrogates, len(spiketrain))) - dither
        dithered_spiketrains.sort(axis=1)

        if edges:
            # Leave out all spikes outside
            # [spiketrain.t_start, spiketrain.t_stop]
            dithered_spiketrains = \
                [train[
                     np.all([t_start < train, train < t_stop], axis=0)]
                 for train in dithered_spiketrains]
        else:
            # Move all spikes outside
            # [spiketrain.t_start, spiketrain.t_stop] to the range's ends
            dithered_spiketrains = np.minimum(
                np.maximum(dithered_spiketrains, t_start),
                t_stop)

        dithered_spiketrains = dithered_spiketrains * units

    elif isinstance(refractory_period, pq.Quantity):
        dithered_spiketrains = _dither_spikes_with_refractory_period(
            spiketrain, dither, n_surrogates, refractory_period)
    else:
        raise ValueError("refractory_period must be of type pq.Quantity")

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        dithered_spiketrains = \
            dithered_spiketrains.rescale(pq.ms).round(decimals).rescale(units)

    # Return the surrogates as list of neo.SpikeTrain
    return [neo.SpikeTrain(
        train, t_start=t_start, t_stop=t_stop)
            for train in dithered_spiketrains]


@deprecated_alias(n='n_surrogates')
def randomise_spikes(spiketrain, n_surrogates=1, decimals=None):
    """
    Generates surrogates of a spike train by spike time randomization.

    The surrogates are obtained by keeping the spike count of the original
    `spiketrain`, but placing the spikes randomly in the interval
    `[spiketrain.t_start, spiketrain.t_stop]`. The generated independent
    neo.SpikeTrain objects follow  Poisson statistics (exponentially
    distributed inter-spike intervals).

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1.
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates.
        If None, machine precision is used.
        Default: None.

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        randomly distributing its spikes in the interval
        `[spiketrain.t_start, spiketrain.t_stop]`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    ...
    >>> st = neo.SpikeTrain([100, 250, 600, 800] * pq.ms, t_stop=1 * pq.s)
    >>> print(randomise_spikes(st))  # doctest: +SKIP
        [<SpikeTrain(array([ 131.23574603,  262.05062963,  549.84371387,
                            940.80503832]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(randomise_spikes(st, n_surrogates=2))  # doctest: +SKIP
        [<SpikeTrain(array([  84.53274955,  431.54011743,  733.09605806,
              852.32426583]) * ms, [0.0 ms, 1000.0 ms])>,
         <SpikeTrain(array([ 197.74596726,  528.93517359,  567.44599968,
              775.97843799]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(randomise_spikes(st, decimals=0))  # doctest: +SKIP
        [<SpikeTrain(array([  29.,  667.,  720.,  774.]) * ms,
              [0.0 ms, 1000.0 ms])>]

    """
    # Create surrogate spike trains as rows of a Quantity array
    sts = ((spiketrain.t_stop - spiketrain.t_start) *
           np.random.random(size=(n_surrogates, len(spiketrain))) +
           spiketrain.t_start).rescale(spiketrain.units)

    # Round the surrogate data to decimal position, if requested
    if decimals is not None:
        sts = sts.round(decimals)

    # Convert the Quantity array to a list of SpikeTrains, and return them
    return [neo.SpikeTrain(np.sort(st), t_start=spiketrain.t_start,
                           t_stop=spiketrain.t_stop)
            for st in sts]


@deprecated_alias(n='n_surrogates')
def shuffle_isis(spiketrain, n_surrogates=1, decimals=None):
    """
    Generates surrogates of a spike train by inter-spike-interval (ISI)
    shuffling.

    The surrogates are obtained by randomly sorting the ISIs of the given input
    `spiketrain`. This generates independent `neo.SpikeTrain` object(s) with
    same ISI distribution and spike count as in `spiketrain`, while
    destroying temporal dependencies and firing rate profile.

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1.
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates.
        If None, machine precision is used.
        Default: None.

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        random ISI shuffling. The time range of the surrogate spike trains is
        the same as in `spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    ...
    >>> st = neo.SpikeTrain([100, 250, 600, 800] * pq.ms, t_stop=1 * pq.s)
    >>> print(shuffle_isis(st))  # doctest: +SKIP
        [<SpikeTrain(array([ 200.,  350.,  700.,  800.]) * ms,
                 [0.0 ms, 1000.0 ms])>]
    >>> print(shuffle_isis(st, n_surrogates=2))  # doctest: +SKIP
        [<SpikeTrain(array([ 100.,  300.,  450.,  800.]) * ms,
              [0.0 ms, 1000.0 ms])>,
         <SpikeTrain(array([ 200.,  350.,  700.,  800.]) * ms,
              [0.0 ms, 1000.0 ms])>]

    """
    if len(spiketrain) > 0:
        isi0 = spiketrain[0] - spiketrain.t_start
        ISIs = np.hstack([isi0, isi(spiketrain)])

        # Round the isis to decimal position, if requested
        if decimals is not None:
            ISIs = ISIs.round(decimals)

        # Create list of surrogate spike trains by random ISI permutation
        sts = []
        for surrogate_id in range(n_surrogates):
            surr_times = np.cumsum(np.random.permutation(ISIs)) * \
                spiketrain.units + spiketrain.t_start
            sts.append(neo.SpikeTrain(
                surr_times, t_start=spiketrain.t_start,
                t_stop=spiketrain.t_stop))

    else:
        sts = [neo.SpikeTrain([] * spiketrain.units,
                              t_start=spiketrain.t_start,
                              t_stop=spiketrain.t_stop)] * n_surrogates

    return sts


@deprecated_alias(n='n_surrogates')
def dither_spike_train(spiketrain, shift, n_surrogates=1, decimals=None,
                       edges=True):
    """
    Generates surrogates of a spike train by spike train shifting.

    The surrogates are obtained by shifting the whole spike train by a
    random amount of time (independent for each surrogate). Thus, ISIs and
    temporal correlations within the spike train are kept. For small shifts,
    the firing rate profile is also kept with reasonable accuracy.

    The surrogates retain the `spiketrain.t_start` and `spiketrain.t_stop`.
    Spikes moved beyond this range are lost or moved to the range's ends,
    depending on the parameter `edges`.

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates.
    shift : pq.Quantity
        Amount of shift. `spiketrain` is shifted by a random amount uniformly
        drawn from the range ]-shift, +shift[.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1.
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates.
        If None, machine precision is used.
        Default: None.
    edges : bool
        For surrogate spikes falling outside the range `[spiketrain.t_start,
        spiketrain.t_stop)`, whether to drop them out (for `edges = True`) or
        set them to the range's closest end (for `edges = False`).
        Default: True.

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        randomly dithering the whole spike train. The time range of the
        surrogate spike trains is the same as in `spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    ...
    >>> st = neo.SpikeTrain([100, 250, 600, 800] * pq.ms, t_stop=1 * pq.s)
    >>> print(dither_spike_train(st, shift = 20*pq.ms))  # doctest: +SKIP
    [<SpikeTrain(array([  96.53801903,  248.57047376,  601.48865767,
     815.67209811]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(dither_spike_train(st, shift = 20*pq.ms, n_surrogates=2))
    [<SpikeTrain(array([  92.89084054,  242.89084054,  592.89084054,
        792.89084054]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([  84.61079043,  234.61079043,  584.61079043,
        784.61079043]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(dither_spike_train(st, shift = 20 * pq.ms,
                                 decimals=0))  # doctest: +SKIP
    [<SpikeTrain(array([  82.,  232.,  582.,  782.]) * ms,
        [0.0 ms, 1000.0 ms])>]

    """
    # Transform spiketrain into a Quantity object (needed for matrix algebra)
    data = spiketrain.view(pq.Quantity)

    # Main: generate the surrogates by spike train shifting
    surr = data.reshape(
        (1, len(data))) + 2 * shift * np.random.random_sample(
        (n_surrogates, 1)) - shift

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


@deprecated_alias(binsize='bin_size', n='n_surrogates')
def jitter_spikes(spiketrain, bin_size, n_surrogates=1):
    """
    Generates surrogates of a spike train by spike jittering.

    The surrogates are obtained by defining adjacent time bins spanning the
    `spiketrain` range, and randomly re-positioning (independently for each
    surrogate) each spike in the time bin it falls into.

    The surrogates retain the `spiketrain.t_start` and `spiketrain.t_stop`.
    Note that within each time bin the surrogate spike trains are locally
    Poissonian (the inter-spike-intervals are exponentially distributed).

    Parameters
    ----------
    spiketrain :  neo.SpikeTrain
        The spike train from which to generate the surrogates.
    bin_size : pq.Quantity
        Size of the time bins within which to randomize the spike times.
        Note: the last bin lasts until `spiketrain.t_stop` and might have
        width different from `bin_size`.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1.

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        randomly replacing its spikes within bins of user-defined width. The
        time range of the surrogate spike trains is the same as in
        `spiketrain`.

    Examples
    --------
    >>> import quantities as pq
    >>> import neo
    ...
    >>> st = neo.SpikeTrain([80, 150, 320, 480] * pq.ms, t_stop=1 * pq.s)
    >>> print(jitter_spikes(st, bin_size=100 * pq.ms))  # doctest: +SKIP
    [<SpikeTrain(array([  98.82898293,  178.45805954,  346.93993867,
        461.34268507]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(jitter_spikes(st, bin_size=100 * pq.ms, n_surrogates=2))
    [<SpikeTrain(array([  97.15720041,  199.06945744,  397.51928207,
        402.40065162]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([  80.74513157,  173.69371317,  338.05860962,
        495.48869981]) * ms, [0.0 ms, 1000.0 ms])>]
    >>> print(jitter_spikes(st, bin_size=100 * pq.ms))  # doctest: +SKIP
    [<SpikeTrain(array([  4.55064897e-01,   1.31927046e+02,   3.57846265e+02,
         4.69370604e+02]) * ms, [0.0 ms, 1000.0 ms])>]

    """
    # Define standard time unit; all time Quantities are converted to
    # scalars after being rescaled to this unit, to use the power of numpy
    std_unit = bin_size.units

    # Compute bin edges for the jittering procedure
    # !: the last bin arrives until spiketrain.t_stop and might have
    # size != bin_size
    start_dl = spiketrain.t_start.rescale(std_unit).magnitude
    stop_dl = spiketrain.t_stop.rescale(std_unit).magnitude

    bin_edges = start_dl + np.arange(start_dl, stop_dl, bin_size.magnitude)
    bin_edges = np.hstack([bin_edges, stop_dl])

    # Create n surrogates with spikes randomly placed in the interval (0,1)
    surr_poiss01 = np.random.random_sample((n_surrogates, len(spiketrain)))

    # Compute the bin id of each spike
    bin_ids = np.array(
        (spiketrain.view(pq.Quantity) /
         bin_size).rescale(pq.dimensionless).magnitude, dtype=int)

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

    When creating a class instance, all necessary preprocessing steps are done
    to use :func:`JointISI.dithering` method.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Input spiketrain to create surrogates from.
    dither : pq.Quantity, optional
        This quantity describes the maximum displacement of a spike, when
        method is 'window'. It is also used for the uniform dithering for
        the spikes, which are outside the regime in the Joint-ISI
        histogram, where Joint-ISI dithering is applicable.
        Default: 15. * pq.ms.
    truncation_limit : pq.Quantity, optional
        The Joint-ISI distribution of :math:`(ISI_i, ISI_{i+1})` is defined
        within the range `[0, inf]`. Since this is computationally not
        feasible, the Joint-ISI distribution is truncated for high ISI.
        The Joint-ISI histogram is calculated for
        :math:`(ISI_i, ISI_{i+1})` from 0 to `truncation_limit`.
        Default: 100 * pq.ms.
    n_bins : int, optional
        The size of the joint-ISI-distribution will be
        `n_bins*n_bins/2`.
        Default: 100.
    sigma : pq.Quantity, optional
        The standard deviation of the Gaussian kernel, with which
        the data is convolved.
        Default: 2. * pq.ms.
    alternate : boolean, optional
        If True, then all even spikes are dithered followed
        by all odd spikes. Otherwise, the spikes are dithered in ascending
        order from the first to the last spike.
        Default: True.
    use_sqrt : boolean, optional
        If True, the joint-ISI histogram is preprocessed by
        applying a square root (following Gerstein et al. 2004).
        Default: False.
    method : {'fast', window'}, optional
        * 'fast': the spike can move in the whole range between the
            previous and subsequent spikes (computationally efficient).
        * 'window': the spike movement is limited to the parameter `dither`
        Default: 'fast'.
    cutoff : boolean, optional
        If True, then the filtering of the Joint-ISI histogram is
        limited on the lower side by the minimal ISI.
        This can be necessary, if in the data there is a certain refractory
        period, which will be destroyed by the convolution with the
        2d-Gaussian function.
        Default: True.
    refractory_period : pq.Quantity, optional
        Defines the refractory period of the dithered `spiketrain` unless
        the smallest ISI of the `spiketrain` is lower than this value.
        Default: 4. * pq.ms.

    Attributes
    ----------
    max_change_index : np.ndarray or int:
        For each ISI the corresponding index in the Joint-ISI distribution.
    max_change_isi : np.ndarray or float:
        The corresponding ISI for each index in :attr:`max_change_index`.

    """

    # The min number of spikes, required for dithering.
    # Otherwise, the original spiketrain is copied N times.
    MIN_SPIKES = 3

    @deprecated_alias(num_bins='n_bins', refr_period='refractory_period')
    def __init__(self,
                 spiketrain,
                 dither=15. * pq.ms,
                 truncation_limit=100. * pq.ms,
                 n_bins=100,
                 sigma=2. * pq.ms,
                 alternate=True,
                 use_sqrt=False,
                 method='fast',
                 cutoff=True,
                 refractory_period=4. * pq.ms
                 ):
        self.spiketrain = spiketrain
        self.truncation_limit = self.get_magnitude(truncation_limit)
        self.n_bins = n_bins

        self.dither = self.get_magnitude(dither)

        self.sigma = self.get_magnitude(sigma)
        self.alternate = alternate

        if method not in ['fast', 'window']:
            raise ValueError("The method can either be 'fast' or 'window', "
                             "but not '{}'".format(method))
        self.method = method

        refractory_period = self.get_magnitude(refractory_period)
        if not self.too_less_spikes:
            minimal_isi = np.min(self.isi)
            refractory_period = min(refractory_period, minimal_isi)
        self.refractory_period = refractory_period

        self.cutoff = cutoff
        self.use_sqrt = use_sqrt
        self._jisih_cumulatives = None

        self.max_change_index = self.isi_to_index(self.dither)
        self.max_change_isi = self.index_to_isi(self.max_change_index)

    @property
    def refr_period(self):
        warnings.warn("'.refr_period' is deprecated; use '.refractory_period'",
                      DeprecationWarning)
        return self.refractory_period

    @property
    def num_bins(self):
        warnings.warn("'.num_bins' is deprecated; use '.n_bins'",
                      DeprecationWarning)
        return self.n_bins

    def get_magnitude(self, quantity):
        """
        Parameters
        ----------
        quantity: pq.Quantity or float

        Returns
        -------
        float
            The magnitude of `quantity`, rescaled to the units of the input
            :attr:`spiketrain`.
        """
        if isinstance(quantity, pq.Quantity):
            return quantity.rescale(self.unit).magnitude
        return quantity

    @property
    def too_less_spikes(self):
        """
        This is a check if the :attr:`spiketrain` has enough spikes to evaluate
        the joint-ISI histogram.

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
        return self.truncation_limit / self.n_bins

    def isi_to_index(self, inter_spike_interval):
        """
        A function that gives for each ISI the corresponding index in the
        Joint-ISI distribution.

        Parameters
        ----------
        inter_spike_interval : np.ndarray or float
            An array of ISIs or a single ISI.

        Returns
        -------
        np.ndarray or int
            The corresponding index for each ISI.
        """
        return np.floor(inter_spike_interval / self.bin_width).astype(int)

    def index_to_isi(self, isi_index):
        """
        Maps `isi_index` back to the original ISI.

        Parameters
        ----------
        isi_index : np.ndarray or int
            The index of ISI.

        Returns
        -------
        np.ndarray or float:
            The corresponding ISI for each index.
        """
        return (isi_index + 0.5) * self.bin_width

    def joint_isi_histogram(self):
        """
        Calculates a 2D histogram of :math:`(ISI_i, ISI_{i+1})` and applies
        square root or gaussian filtering if necessary.

        Returns
        -------
        joint_isi_histogram : np.ndarray
        """
        if self.too_less_spikes:
            return None
        isis = self.isi
        joint_isi_histogram = np.histogram2d(
            isis[:-1], isis[1:],
            bins=[self.n_bins, self.n_bins],
            range=[[0., self.truncation_limit],
                   [0., self.truncation_limit]])[0]

        if self.use_sqrt:
            joint_isi_histogram = np.sqrt(joint_isi_histogram)

        if self.sigma:
            if self.cutoff:
                start_index = self.isi_to_index(self.refractory_period)
                joint_isi_histogram[
                    start_index:, start_index:] = gaussian_filter(
                        joint_isi_histogram[start_index:, start_index:],
                        sigma=self.sigma / self.bin_width)
                joint_isi_histogram[:start_index, :] = 0
                joint_isi_histogram[:, :start_index] = 0
            else:
                joint_isi_histogram = gaussian_filter(
                    joint_isi_histogram, sigma=self.sigma / self.bin_width)
        return joint_isi_histogram

    @staticmethod
    def normalize_cumulative_distribution(array):
        """
        This function normalizes parts of a cumulative distribution function
        to be a cumulative distribution function again.

        Parameters
        ----------
        array: np.ndarray
            A monotonously increasing array as a part of an unnormalized
             cumulative distribution function.

        Returns
        -------
        np.ndarray
            Monotonously increasing array from 0 to 1.
        """
        if array[-1] - array[0] > 0.:
            return (array - array[0]) / (array[-1] - array[0])
        return np.zeros_like(array)

    def dithering(self, n_surrogates=1):
        """
        Implementation of Joint-ISI-dithering for spike trains that pass the
        threshold of the dense rate. If not, a uniform dithered spike train is
        given back. The implementation continued the ideas of Louis et al.
        (2010) and Gerstein (2004).

        Parameters
        ----------
        n_surrogates: int
            The number of dithered spiketrains to be returned.
            Default: 1.

        Returns
        ----------
        dithered_sts: list of neo.SpikeTrain
            Spike trains, that are dithered versions of the given
            :attr:`spiketrain`
        """
        if self.too_less_spikes:
            return [self.spiketrain] * n_surrogates

        # Checks, whether the preprocessing is already done.
        if self._jisih_cumulatives is None:
            self._determine_cumulative_functions()

        dithered_sts = []
        isi_to_dither = self.isi
        for _ in range(n_surrogates):
            dithered_isi = self._get_dithered_isi(isi_to_dither)

            dithered_st = self.spiketrain[0].magnitude + \
                np.r_[0., np.cumsum(dithered_isi)]
            dithered_st = neo.SpikeTrain(dithered_st * self.unit,
                                         t_start=self.spiketrain.t_start,
                                         t_stop=self.spiketrain.t_stop)
            dithered_sts.append(dithered_st)
        return dithered_sts

    def _determine_cumulative_functions(self):
        rotated_jisih = np.rot90(self.joint_isi_histogram())

        if self.method == 'fast':
            self._jisih_cumulatives = []
            for double_index in range(self.n_bins):
                # Taking anti-diagonals of the original joint-ISI histogram
                diagonal = np.diagonal(
                    rotated_jisih, offset=-self.n_bins + double_index + 1)
                jisih_cum = self.normalize_cumulative_distribution(
                    np.cumsum(diagonal))
                self._jisih_cumulatives.append(jisih_cum)
            self._jisih_cumulatives = np.array(self._jisih_cumulatives)
        else:
            self._jisih_cumulatives = self._window_cumulatives(rotated_jisih)

    def _window_cumulatives(self, rotated_jisih):
        jisih_diag_cums = self._window_diagonal_cumulatives(rotated_jisih)
        jisih_cumulatives = np.zeros(
            (self.n_bins, self.n_bins,
             2 * self.max_change_index + 1))
        for curr_isi_id in range(self.n_bins):
            for next_isi_id in range(self.n_bins - curr_isi_id):
                double_index = next_isi_id + curr_isi_id
                cum_slice = jisih_diag_cums[
                    double_index,
                    curr_isi_id: curr_isi_id + 2 * self.max_change_index + 1]

                normalized_cum = self.normalize_cumulative_distribution(
                    cum_slice)
                jisih_cumulatives[curr_isi_id][next_isi_id] = normalized_cum
        return jisih_cumulatives

    def _window_diagonal_cumulatives(self, rotated_jisih):
        # An element of the first axis is defined as the sum of indices
        # for previous and subsequent ISI.

        jisih_diag_cums = np.zeros((self.n_bins,
                                    self.n_bins
                                    + 2 * self.max_change_index))

        # double_index corresponds to the sum of the indices for the previous
        # and the subsequent ISI.
        for double_index in range(self.n_bins):
            cum_diag = np.cumsum(np.diagonal(rotated_jisih,
                                             - self.n_bins
                                             + double_index + 1))

            right_padding = jisih_diag_cums.shape[1] - \
                len(cum_diag) - self.max_change_index

            jisih_diag_cums[double_index] = np.pad(
                cum_diag,
                pad_width=(self.max_change_index, right_padding),
                mode='constant',
                constant_values=(cum_diag[0], cum_diag[-1])
            )

        return jisih_diag_cums

    def _get_dithered_isi(self, isi_to_dither):
        dithered_isi = np.copy(isi_to_dither)
        # if alternate is true, a sampling_rhythm of 2 means that we have two
        # partitions of spikes, the "odd" and the "even" spikes and first
        # dither the "even" ones and then the "odd" ones.
        # if alternate is false, we just go dither from the first to the last
        # spike, which corresponds to a sampling_rhythm of 1.
        sampling_rhythm = self.alternate + 1
        number_of_isis = len(dithered_isi)

        for start in range(sampling_rhythm):
            dithered_isi_indices = self.isi_to_index(dithered_isi)
            for i in range(start, number_of_isis - 1,
                           sampling_rhythm):
                step = self._get_dithering_step(
                    dithered_isi,
                    dithered_isi_indices,
                    i)
                dithered_isi[i] += step
                dithered_isi[i + 1] -= step

        return dithered_isi

    def _get_dithering_step(self,
                            dithered_isi,
                            dithered_isi_indices,
                            i):
        curr_isi_id = dithered_isi_indices[i]
        next_isi_id = dithered_isi_indices[i + 1]
        double_index = curr_isi_id + next_isi_id
        if double_index < self.n_bins:
            if self.method == 'fast':
                cum_dist_func = self._jisih_cumulatives[
                    double_index]
                compare_isi = self.index_to_isi(curr_isi_id)
            else:
                cum_dist_func = self._jisih_cumulatives[
                    curr_isi_id][next_isi_id]
                compare_isi = self.max_change_isi

            if cum_dist_func[-1] > 0.:
                # when the method is 'fast', new_isi_id is where the current
                # ISI id should go to.
                new_isi_id = np.searchsorted(cum_dist_func, random.random())
                step = self.index_to_isi(new_isi_id) - compare_isi
                return step

        return self._uniform_dither_not_jisi_movable_spikes(
            dithered_isi[i],
            dithered_isi[i + 1])

    def _uniform_dither_not_jisi_movable_spikes(self,
                                                curr_isi,
                                                next_isi):
        left_dither = min(curr_isi - self.refractory_period, self.dither)
        right_dither = min(next_isi - self.refractory_period, self.dither)
        step = random.random() * (right_dither + left_dither) - left_dither
        return step


@deprecated_alias(n='n_surrogates', surr_method='method')
def surrogates(spiketrain, n_surrogates=1, method='dither_spike_train',
               dt=None, decimals=None, edges=True):
    """
    Generates surrogates of a `spiketrain` by a desired generation
    method.

    This routine is a wrapper for the other surrogate generators in the
    module.

    The surrogates retain the `spiketrain.t_start` and `spiketrain.t_stop` of
    the original `spiketrain`.


    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        The spike train from which to generate the surrogates
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1.
    method : str, optional
        The method to use to generate surrogate spike trains. Can be one of:
        * 'dither_spike_train': see `surrogates.dither_spike_train` [dt needed]
        * 'dither_spikes': see `surrogates.dither_spikes` [dt needed]
        * 'jitter_spikes': see `surrogates.jitter_spikes` [dt needed]
        * 'randomise_spikes': see `surrogates.randomise_spikes`
        * 'shuffle_isis': see `surrogates.shuffle_isis`
        * 'joint_isi_dithering': see `surrogates.joint_isi_dithering`
        Default: 'dither_spike_train'.
    dt : pq.Quantity, optional
        For methods shifting spike times randomly around their original time
        (`dither_spikes`, `dither_spike_train`) or replacing them randomly
        within a certain window (`jitter_spikes`), dt represents the size of
        that shift / window. For other methods, dt is ignored.
        Default: None.
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates
        If None, machine precision is used.
        Default: None.
    edges : bool
        For surrogate spikes falling outside the range `[spiketrain.t_start,
        spiketrain.t_stop)`, whether to drop them out (for `edges = True`) or
        set them to the range's closest end (for `edges = False`).
        Default: True.

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain`
        according to chosen surrogate type. The time range of the surrogate
        spike trains is the same as in `spiketrain`.
    """

    if not isinstance(spiketrain, neo.SpikeTrain):
        raise ValueError("spiketrain must be of instance neo.SpikeTrain")

    # Define the surrogate function to use, depending on the specified method
    surrogate_types = {
        'dither_spike_train': dither_spike_train,
        'dither_spikes': dither_spikes,
        'jitter_spikes': jitter_spikes,
        'randomise_spikes': randomise_spikes,
        'shuffle_isis': shuffle_isis,
        'joint_isi_dithering': JointISI(spiketrain).dithering,
    }

    if method not in surrogate_types.keys():
        raise ValueError("Specified surrogate method ('{}') "
                         "is not valid".format(method))
    method = surrogate_types[method]

    # PYTHON2: replace with inspect.signature()
    if dt is None and method in (dither_spike_train, dither_spikes,
                                 jitter_spikes):
        raise ValueError("{}() method requires 'dt' parameter to be "
                         "not None".format(method.__name__))

    if method in (dither_spike_train, dither_spikes):
        return method(spiketrain, dt, n=n_surrogates, decimals=decimals,
                      edges=edges)
    if method in (randomise_spikes, shuffle_isis):
        return method(spiketrain, n=n_surrogates, decimals=decimals)
    if method == jitter_spikes:
        return method(spiketrain, dt, n=n_surrogates)
    # method == 'joint_isi_dithering':
    return method(n_surrogates)
