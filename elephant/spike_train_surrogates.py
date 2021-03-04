# -*- coding: utf-8 -*-
"""
Module to generate surrogates of a spike train by randomising its spike times
in different ways (see :cite:`surrogates-Gerstein2004_203`,
:cite:`surrogates-Louis2010_127`, and :cite:`surrogates-Louis2010_359`).
Different methods destroy different features of the original data.


Main function
-------------
.. autosummary::
    :toctree: _toctree/spike_train_surrogates

    surrogates


Surrogate types
---------------
.. autosummary::
    :toctree: _toctree/spike_train_surrogates

    JointISI
    dither_spikes
    randomise_spikes
    shuffle_isis
    dither_spike_train
    jitter_spikes
    bin_shuffling
    trial_shifting

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import random
import warnings
import copy

import neo
import numpy as np
import quantities as pq
from scipy.ndimage import gaussian_filter

from elephant.statistics import isi
import elephant.conversion as conv
from elephant.utils import deprecated_alias

__all__ = [
    "dither_spikes",
    "randomise_spikes",
    "shuffle_isis",
    "dither_spike_train",
    "jitter_spikes",
    "bin_shuffling",
    "JointISI",
    "trial_shifting",
    "surrogates"
]

# List of all available surrogate methods
SURR_METHODS = ('dither_spike_train', 'dither_spikes', 'jitter_spikes',
                'randomise_spikes', 'shuffle_isis', 'joint_isi_dithering',
                'dither_spikes_with_refractory_period', 'trial_shifting',
                'bin_shuffling', 'isi_dithering')


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
    spiketrain : neo.SpikeTrain
        The spike train from which to generate the surrogates.
    dither : pq.Quantity
        Amount of dithering. A spike at time `t` is placed randomly within
        `(t-dither, t+dither)`.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates at a
        millisecond level.
        If None, machine precision is used.
        Default: None
    edges : bool, optional
        For surrogate spikes falling outside the range
        `[spiketrain.t_start, spiketrain.t_stop)`, whether to drop them out
        (for `edges = True`) or set them to the range's closest end
        (for `edges = False`).
        Default: True
    refractory_period : pq.Quantity or None, optional
        The dither range of each spike is adjusted such that the spike can not
        fall into the `refractory_period` of the previous or next spike.
        To account this, the refractory period is estimated as the smallest ISI
        of the spike train. The given argument `refractory_period` here is thus
        an initial estimation.
        Note, that with this option a spike cannot "jump" over the previous or
        next spike as it is normally possible.
        If set to None, no refractoriness is in dithering.
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
    return [neo.SpikeTrain(train, t_start=t_start, t_stop=t_stop,
                           sampling_rate=spiketrain.sampling_rate)
            for train in dithered_spiketrains]


@deprecated_alias(n='n_surrogates')
def randomise_spikes(spiketrain, n_surrogates=1, decimals=None):
    """
    Generates surrogates of a spike train by spike time randomization.

    The surrogates are obtained by keeping the spike count of the original
    `spiketrain`, but placing the spikes randomly in the interval
    `[spiketrain.t_start, spiketrain.t_stop]`. The generated independent
    `neo.SpikeTrain` objects follow  Poisson statistics (exponentially
    distributed inter-spike intervals).

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        The spike train from which to generate the surrogates.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates.
        If None, machine precision is used.
        Default: None

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
                           t_stop=spiketrain.t_stop,
                           sampling_rate=spiketrain.sampling_rate)
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
    spiketrain : neo.SpikeTrain
        The spike train from which to generate the surrogates.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates.
        If None, machine precision is used.
        Default: None

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
    if len(spiketrain) == 0:
        return [neo.SpikeTrain([] * spiketrain.units,
                               t_start=spiketrain.t_start,
                               t_stop=spiketrain.t_stop,
                               sampling_rate=spiketrain.sampling_rate)
                for _ in range(n_surrogates)]

    # A correct sorting is necessary, to calculate the ISIs
    spiketrain = spiketrain.copy()
    spiketrain.sort()
    isi0 = spiketrain[0] - spiketrain.t_start
    isis = np.hstack([isi0, isi(spiketrain)])

    # Round the isis to decimal position, if requested
    if decimals is not None:
        isis = isis.round(decimals)

    # Create list of surrogate spike trains by random ISI permutation
    sts = []
    for surrogate_id in range(n_surrogates):
        surr_times = np.cumsum(np.random.permutation(isis)) * \
            spiketrain.units + spiketrain.t_start
        sts.append(neo.SpikeTrain(
            surr_times, t_start=spiketrain.t_start,
            t_stop=spiketrain.t_stop,
            sampling_rate=spiketrain.sampling_rate))
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
    spiketrain : neo.SpikeTrain
        The spike train from which to generate the surrogates.
    shift : pq.Quantity
        Amount of shift. `spiketrain` is shifted by a random amount uniformly
        drawn from the range `(-shift, +shift)`.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1
    decimals : int or None, optional
        Number of decimal points for every spike time in the surrogates.
        If None, machine precision is used.
        Default: None
    edges : bool, optional
        For surrogate spikes falling outside the range `[spiketrain.t_start,
        spiketrain.t_stop)`, whether to drop them out (for `edges = True`) or
        set them to the range's closest end (for `edges = False`).
        Default: True

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
                           t_stop=spiketrain.t_stop,
                           sampling_rate=spiketrain.sampling_rate
                           ).rescale(spiketrain.units)
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
    spiketrain : neo.SpikeTrain
        The spike train from which to generate the surrogates.
    bin_size : pq.Quantity
        Size of the time bins within which to randomize the spike times.
        Note: the last bin lasts until `spiketrain.t_stop` and might have
        width different from `bin_size`.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1

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
                           t_stop=spiketrain.t_stop,
                           sampling_rate=spiketrain.sampling_rate
                           ).rescale(spiketrain.units)
            for s in surr]


def bin_shuffling(
        spiketrain, max_displacement, bin_size=None, n_surrogates=1,
        sliding=False):
    """
    Bin shuffling surrogate generation.

    The function shuffles the entries of a binned spike train entries inside
    windows with a fixed maximal displacement. The windows are either exclusive
    or sliding.

    Parameters
    ----------
    spiketrain : conv.BinnedSpikeTrain or neo.SpikeTrain
        The binned spike train or a continuous time spike train
        to create surrogates of.
    max_displacement : int
        Number of bins that a single spike can be displaced.
    bin_size : pq.Quantity or None
        the bin size needs to be specified only if a not-binned spike train
        is passed to the method
    n_surrogates : int, optional
        Number of surrogates to create.
        Default: 1
    sliding : bool, optional
        If True, the window is slided bin by bin
        (only implemented for binned spike trains).
        Default: False

    Returns
    -------
    binned_surrogates : list of conv.BinnedSpikeTrain or list of neo.SpikeTrain
        Each entry of the list is a surrogate spike train either binned or in
        continuous time.
    """
    if isinstance(spiketrain, neo.SpikeTrain):
        if bin_size is None:
            raise ValueError(
                'If you want to create surrogates from neo.SpikeTrain objects,'
                'you need to specify the bin_size')
        if sliding:
            warnings.warn(
                'The sliding option is not implemented yet for bin shuffling'
                ' on continuos time spike trains. Results are given for'
                ' sliding=False.', UserWarning)
        return _continuous_time_bin_shuffling(
            spiketrain, max_displacement=max_displacement, bin_size=bin_size,
            n_surrogates=n_surrogates)

    displacement_window = 2 * max_displacement

    binned_spiketrain_bool = spiketrain.to_bool_array()[0]
    st_length = len(binned_spiketrain_bool)

    surrogate_spiketrains = []
    for surrogate_id in range(n_surrogates):
        surrogate_spiketrain = np.copy(binned_spiketrain_bool)
        if sliding:
            for window_position in range(st_length - displacement_window):
                # shuffling the binned spike train within the window
                np.random.shuffle(
                    surrogate_spiketrain[
                        window_position:
                        window_position + displacement_window])
        else:
            windows = st_length // displacement_window
            windows_remainder = st_length % displacement_window
            for window_position in range(windows):
                # shuffling the binned spike train within the window
                np.random.shuffle(
                    surrogate_spiketrain[
                        window_position * displacement_window:
                        (window_position + 1) * displacement_window])
            if windows_remainder != 0:
                np.random.shuffle(
                    surrogate_spiketrain[windows * displacement_window:])
        surrogate_spiketrain = surrogate_spiketrain.reshape((1, st_length))

        surrogate_spiketrains.append(
            conv.BinnedSpikeTrain(
                surrogate_spiketrain,
                bin_size=spiketrain.bin_size,
                t_start=spiketrain.t_start,
                t_stop=spiketrain.t_stop))
    return surrogate_spiketrains


def _continuous_time_bin_shuffling(spiketrain, max_displacement, bin_size,
                                   n_surrogates=1):
    """

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
    max_displacement : int
        number of bins that a single spike can be displaced
    bin_size : pq.Quantity
    n_surrogates : int, optional
        Default : 1

    Returns
    -------
    list of neo.SpikeTrain
    """
    units = spiketrain.units
    bin_size = bin_size.rescale(units).item()
    t_start = spiketrain.t_start.item()
    t_stop = spiketrain.t_stop.item()
    spiketrain_shifted = spiketrain.magnitude - t_start
    displacement_window = 2 * max_displacement

    binned_duration = int((t_stop - t_start) // bin_size)

    bin_indices = (spiketrain_shifted // bin_size).astype(int)

    split_indices = np.searchsorted(
        bin_indices,
        np.arange(displacement_window, binned_duration, displacement_window))

    bin_indices = np.split(
        bin_indices,
        split_indices)

    surrogate_spiketrains = []
    for surrogate_id in range(n_surrogates):
        surrogate_bin_indices = np.empty(shape=len(bin_indices),
                                         dtype=np.ndarray)
        for i, bin_indices_slice in enumerate(bin_indices):
            window_start = i*displacement_window

            random_indices = np.random.permutation(displacement_window)
            surrogate_bin_indices[i] = \
                random_indices[bin_indices_slice - window_start] \
                + window_start

        surrogate_bin_indices = np.concatenate(surrogate_bin_indices)

        bin_remainders = bin_size * np.random.random(len(spiketrain))

        surrogate_spiketrain = \
            surrogate_bin_indices * bin_size + bin_remainders + t_start

        # ensure last and first spike being inside the boundaries
        surrogate_spiketrain = surrogate_spiketrain[
            np.all((surrogate_spiketrain > t_start,
                    surrogate_spiketrain < t_stop),
                   axis=0)]

        surrogate_spiketrain.sort()

        surrogate_spiketrain = neo.SpikeTrain(
            surrogate_spiketrain,
            units=units,
            t_start=t_start,
            t_stop=t_stop,
            copy=False,
        )

        surrogate_spiketrains.append(surrogate_spiketrain)
    return surrogate_spiketrains


class JointISI(object):
    r"""
    Joint-ISI dithering implementation, based on the ideas from
    :cite:`surrogates-Gerstein2004_203` and :cite:`surrogates-Louis2010_127`.

    The main function is :func:`JointISI.dithering`.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Input spiketrain to create surrogates of.
    dither : pq.Quantity, optional
        This quantity describes the maximum displacement of a spike, when
        `method` is 'window'. It is also used for the uniform dithering for
        the spikes, which are outside the regime in the Joint-ISI
        histogram, where Joint-ISI dithering is applicable.
        Default: 15. * pq.ms
    truncation_limit : pq.Quantity, optional
        The Joint-ISI distribution of :math:`(ISI_i, ISI_{i+1})` is defined
        within the range :math:`[0, \infty)`. Since this is computationally not
        feasible, the Joint-ISI distribution is truncated for high ISI.
        The Joint-ISI histogram is calculated for
        :math:`(ISI_i, ISI_{i+1})` from 0 to `truncation_limit`.
        Default: 100. * pq.ms
    n_bins : int, optional
        The size of the joint-ISI-distribution will be
        `n_bins*n_bins/2`.
        Default: 100
    sigma : pq.Quantity, optional
        The standard deviation of the Gaussian kernel, with which
        the data is convolved.
        Default: 2. * pq.ms
    alternate : bool, optional
        If True, then all even spikes are dithered followed
        by all odd spikes. Otherwise, the spikes are dithered in ascending
        order from the first to the last spike.
        Default: True
    use_sqrt : bool, optional
        If True, the joint-ISI histogram is preprocessed by
        applying a square root (following :cite:`surrogates-Gerstein2004_203`).
        Default: False
    method : {'fast', 'window'}, optional
        * 'fast': the spike can move in the whole range between the
            previous and subsequent spikes (computationally efficient).
        * 'window': the spike movement is limited to the parameter `dither`.

        Default: 'window'
    cutoff : bool, optional
        If True, then the filtering of the Joint-ISI histogram is
        limited on the lower side by the minimal ISI.
        This can be necessary, if in the data there is a certain refractory
        period, which will be destroyed by the convolution with the
        2d-Gaussian function.
        Default: True
    refractory_period : pq.Quantity, optional
        Defines the refractory period of the dithered `spiketrain` unless
        the smallest ISI of the `spiketrain` is lower than this value.
        Default: 4. * pq.ms
    isi_dithering : bool, optional
        If True, the Joint-ISI distribution is evaluated as the outer product
        of the ISI-distribution with itself. Thus, all serial correlations are
        destroyed.
        Default: False
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
                 method='window',
                 cutoff=True,
                 refractory_period=4. * pq.ms,
                 isi_dithering=False):

        if not isinstance(spiketrain, neo.SpikeTrain):
            raise TypeError('spiketrain must be of type neo.SpikeTrain')

        # A correct sorting is necessary to calculate the ISIs
        spiketrain = spiketrain.copy()
        spiketrain.sort()
        self.spiketrain = spiketrain
        self.truncation_limit = self._get_magnitude(truncation_limit)
        self.n_bins = n_bins

        self.dither = self._get_magnitude(dither)

        self.sigma = self._get_magnitude(sigma)
        self.alternate = alternate

        if method not in ('fast', 'window'):
            raise ValueError("The method can either be 'fast' or 'window', "
                             "but not '{}'".format(method))
        self.method = method

        refractory_period = self._get_magnitude(refractory_period)
        if not self._too_less_spikes:
            minimal_isi = np.min(self.isi)
            refractory_period = min(refractory_period, minimal_isi)
        self.refractory_period = refractory_period

        self.cutoff = cutoff
        self.use_sqrt = use_sqrt
        self._jisih_cumulatives = None

        self._max_change_index = self._isi_to_index(self.dither)
        self._max_change_isi = self._index_to_isi(self._max_change_index)

        self.isi_dithering = isi_dithering

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

    def _get_magnitude(self, quantity):
        """
        Parameters
        ----------
        quantity : pq.Quantity or float

        Returns
        -------
        magnitude : float
            The magnitude of `quantity`, rescaled to the units of the input
            :attr:`spiketrain`.
        """
        if isinstance(quantity, pq.Quantity):
            return quantity.rescale(self._unit).magnitude
        return quantity

    @property
    def _too_less_spikes(self):
        """
        This is a check if the :attr:`spiketrain` has enough spikes to evaluate
        the joint-ISI histogram.

        Returns
        -------
        bool
            If True, the spike train is so sparse, that this algorithm can't be
            applied properly. Than in dithering() copies of the spiketrains are
            returned.
        """
        return len(self.spiketrain) < self.MIN_SPIKES

    @property
    def _unit(self):
        """
        The unit of the spiketrain. Thus, the unit of the output surrogates.
        """
        return self.spiketrain.units

    @property
    def isi(self):
        """
        The inter-spike intervals of the spiketrain.

        Returns
        -------
        np.ndarray or None
            An array of inter-spike intervals of the `spiketrain`.
            None, if not enough spikes in the `spiketrain`.
        """
        if self._too_less_spikes:
            return None
        return isi(self.spiketrain.magnitude)

    @property
    def bin_width(self):
        return self.truncation_limit / self.n_bins

    def _isi_to_index(self, inter_spike_interval):
        """
        A function that gives for each ISI the corresponding index in the
        Joint-ISI distribution.

        Parameters
        ----------
        inter_spike_interval : np.ndarray or float
            An array of ISIs or a single ISI.

        Returns
        -------
        indices : np.ndarray or int
            The corresponding indices/index for each ISI. When the input is an
            array, also the output is.
        """
        return np.floor(inter_spike_interval / self.bin_width).astype(int)

    def _index_to_isi(self, isi_index):
        """
        Maps `isi_index` back to the original ISI.

        Parameters
        ----------
        isi_index : np.ndarray or int
            The index of ISI.

        Returns
        -------
        np.ndarray or float
            The corresponding ISI(s) for each indices/index. When the input is
            an array, also the output is.
        """
        return (isi_index + 0.5) * self.bin_width

    def joint_isi_histogram(self):
        """
        Calculates a 2D histogram of :math:`(ISI_i, ISI_{i+1})` and applies
        square root or gaussian filtering if necessary.

        Returns
        -------
        joint_isi_histogram : np.ndarray or None
            A np.ndarray with shape `n_bins` x `n_bins` containing the
            joint-ISI histogram.
            None, if not enough spikes in the `spiketrain`.
        """
        if self._too_less_spikes:
            return None
        isis = self.isi
        if not self.isi_dithering:
            joint_isi_histogram = np.histogram2d(
                isis[:-1], isis[1:],
                bins=[self.n_bins, self.n_bins],
                range=[[0., self.truncation_limit],
                       [0., self.truncation_limit]])[0]
        else:
            isi_histogram = np.histogram(
                isis,
                bins=self.n_bins,
                range=[0., self.truncation_limit])[0]
            joint_isi_histogram = np.outer(isi_histogram, isi_histogram)

        if self.use_sqrt:
            joint_isi_histogram = np.sqrt(joint_isi_histogram)

        if self.sigma:
            if self.cutoff:
                start_index = self._isi_to_index(self.refractory_period)
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
    def _normalize_cumulative_distribution(array):
        """
        This function normalizes the cut-off of a cumulative distribution
        function so that the first element is again zero and the last element
        is one.

        Parameters
        ----------
        array : np.ndarray
            A monotonously increasing array as a part of an unnormalized
            cumulative distribution function.

        Returns
        -------
        np.ndarray
            Monotonously increasing array from 0 to 1.
            If `array` does not contain all equal elements, a only-zeros array
            is returned.
        """
        if array[-1] - array[0] > 0.:
            return (array - array[0]) / (array[-1] - array[0])
        return np.zeros_like(array)

    def dithering(self, n_surrogates=1):
        """
        Implementation of Joint-ISI-dithering for spike trains that pass the
        threshold of the dense rate. If not, a uniform dithered spike train is
        given back.

        Parameters
        ----------
        n_surrogates : int
            The number of dithered spiketrains to be returned.
            Default: 1

        Returns
        ----------
        dithered_sts : list of neo.SpikeTrain
            Spike trains, that are dithered versions of the given
            :attr:`spiketrain`.
        """
        if self._too_less_spikes:
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
            sampling_rate = self.spiketrain.sampling_rate

            # Due to rounding errors, the last spike may be above t_stop.
            # If the case, this is set to t_stop.
            if dithered_st[-1] > self.spiketrain.t_stop:
                dithered_st[-1] = self.spiketrain.t_stop

            dithered_st = neo.SpikeTrain(dithered_st * self._unit,
                                         t_start=self.spiketrain.t_start,
                                         t_stop=self.spiketrain.t_stop,
                                         sampling_rate=sampling_rate)
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
                jisih_cum = self._normalize_cumulative_distribution(
                    np.r_[0., np.cumsum(diagonal)])
                self._jisih_cumulatives.append(jisih_cum)
            self._jisih_cumulatives = np.array(
                self._jisih_cumulatives, dtype=object)
        else:
            self._jisih_cumulatives = self._window_cumulatives(rotated_jisih)

    def _window_cumulatives(self, rotated_jisih):
        jisih_diag_cums = self._window_diagonal_cumulatives(rotated_jisih)
        jisih_cumulatives = np.zeros(
            (self.n_bins, self.n_bins,
             2 * self._max_change_index + 1))
        for curr_isi_id in range(self.n_bins):
            for next_isi_id in range(self.n_bins - curr_isi_id):
                double_index = next_isi_id + curr_isi_id
                cum_slice = jisih_diag_cums[
                    double_index,
                    curr_isi_id: curr_isi_id + 2 * self._max_change_index + 1]

                normalized_cum = self._normalize_cumulative_distribution(
                    cum_slice)
                jisih_cumulatives[curr_isi_id][next_isi_id] = normalized_cum
        return jisih_cumulatives

    def _window_diagonal_cumulatives(self, rotated_jisih):
        # An element of the first axis is defined as the sum of indices
        # for previous and subsequent ISI.

        jisih_diag_cums = np.zeros((self.n_bins,
                                    self.n_bins
                                    + 2 * self._max_change_index))

        # double_index corresponds to the sum of the indices for the previous
        # and the subsequent ISI.
        for double_index in range(self.n_bins):
            anti_diagonal = np.diagonal(
                rotated_jisih, - self.n_bins + double_index + 1)

            right_padding = jisih_diag_cums.shape[1] - \
                len(anti_diagonal) - self._max_change_index

            cumulated_diagonal = np.cumsum(anti_diagonal)

            padded_cumulated_diagonal = np.pad(
                cumulated_diagonal,
                pad_width=(self._max_change_index, right_padding),
                mode='constant',
                constant_values=(0., cumulated_diagonal[-1]))

            jisih_diag_cums[double_index] = padded_cumulated_diagonal

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
            dithered_isi_indices = self._isi_to_index(dithered_isi)
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
                compare_isi = self._index_to_isi(curr_isi_id + 1)
            else:
                cum_dist_func = self._jisih_cumulatives[
                    curr_isi_id][next_isi_id]
                compare_isi = self._max_change_isi

            if cum_dist_func[-1] > 0.:
                # when the method is 'fast', new_isi_id is where the current
                # ISI id should go to.
                new_isi_id = np.searchsorted(cum_dist_func, random.random())
                step = self._index_to_isi(new_isi_id)\
                    - compare_isi
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


def trial_shifting(spiketrains, dither, n_surrogates=1):
    """
    Generates surrogates of a spike train by trial shifting.

    It shifts by a random uniform amount independently different trials,
    which are the elements of a list of spiketrains.

    The shifting is done independently for each surrogate.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        A list of spike trains of the same neuron
        where each element corresponds to one trial.
    dither : pq.Quantity
        Amount of dithering.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1

    Returns
    -------
    surrogate_spiketrains : list of list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        randomly dithering its spikes. The range of the surrogate spike trains
        is the same as of `spiketrain`.
    """
    dither = dither.simplified.magnitude

    units = spiketrains[0].units
    t_starts = [single_trial_st.t_start.simplified.magnitude
                for single_trial_st in spiketrains]
    t_stops = [single_trial_st.t_stop.simplified.magnitude
               for single_trial_st in spiketrains]
    sampling_rates = [single_trial_st.sampling_rate
                      for single_trial_st in spiketrains]
    spiketrains = [single_trial_st.simplified.magnitude
                   for single_trial_st in spiketrains]

    surrogate_spiketrains = \
        _trial_shifting(spiketrains, dither, t_starts, t_stops,
                        n_surrogates)

    surrogate_spiketrains = \
        [[neo.SpikeTrain(
            surrogate_spiketrain[trial_id] * pq.s,
            t_start=t_starts[trial_id] * pq.s,
            t_stop=t_stops[trial_id] * pq.s,
            units=units,
            sampling_rate=sampling_rates[trial_id])
          for trial_id in range(len(surrogate_spiketrain))]
         for surrogate_spiketrain in surrogate_spiketrains]

    return surrogate_spiketrains


def _trial_shifting(spiketrains, dither, t_starts, t_stops, n_surrogates):
    """
    Inner nucleus of the trial shuffling procedure.
    """
    surrogate_spiketrains = []
    for surrogate_id in range(n_surrogates):
        copied_spiketrain = copy.deepcopy(spiketrains)
        surrogate_spiketrain = []
        # looping over all trials
        for trial_id, single_trial_st in enumerate(copied_spiketrain):
            single_trial_st += dither * (2 * random.random() - 1)
            single_trial_st = np.remainder(
                single_trial_st - t_starts[trial_id],
                t_stops[trial_id] - t_starts[trial_id]
            ) + t_starts[trial_id]
            single_trial_st.sort()

            surrogate_spiketrain.append(single_trial_st)

        surrogate_spiketrains.append(surrogate_spiketrain)
    return surrogate_spiketrains


def _trial_shifting_of_concatenated_spiketrain(
        spiketrain, dither, trial_length, trial_separation, n_surrogates=1):
    """
    Generates surrogates of a spike train by trial shifting.

    It shifts by a random uniform amount independently different trials,
    individuated by the `trial_length` and the possible buffering period
    `trial_separation` present in between trials.

    The shifting is done independently for each surrogate.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        A single spike train, where the trials are concatenated.
    dither : pq.Quantity
        Amount of dithering.
    trial_length : pq.Quantity
        The length of the single-trial spiketrain.
    trial_separation : pq.Quantity
        Buffering in between trials in the concatenation of the spiketrain.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1

    Returns
    -------
    surrogate_spiketrains : list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain` by
        randomly dithering its spikes. The range of the surrogate spike trains
        is the same as of `spiketrain`.
    """
    units = spiketrain.units
    t_start = spiketrain.t_start.simplified.magnitude
    t_stop = spiketrain.t_stop.simplified.magnitude
    trial_length = trial_length.simplified.magnitude
    trial_separation = trial_separation.simplified.magnitude
    dither = dither.simplified.magnitude
    n_trials = int((t_stop - t_start) // (trial_length + trial_separation))
    t_starts = t_start + \
        np.arange(n_trials) * (trial_length + trial_separation)
    t_stops = t_starts + trial_length
    spiketrains = spiketrain.simplified.magnitude
    spiketrains = [spiketrains[(spiketrains >= t_starts[trial_id]) &
                               (spiketrains <= t_stops[trial_id])]
                   for trial_id in range(n_trials)]

    surrogate_spiketrains = _trial_shifting(
        spiketrains, dither, t_starts, t_stops, n_surrogates)

    surrogate_spiketrains = [neo.SpikeTrain(
        np.hstack(surrogate_spiketrain) * pq.s,
        t_start=t_start * pq.s,
        t_stop=t_stop * pq.s,
        units=units,
        sampling_rate=spiketrain.sampling_rate)
        for surrogate_spiketrain in surrogate_spiketrains]
    return surrogate_spiketrains


@deprecated_alias(n='n_surrogates', surr_method='method')
def surrogates(spiketrain, n_surrogates=1, method='dither_spike_train',
               dt=None, **kwargs):
    """
    Generates surrogates of a `spiketrain` by a desired generation
    method.

    This routine is a wrapper for the other surrogate generators in the
    module.

    The surrogates retain the `spiketrain.t_start` and `spiketrain.t_stop` of
    the original `spiketrain`.


    Parameters
    ----------
    spiketrain : neo.SpikeTrain or list of neo.SpikeTrain
        The spike train from which to generate the surrogates.
        The only method that accepts a list of spike trains instead of a single
        spike train to generate the surrogates from is 'trial_shifting'.
    n_surrogates : int, optional
        Number of surrogates to be generated.
        Default: 1
    method : str, optional
        The method to use to generate surrogate spike trains. Can be one of:

        * 'dither_spike_train': see `surrogates.dither_spike_train()`
            [`dt` needed]
        * 'dither_spikes': see `surrogates.dither_spikes()` [`dt` needed]
        * 'jitter_spikes': see `surrogates.jitter_spikes()` [`dt` needed]
        * 'randomise_spikes': see `surrogates.randomise_spikes()`
        * 'shuffle_isis': see `surrogates.shuffle_isis()`
        * 'joint_isi_dithering': see `surrogates.joint_isi_dithering()`
            [`dt` needed]
        * 'trial_shifting': see `surrogates.trial_shifting` [`dt` needed]
            If used on a neo.SpikeTrain, specify the key-word argument
            `trial_length` and `trial_separation` of type pq.Quantity.
            Else, `spiketrain` has to be a list of neo.SpikeTrain.
        * 'bin_shuffling': see `surrogates.bin_shuffling()` [`dt` needed]
            If used in this module, specify the key-word argument `bin_size`
            of type pq.Quantity.

        Default: 'dither_spike_train'
    dt : pq.Quantity, optional
        For methods shifting spike times or spike trains randomly around
        their original time
        (`dither_spikes`, `dither_spike_train`) or replacing them randomly
        within a certain window (`jitter_spikes`), dt represents the size of
        that shift / window. For other methods, dt is ignored.
        Default: None
    kwargs
        Keyword arguments passed to the chosen surrogate method.

    Returns
    -------
    list of neo.SpikeTrain
        Each surrogate spike train obtained independently from `spiketrain`
        according to chosen surrogate type. The time range of the surrogate
        spike trains is the same as in `spiketrain`.

    Raises
    ------
    TypeError
        If `spiketrain` is not either a `neo.SpikeTrain` object or a list of
        `neo.SpikeTrain`.

    ValueError
        If `method` is not one of the surrogate methods defined in this module.

        If `dt` is None and `method` is not 'randomise_spikes' nor
        'shuffle_isis'.
    """

    if isinstance(spiketrain, list):
        if not isinstance(spiketrain[0], neo.SpikeTrain):
            raise TypeError('spiketrain must be an instance neo.SpikeTrain or'
                            ' a list of neo.SpikeTrain')
    elif not isinstance(spiketrain, neo.SpikeTrain):
        raise TypeError('spiketrain must be an instance neo.SpikeTrain or'
                        ' a list of neo.SpikeTrain')

    if method == "dither_spikes_with_refractory_period":
        warnings.warn("'dither_spikes_with_refractory_period' is deprecated "
                      "in favor of 'dither_spikes'", DeprecationWarning)

    # Define the surrogate function to use, depending on the specified method
    surrogate_types = {
        'dither_spike_train': dither_spike_train,
        'dither_spikes': dither_spikes,
        'dither_spikes_with_refractory_period': dither_spikes,
        'jitter_spikes': jitter_spikes,
        'randomise_spikes': randomise_spikes,
        'shuffle_isis': shuffle_isis,
        'bin_shuffling': bin_shuffling,
        'trial_shifting': trial_shifting,
        'joint_isi_dithering': lambda n: JointISI(
            spiketrain, **kwargs).dithering(n),
        'isi_dithering': lambda n: JointISI(
            spiketrain, isi_dithering=True, **kwargs).dithering(n)
    }

    if method not in surrogate_types.keys():
        raise ValueError("Specified surrogate method ('{}') "
                         "is not valid".format(method))
    method = surrogate_types[method]

    if dt is None and method not in (randomise_spikes, shuffle_isis):
        raise ValueError(f"'{method.__name__}' method requires 'dt' parameter "
                         f"to be set")

    if method in (dither_spike_train, dither_spikes):
        return method(
            spiketrain, dt, n_surrogates=n_surrogates, **kwargs)
    if method in (randomise_spikes, shuffle_isis):
        return method(spiketrain, n_surrogates=n_surrogates, **kwargs)
    if method is jitter_spikes:
        return method(spiketrain, dt, n_surrogates=n_surrogates)
    if method is trial_shifting:
        if isinstance(spiketrain, list):
            return method(
                spiketrain, dither=dt, n_surrogates=n_surrogates)
        return _trial_shifting_of_concatenated_spiketrain(
            spiketrain, dither=dt, n_surrogates=n_surrogates, **kwargs)
    if method is bin_shuffling:
        max_displacement = int(
            dt.simplified.magnitude / kwargs['bin_size'].simplified.magnitude)
        return method(
            spiketrain, max_displacement=max_displacement,
            bin_size=kwargs['bin_size'], n_surrogates=n_surrogates)
    # surr_method is 'joint_isi_dithering' or isi_dithering:
    return method(n_surrogates)
