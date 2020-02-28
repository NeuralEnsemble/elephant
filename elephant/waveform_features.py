# -*- coding: utf-8 -*-
"""
Features of waveforms (e.g waveform_snr).

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings

import numpy as np


def waveform_width(waveform, cutoff=0.75):
    """
    Calculate the width (trough-to-peak TTP) of a waveform.

    Searches for an index of a minimum within first `cutoff` of the waveform
    vector, next for a maximum after the identified minimum, and returns the
    difference between them.

    Parameters
    ----------
    waveform : np.ndarray or list or pq.Quantity
        Time course of a single waveform
    cutoff : float, optional
        Defines the normalized range `[0, cutoff]` of the input sequence for
        computing the minimum. Must be in `[0, 1)` range.
        Default: 0.75.

    Returns
    -------
    width : int
        Width of a waveform expressed as a number of data points

    Raises
    ------
    ValueError
        If `waveform` is not a one-dimensional vector with at least two
        numbers.

        If `cutoff` is not in `[0, 1)` range.

    """
    waveform = np.squeeze(waveform)
    if np.ndim(waveform) != 1:
        raise ValueError('Expected 1-dimensional waveform.')
    if len(waveform) < 2:
        raise ValueError('Too short waveform.')
    if not (0 <= cutoff < 1):
        raise ValueError('Cuttoff must be in range [0, 1).')

    min_border = max(1, int(len(waveform) * cutoff))
    idx_min = np.argmin(waveform[:min_border])
    idx_max = np.argmax(waveform[idx_min:]) + idx_min
    width = idx_max - idx_min

    return width


def waveform_snr(spiketrain):
    """
    Return the signal-to-noise ratio of the waveforms of a SpikeTrain.

    Signal-to-noise ratio is defined as the difference in mean peak-to-trough
    voltage divided by twice the mean SD. The mean SD is computed by
    measuring the SD of the spike waveform over all acquired spikes
    at each of the sample time points of the waveform and then averaging [1]_.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        The spike times with attached waveforms.

    Returns
    -------
    snr : float
        signal-to-noise ratio according to [1]_

    Raises
    ------
    ValueError
        If `spiketrain` has no attached waveforms.

    References
    ----------

    .. [1] Hatsopoulos, N. G., Xu, Q. & Amit, Y.
           Encoding of Movement Fragments in the Motor Cortex.
           J. Neurosci. 27, 5105–5114 (2007).

    """
    # check whether spiketrain contains waveforms
    if spiketrain.waveforms is None:
        raise ValueError('There are no waveforms attached to this \
                         neo.Spiketrain. Did you forget to set \
                         load_waveforms=True in neoIO.read_block()?')

    # average over all waveforms for each bin
    mean_waveform = np.mean(spiketrain.waveforms.magnitude, axis=0)[0]
    # standard deviation over all waveforms for each bin
    std_waveform = np.std(spiketrain.waveforms.magnitude, axis=0)[0]
    mean_std = np.mean(std_waveform)

    # signal
    peak_to_trough_voltage = np.max(mean_waveform) - np.min(mean_waveform)
    # noise
    noise = 2 * mean_std

    if noise == 0:
        warnings.warn('The noise was evaluated to 0.')
        snr = np.nan
    else:
        snr = peak_to_trough_voltage / noise
    return snr
