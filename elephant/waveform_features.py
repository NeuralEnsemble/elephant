# -*- coding: utf-8 -*-
"""
.. autosummary::
    :toctree: _toctree/waveform_features

    waveform_width
    waveform_snr

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings

import neo
import numpy as np

__all__ = [
    "waveform_width",
    "waveform_snr"
]


def waveform_width(waveform, cutoff=0.75):
    """
    Calculate the width (trough-to-peak TTP) of a waveform.

    Searches for an index of a minimum within first `cutoff` of the waveform
    vector, next for a maximum after the identified minimum, and returns the
    difference between them.

    Parameters
    ----------
    waveform : array-like
        Time course of a single waveform. Accepts a list, a numpy array or a
        quantity.
    cutoff : float, optional
        Defines the normalized range `[0, cutoff]` of the input sequence for
        computing the minimum. Must be in `[0, 1)` range.
        Default: 0.75

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

    Examples
    --------
    >>> from elephant.waveform_features import waveform_width
    >>> waveform_width([20, 25, 10, -5, -2, 7, 15], cutoff=0.75)
    3

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


def waveform_snr(waveforms):
    """
    Return the signal-to-noise ratio of the waveforms of one or more
    spike trains :cite:`waveforms-Hatsopoulos2007_5105`.

    Signal-to-noise ratio is defined as the difference in mean peak-to-trough
    voltage divided by twice the mean SD. The mean SD is computed by
    measuring the SD of the spike waveform over all acquired spikes
    at each of the sample time points of the waveform and then averaging.

    Parameters
    ----------
    waveforms : array-like
        A list or a quantity or a numpy array of waveforms of shape
        ``(n_waveforms, time)`` in case of a single spike train or
        ``(n_waveforms, n_spiketrains, time)`` in case of one or more spike
        trains.

    Returns
    -------
    snr : float or np.ndarray
        Signal-to-noise ratio according to
        :cite:`waveforms-Hatsopoulos2007_5105`. If the input `waveforms`
        shape is ``(n_waveforms, time)`` or ``(n_waveforms, 1, time)``, a
        single float is returned. Otherwise, if the shape is
        ``(n_waveforms, n_spiketrains, time)``, a numpy array of length
        ``n_spiketrains`` is returned.

    Notes
    -----
    The waveforms of a `neo.SpikeTrain` can be extracted as
    `spiketrain.waveforms`, if it's loaded from a file, in which case you need
    to set ``load_waveforms=True`` in ``neo.read_block()``.

    Examples
    --------
    >>> from elephant.waveform_features import waveform_snr
    >>> waveforms = [[20, 25, 10, -5, -2, 7, 15], [17, 29, 11, -4, 0, 5, 20]]
    >>> waveform_snr(waveforms)
    12.249999999999998

    """
    if isinstance(waveforms, neo.SpikeTrain):
        warnings.warn("spiketrain input is deprecated; pass "
                      "'spiketrain.waveforms' directly.", DeprecationWarning)
        waveforms = waveforms.waveforms
    # asarray removes quantities, if present
    waveforms = np.squeeze(np.asarray(waveforms))
    # average over all waveforms for each bin
    mean_waveform = waveforms.mean(axis=0)
    # standard deviation over all waveforms over all bins
    std_waveform = waveforms.std(axis=0).mean(axis=-1)

    # peak to trough voltage signal
    peak_range = mean_waveform.max(axis=-1) - mean_waveform.min(axis=-1)
    # noise
    noise = 2 * std_waveform

    snr = peak_range / noise
    if np.isnan(snr).any():
        warnings.warn('The waveforms noise was evaluated to 0. Returning NaN')

    return snr
