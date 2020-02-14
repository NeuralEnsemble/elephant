# -*- coding: utf-8 -*-
"""
.. include:: causality-overview.rst

.. current_module elephant.causality

Overview of Functions
---------------------
Various formulations of Granger causality have been developed. In this module you will find function for time-series data to test pairwise Granger causality (`pairwise_granger`).

Time-series Granger causality
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: causality/

    pairwise_granger


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
from collections import namedtuple
from neo.core import AnalogSignal

Causality = namedtuple('causality',
                       'directional_causality_x_y directional_causality_y_x instantaneous_causality total_interdependence')


# TODO: The result of pairwise granger outputs three arrays and one float.

def _lag_covariances(signals, dimension, max_lag):
    """
    Determine covariances of time series and time shift of itself up to a
    maximal lag
    Parameters
    ----------
    signals: np.ndarray
        time series data
    dimension : int
        number of time series
    max_lag: int
        maximal time lag to be considered
    Returns
    -------
    lag_corr : np.ndarray
        correlations matrices of lagged signals
    """

    length = np.size(signals[0])

    assert (length >= max_lag), 'maximum lag larger than size of data'

    lag_covariances = []
    series_mean = signals.mean(1)

    for i in range(max_lag+1):
        temp_corr = np.zeros((dimension, dimension))
        for time in range(i, length):
            temp_corr += np.outer(signals[:, time - i] - series_mean,
            signals[:, time] - series_mean)
        lag_covariances.append(temp_corr/(length-i))

    return np.asarray(lag_covariances)


def _yule_walker_matrix(data, dimension, order):
    """
    Generate matrix for Yule-Walker equation
    Parameters
    ----------
    data : np.ndarray
        correlation of data shifted with lags up to order
    dimension : int
        dimensionality of data (e.g. number of channels)
    order : int
        order of the autoregressive model
    Returns
    -------
    yule_walker_matrix : np.ndarray
        matrix in Yule-Walker equation
    """

    lag_covariances = _lag_covariances(data, dimension, order)

    yule_walker_matrix = np.zeros((dimension*order, dimension*order))

    for block_row in range(order):
        for block_column in range(block_row, order):
            yule_walker_matrix[block_row*dimension: (block_row+1)*dimension,
                               block_column*dimension:
                               (block_column+1)*dimension] = lag_covariances[block_column-block_row]

            yule_walker_matrix[block_column*dimension: (block_column+1)*dimension,
                               block_row*dimension:
                               (block_row+1)*dimension] = lag_covariances[block_column-block_row].T
    return yule_walker_matrix, lag_covariances


def _vector_arm(signals, dimension, order):
    """
    Determine coefficients of autoregressive model from time series data
    Parameters
    ----------
    signals : np.ndarray
        time series data
    order : int
        order of the autoregressive model
    Returns
    -------
    coeffs: np.ndarray
        coefficients of the autoregressive model
        ry
    covar_mat : np.ndarray
        covariance matrix of
    """

    yule_walker_matrix, lag_covariances = _yule_walker_matrix(signals, dimension, order)

    positive_lag_covariances = np.reshape(lag_covariances[1:], (dimension*order, dimension))

    lstsq_coeffs = np.linalg.lstsq(yule_walker_matrix, positive_lag_covariances)[0]

    coeffs = []
    for index in range(order):
        coeffs.append(lstsq_coeffs[index*dimension:(index+1)*dimension, ].T)

    coeffs = np.stack(coeffs)

    cov_matrix = np.zeros((dimension, dimension))

    #cov_matrix = lag_covariances[0]
    for i in range(order):
        cov_matrix += np.matmul(coeffs[i], lag_covariances[i+1])

    return coeffs, cov_matrix


def pairwise_granger(signals, order):
    """
    Determine Granger Causality of two time series
    Note: order parameter should be removed
    Parameters
    ----------
    signals : np.ndarray or neo.AnalogSignal
        time series data
    order : int
        order of autoregressive model (should be removed)
    Returns
    -------
    causality : namedTuple, where:
    causality.directional_causality_x_y : float
    causality.directional_causality_y_x : float
    causality.instantaneous_causality : float
    causality.total_interdependence : float
    """
    # TODO: remove order parameter
    if order <= 0:
        raise ValueError(f"The order parameter should be positive. Not {order}")

    if isinstance(signals, AnalogSignal):
        signals = np.asarray(signals)
        signals = np.rollaxis(signals, 0, len(signals.shape))
    else:
        signals = np.asarray(signals)

    signal_x = np.asarray([signals[0, :]])
    signal_y = np.asarray([signals[1, :]])

    coeffs_x, var_x = _vector_arm(signal_x, 1, order)
    coeffs_y, var_y = _vector_arm(signal_y, 1, order)
    print(var_x)
    print(var_y)
    coeffs_xy, cov_xy = _vector_arm(signals, 2, order)

    directional_causality_x_y = np.log(var_x[0]/cov_xy[0, 0])
    print(f'directional_x_y is {var_x[0]/cov_xy[0, 0]}. The variance of x is {var_x[0]}, the covariance_xy is {cov_xy[0, 0]}')
    directional_causality_y_x = np.log(var_y[0]/cov_xy[1, 1])
    print(f'directional_x_y is {var_y[0]/cov_xy[0, 0]}. The variance of x is {var_y[0]}, the covariance_xy is {cov_xy[0, 0]}')

    cov_determinant = np.linalg.det(cov_xy)

    instantaneous_causality = np.log((cov_xy[0, 0]*cov_xy[1, 1])/cov_determinant)
    instantaneous_causality = np.asarray(instantaneous_causality)

    total_interdependence = np.log(var_x[0]*var_y[0]/cov_determinant)

    print(coeffs_xy)
    print(cov_xy)

    return Causality(directional_causality_x_y=directional_causality_x_y,
                     directional_causality_y_x=directional_causality_y_x,
                     instantaneous_causality=instantaneous_causality,
                     total_interdependence=total_interdependence)


