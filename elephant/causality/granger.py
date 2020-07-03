# -*- coding: utf-8 -*-
"""
.. include:: causality-overview.rst

.. current_module elephant.causality

Overview of Functions
---------------------
Various formulations of Granger causality have been developed. In this module
you will find function for time-series data to test pairwise Granger causality
(`pairwise_granger`).

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

Causality = namedtuple('Causality',
                       ['directional_causality_x_y',
                        'directional_causality_y_x',
                        'instantaneous_causality',
                        'total_interdependence'])


def bic(cov, order, dimension, length):
    """
    Calculate Bayesian Information Criterion

    Parameters
    ----------
    cov : np.ndarray
        covariance matrix of auto regressive model
    order : int
        order of autoregressive model
    dimension : int
        dimensionality of the data
    length : int
        number of time samples

    Returns
    -------
    criterion : float
       Bayesian Information Criterion
    """
    sign, log_det_cov = np.linalg.slogdet(cov)
    criterion = 2 * log_det_cov \
        + 2*(dimension**2)*order*np.log(length)/length

    return criterion


def aic(cov, order, dimension, length):
    """
    Calculate Akaike Information Criterion

    Parameters
    ----------
    cov : np.ndarray
        covariance matrix of auto regressive model
    order : int
        order of autoregressive model
    dimension : int
        dimensionality of the data
    length : int
        number of time samples

    Returns
    -------
    criterion : float
        Akaike Information Criterion
    """
    sign, log_det_cov = np.linalg.slogdet(cov)
    criterion = 2 * log_det_cov \
        + 2*(dimension**2)*order/length

    return criterion


def _lag_covariances(signals, dimension, max_lag):
    r"""
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

    Covariance of shifted signals calculated according to the following
    formula:

        x: d-dimensional signal
        x^T: transpose of d-dimensional signal
        N: number of time points
        \tau: lag

        C(\tau) = \sum_{i=0}^{N-\tau} x[i]*x^T[\tau+i]

    """
    length = np.size(signals[0])

    if length < max_lag:
        raise ValueError("Maximum lag larger than size of data")

    # centralize time series
    signals_mean = (signals - np.mean(signals, keepdims=True)).T

    lag_covariances = np.zeros((max_lag+1, dimension, dimension))

    # determine lagged covariance for different time lags
    for lag in range(0, max_lag+1):
        lag_covariances[lag] = \
                np.mean(np.einsum('ij,ik -> ijk', signals_mean[:length-lag],
                                  signals_mean[lag:]), axis=0)

    return lag_covariances


def _yule_walker_matrix(data, dimension, order):
    r"""
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


    Yule-Walker Matrix M is a block-structured symmetric matrix with
    dimension (d \cdot p)\times(d \cdot p)

    where
    d: dimension of signal
    p: order of autoregressive model
    C(\tau): time-shifted covariances \tau -> d \times d matrix

    The blocks of size (d \times d) are set as follows:

    M_ij = C(j-i)^T

    where 1 \leq i \leq j \leq p. The other entries are determined by symmetry.

    """

    lag_covariances = _lag_covariances(data, dimension, order)

    yule_walker_matrix = np.zeros((dimension*order, dimension*order))

    for block_row in range(order):
        for block_column in range(block_row, order):
            yule_walker_matrix[block_row*dimension: (block_row+1)*dimension,
                               block_column*dimension:
                               (block_column+1)*dimension] = \
                lag_covariances[block_column-block_row].T

            yule_walker_matrix[block_column*dimension:
                               (block_column+1)*dimension,
                               block_row*dimension:
                               (block_row+1)*dimension] = \
                lag_covariances[block_column-block_row]
    return yule_walker_matrix, lag_covariances


def _vector_arm(signals, dimension, order):
    r"""
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

    Coefficients of autoregressive model calculated via solving the linear
    equation

    M A = C

    where
    M: Yule-Waler Matrix
    A: Coefficients of autoregressive model
    C: Time-shifted covariances with positive lags

    Covariance matrix C_0 is then given by

    C_0 = C[0] - \sum_{i=0}^{p-1} A[i]C[i+1]

    where p is the orde of the autoregressive model.

    """

    yule_walker_matrix, lag_covariances = \
        _yule_walker_matrix(signals, dimension, order)

    positive_lag_covariances = np.reshape(lag_covariances[1:],
                                          (dimension*order, dimension))

    lstsq_coeffs = \
        np.linalg.lstsq(yule_walker_matrix, positive_lag_covariances)[0]

    coeffs = []
    for index in range(order):
        coeffs.append(lstsq_coeffs[index*dimension:(index+1)*dimension, ].T)

    coeffs = np.stack(coeffs)

    cov_matrix = np.copy(lag_covariances[0])
    for i in range(order):
        cov_matrix -= np.matmul(coeffs[i], lag_covariances[i+1])

    return coeffs, cov_matrix


def _optimal_vector_arm(signals, dimension, max_order,
                        information_criterion=aic):
    """
    Determine optimal auto regressive model by choosing optimal order via
    Information Criterion

    Parameters
    ----------
    signals : np.ndarray
        time series data
    dimension : int
        dimensionality of the data
    max_order : int
        maximal order to consider
    information_criterion : callable
        'bic' for Bayesian information_criterion,
        'aic' for Akaike information criterion

    Returns
    -------
    optimal_coeffs: np.ndarray
        coefficients of the autoregressive model
    optimal_cov_mat : np.ndarray
        covariance matrix of
    optimal_order : int
        optimal order
    """

    length = np.size(signals[0])

    optimal_ic = np.infty
    optimal_order = 1
    optimal_coeffs = np.zeros((dimension, dimension, optimal_order))
    optimal_cov_matrix = np.zeros((dimension, dimension))

    for order in range(1, max_order + 1):
        coeffs, cov_matrix = _vector_arm(signals, dimension, order)

        temp_ic = information_criterion(cov_matrix, order, dimension, length)

        if temp_ic < optimal_ic:
            optimal_ic = temp_ic
            optimal_order = order
            optimal_coeffs = coeffs
            optimal_cov_matrix = cov_matrix

    return optimal_coeffs, optimal_cov_matrix, optimal_order


def pairwise_granger(signals, max_order, information_criterion=aic):
    r"""
    Determine Granger Causality of two time series

    Parameters
    ----------
    signals : np.ndarray or neo.AnalogSignal
        time series data
    max_order : int
        maximal order of autoregressive model
    information_criterion : callable
        'bic' for Bayesian information_criterion,
        'aic' for Akaike information criterion

    Returns
    -------
    causality : namedTuple, where:
    causality.directional_causality_x_y : float
    causality.directional_causality_y_x : float
    causality.instantaneous_causality : float
    causality.total_interdependence : float

    Denote covariance matrix of signals
        X by C|X  - a real number
        Y by C|Y - a real number
        (X,Y) by C|XY - a (2 \times 2) matrix

    directional causality X -> Y given by
        log(C|X / C|XY_00)
    directional causality Y -> X given by
        log(C|Y / C|XY_11)
    instantaneous causality of X,Y given by
        log(C|XY_00 / C|XY_11)
    total interdependence of X,Y given by
        log( {C|X \cdot C|Y} / det{C|XY} )



    """

    if isinstance(signals, AnalogSignal):
        signals = np.asarray(signals)
        signals = np.rollaxis(signals, 0, len(signals.shape))
    else:
        signals = np.asarray(signals)

    signal_x = np.asarray([signals[0, :]])
    signal_y = np.asarray([signals[1, :]])

    coeffs_x, var_x, p_1 = _optimal_vector_arm(signal_x, 1, max_order,
                                               information_criterion)
    coeffs_y, var_y, p_2 = _optimal_vector_arm(signal_y, 1, max_order,
                                               information_criterion)
    coeffs_xy, cov_xy, p_3 = _optimal_vector_arm(signals, 2, max_order,
                                                 information_criterion)

    sign, log_det_cov = np.linalg.slogdet(cov_xy)

    if sign <= 0:

        raise ValueError(
            "Determinant of covariance matrix must be always positive: "
            "In this case its sign is {}".format(sign))

    directional_causality_y_x = np.log(var_x[0]) - np.log(cov_xy[0, 0])
    directional_causality_x_y = np.log(var_y[0]) - np.log(cov_xy[1, 1])

    instantaneous_causality = \
        np.log(cov_xy[0, 0]) + np.log(cov_xy[1, 1]) - log_det_cov
    instantaneous_causality = np.asarray(instantaneous_causality)

    total_interdependence = np.log(var_x[0]) + np.log(var_y[0]) - log_det_cov

    '''
    Round GC according to following scheme:
        Note that standard error scales as 1/sqrt(sample_size)
        Calculate  significant figures according to standard error
    '''
    length = np.size(signal_x)
    asymptotic_std_error = 1/np.sqrt(length)
    est_sig_figures = int((-1)*np.around(np.log10(asymptotic_std_error)))

    directional_causality_x_y_round = np.around(directional_causality_x_y,
                                                est_sig_figures)
    directional_causality_y_x_round = np.around(directional_causality_y_x,
                                                est_sig_figures)
    instantaneous_causality_round = np.around(instantaneous_causality,
                                              est_sig_figures)
    total_interdependence_round = np.around(total_interdependence,
                                            est_sig_figures)

    return Causality(
        directional_causality_x_y=directional_causality_x_y_round.item(),
        directional_causality_y_x=directional_causality_y_x_round.item(),
        instantaneous_causality=instantaneous_causality_round.item(),
        total_interdependence=total_interdependence_round.item())