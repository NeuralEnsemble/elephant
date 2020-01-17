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

# TODO: Pairwise Granger implementation
import numpy as np


def _lag_covariances(time_series, dimension, max_lag):
    '''
    Determine covariances of time series and time shift of itself up to a
    maximal lag
    Parameters
    ----------
    time_series: np.ndarray
        time series data
    dimension : int
        number of time series
    max_lag: int
        maximal time lag to be considered
    Returns
    -------
    lag_corr : np.ndarray
        correlations matrices of lagged signals
    '''

    length = np.size(time_series[0])

    assert (length >= max_lag), 'maximum lag larger than size of data'

    lag_covs = []
    series_mean = time_series.mean(1)

    for i in range(max_lag+1):
        temp_corr = np.zeros((dimension, dimension))
        for time in range(i, length):
            temp_corr+= np.outer(time_series[ : ,time - i] - series_mean,
            time_series[ : ,time] - series_mean)
        lag_covs.append(temp_corr/(length-i))

    return np.asarray(lag_covs)


def _Yule_Walker_matrix(data, dimension, order):
    '''
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
    Yule_Walker_matrix : np.ndarray
        matrix in Yule-Walker equation
    '''

    lag_covs = _lag_covariances(data, dimension, order)

    Yule_Walker_matrix = np.zeros((dimension*order, dimension*order))


    for block_row in range(order):
        for block_column in range(block_row, order):
            Yule_Walker_matrix[block_row*dimension : (block_row+1)*dimension,
                               block_column*dimension :
                               (block_column+1)*dimension] = lag_covs[block_column-block_row]

            Yule_Walker_matrix[block_column*dimension : (block_column+1)*dimension,
                               block_row*dimension :
                               (block_row+1)*dimension] = lag_covs[block_column-block_row].T
    return Yule_Walker_matrix, lag_covs


def _vector_arm(time_series, dimension, order):
    '''
    Determine coefficients of autoregressive model from time series data
    Parameters
    ----------
    time_series : np.ndarray
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
    '''

    Yule_Walker_matrix, lag_covs = _Yule_Walker_matrix(time_series,  dimension, order)

    solution_vector = np.reshape(lag_covs[1:], (dimension*order, dimension))

    coeffs_pre = np.linalg.lstsq(Yule_Walker_matrix, solution_vector)[0]

    coeffs = []
    for index in range(order):
        coeffs.append(coeffs_pre[index*dimension:(index+1)*dimension, ].T)

    coeffs = np.stack(coeffs)

    cov_mat = np.zeros((dimension, dimension))

    #cov_mat = lag_covs[0]
    for i in range(order):
        cov_mat += np.matmul(coeffs[i],lag_covs[i+1])

    return coeffs, cov_mat


def pairwise_granger(time_series, order):
    '''
    Determine Granger Causality of two time series
    Note: order paramter should be remove
    Parameters
    ----------
    time_series : np.ndarray
        time series data
    order : int
        order of autoregressive model (should be removed)
    Returns
    -------
    directional_causality_x_y : float
    directional_causality_y_x : float
    instantaneous_causality : float
    total_interdependence : float
    '''

    time_series_x_comp = np.asarray([time_series[0,:]])
    time_series_y_comp = np.asarray([time_series[1,:]])

    coeffs_x, var_x = _vector_arm(time_series_x_comp, 1, order)
    coeffs_y, var_y = _vector_arm(time_series_y_comp, 1, order)
    print(var_x)
    print(var_y)
    coeffs_xy, cov_xy = _vector_arm(time_series, 2, order)

    direct_caus_x_y = np.log(var_x[0]/cov_xy[0,0])
    direct_causy_y_x = np.log(var_y[0]/cov_xy[1,1])

    cov_det = np.linalg.det(cov_xy)

    inst_caus = np.log((cov_xy[0,0]*cov_xy[1,1])/cov_det)

    tot_interdep = np.log(var_x[0]*var_y[0]/cov_det)


    print(coeffs_xy)
    print(cov_xy)

    return direct_caus_x_y, direct_causy_y_x, inst_caus, tot_interdep