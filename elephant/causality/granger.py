# -*- coding: utf-8 -*-
"""
This module provides function to estimate causal influences of signals on each
other.


Granger causality
*****************
Granger causality is a method to determine causal influence of one signal on
another based on autoregressive modelling. It was developed by Nobel prize
laureate Clive Granger and has been adopted in various numerical fields ever
since :cite:`granger-Granger69_424`. In its simplest form, the
method tests whether the past values of one signal help to reduce the
prediction error of another signal, compared to the past values of the latter
signal alone. If it does reduce the prediction error, the first signal is said
to Granger cause the other signal.

Limitations
-----------
The user must be mindful of the method's limitations, which are assumptions of
covariance stationary data, linearity imposed by the underlying autoregressive
modelling as well as the fact that the variables not included in the model will
not be accounted for :cite:`granger-Seth07_1667`.

Implementation
--------------
The mathematical implementation of Granger causality methods in this module
closely follows :cite:`granger-Ding06_0608035`.


Overview of Functions
---------------------
Various formulations of Granger causality have been developed. In this module
you will find function for time-series data to test pairwise Granger causality
(`pairwise_granger`).

Time-series Granger causality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _toctree/causality/

    pairwise_granger
    conditional_granger


Tutorial
--------

:doc:`View tutorial <../tutorials/granger_causality>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/granger_causality.ipynb

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
from collections import namedtuple

import numpy as np
from neo.core import AnalogSignal


__all__ = (
    "Causality",
    "pairwise_granger",
    "conditional_granger"
)


# the return type of pairwise_granger() function
Causality = namedtuple('Causality',
                       ['directional_causality_x_y',
                        'directional_causality_y_x',
                        'instantaneous_causality',
                        'total_interdependence'])


def _bic(cov, order, dimension, length):
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


def _aic(cov, order, dimension, length):
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
    signals : np.ndarray
        time series data
    dimension : int
        number of time series
    max_lag : int
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

        where 1 \leq i \leq j \leq p. The other entries are determined by
        symmetry.
    lag_covariances : np.ndarray

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
    Determine coefficients of autoregressive model from time series data.

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

    Parameters
    ----------
    signals : np.ndarray
        time series data
    order : int
        order of the autoregressive model

    Returns
    -------
    coeffs : np.ndarray
        coefficients of the autoregressive model
        ry
    covar_mat : np.ndarray
        covariance matrix of

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
                        information_criterion='aic'):
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
    information_criterion : str
        A function to compute the information criterion:
            `bic` for Bayesian information_criterion,
            `aic` for Akaike information criterion
        Default: aic

    Returns
    -------
    optimal_coeffs : np.ndarray
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

        if information_criterion == 'aic':
            temp_ic = _aic(cov_matrix, order, dimension, length)
        elif information_criterion == 'bic':
            temp_ic = _bic(cov_matrix, order, dimension, length)
        else:
            raise ValueError("The specified information criterion is not"
                             "available. Please use 'aic' or 'bic'.")

        if temp_ic < optimal_ic:
            optimal_ic = temp_ic
            optimal_order = order
            optimal_coeffs = coeffs
            optimal_cov_matrix = cov_matrix

    return optimal_coeffs, optimal_cov_matrix, optimal_order


def pairwise_granger(signals, max_order, information_criterion='aic'):
    r"""
    Determine Granger Causality of two time series

    Parameters
    ----------
    signals : (N, 2) np.ndarray or neo.AnalogSignal
        A matrix with two time series (second dimension) that have N time
        points (first dimension).
    max_order : int
        Maximal order of autoregressive model.
    information_criterion : {'aic', 'bic'}, optional
        A function to compute the information criterion:
            `bic` for Bayesian information_criterion,
            `aic` for Akaike information criterion,
        Default: 'aic'

    Returns
    -------
    Causality
        A `namedtuple` with the following attributes:
            directional_causality_x_y : float
                The Granger causality value for X influence onto Y.

            directional_causality_y_x : float
                The Granger causality value for Y influence onto X.

            instantaneous_causality : float
                The remaining channel interdependence not accounted for by
                the directional causalities (e.g. shared input to X and Y).

            total_interdependence : float
                The sum of the former three metrics. It measures the dependence
                of X and Y. If the total interdependence is positive, X and Y
                are not independent.

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

    Raises
    ------
    ValueError
        If the provided signal does not have a shape of Nx2.

        If the determinant of the prediction error covariance matrix is not
        positive.

    Warns
    -----
    UserWarning
        If the log determinant of the prediction error covariance matrix is
        below the tolerance level of 1e-7.

    Notes
    -----
    The formulas used in this implementation follows
    :cite:`granger-Ding06_0608035`. The only difference being that we change
    the equation 47 in the following way:
    -R(k) - A(1)R(k - 1) - ... - A(m)R(k - m) = 0.
    This forumlation allows for the usage of R values without transposition
    (i.e. directly) in equation 48.

    Examples
    --------
    Example 1. Independent variables.

    >>> import numpy as np
    >>> from elephant.causality.granger import pairwise_granger
    >>> pairwise_granger(np.random.uniform(size=(1000, 2)), max_order=2)
    Causality(directional_causality_x_y=0.0,
             directional_causality_y_x=-0.0,
             instantaneous_causality=0.0,
             total_interdependence=0.0)

    Example 2. Dependent variables. Y depends on X but not vice versa.

    .. math::
        \begin{array}{ll}
            X_t \sim \mathcal{N}(0, 1) \\
            Y_t = 3.5 \cdot X_{t-1} + \epsilon, \;
                  \epsilon \sim\mathcal{N}(0, 1)
        \end{array}

    In this case, the directional causality is non-zero.

    >>> x = np.random.randn(1001)
    >>> y = 3.5 * x[:-1] + np.random.randn(1000)
    >>> signals = np.array([x[1:], y]).T  # N x 2 matrix
    >>> pairwise_granger(signals, max_order=1)
    Causality(directional_causality_x_y=2.64,
              directional_causality_y_x=0.0,
              instantaneous_causality=0.0,
              total_interdependence=2.64)

    """
    if isinstance(signals, AnalogSignal):
        signals = signals.magnitude

    if not (signals.ndim == 2 and signals.shape[1] == 2):
        raise ValueError("The input 'signals' must be of dimensions Nx2.")

    # transpose (N,2) -> (2,N) for mathematical convenience
    signals = signals.T

    # signal_x and signal_y are (1, N) arrays
    signal_x, signal_y = np.expand_dims(signals, axis=1)

    coeffs_x, var_x, p_1 = _optimal_vector_arm(signal_x, 1, max_order,
                                               information_criterion)
    coeffs_y, var_y, p_2 = _optimal_vector_arm(signal_y, 1, max_order,
                                               information_criterion)
    coeffs_xy, cov_xy, p_3 = _optimal_vector_arm(signals, 2, max_order,
                                                 information_criterion)

    sign, log_det_cov = np.linalg.slogdet(cov_xy)
    tolerance = 1e-7

    if sign <= 0:
        raise ValueError(
            "Determinant of covariance matrix must be always positive: "
            "In this case its sign is {}".format(sign))

    if log_det_cov <= tolerance:
        warnings.warn("The value of the log determinant is at or below the "
                      "tolerance level. Proceeding with computation.",
                      UserWarning)

    directional_causality_y_x = np.log(var_x[0]) - np.log(cov_xy[0, 0])
    directional_causality_x_y = np.log(var_y[0]) - np.log(cov_xy[1, 1])

    instantaneous_causality = \
        np.log(cov_xy[0, 0]) + np.log(cov_xy[1, 1]) - log_det_cov
    instantaneous_causality = np.asarray(instantaneous_causality)

    total_interdependence = np.log(var_x[0]) + np.log(var_y[0]) - log_det_cov

    # Round GC according to following scheme:
    #     Note that standard error scales as 1/sqrt(sample_size)
    #     Calculate  significant figures according to standard error
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


def conditional_granger(signals, max_order, information_criterion='aic'):
    r"""
    Determine conditional Granger Causality of the second time series on the
    first time series, given the third time series. In other words, for time
    series X_t, Y_t and Z_t, this function tests if Y_t influences X_t via Z_t.

    Parameters
    ----------
    signals : (N, 3) np.ndarray or neo.AnalogSignal
        A matrix with three time series (second dimension) that have N time
        points (first dimension). The time series to be conditioned on is the
        third.
    max_order : int
        Maximal order of autoregressive model.
    information_criterion : {'aic', 'bic'}, optional
        A function to compute the information criterion:
            `bic` for Bayesian information_criterion,
            `aic` for Akaike information criterion,
        Default: 'aic'

    Returns
    -------
    conditional_causality_xy_z_round : float
        The value of conditional causality of Y_t on X_t given Z_t. Zero value
        indicates that causality of Y_t on X_t is solely dependent on Z_t.

    Raises
    ------
    ValueError
        If the provided signal does not have a shape of Nx3.

    Notes
    -----
    The formulas used in this implementation follows
    :cite:`granger-Ding06_0608035`. Specifically, the Eq 35.
    """
    if isinstance(signals, AnalogSignal):
        signals = signals.magnitude

    if not (signals.ndim == 2 and signals.shape[1] == 3):
        raise ValueError("The input 'signals' must be of dimensions Nx3.")

    # transpose (N,3) -> (3,N) for mathematical convenience
    signals = signals.T

    # signal_x, signal_y and signal_z are (1, N) arrays
    signal_x, signal_y, signal_z = np.expand_dims(signals, axis=1)

    signals_xz = np.vstack([signal_x, signal_z])

    coeffs_xz, cov_xz, p_1 = _optimal_vector_arm(
        signals_xz, dimension=2, max_order=max_order,
        information_criterion=information_criterion)
    coeffs_xyz, cov_xyz, p_2 = _optimal_vector_arm(
        signals, dimension=3, max_order=max_order,
        information_criterion=information_criterion)

    conditional_causality_xy_z = np.log(cov_xz[0, 0]) - np.log(cov_xyz[0, 0])

    # Round conditional GC according to following scheme:
    #     Note that standard error scales as 1/sqrt(sample_size)
    #     Calculate  significant figures according to standard error
    length = np.size(signal_x)
    asymptotic_std_error = 1/np.sqrt(length)
    est_sig_figures = int((-1)*np.around(np.log10(asymptotic_std_error)))

    conditional_causality_xy_z_round = np.around(conditional_causality_xy_z,
                                                 est_sig_figures)

    return conditional_causality_xy_z_round
