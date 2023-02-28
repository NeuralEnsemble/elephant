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
Granger causality analysis can be extended to the spectral domain investigating
the influnece signals have onto each other in a frequency resolved manner. It
relies on estimating the cross-spectrum of time series and decomposing them
into a transfer function and a noise covariance matrix. The method to estimate
the spectral Granger causality is non-parametric in the sense that it does not
require to fit an autoregressive model (see :cite:`granger-Dhamala02_354`).

Limitations
-----------
The user must be mindful of the method's limitations, which are assumptions of
covariance stationary data, linearity imposed by the underlying autoregressive
modelling as well as the fact that the variables not included in the model will
not be accounted for :cite:`granger-Seth07_1667`.
When estimating spectral Granger causality the user must be familiar with
basics the multitaper method to estimate power- and cross-spectra (e.g.
sampling frequency, DPSS, time-half bandwidth product).

Implementation
--------------
The mathematical implementation of Granger causality methods in this module
closely follows :cite:`granger-Ding06_0608035`.
The implementation of spectral Granger causality follows
:cite:`granger-Dhamala02_354`, :cite:`granger-Wend13_20110610` and
:cite:`granger-Wilson72` for the spectral matrix factorization.


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

Spectral Granger causality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _toctree/causality/

    pairwise_spectral_granger


Tutorial
--------

:doc:`View tutorial <../tutorials/granger_causality>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/granger_causality.ipynb

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
from collections import namedtuple

import numpy as np
import quantities as pq
import neo
from neo.core import AnalogSignal
from elephant.spectral import segmented_multitaper_cross_spectrum, multitaper_psd


__all__ = (
    "Causality",
    "pairwise_granger",
    "conditional_granger",
    "pairwise_spectral_granger"
)


# the return type of pairwise_granger(), pairwise_spectral_granger() function
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


def _bracket_operator(spectrum, num_freqs, num_signals):
    r"""
    Implementation of the $[ \cdot ]^{+}$ from 'The Factorization of Matricial
    Spectral Densities', Wilson 1972, SiAM J Appl Math, Definition 1.2 (ii).

    Paramaters
    ----------
    spectrum : np.ndarray
        Cross-spectrum of multivariate time series
    num_freqs : int
        Number of frequencies
    num_signals : int
        Number of time series

    Returns
    -------
    causal_part : np.ndarray
        Causal part of cross-spectrum of multivariate time series
    """
    # Get coefficients from spectrum
    causal_part = np.fft.ifft(spectrum, axis=0)
    # Version 1
    # Throw away of acausal part
    causal_part[(num_freqs + 1) // 2:] = 0

    # Treat zero frequency part
    causal_part[0] /= 2

    # Back-transformation
    causal_part = np.fft.fft(causal_part, axis=0)

    # Adjust zero frequency part to ensure convergence by setting entries
    # below diagonal to zero at zero-frequency
    if num_signals > 1:
        indices = np.tril_indices(num_signals, k=-1)
        causal_part[0, indices[0], indices[1]] = 0

    return causal_part


def _dagger(matrix_array):
    r"""
    Return Hermitian conjugate of matrix array.

    Parameters
    ----------
    matrix_array : np.ndarray
        Array of matrices the Hermitian conjugate of which needs to be
        determined

    Returns
    -------
    matrix_array_dagger : np.ndarray
        Hermitian conjugate of matrix_array
    """

    if matrix_array.ndim == 2:
        matrix_array_dagger = np.transpose(matrix_array.conj(), axes=(1, 0))

    else:
        matrix_array_dagger = np.transpose(matrix_array.conj(), axes=(0, 2, 1))

    return matrix_array_dagger


def _spectral_factorization(cross_spectrum, num_iterations):
    r"""
    Determine the spectral matrix factorization of the cross-spectrum
    (denoted by S) of multiple time series:
        S = H \Sigma H^{\dagger}
    Here, \Sigma is the covariance matrix, H the transfer function.
    We follow the algorithm outlined in 'The Factorization of Matricial
    Spectral Densities', Wilson 1972, SiAM J Appl Math.
    The algorithm iteratively calculates approximations for the spectral
    factorization and terminates if either the maximum number of iterations is
    reached or the difference between the cross spectrum and the approximate
    cross spectrum calculated from the covariance matrix and transfer function
    is sufficiently small.

    Parameters
    ----------
    cross_spectrum : np.ndarray
        Cross spectrum to be decomposed in covariance matrix and transfer
        function
    num_iterations : int
        Maximal number of iterations of iterative algorithm

    Returns
    ------
    cov_matrix : np.ndarray
        Covariance matrix of spectral factorization
    transfer_function : np.ndarray
        Transfer function of spectral factorization
    """
    cross_spectrum = np.transpose(cross_spectrum, axes=(2, 0, 1))

    # spectral_density_function = np.fft.ifft(cross_spectrum, axis=0)
    spectral_density_function = np.copy(cross_spectrum)

    # Resolve dimensions
    num_freqs = np.shape(spectral_density_function)[0]
    num_signals = np.shape(spectral_density_function)[1]

    # Initialization
    identity = np.identity(num_signals)
    factorization = np.zeros(np.shape(spectral_density_function),
                             dtype='complex128')

    # Estimate initial conditions
    try:
        initial_cond = np.linalg.cholesky(cross_spectrum[0].real).T
    except np.linalg.LinAlgError:
        raise ValueError('Could not calculate Cholesky decomposition of real'
                         + ' part of zero frequency estimate of cross-spectrum'
                         + '. This might suggest a problem with the input')

    factorization += initial_cond
    # Iteration for calculating spectral factorization
    for i in range(num_iterations):

        factorization_old = np.copy(factorization)

        # Implementation of Eq. 3.1 from "The Factorization of Matricial
        # Spectral Densities", Wilson 1972, SiAM J Appl Math
        X = np.linalg.solve(factorization,
                            spectral_density_function)
        Y = np.linalg.solve(factorization,
                            _dagger(X))
        Y += identity
        Y = _bracket_operator(Y, num_freqs, num_signals)

        factorization = np.matmul(factorization, Y)

        diff = factorization - factorization_old
        error = np.max(np.abs(diff))
        if error < 1e-12:
            print(f'Spectral factorization converged after {i} steps')
            break

    cov_matrix = np.matmul(factorization[0],
                           _dagger(factorization[0]))

    transfer_function = np.matmul(factorization,
                                  np.linalg.inv(factorization[0]))

    return cov_matrix, transfer_function


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
    >>> pairwise_granger(np.random.uniform(size=(1000, 2)), max_order=2)  # noqa
    Causality(directional_causality_x_y=-0.0, directional_causality_y_x=0.0, instantaneous_causality=0.0, total_interdependence=0.0)

    Example 2. Dependent variables. Y depends on X but not vice versa.

    .. math::
        \begin{array}{ll}
            X_t \sim \mathcal{N}(0, 1) \\
            Y_t = 3.5 \cdot X_{t-1} + \epsilon, \;
                  \epsilon \sim\mathcal{N}(0, 1)
        \end{array}

    In this case, the directional causality is non-zero.

    >>> np.random.seed(31)
    >>> x = np.random.randn(1001)
    >>> y = 3.5 * x[:-1] + np.random.randn(1000)
    >>> signals = np.array([x[1:], y]).T  # N x 2 matrix
    >>> pairwise_granger(signals, max_order=1)  # noqa
    Causality(directional_causality_x_y=2.64, directional_causality_y_x=-0.0, instantaneous_causality=0.0, total_interdependence=2.64)

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


def pairwise_spectral_granger(signal_i, signal_j, n_segments=8,
                              len_segment=None, frequency_resolution=None,
                              overlap=0.5, fs=1, nw=4, num_tapers=None,
                              peak_resolution=None, num_iterations=300):
    r"""Determine spectral Granger Causality of two signals.

    Parameters
    ----------
    signal_i : neo.AnalogSignal or pq.Quantity or np.ndarray
        First time series data of the pair between which spectral Granger
        causality is computed.
    signal_j : neo.AnalogSignal or pq.Quantity or np.ndarray
        Second time series data of the pair between which spectral Granger
        causality is computed.
        The shapes and the sampling frequencies of `signal_i` and `signal_j`
        must be identical. When `signal_i` and `signal_j` are not
        `neo.AnalogSignal`, sampling frequency should be specified through the
        keyword argument `fs`. Otherwise, the default value is used
        (`fs` = 1.0).
    n_segments : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_segment` or `frequency_resolution` is
        given.
        Default: 8.
    len_segment : int, optional
        Length of segments. This parameter is ignored if `frequency_resolution`
        is given. If None, it will be determined from other parameters.
        Default: None.
    frequency_resolution : pq.Quantity or float, optional
        Desired frequency resolution of the obtained spectral Granger causality
        estimate in terms of the interval between adjacent frequency bins. When
        given as a `float`, it is taken as frequency in Hz.
        If None, it will be determined from other parameters.
        Default: None.
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped).
    fs : float, optional
        Specifies the sampling frequency of the input time series
        Default: 1.0
    nw : float, optional
        Time bandwidth product
        Default: 4.0
    num_tapers : int, optional
        Number of tapers used in 1. to obtain estimate of PSD. By default
        [2*nw] - 1 is chosen.
        Default: None
    peak_resolution : pq.Quantity float, optional
        Quantity in Hz determining the number of tapers used for analysis.
        Fine peak resolution --> low numerical value --> low number of tapers
        High peak resolution --> high numerical value --> high number of tapers
        When given as a `float`, it is taken as frequency in Hz.
        Default: None.
    num_iterations : int
        Number of iterations for algorithm to estimate spectral factorization.
        Default: 300

    Returns
    -------
    freqs : np.ndarray
        Frequencies associated with the magnitude-squared spectral Granger
        Causality estimate
    Causality
        A `namedtuple` with the following attributes:
            directional_causality_x_y : np.ndarray
                Spectrally resolved Granger causality of X influence onto Y.

            directional_causality_y_x : np.ndarray
                Spectrally resolved Granger causality of Y influence onto X.

            instantaneous_causality : np.ndarray
                The remaining spectrally resolved channel interdependence not
                accounted for by the directional causalities (e.g. shared input
                to X and Y).

            total_interdependence : np.ndarray
                The sum of the former three metrics. It measures the dependence
                of X and Y. If the total interdependence is positive, X and Y
                are not independent.
    """
    if isinstance(signal_i, neo.core.AnalogSignal) and \
            isinstance(signal_j, neo.core.AnalogSignal):
        signals = signal_i.merge(signal_j)
    elif isinstance(signal_i, np.ndarray) and isinstance(signal_j, np.ndarray):
        signals = np.vstack([signal_i, signal_j])

    # Calculate cross spectrum for signals
    freqs, S = segmented_multitaper_cross_spectrum(
        signals=signals, n_segments=n_segments, len_segment=len_segment,
        frequency_resolution=frequency_resolution, overlap=overlap, fs=fs,
        nw=nw, num_tapers=num_tapers, peak_resolution=peak_resolution,
        return_onesided=False)

    # Remove units attached by the multitaper_cross_spectrum
    if isinstance(S, pq.Quantity):
        S = S.magnitude

    C, H = _spectral_factorization(S, num_iterations=num_iterations)

    # Take positive frequencies
    mask = (freqs >= 0)
    freqs = freqs[mask]

    # Get cross-spectrum and transfer function for positive frequencies
    S = np.transpose(S, axes=(2, 0, 1))[mask]
    ########## Experimental line #######
    S = np.conjugate(S)
    H = H[mask]

    # Calculate spectral Granger Causality.
    # Formulae follow Wen et al., 2012, Phil Trans R Soc
    H_tilde_xx = H[:, 0, 0] + C[0, 1]/C[0, 0]*H[:, 0, 1]
    H_tilde_yy = H[:, 1, 1] + C[0, 1]/C[1, 1]*H[:, 1, 0]

    directional_causality_y_x = np.log(S[:, 0, 0].real /
                                       (H_tilde_xx
                                        * C[0, 0]
                                        * H_tilde_xx.conj()).real)

    directional_causality_x_y = np.log(S[:, 1, 1].real /
                                       (H_tilde_yy
                                        * C[1, 1]
                                        * H_tilde_yy.conj()).real)

    instantaneous_causality = np.log(
        (H_tilde_xx * C[0, 0] * H_tilde_xx.conj()).real
        * (H_tilde_yy * C[1, 1] * H_tilde_yy.conj()).real)
    instantaneous_causality -= np.linalg.slogdet(S)[1]

    total_interdependence = (directional_causality_x_y
                             + directional_causality_y_x
                             + instantaneous_causality)

    spectral_causality = Causality(
        directional_causality_x_y=directional_causality_x_y,
        directional_causality_y_x=directional_causality_y_x,
        instantaneous_causality=instantaneous_causality,
        total_interdependence=total_interdependence)

    return freqs, spectral_causality


if __name__ == '__main__':

    # Test spectral factorization
    #np.random.seed(12321)
    #length_2d = 2**12
    #signal = np.zeros((2, length_2d))

    #order = 2
    #weights_1 = np.array([[0.9, 0], [0.16, 0.8]]).T
    #weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]]).T

    #weights = np.stack((weights_1, weights_2))

    #noise_covariance = np.array([[1., 0.4], [0.4, 0.7]])

    #for i in range(length_2d):
    #    for lag in range(order):
    #        signal[:, i] += np.dot(weights[lag],
    #                               signal[:, i - lag - 1])
    #    rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
    #    signal[0, i] += rnd_var[0]
    #    signal[1, i] += rnd_var[1]

    #signal = signal[:, 100:]
    #length_2d -= 100
    #n = length_2d

    #x = signal[0]
    #y = signal[1]

    #f, psd_1 = multitaper_psd(x, num_tapers=15, n_segments=8)
    #f, psd_2 = multitaper_psd(y, num_tapers=15, n_segments=8)

    #_,  cross_spectrum = segmented_multitaper_cross_spectrum(signal,
    #                                                           n_segments=8,
    #                                                           num_tapers=15,
    #                                                           return_onesided=True)

    #cov_matrix, transfer_function = _spectral_factorization(cross_spectrum,
    #                                                        num_iterations=100)

    #A = np.matmul(np.matmul(transfer_function, cov_matrix),
    #              _dagger(transfer_function))


    #print('################')
    #from matplotlib import pyplot as plt

    #plt.plot(f, cross_spectrum[0,0, :(n+2)//2], label='True')
    #plt.plot(f, A[:(n+2)//2, 0, 0], label='Mult')
    #plt.legend()
    #plt.show()

    #plt.plot(f, np.real(cross_spectrum[0, 1, :(n+2)//2]), label='True')
    #plt.plot(f, np.real(A[:(n+2)//2, 0, 1]), label='Mult')
    #plt.legend()
    #plt.show()

    #plt.plot(f, np.imag(cross_spectrum[0,1, :(n+2)//2]), label='True')
    #plt.plot(f, np.imag(A[:(n+2)//2, 0, 1]), label='Mult')
    #plt.legend()
    #plt.show()

    #plt.plot(f, np.real(cross_spectrum[1,0, :(n+2)//2]), label='True')
    #plt.plot(f, np.real(A[:(n+2)//2, 1, 0]), label='Mult')
    #plt.legend()
    #plt.show()

    #plt.plot(f, np.imag(cross_spectrum[1,0, :(n+2)//2]), label='True')
    #plt.plot(f, np.imag(A[:(n+2)//2, 1, 0]), label='Mult')
    #plt.legend()
    #plt.show()

    #plt.plot(f, cross_spectrum[1,1,:(n+2)//2], label='True')
    #plt.plot(f, A[:(n+2)//2, 1, 1], label='Mult')
    #plt.legend()
    #plt.show()

    # Test spectral granger
    length_2d = 2**14
    signal = np.zeros((2, length_2d))

    order = 2
    weights_1 = np.array([[0.9, 0], [0.16, 0.8]]).T
    weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]]).T

    weights = np.stack((weights_1, weights_2))

    noise_covariance = np.array([[1., 0.4], [0.4, 0.7]])

    for i in range(length_2d):
        for lag in range(order):
            signal[:, i] += np.dot(weights[lag],
                                   signal[:, i - lag - 1])
        rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
        signal[0, i] += rnd_var[0]
        signal[1, i] += rnd_var[1]

    # from elephant.test.test_causality import ConditionalGrangerTestCase
    # signal3 = ConditionalGrangerTestCase._generate_ground_truth(length_2d=2**10, causality_type='indirect').T
    # temp_signal = signal
    # signal = signal3[:2, :]

    np.save('/home/jurkus/repositories/r_spectral_granger/signal.npy', signal)
    freq, cross_spectrum = segmented_multitaper_cross_spectrum(signal, nw=5, return_onesided=True)

    # from spectral_connectivity import Connectivity, Multitaper


    # Good choices
    # length_2d=2 ** 16; len_segment=2**8, num_tapers=25
    # length_2d=2 ** 14; len_segment=2**8, num_tapers=25
    # length_2d=2 ** 16; frequency_resolution=0.01
    f, spectral_causality = pairwise_spectral_granger(signal[0], signal[1], len_segment=2**8, num_tapers=25, fs=1, num_iterations=50)
    # f, spectral_causality = pairwise_spectral_granger(signal[0], signal[1], frequency_resolution=0.01, fs=1, num_iterations=50)

    # Spectral connectivity package

    ##connectivity = Connectivity(
    ##    fourier_coefficients=multitaper.fft(),
    ##    expectation_type="tapers",
    ##    frequencies=multitaper.frequencies,
    ##    time=multitaper.time,
    ##    blocks=1,)
    #fcs, cross_spectrum = segmented_multitaper_cross_spectrum(signal,
    #                                                     len_segment=2**9,
    #                                                     nw=5,
    #                                                     return_onesided=True)
    #cov_matrix, transfer_function = _spectral_factorization(cross_spectrum,
    #                                                        num_iterations=100)

    #c_lfp = Connectivity.from_multitaper(multitaper)

    fn = np.arange(0, np.pi , 0.01)



    # Theoretical prediction for granger causality
    freqs_for_theo = np.array([1, 2])[:, np.newaxis] * fn
    A_theo = (np.identity(2)[np.newaxis]
              - weights_1.T * np.exp(
                  - 1j * freqs_for_theo[0][:, np.newaxis, np.newaxis]))
    A_theo -= weights_2.T * np.exp(
        - 1j * freqs_for_theo[1][:, np.newaxis, np.newaxis])

    H_theo = np.array([[A_theo[:, 1, 1], -A_theo[:, 0, 1]],
                      [-A_theo[:, 1, 0], A_theo[:, 0, 0]]])
    H_theo /= np.linalg.det(A_theo)
    H_theo = np.moveaxis(H_theo, 2, 0)

    S_theo = np.matmul(np.matmul(H_theo, noise_covariance), _dagger(H_theo))

    H_tilde_xx = (H_theo[:, 0, 0]
                  + noise_covariance[0, 1]/noise_covariance[0, 0]*H_theo[:, 0,
                                                                         1])
    H_tilde_yy = (H_theo[:, 1, 1]
                  + noise_covariance[0, 1]/noise_covariance[1, 1]*H_theo[:, 1,
                                                                         0])

    directional_causality_y_x = np.log(S_theo[:, 0, 0].real /
                                       (H_tilde_xx
                                        * noise_covariance[0, 0]
                                        * H_tilde_xx.conj()).real)

    directional_causality_x_y = np.log(S_theo[:, 1, 1].real /
                                       (H_tilde_yy
                                        * noise_covariance[1, 1]
                                        * H_tilde_yy.conj()).real)

    instantaneous_causality = np.log(
        (H_tilde_xx * noise_covariance[0, 0] * H_tilde_xx.conj()).real
        * (H_tilde_yy * noise_covariance[1, 1] * H_tilde_yy.conj()).real)
    instantaneous_causality -= np.linalg.slogdet(S_theo)[1]

    from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(fn, directional_causality_x_y, label='x->y')
    # plt.plot(fn, directional_causality_y_x, label='y->x')
    # plt.plot(fn, instantaneous_causality, label='inst')
    # plt.legend()
    # plt.title('Theoretical prediction')
    # plt.show()
    #
    # from matplotlib import pyplot as plt
    #
    # plt.figure()
    # plt.plot(f, spectral_causality[0], label='x->y')
    # plt.plot(f, spectral_causality[1], label='y->x')
    # plt.plot(f, spectral_causality[2], label='inst')
    # plt.plot(f, spectral_causality[3], label='tot')
    # plt.legend()
    # plt.title('Granger (Elephant) estimate')
    # plt.show()

    plt.figure()
    plt.plot(fn * f.max() / fn.max(), directional_causality_x_y, 'r:', label='Theoretical x->y')
    plt.plot(fn * f.max() / fn.max(), directional_causality_y_x, 'b:', label='Theoretical y->x')
    plt.plot(fn * f.max() / fn.max(), instantaneous_causality, 'k:', label='Theoretical inst')
    plt.plot(f, spectral_causality[0], 'r', label='Estimated x->y')
    plt.plot(f, spectral_causality[1], 'b', label='Estimated y->x')
    plt.plot(f, spectral_causality[2], 'k', label='Estimated inst')
    plt.legend()
    plt.title('Granger (Elephant) vs Theoretical estimate')
    plt.show()

    plt.figure()
    plt.plot(fn * f.max() / fn.max(), directional_causality_x_y, label='x->y true')
    plt.plot(fn * f.max() / fn.max(), directional_causality_y_x, label='y->x true')
    plt.plot(f, spectral_causality[0], label='x->y')
    plt.plot(f, spectral_causality[1], label='y->x')
    plt.legend()
    plt.title('Theoretical vs Elephant')
    plt.show()


    from elephant.spectral import multitaper_coherence
    freqs, coh, phase_lag = multitaper_coherence(signal[0, :], signal[1, :], len_segment=2**8, num_tapers=25, fs=1,)

    tot = spectral_causality[3]
    der_tot = -np.log(1 - coh)[:len(tot)]
    print(f"Max error of total interdependence values: "
          f"{(tot - der_tot)[np.argmax(np.abs(tot - der_tot))]}")


    # Signal used: length_2d = 2 ** 14
    gc_mat = np.load('/home/jurkus/repositories/r_spectral_granger/gc_matrix.npy')

    fig = plt.figure()
    ax = fig.gca()
    plt.plot(fn * f.max() / fn.max(), directional_causality_x_y, 'r--', label='Theoretical: x->y')
    plt.plot(fn * f.max() / fn.max(), directional_causality_y_x, 'g--', label='Theoretical: y->x')
    plt.plot(f, spectral_causality[0], 'r', label='Elephant: x->y')
    plt.plot(f, spectral_causality[1], 'g', label='Elephant: y->x')
    plt.plot(gc_mat[:, 0], gc_mat[:, 1], 'r:', label='R: x->y')
    plt.plot(gc_mat[:, 0], gc_mat[:, 2], 'g:', label='R: y->x')
    info_str = f'length_2d={length_2d}\nlen_segment=2**8\nnum_tapers=25'
    plt.text(0, 1, info_str, ha='left', va='top', transform=ax.transAxes)
    plt.legend()
    plt.title('Theoretical vs Elephant vs R')
    plt.show()
