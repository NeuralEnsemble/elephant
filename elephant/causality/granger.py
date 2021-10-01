# -*- coding: utf-8 -*-
"""
.. current_module elephant.causality

Overview
--------
This module provides function to estimate causal influences of signals on each
other.


Granger causality
~~~~~~~~~~~~~~~~~
Granger causality is a method to determine causal influence of one signal on
another based on autoregressive modelling. It was developed by Nobel prize
laureate Clive Granger and has been adopted in various numerical fields ever
since :cite:`granger-Granger69_424`. In its simplest form, the
method tests whether the past values of one signal help to reduce the
prediction error of another signal, compared to the past values of the latter
signal alone. If it does reduce the prediction error, the first signal is said
to Granger cause the other signal.

Limitations
+++++++++++
The user must be mindful of the method's limitations, which are assumptions of
covariance stationary data, linearity imposed by the underlying autoregressive
modelling as well as the fact that the variables not included in the model will
not be accounted for :cite:`granger-Seth07_1667`.

Implementation
++++++++++++++
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
    :toctree: toctree/causality/

    pairwise_granger
    conditional_granger


Tutorial
--------

:doc:`View tutorial <../tutorials/granger_causality>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/granger_causality.ipynb


References
----------

.. bibliography:: ../bib/elephant.bib
   :labelprefix: gr
   :keyprefix: granger-
   :style: unsrt


:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
from collections import namedtuple

import numpy as np
from neo.core import AnalogSignal
from elephant.spectral import multitaper_cross_spectrum, multitaper_psd


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
    coeffs: np.ndarray
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
    '''
    Implementation of the [ \cdot ]^{+} from "The Factorization of Matricial
    Spectral Densities", Wilson 1972, SiAM J Appl Math, Definition 1.2 (ii)

    Paramaters
    ----------

    spectrum : np.ndarray

    '''

    # Get coefficients from spectrum
    causal_part = np.fft.ifft(spectrum, axis=0)
    # Throw away of acausal part
    causal_part[(num_freqs + 1) // 2:] = 0

    # Treat zero frequency part
    causal_part[0] /= 2

    # Back-transformation
    causal_part = np.fft.fft(causal_part, axis=0)

    # Adjust zero frequency part to ensure convergence
    indices = np.tril_indices(num_signals, k=-1)
    causal_part[0, indices[0], indices[1]] = 0
    '''
    ## Version 2
    # Treat zero frequency term
    causal_part[0] /= 2

    # Ensure convergence
    indices = np.tril_indices(num_signals, k=-1)
    causal_part[0, indices[0], indices[1]] = 0

    # Throw away acausal part
    causal_part[(num_freqs + 1) // 2:] = 0

    # Back-transformation
    causal_part = np.fft.fft(causal_part, axis=0)
    '''

    return causal_part


def _dagger(matrix_array):
    '''
    Return Hermitian conjugate of matrix array
    '''

    if matrix_array.ndim == 2:
        return np.transpose(matrix_array.conj(), axes=(1, 0))

    else:
        return np.transpose(matrix_array.conj(), axes=(0, 2, 1))


def _spectral_factorization(cross_spectrum, num_iterations):
    '''
    '''

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
        raise NotImplementedError('ToDo - non converging Cholesky')

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
        if error < 1e-10:
            pass

    cov_matrix = np.matmul(factorization[0].real,
                           np.transpose(factorization[0].real))

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
        Default: 'aic'.

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
        Default: 'aic'.

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


def pairwise_spectral_granger(signals, fs=1, nw=4.0, num_tapers=None,
                              peak_resolution=None, num_iterations=20):

    length = np.size(signals[0])
    signals[0] -= np.mean(signals[0])
    signals[1] -= np.mean(signals[1])

    freqs, _, S = multitaper_cross_spectrum(signals.T,
                                            fs=fs,
                                            nw=nw,
                                            num_tapers=num_tapers,
                                            peak_resolution=peak_resolution,
                                            return_onesided=False)

    C, H = _spectral_factorization(S, num_iterations=num_iterations)

    # Take positive frequencies
    freqs = freqs[:(length+1)//2]
    S = S[:(length+1)//2]
    H = H[:(length+1)//2]

    spectral_granger_y_x = np.log(S[:, 0, 0]
                                  / (S[:, 0, 0]
                                     - (C[1, 1] - C[0, 1]**2/C[0, 0])
                                     * np.abs(H[:, 0, 1])**2))

    spectral_granger_x_y = np.log(S[:, 1, 1]
                                  / (S[:, 1, 1]
                                     - (C[0, 0] - C[1, 0]**2/C[1, 1])
                                     * np.abs(H[:, 1, 0])**2))


    return freqs, spectral_granger_y_x, spectral_granger_x_y

def ding_pairwise_spectral_granger(signals, fs=1, nw=4.0, num_tapers=None,
                                   peak_resolution=None, num_iterations=20):

    length = np.size(signals[0])
    signals[0] -= np.mean(signals[0])
    signals[1] -= np.mean(signals[1])

    freqs, _, S = multitaper_cross_spectrum(signals.T,
                                            fs=fs,
                                            nw=nw,
                                            num_tapers=num_tapers,
                                            peak_resolution=peak_resolution,
                                            return_onesided=False)

    C, H = _spectral_factorization(S, num_iterations=num_iterations)

    # Take positive frequencies
    freqs = freqs[:(length+1)//2]
    S = S[:(length+1)//2]
    H = H[:(length+1)//2]

    H_tilde_xx = H[:, 0, 0] + C[0, 1]/C[0, 0]*H[:, 0, 1]
    H_tilde_yy = H[:, 1, 1] + C[0, 1]/C[1, 1]*H[:, 1, 0]

    granger_y_x = np.log(S[:, 0, 0] /
                                  (H_tilde_xx
                                   * C[0, 0]
                                   * H_tilde_xx.conj()))

    granger_x_y = np.log(S[:, 1, 1] /
                                  (H_tilde_yy
                                   * C[1, 1]
                                   * H_tilde_yy.conj()))

    instantaneous_causality = np.log(
        (H_tilde_xx * C[0, 0] * H_tilde_xx.conj())
        * (H_tilde_yy * C[1, 1] * H_tilde_yy.conj()))
    instantaneous_causality -= np.linalg.slogdet(S)[1]

    total_interdependence = granger_x_y + granger_y_x + instantaneous_causality

    return freqs, granger_y_x, granger_x_y, instantaneous_causality, total_interdependence


if __name__ == '__main__':

    '''

    # Test spectral factorization
    np.random.seed(12321)
    length_2d = 1124
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

    signal = signal[:, 100:]
    length_2d -= 100
    n = length_2d

    x = signal[0]
    y = signal[1]

    f, psd_1 = multitaper_psd(x, num_tapers=15)
    f, psd_2 = multitaper_psd(y, num_tapers=15)

    _, _, cross_spectrum = multitaper_cross_spectrum(signal.T, num_tapers=15)

    from matplotlib import pyplot as plt
    plt.plot(f, psd_1)
    plt.plot(f, 2*cross_spectrum[:(n+2)//2, 0, 0])
    plt.show()

    plt.plot(f, psd_2)
    plt.plot(f, 2*cross_spectrum[:(n+2)//2, 1, 1])
    plt.show()

    cov_matrix, transfer_function = _spectral_factorization(cross_spectrum,
                                                            num_iterations=50)

    A = np.matmul(np.matmul(transfer_function, cov_matrix),
                  _dagger(transfer_function))



    print('################')

    plt.plot(f, cross_spectrum[:(n+2)//2, 0, 0], label='True')
    plt.plot(f, A[:(n+2)//2, 0, 0], label='Mult')
    plt.legend()
    plt.show()

    plt.plot(f, np.real(cross_spectrum[:(n+2)//2, 0, 1]), label='True')
    plt.plot(f, np.real(A[:(n+2)//2, 0, 1]), label='Mult')
    plt.legend()
    plt.show()

    plt.plot(f, np.imag(cross_spectrum[:(n+2)//2, 0, 1]), label='True')
    plt.plot(f, np.imag(A[:(n+2)//2, 0, 1]), label='Mult')
    plt.legend()
    plt.show()

    plt.plot(f, np.real(cross_spectrum[:(n+2)//2, 1, 0]), label='True')
    plt.plot(f, np.real(A[:(n+2)//2, 1, 0]), label='Mult')
    plt.legend()
    plt.show()

    plt.plot(f, np.imag(cross_spectrum[:(n+2)//2, 1, 0]), label='True')
    plt.plot(f, np.imag(A[:(n+2)//2, 1, 0]), label='Mult')
    plt.legend()
    plt.show()

    plt.plot(f, cross_spectrum[:(n+2)//2, 1, 1], label='True')
    plt.plot(f, A[:(n+2)//2, 1, 1], label='Mult')
    plt.legend()
    plt.show()
    '''

    # Test spectral granger
    xy = []
    yx = []
    ding_xy = []
    ding_yx = []
    ding_inst = []
    ding_tot = []
    psd_x = []
    psd_y = []

<<<<<<< HEAD
    for i in range(300):
=======
    for i in range(50):
>>>>>>> d371fe38a90acd616f398b2eb43dab41f404e0e8
        np.random.seed(i**2+134)
        length_2d = 1124
        signal = np.zeros((2, length_2d))

        order = 2
        weights_1 = np.array([[0.9, 0], [0.16, 0.8]]).T
        weights_2 = np.array([[-0.5, 0], [-0.2, -0.5]]).T

        weights = np.stack((weights_1, weights_2))

<<<<<<< HEAD
        noise_covariance = np.array([[1., 0.4], [0.4, 0.7]])
=======
        noise_covariance = np.array([[1., 0.], [0., 0.7]])
>>>>>>> d371fe38a90acd616f398b2eb43dab41f404e0e8

        for i in range(length_2d):
            for lag in range(order):
                signal[:, i] += np.dot(weights[lag],
                                       signal[:, i - lag - 1])
            rnd_var = np.random.multivariate_normal([0, 0], noise_covariance)
            signal[0, i] += rnd_var[0]
            signal[1, i] += rnd_var[1]

        signal = signal[:, 100:]
        length_2d -= 100

        f, _, cross_spec = multitaper_cross_spectrum(signal.T, num_tapers=15)
        psd_x.append(cross_spec[:(length_2d+1)//2, 0, 0])
        psd_y.append(cross_spec[:(length_2d+1)//2, 1, 1])

        f, y_x, x_y = pairwise_spectral_granger(signal, num_tapers=15,
                                                num_iterations=50)

        f, ding_y_x, ding_x_y, inst, tot = ding_pairwise_spectral_granger(signal, num_tapers=15,
                                                               num_iterations=50)

        xy.append(x_y)
        yx.append(y_x)

        ding_xy.append(ding_x_y)
        ding_yx.append(ding_y_x)
        ding_inst.append(inst)
        ding_tot.append(tot)

    xy = np.array(xy)
    yx = np.array(yx)

    x_y = np.mean(xy, axis=0)
    y_x = np.mean(yx, axis=0)

    ding_xy = np.array(ding_xy)
    ding_yx = np.array(ding_yx)
    ding_inst = np.array(ding_inst)
    ding_tot = np.array(ding_tot)

    ding_x_y = np.mean(ding_xy, axis=0)
    ding_y_x = np.mean(ding_yx, axis=0)
    ding_inst = np.mean(ding_inst, axis=0)
    ding_tot = np.mean(ding_tot, axis=0)

    psd_x = np.array(psd_x)
    psd_y = np.array(psd_y)

    psd_x = np.mean(psd_x, axis=0)
    psd_y = np.mean(psd_y, axis=0)

    from matplotlib import pyplot as plt

    #plt.plot(f, 2*cross_spec[:, 0, 0], label='1')
    #plt.plot(f, 2*cross_spec[:, 1, 1], label='2')
<<<<<<< HEAD
    plt.plot(f, psd_x, label='1')
    plt.plot(f, psd_y, label='2')
    #plt.legend()
    plt.show()
=======
    #plt.plot(f, psd_x, label='1')
    #plt.plot(f, psd_y, label='2')
    #plt.legend()
    #plt.show()
>>>>>>> d371fe38a90acd616f398b2eb43dab41f404e0e8


    plt.plot(f, y_x, label='y->x')
    plt.plot(f, x_y, label='x->y')
    plt.plot(f, ding_y_x, label='y->x')
    plt.plot(f, ding_x_y, label='x->y')
    plt.legend()
    plt.show()

    plt.plot(f, ding_y_x, label='y->x')
    plt.plot(f, ding_x_y, label='x->y')
    plt.plot(f, ding_inst, label='inst')
    plt.plot(f, ding_tot, label='tot')
    plt.legend()
    plt.show()
