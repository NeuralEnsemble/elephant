# -*- coding: utf-8 -*-
"""
Unitary Event (UE) analysis is a statistical method to analyze in a time
resolved manner excess spike correlation between simultaneously recorded
neurons by comparing the empirical spike coincidences (precision of a few ms)
to the expected number based on the firing rates of the neurons
(see :cite:`unitary_event_analysis-Gruen99_67`).

Background
----------

It has been proposed that cortical neurons organize dynamically into functional
groups (“cell assemblies”) by the temporal structure of their joint spiking
activity. The Unitary Events analysis method detects conspicuous patterns of
synchronous spike activity among simultaneously recorded single neurons. The
statistical significance of a pattern is evaluated by comparing the empirical
number of occurrences to the number expected given the firing rates of the
neurons. Key elements of the method are the proper formulation of the null
hypothesis and the derivation of the corresponding count distribution of
synchronous spike events used in the significance test. The analysis is
performed in a sliding window manner and yields a time-resolved measure of
significant spike synchrony. For further reading, see
:cite:`unitary_event_analysis-Riehle97_1950,unitary_event_analysis-Gruen02_43,\
unitary_event_analysis-Gruen02_81,unitary_event_analysis-Gruen03,\
unitary_event_analysis-Gruen09_1126,unitary_event_analysis-Gruen99_67`.


Tutorial
--------

:doc:`View tutorial <../tutorials/unitary_event_analysis>`

Run tutorial interactively:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/NeuralEnsemble/elephant/master
            ?filepath=doc/tutorials/unitary_event_analysis.ipynb


Functions overview
------------------

.. autosummary::
    :toctree: _toctree/unitary_event_analysis/

    jointJ_window_analysis

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
from collections import defaultdict

import neo
import numpy as np
import quantities as pq
import scipy

import elephant.conversion as conv
from elephant.utils import is_binary, deprecated_alias

__all__ = [
    "hash_from_pattern",
    "inverse_hash_from_pattern",
    "n_emp_mat",
    "n_emp_mat_sum_trial",
    "n_exp_mat",
    "n_exp_mat_sum_trial",
    "gen_pval_anal",
    "jointJ",
    "jointJ_window_analysis"
]


def hash_from_pattern(m, base=2):
    """
    Calculate for a spike pattern or a matrix of spike patterns
    (provide each pattern as a column) composed of N neurons a
    unique number.


    Parameters
    ----------
    m: np.ndarray or list
        2-dim ndarray
        spike patterns represented as a binary matrix (i.e., matrix of 0's and
        1's).
        Rows and columns correspond to patterns and neurons, respectively.
    base: integer
        The base for hashes calculation.
        Default: 2

    Returns
    -------
    np.ndarray
        An array containing the hash values of each pattern,
        shape: (number of patterns).

    Raises
    ------
    ValueError
        If matrix `m` has wrong orientation.

    Examples
    --------
    With `base=2`, the hash of `[0, 1, 1]` is `0*2^2 + 1*2^1 + 1*2^0 = 3`.

    >>> import numpy as np
    >>> hash_from_pattern([0, 1, 1])
    3

    >>> import numpy as np
    >>> m = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
    ...               [0, 0, 1, 0, 1, 0, 1, 1],
    ...               [0, 0, 0, 1, 0, 1, 1, 1]])

    >>> hash_from_pattern(m)
    array([0, 4, 2, 1, 6, 5, 3, 7])

    """
    m = np.asarray(m)
    n_neurons = m.shape[0]

    # check the entries of the matrix
    if not is_binary(m):
        raise ValueError('Patterns should be binary: 0 or 1')

    # generate the representation
    # don't use numpy - it's upperbounded by int64
    powers = [base ** x for x in range(n_neurons)][::-1]

    # calculate the binary number by use of scalar product
    return np.dot(powers, m)


def inverse_hash_from_pattern(h, N, base=2):
    """
    Calculate the binary spike patterns (matrix) from hash values `h`.

    Parameters
    ----------
    h: list of int
        Array-like of integer hash values of length of the number of patterns.
    N: integer
        The number of neurons.
    base: integer
        The base, used to generate the hash values.
        Default: 2

    Returns
    -------
    m: (N, P) np.ndarray
       A matrix of shape: (N, number of patterns)

    Raises
    ------
    ValueError
        If the hash is not compatible with the number of neurons.
        The hash value should not be larger than the largest
        possible hash number with the given number of neurons
        (e.g. for N = 2, max(hash) = 2^1 + 2^0 = 3, or for N = 4,
        max(hash) = 2^3 + 2^2 + 2^1 + 2^0 = 15).

    Examples
    ---------
    >>> import numpy as np
    >>> h = np.array([3, 7])
    >>> N = 4
    >>> inverse_hash_from_pattern(h, N)
    array([[1, 1],
        [1, 1],
        [0, 1],
        [0, 0]])

    """
    h = np.asarray(h)  # this will cast to object type if h > int64
    if not all(isinstance(v, int) for v in h.tolist()):
        # .tolist() is necessary because np.int[64] is not int
        raise ValueError("hash values should be integers")

    # check if the hash values are not greater than the greatest possible
    # value for N neurons with the given base
    powers = np.array([base ** x for x in range(N)])[::-1]
    if any(h > sum(powers)):
        raise ValueError(f"hash value {h} is not compatible with the number "
                         f"of neurons {N}")
    m = h // np.expand_dims(powers, axis=1)
    m %= base  # m is a binary matrix now
    m = m.astype(int)  # convert object to int if the hash was > int64
    return m


def n_emp_mat(mat, pattern_hash, base=2):
    """
    Count the occurrences of spike coincidence patterns in the given spike
    trains.

    Parameters
    ----------
    mat : (N, M) np.ndarray
        Binned spike trains of N neurons. Rows and columns correspond
        to neurons and temporal bins, respectively.
    pattern_hash: list of int
        List of hash values, representing the spike coincidence patterns
        of which occurrences are counted.
    base: integer
        The base, used to generate the hash values.
        Default: 2

    Returns
    -------
    N_emp: np.ndarray
        The number of occurrences of the given patterns in the given
        spiketrains.
    indices: list of list
        List of lists of int.
        Indices indexing the bins where the given spike patterns are found
        in `mat`. Same length as `pattern_hash`.
        `indices[i] = N_emp[i] = pattern_hash[i]`

    Raises
    ------
    ValueError
        If `mat` is not a binary matrix.

    Examples
    --------
    >>> mat = np.array([[1, 0, 0, 1, 1],
    ...                 [1, 0, 0, 1, 0]])
    >>> pattern_hash = np.array([1,3])
    >>> n_emp, n_emp_indices = n_emp_mat(mat, pattern_hash)
    >>> print(n_emp)
    [ 0.  2.]
    >>> print(n_emp_indices)
    [array([]), array([0, 3])]

    """
    # check if the mat is zero-one matrix
    if not is_binary(mat):
        raise ValueError("entries of mat should be either one or zero")
    h = hash_from_pattern(mat, base=base)
    N_emp = np.zeros(len(pattern_hash))
    indices = []
    for idx_ph, ph in enumerate(pattern_hash):
        indices_tmp = np.where(h == ph)[0]
        indices.append(indices_tmp)
        N_emp[idx_ph] = len(indices_tmp)
    return N_emp, indices


def n_emp_mat_sum_trial(mat, pattern_hash):
    """
    Calculate empirical number of observed patterns, summed across trials.

    Parameters
    ----------
    mat: np.ndarray
        Binned spike trains are represented as a binary matrix (i.e., matrix of
        0's and 1's), segmented into trials. Trials should contain an identical
        number of neurons and an identical number of time bins.
         the entries are zero or one
         0-axis --> trials
         1-axis --> neurons
         2-axis --> time bins
    pattern_hash: list of int
         Array of hash values of length of the number of patterns.

    Returns
    -------
    N_emp: np.ndarray
        The number of occurences of the given spike patterns in the given spike
        trains, summed across trials. Same length as `pattern_hash`.
    idx_trials: list of int
        List of indices of `mat` for each trial in which the specific pattern
        has been observed.
        0-axis --> trial
        1-axis --> list of indices for the chosen trial per entry of
                    `pattern_hash`

    Raises
    ------
    ValueError
        If `mat` has the wrong orientation.
        If `mat` is not a binary matrix.

    Examples
    ---------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
    ...                  [0, 1, 1, 1, 0],
    ...                  [0, 1, 1, 0, 1]],
    ...                 [[1, 1, 1, 1, 1],
    ...                  [0, 1, 1, 1, 1],
    ...                  [1, 1, 0, 1, 0]]])
    >>> pattern_hash = np.array([4,6])
    >>> n_emp_sum_trial, n_emp_sum_trial_idx = \
    ...                   n_emp_mat_sum_trial(mat, pattern_hash)
    >>> n_emp_sum_trial
    array([ 1.,  3.])
    >>> n_emp_sum_trial_idx
    [[array([0]), array([3])], [array([], dtype=int64), array([2, 4])]]

    """
    num_patt = len(pattern_hash)
    N_emp = np.zeros(num_patt)

    idx_trials = []
    # check if the mat is zero-one matrix
    if not is_binary(mat):
        raise ValueError("entries of mat should be either one or zero")

    for mat_tr in mat:
        N_emp_tmp, indices_tmp = n_emp_mat(mat_tr, pattern_hash, base=2)
        idx_trials.append(indices_tmp)
        N_emp += N_emp_tmp

    return N_emp, idx_trials


def _n_exp_mat_analytic(mat, pattern_hash):
    """
    Calculates the expected joint probability for each spike pattern
    analytically.
    """
    marg_prob = np.mean(mat, 1, dtype=float)
    # marg_prob needs to be a column vector, so we
    # build a two dimensional array with 1 column
    # and len(marg_prob) rows
    marg_prob = np.expand_dims(marg_prob, axis=1)
    n_neurons = mat.shape[0]
    m = inverse_hash_from_pattern(pattern_hash, n_neurons)
    nrep = m.shape[1]
    # multipyling the marginal probability of neurons with regard to the
    # pattern
    pmat = np.multiply(m, np.tile(marg_prob, (1, nrep))) + \
        np.multiply(1 - m, np.tile(1 - marg_prob, (1, nrep)))
    return np.prod(pmat, axis=0) * float(mat.shape[1])


def _n_exp_mat_surrogate(mat, pattern_hash, n_surrogates=1):
    """
    Calculates the expected joint probability for each spike pattern with spike
    time randomization surrogate
    """
    if len(pattern_hash) > 1:
        raise ValueError('surrogate method works only for one pattern!')
    N_exp_array = np.zeros(n_surrogates)
    for rz_idx, rz in enumerate(np.arange(n_surrogates)):
        # row-wise shuffling all elements of zero-one matrix
        mat_surr = np.copy(mat)
        for row in mat_surr:
            np.random.shuffle(row)
        N_exp_array[rz_idx] = n_emp_mat(mat_surr, pattern_hash)[0][0]
    return N_exp_array


def n_exp_mat(mat, pattern_hash, method='analytic', n_surrogates=1):
    """
    Calculates the expected joint probability for each spike pattern.

    Parameters
    ----------
    mat: np.ndarray
         The entries are in the range [0, 1].
         The only possibility when the entries are floating point values is
         when the `mat` is calculated with the flag `analytic_TrialAverage`
         in `n_exp_mat_sum_trial()`.
         Otherwise, the entries are binary.
         0-axis --> neurons
         1-axis --> time bins
    pattern_hash: list of int
         List of hash values, length: number of patterns
    method: {'analytic', 'surr'}, optional
         The method with which the expectation is calculated.
         'analytic' -- > analytically
         'surr' -- > with surrogates (spike time randomization)
         Default: 'analytic'
    n_surrogates: int
         number of surrogates for constructing the distribution of expected
         joint probability.
         Default: 1 and this number is needed only when method = 'surr'

    Returns
    -------
    np.ndarray
        if method is 'analytic':
            An array containing the expected joint probability of each pattern,
            shape: (number of patterns,)
        if method is 'surr':
            0-axis --> different realizations, length = number of surrogates
            1-axis --> patterns

    Raises
    ------
    ValueError
        If `mat` has the wrong orientation.

    Examples
    --------
    >>> mat = np.array([[1, 1, 1, 1],
    ...                 [0, 1, 0, 1],
    ...                 [0, 0, 1, 0]])
    >>> pattern_hash = np.array([5,6])
    >>> n_exp_anal = n_exp_mat(mat, pattern_hash, method='analytic')
    >>> n_exp_anal
    [ 0.5 1.5 ]
    >>> n_exp_surr = n_exp_mat(mat, pattern_hash, method='surr',
    ...                        n_surrogates=5000)
    >>> print(n_exp_surr)
    [[ 1.  1.]
     [ 2.  0.]
     [ 2.  0.]
     ...,
     [ 2.  0.]
     [ 2.  0.]
     [ 1.  1.]]

    """
    # check if the mat is in the range [0, 1]
    if not np.all((mat >= 0) & (mat <= 1)):
        raise ValueError("entries of mat should be in range [0, 1]")

    if method == 'analytic':
        return _n_exp_mat_analytic(mat, pattern_hash)
    if method == 'surr':
        return _n_exp_mat_surrogate(mat, pattern_hash,
                                    n_surrogates=n_surrogates)


def n_exp_mat_sum_trial(mat, pattern_hash, method='analytic_TrialByTrial',
                        n_surrogates=1):
    """
    Calculates the expected joint probability for each spike pattern sum over
    trials.

    Parameters
    ----------
    mat: np.ndarray
        Binned spike trains represented as a binary matrix (i.e., matrix of
        0's and 1's), segmented into trials. Trials should contain an identical
        number of neurons and an identical number of time bins.
        The entries of mat should be a list of a list where 0-axis is trials
        and 1-axis is neurons.
         0-axis --> trials
         1-axis --> neurons
         2-axis --> time bins
    pattern_hash: list of int
         List of hash values, length: number of patterns
    method: str
         method with which the unitary events whould be computed
         'analytic_TrialByTrial' -- > calculate the expectency
         (analytically) on each trial, then sum over all trials.
         'analytic_TrialAverage' -- > calculate the expectency
         by averaging over trials.
         (cf. Gruen et al. 2003)
         'surrogate_TrialByTrial' -- > calculate the distribution
         of expected coincidences by spike time randomzation in
         each trial and sum over trials.
         Default: 'analytic_trialByTrial'.
    n_surrogates: int, optional
         The number of surrogate to be used.
         Default: 1

    Returns
    -------
    n_exp: np.ndarray
         An array containing the expected joint probability of
         each pattern summed over trials,shape: (number of patterns,)

    Raises
    ------
    ValueError
        If `method` is not one of the specified above.

    Examples
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
    ...                  [0, 1, 1, 1, 0],
    ...                  [0, 1, 1, 0, 1]],
    ...                 [[1, 1, 1, 1, 1],
    ...                  [0, 1, 1, 1, 1],
    ...                  [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([5,6])
    >>> n_exp_anal = n_exp_mat_sum_trial(mat, pattern_hash)
    >>> print(n_exp_anal)
        array([ 1.56,  2.56])
    """
    if method == 'analytic_TrialByTrial':
        n_exp = np.zeros(len(pattern_hash))
        for mat_tr in mat:
            n_exp += n_exp_mat(mat_tr, pattern_hash,
                               method='analytic')
    elif method == 'analytic_TrialAverage':
        n_exp = n_exp_mat(
            np.mean(mat, axis=0), pattern_hash,
            method='analytic') * mat.shape[0]
    elif method == 'surrogate_TrialByTrial':
        n_exp = np.zeros(n_surrogates)
        for mat_tr in mat:
            n_exp += n_exp_mat(mat_tr, pattern_hash,
                               method='surr', n_surrogates=n_surrogates)
    else:
        raise ValueError(
            "The method only works on the zero_one matrix at the moment")
    return n_exp


def gen_pval_anal(mat, pattern_hash, method='analytic_TrialByTrial',
                  n_surrogates=1):
    """
    Compute the expected coincidences and a function to calculate the
    p-value for the given empirical coincidences.

    This function generates a poisson distribution with the expected
    value calculated by `mat`. It returns a function that gets
    the empirical coincidences, `n_emp`, and calculates a p-value
    as the area under the poisson distribution from `n_emp` to infinity.

    Parameters
    ----------
    mat: np.ndarray
        Binned spike trains represented as a binary matrix (i.e., matrix of
        0's and 1's), segmented into trials. Trials should contain an identical
        number of neurons and an identical number of time bins.
        The entries of mat should be a list of a list where 0-axis is trials
        and 1-axis is neurons.
         0-axis --> trials
         1-axis --> neurons
         2-axis --> time bins
    pattern_hash: list of int
         List of hash values, length: number of patterns
    method: string
         method with which the unitary events whould be computed
         'analytic_TrialByTrial' -- > calculate the expectency
         (analytically) on each trial, then sum over all trials.
         ''analytic_TrialAverage' -- > calculate the expectency
         by averaging over trials.
         Default: 'analytic_trialByTrial'
         (cf. Gruen et al. 2003)
    n_surrogates: integer, optional
         number of surrogate to be used
         Default: 1

    Returns
    --------
    pval_anal: callable
         The function that calculates the p-value for the given empirical
         coincidences.
    n_exp: list
        List of expected coincidences.

    Raises
    ------
    ValueError
        If `method` is not one of the specified above.

    Examples
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
    ...                  [0, 1, 1, 1, 0],
    ...                  [0, 1, 1, 0, 1]],
    ...                 [[1, 1, 1, 1, 1],
    ...                  [0, 1, 1, 1, 1],
    ...                  [1, 1, 0, 1, 0]]])
    >>> pattern_hash = np.array([5, 6])
    >>> pval_anal, n_exp = gen_pval_anal(mat, pattern_hash)
    >>> n_exp
    array([ 1.56,  2.56])

    """
    if method == 'analytic_TrialByTrial' or method == 'analytic_TrialAverage':
        n_exp = n_exp_mat_sum_trial(mat, pattern_hash, method=method)

        def pval(n_emp):
            p = 1. - scipy.special.gammaincc(n_emp, n_exp)
            return p
    elif method == 'surrogate_TrialByTrial':
        n_exp = n_exp_mat_sum_trial(
            mat, pattern_hash, method=method, n_surrogates=n_surrogates)

        def pval(n_emp):
            hist = np.bincount(np.int64(n_exp))
            exp_dist = hist / float(np.sum(hist))
            if len(n_emp) > 1:
                raise ValueError('In surrogate method the p_value can be'
                                 'calculated only for one pattern!')
            return np.sum(exp_dist[int(n_emp[0]):])
    else:
        raise ValueError("Method is not allowed: {method}".format(
            method=method))

    return pval, n_exp


def jointJ(p_val):
    """
    Surprise measurement.

    Logarithmic transformation of joint-p-value into surprise measure
    for better visualization as the highly significant events are
    indicated by very low joint-p-values.

    Parameters
    ----------
    p_val : float or list of float
        List of p-values of statistical tests for different pattern.

    Returns
    -------
    Js: list of float
        List of surprise measures.

    Examples
    --------
    >>> p_val = np.array([0.31271072,  0.01175031])
    >>> jointJ(p_val)
    array([0.3419968 ,  1.92481736])

    """
    p_arr = np.asarray(p_val)
    with np.errstate(divide='ignore'):
        # Ignore 'Division by zero' warning which happens when p_arr is 1.0 or
        # 0.0 (no spikes).
        Js = np.log10(1 - p_arr) - np.log10(p_arr)
    return Js


def _rate_mat_avg_trial(mat):
    """
    Calculates the average firing rate of each neurons across trials.
    """
    n_trials, n_neurons, n_bins = np.shape(mat)
    psth = np.zeros(n_neurons, dtype=np.float32)
    for tr, mat_tr in enumerate(mat):
        psth += np.sum(mat_tr, axis=1)
    return psth / (n_bins * n_trials)


def _bintime(t, bin_size):
    """
    Change the real time to `bin_size` units.
    """
    t_dl = t.rescale('ms').magnitude
    bin_size_dl = bin_size.rescale('ms').item()
    return np.floor(np.array(t_dl) / bin_size_dl).astype(int)


@deprecated_alias(winsize='win_size', winstep='win_step')
def _winpos(t_start, t_stop, win_size, win_step, position='left-edge'):
    """
    Calculate the position of the analysis window.
    """
    t_start_dl = t_start.rescale('ms').item()
    t_stop_dl = t_stop.rescale('ms').item()
    winsize_dl = win_size.rescale('ms').item()
    winstep_dl = win_step.rescale('ms').item()

    # left side of the window time
    if position == 'left-edge':
        ts_winpos = np.arange(
            t_start_dl, t_stop_dl - winsize_dl + winstep_dl,
            winstep_dl) * pq.ms
    else:
        raise ValueError(
            'the current version only returns left-edge of the window')
    return ts_winpos


def _UE(mat, pattern_hash, method='analytic_TrialByTrial', n_surrogates=1):
    """
    Return the default results of unitary events analysis
    (Surprise, empirical coincidences and index of where it happened
    in the given mat, n_exp and average rate of neurons)
    """
    rate_avg = _rate_mat_avg_trial(mat)
    n_emp, indices = n_emp_mat_sum_trial(mat, pattern_hash)
    if method == 'surrogate_TrialByTrial':
        dist_exp, n_exp = gen_pval_anal(
            mat, pattern_hash, method, n_surrogates=n_surrogates)
        n_exp = np.mean(n_exp)
    elif method == 'analytic_TrialByTrial' or \
            method == 'analytic_TrialAverage':
        dist_exp, n_exp = gen_pval_anal(mat, pattern_hash, method)
    pval = dist_exp(n_emp)
    Js = jointJ(pval)
    return Js, rate_avg, n_exp, n_emp, indices


@deprecated_alias(data='spiketrains', binsize='bin_size', winsize='win_size',
                  winstep='win_step', n_surr='n_surrogates')
def jointJ_window_analysis(spiketrains, bin_size=5 * pq.ms,
                           win_size=100 * pq.ms, win_step=5 * pq.ms,
                           pattern_hash=None, method='analytic_TrialByTrial',
                           t_start=None, t_stop=None, binary=True,
                           n_surrogates=100):
    """
    Calculates the joint surprise in a sliding window fashion.

    Implementation is based on :cite:`unitary_event_analysis-Gruen99_67`.

    Parameters
    ----------
    spiketrains : list
        A list of spike trains (`neo.SpikeTrain` objects) in different trials:
          * 0-axis --> Trials

          * 1-axis --> Neurons

          * 2-axis --> Spike times
    bin_size : pq.Quantity, optional
        The size of bins for discretizing spike trains.
        Default: 5 ms
    win_size : pq.Quantity, optional
        The size of the window of analysis.
        Default: 100 ms
    win_step : pq.Quantity, optional
        The size of the window step.
        Default: 5 ms
    pattern_hash : int or list of int or None, optional
        A list of interested patterns in hash values (see `hash_from_pattern`
        and `inverse_hash_from_pattern` functions). If None, all neurons
        are participated.
        Default: None
    method : str, optional
        The method with which to compute the unitary events:
          * 'analytic_TrialByTrial': calculate the analytical expectancy
            on each trial, then sum over all trials;

          * 'analytic_TrialAverage': calculate the expectancy by averaging over
            trials (cf. Gruen et al. 2003);

          * 'surrogate_TrialByTrial': calculate the distribution of expected
            coincidences by spike time randomization in each trial and sum over
            trials.
        Default: 'analytic_trialByTrial'
    t_start, t_stop : float or pq.Quantity, optional
        The start and stop times to use for the time points.
        If not specified, retrieved from the `t_start` and `t_stop` attributes
        of the input spiketrains.
    binary : bool, optional
        Binarize the binned spike train objects (True) or not. Only the binary
        matrices are supported at the moment.
        Default: True
    n_surrogates : int, optional
        The number of surrogates to be used.
        Default: 100

    Returns
    -------
    dict
        The values of the following keys have the shape of

          * different window --> 0-axis
          * different pattern hash --> 1-axis

        'Js': list of float
          JointSurprise of different given patterns within each window.
        'indices': list of list of int
          A list of indices of pattern within each window.
        'n_emp': list of int
          The empirical number of each observed pattern.
        'n_exp': list of float
          The expected number of each pattern.
        'rate_avg': list of float
          The average firing rate of each neuron.

        Additionally, 'input_parameters' key stores the input parameters.

    Raises
    ------
    ValueError
        If `data` is not in the format, specified above.
    NotImplementedError
        If `binary` is not True. The method works only with binary matrices at
        the moment.

    Warns
    -----
    UserWarning
        The ratio between `winsize` or `winstep` and `bin_size` is not an
        integer.

    """
    if not isinstance(spiketrains[0][0], neo.SpikeTrain):
        raise ValueError(
            "structure of the data is not correct: 0-axis should be trials, "
            "1-axis units and 2-axis neo spike trains")

    if t_start is None:
        t_start = spiketrains[0][0].t_start
    if t_stop is None:
        t_stop = spiketrains[0][0].t_stop

    n_trials = len(spiketrains)
    n_neurons = len(spiketrains[0])
    if pattern_hash is None:
        pattern = [1] * n_neurons
        pattern_hash = hash_from_pattern(pattern)
    if np.issubdtype(type(pattern_hash), np.integer):
        pattern_hash = [int(pattern_hash)]

    # position of all windows (left edges)
    t_winpos = _winpos(t_start, t_stop, win_size, win_step,
                       position='left-edge')
    t_winpos_bintime = _bintime(t_winpos, bin_size)

    winsize_bintime = _bintime(win_size, bin_size)
    winstep_bintime = _bintime(win_step, bin_size)

    if winsize_bintime * bin_size != win_size:
        warnings.warn(f"The ratio between the win_size ({win_size}) and the "
                      f"bin_size ({bin_size}) is not an integer")

    if winstep_bintime * bin_size != win_step:
        warnings.warn(f"The ratio between the win_step ({win_step}) and the "
                      f"bin_size ({bin_size}) is not an integer")

    input_parameters = dict(pattern_hash=pattern_hash, bin_size=bin_size,
                            win_size=win_size, win_step=win_step,
                            method=method, t_start=t_start, t_stop=t_stop,
                            n_surrogates=n_surrogates)

    n_bins = int(((t_stop - t_start) / bin_size).simplified.item())

    mat_tr_unit_spt = np.zeros((len(spiketrains), n_neurons, n_bins),
                               dtype=np.int32)
    for trial, sts in enumerate(spiketrains):
        bs = conv.BinnedSpikeTrain(list(sts), t_start=t_start, t_stop=t_stop,
                                   bin_size=bin_size)
        if not binary:
            raise NotImplementedError(
                "The method works only with binary matrices at the moment")
        mat_tr_unit_spt[trial] = bs.to_bool_array()

    n_windows = len(t_winpos)
    n_hashes = len(pattern_hash)
    Js_win, n_exp_win, n_emp_win = np.zeros((3, n_windows, n_hashes),
                                            dtype=np.float32)
    rate_avg = np.zeros((n_windows, n_hashes, n_neurons), dtype=np.float32)
    indices_win = defaultdict(list)

    for i, win_pos in enumerate(t_winpos_bintime):
        mat_win = mat_tr_unit_spt[:, :, win_pos:win_pos + winsize_bintime]
        Js_win[i], rate_avg[i], n_exp_win[i], n_emp_win[
            i], indices_lst = _UE(mat_win, pattern_hash=pattern_hash,
                                  method=method, n_surrogates=n_surrogates)
        for j in range(n_trials):
            if len(indices_lst[j][0]) > 0:
                indices_win[f"trial{j}"].append(indices_lst[j][0] + win_pos)
    for key in indices_win.keys():
        indices_win[key] = np.hstack(indices_win[key])
    return {'Js': Js_win, 'indices': indices_win, 'n_emp': n_emp_win,
            'n_exp': n_exp_win, 'rate_avg': rate_avg / bin_size,
            'input_parameters': input_parameters}
