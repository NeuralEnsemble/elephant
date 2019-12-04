# -*- coding: utf-8 -*-
"""
Unitary Event (UE) analysis is a statistical method that
 enables to analyze in a time resolved manner excess spike correlation
 between simultaneously recorded neurons by comparing the empirical
 spike coincidences (precision of a few ms) to the expected number
 based on the firing rates of the neurons.

References:
  - Gruen, Diesmann, Grammont, Riehle, Aertsen (1999) J Neurosci Methods,
    94(1): 67-79.
  - Gruen, Diesmann, Aertsen (2002a,b) Neural Comput, 14(1): 43-80; 81-19.
  - Gruen S, Riehle A, and Diesmann M (2003) Effect of cross-trial
    nonstationarity on joint-spike events Biological Cybernetics 88(5):335-351.
  - Gruen S (2009) Data-driven significance estimation of precise spike
    correlation. J Neurophysiology 101:1126-1140 (invited review)

:copyright: Copyright 2015-2016 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sys
import warnings
from functools import wraps

import neo
import numpy as np
import quantities as pq
import scipy

import elephant.conversion as conv
from elephant.utils import is_binary


def decorate_deprecated_N(func):
    @wraps(func)
    def decorated_func(*args, **kwargs):
        N = None
        if 'N' in kwargs:
            N = kwargs.pop('N')
        elif len(args) > 1 and isinstance(args[1], int):
            args = list(args)
            N = args.pop(1)
        if N is not None:
            warnings.warn("'N' is deprecated in '{func_name}' and will be "
                          "removed in the next Elephant release. Now 'N' is "
                          "extracted from the data shape.".format(
                            func_name=func.__name__), DeprecationWarning)
        return func(*args, **kwargs)

    return decorated_func


@decorate_deprecated_N
def hash_from_pattern(m, base=2):
    """
    Calculate for a spike pattern or a matrix of spike patterns
    (provide each pattern as a column) composed of N neurons a
    unique number.


    Parameters
    -----------
    m: np.ndarray
        2-dim ndarray
        spike patterns represented as a binary matrix (i.e., matrix of 0's and
        1's).
        Rows and columns correspond to patterns and neurons, respectively.
    base: integer
        base for calculation of hash values from binary
        sequences (= pattern).
        Default is 2

    Returns
    --------
    np.ndarray
        An array containing the hash values of each pattern,
        shape: (number of patterns).

    Raises
    -------
    ValueError
        if matrix `m` has wrong orientation

    Examples
    ---------
    descriptive example:
    m = [0
         1
         1]
    N = 3
    base = 2
    hash = 0*2^2 + 1*2^1 + 1*2^0 = 3

    second example:
    >>> import numpy as np
    >>> m = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
    >>>               [0, 0, 1, 0, 1, 0, 1, 1],
    >>>               [0, 0, 0, 1, 0, 1, 1, 1]])

    >>> hash_from_pattern(m)
        array([0, 4, 2, 1, 6, 5, 3, 7])
    """
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
    Calculate the 0-1 spike patterns (matrix) from hash values

    Parameters
    -----------
    h: list
        list or array of integer hash values, length: number of patterns
    N: integer
        number of neurons
    base: integer
        base for calculation of the number from binary
        sequences (= pattern).
        Default is 2

    Raises
    -------
       ValueError: if the hash is not compatible with the number
       of neurons hash value should not be larger than the biggest
       possible hash number with given number of neurons
       (e.g. for N = 2, max(hash) = 2^1 + 2^0 = 3
         , or for N = 4, max(hash) = 2^3 + 2^2 + 2^1 + 2^0 = 15)

    Returns
    --------
    m: np.ndarray
       A matrix of shape: (N, number of patterns)

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
    if sys.version_info < (3,):
        integer_types = (int, long)
    else:
        integer_types = (int,)
    if not all(isinstance(v, integer_types) for v in h.tolist()):
        # .tolist() is necessary because np.int[64] is not int
        raise ValueError("hash values should be integers")

    # check if the hash values are not greater than the greatest possible
    # value for N neurons with the given base
    powers = np.array([base ** x for x in range(N)])[::-1]
    if any(h > sum(powers)):
        raise ValueError(
            "hash value is not compatible with the number of neurons N")
    m = h // np.expand_dims(powers, axis=1)
    m %= base  # m is a binary matrix now
    m = m.astype(int)  # convert object to int if the hash was > int64
    return m


@decorate_deprecated_N
def n_emp_mat(mat, pattern_hash, base=2):
    """
    Count the occurrences of spike coincidence patterns
    in the given spike trains.

    Parameters
    -----------
    mat : np.ndarray
        2-dim ndarray
        binned spike trains of N neurons. Rows and columns correspond
        to neurons and temporal bins, respectively.
    pattern_hash: list
        List of hash values, representing the spike coincidence patterns
        of which occurrences are counted.
    base: integer
        Base which was used to generate the hash values.
        Default is 2

    Returns
    --------
    N_emp: np.ndarray
        number of occurrences of the given patterns in the given spike trains
    indices: list
        list of lists of integers
        indices indexing the bins where the given spike patterns are found
        in `mat`. Same length as `pattern_hash`
        indices[i] = N_emp[i] = pattern_hash[i]

    Raises
    -------
    ValueError
        If mat is not zero-one matrix.

    Examples
    ---------
    >>> mat = np.array([[1, 0, 0, 1, 1],
    >>>                 [1, 0, 0, 1, 0]])
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


@decorate_deprecated_N
def n_emp_mat_sum_trial(mat, pattern_hash):
    """
    Calculates empirical number of observed patterns summed across trials

    Parameters
    -----------
    mat: np.ndarray
        3d numpy array or elephant BinnedSpikeTrain object
        Binned spike trains represented as a binary matrix (i.e., matrix of
        0's and 1's), segmented into trials. Trials should contain an identical
        number of neurons and an identical number of time bins.
         the entries are zero or one
         0-axis --> trials
         1-axis --> neurons
         2-axis --> time bins
    pattern_hash: list
         Array of hash values, length: number of patterns.

    Returns
    --------
    N_emp: np.ndarray
        numbers of occurences of the given spike patterns in the given spike
        trains, summed across trials. Same length as `pattern_hash`.
    idx_trials: list
        list of indices of mat for each trial in which
        the specific pattern has been observed.
        0-axis --> trial
        1-axis --> list of indices for the chosen trial per
        entry of `pattern_hash`

    Raises
    -------
       ValueError: if matrix mat has wrong orientation
       ValueError: if mat is not zero-one matrix

    Examples
    ---------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 0],
                 [0, 1, 1, 0, 1]],

                 [[1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [1, 1, 0, 1, 0]]])

    >>> pattern_hash = np.array([4,6])
    >>> N = 3
    >>> n_emp_sum_trial, n_emp_sum_trial_idx = \
    >>>                   n_emp_mat_sum_trial(mat, N,pattern_hash)
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


@decorate_deprecated_N
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


@decorate_deprecated_N
def _n_exp_mat_surrogate(mat, pattern_hash, n_surr=1):
    """
    Calculates the expected joint probability for each spike pattern with spike
    time randomization surrogate
    """
    if len(pattern_hash) > 1:
        raise ValueError('surrogate method works only for one pattern!')
    N_exp_array = np.zeros(n_surr)
    for rz_idx, rz in enumerate(np.arange(n_surr)):
        # row-wise shuffling all elements of zero-one matrix
        mat_surr = np.copy(mat)
        [np.random.shuffle(row) for row in mat_surr]
        N_exp_array[rz_idx] = n_emp_mat(mat_surr, pattern_hash)[0][0]
    return N_exp_array


@decorate_deprecated_N
def n_exp_mat(mat, pattern_hash, method='analytic', n_surr=1):
    """
    Calculates the expected joint probability for each spike pattern

    Parameters
    -----------
    mat: np.ndarray
         The entries are in the range [0, 1].
         The only possibility when the entries are floating point values is
         when the `mat` is calculated with the flag `analytic_TrialAverage`
         in `n_exp_mat_sum_trial()`.
         Otherwise, the entries are binary.
         0-axis --> neurons
         1-axis --> time bins
    pattern_hash: list
         List of hash values, length: number of patterns
    method: string
         method with which the expectency should be caculated
         'analytic' -- > analytically
         'surr' -- > with surrogates (spike time randomization)
         Default is 'analytic'
    n_surr: integer
         number of surrogates for constructing the distribution of expected
         joint probability.
         Default is 1 and this number is needed only when method = 'surr'

    Raises
    -------
       ValueError: if matrix m has wrong orientation

    Returns
    --------
    np.ndarray
        if method is analytic:
            An array containing the expected joint probability of each pattern,
            shape: (number of patterns,)
        if method is surr:
            0-axis --> different realizations, length = number of surrogates
            1-axis --> patterns

    Examples
    ---------
    >>> mat = np.array([[1, 1, 1, 1],
    >>>                 [0, 1, 0, 1],
    >>>                 [0, 0, 1, 0]])
    >>> pattern_hash = np.array([5,6])
    >>> n_exp_anal = n_exp_mat(mat, pattern_hash, method='analytic')
    >>> n_exp_anal
        [ 0.5 1.5 ]
    >>>
    >>>
    >>> n_exp_surr = n_exp_mat(mat, pattern_hash, method='surr', n_surr=5000)
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
        return _n_exp_mat_surrogate(mat, pattern_hash, n_surr=n_surr)


def n_exp_mat_sum_trial(mat, pattern_hash, method='analytic_TrialByTrial',
                        n_surr=1):
    """
    Calculates the expected joint probability
    for each spike pattern sum over trials

    Parameters
    -----------
    mat: np.ndarray
        3d numpy array or elephant BinnedSpikeTrain object
        Binned spike trains represented as a binary matrix (i.e., matrix of
        0's and 1's), segmented into trials. Trials should contain an identical
        number of neurons and an identical number of time bins.
        The entries of mat should be a list of a list where 0-axis is trials
        and 1-axis is neurons.
         0-axis --> trials
         1-axis --> neurons
         2-axis --> time bins
    pattern_hash: list
         List of hash values, length: number of patterns
    method: string
         method with which the unitary events whould be computed
         'analytic_TrialByTrial' -- > calculate the expectency
         (analytically) on each trial, then sum over all trials.
         'analytic_TrialAverage' -- > calculate the expectency
         by averaging over trials.
         (cf. Gruen et al. 2003)
         'surrogate_TrialByTrial' -- > calculate the distribution
         of expected coincidences by spike time randomzation in
         each trial and sum over trials.
         Default is 'analytic_trialByTrial'
    n_surr: integer, optional
         number of surrogate to be used
         Default is 1

    Returns
    --------
    n_exp: np.ndarray
         An array containing the expected joint probability of
         each pattern summed over trials,shape: (number of patterns,)

    Examples
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
    >>>                  [0, 1, 1, 1, 0],
    >>>                  [0, 1, 1, 0, 1]],
    >>>                 [[1, 1, 1, 1, 1],
    >>>                  [0, 1, 1, 1, 1],
    >>>                  [1, 1, 0, 1, 0]]])

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
        n_exp = np.zeros(n_surr)
        for mat_tr in mat:
            n_exp += n_exp_mat(mat_tr, pattern_hash,
                               method='surr', n_surr=n_surr)
    else:
        raise ValueError(
            "The method only works on the zero_one matrix at the moment")
    return n_exp


def gen_pval_anal(mat, pattern_hash, method='analytic_TrialByTrial',
                  n_surr=1):
    """
    computes the expected coincidences and a function to calculate
    p-value for given empirical coincidences

    this function generate a poisson distribution with the expected
    value calculated by mat. it returns a function which gets
    the empirical coincidences, `n_emp`,  and calculates a p-value
    as the area under the poisson distribution from `n_emp` to infinity

    Parameters
    -----------
    mat: np.ndarray
        3d numpy array or elephant BinnedSpikeTrain object
        Binned spike trains represented as a binary matrix (i.e., matrix of
        0's and 1's), segmented into trials. Trials should contain an identical
        number of neurons and an identical number of time bins.
        The entries of mat should be a list of a list where 0-axis is trials
        and 1-axis is neurons.
         0-axis --> trials
         1-axis --> neurons
         2-axis --> time bins
    pattern_hash: list
         List of hash values, length: number of patterns
    method: string
         method with which the unitary events whould be computed
         'analytic_TrialByTrial' -- > calculate the expectency
         (analytically) on each trial, then sum over all trials.
         ''analytic_TrialAverage' -- > calculate the expectency
         by averaging over trials.
         Default is 'analytic_trialByTrial'
         (cf. Gruen et al. 2003)
    n_surr: integer, optional
         number of surrogate to be used
         Default is 1

    Returns
    --------
    pval_anal: callable
         a function which calculates the p-value for
         the given empirical coincidences
    n_exp: list
        List of expected coincidences.

    Examples
    --------
    >>> mat = np.array([[[1, 1, 1, 1, 0],
    >>>                  [0, 1, 1, 1, 0],
    >>>                  [0, 1, 1, 0, 1]],
    >>>                 [[1, 1, 1, 1, 1],
    >>>                  [0, 1, 1, 1, 1],
    >>>                  [1, 1, 0, 1, 0]]])

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
            mat, pattern_hash, method=method, n_surr=n_surr)

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
    """Surprise measurement

    logarithmic transformation of joint-p-value into surprise measure
    for better visualization as the highly significant events are
    indicated by very low joint-p-values

    Parameters
    -----------
    p_val: list
        List of p-values (float) of statistical tests for different pattern.

    Returns
    --------
    Js: list
        List of surprise measures (float).

    Examples:
    ---------
    >>> p_val = np.array([0.31271072,  0.01175031])
    >>> jointJ(p_val)
        array([0.3419968 ,  1.92481736])
    """
    p_arr = np.asarray(p_val)
    Js = np.log10(1 - p_arr) - np.log10(p_arr)
    return Js


def _rate_mat_avg_trial(mat):
    """
    calculates the average firing rate of each neurons across trials
    """
    n_trials, n_neurons, n_bins = np.shape(mat)
    psth = np.zeros(n_neurons, dtype=np.float32)
    for tr, mat_tr in enumerate(mat):
        psth += np.sum(mat_tr, axis=1)
    return psth / (n_bins * n_trials)


def _bintime(t, binsize):
    """
    change the real time to bintime
    """
    t_dl = t.rescale('ms').magnitude
    binsize_dl = binsize.rescale('ms').magnitude
    return np.floor(np.array(t_dl) / binsize_dl).astype(int)


def _winpos(t_start, t_stop, winsize, winstep, position='left-edge'):
    """
    Calculates the position of the analysis window
    """
    t_start_dl = t_start.rescale('ms').magnitude
    t_stop_dl = t_stop.rescale('ms').magnitude
    winsize_dl = winsize.rescale('ms').magnitude
    winstep_dl = winstep.rescale('ms').magnitude

    # left side of the window time
    if position == 'left-edge':
        ts_winpos = np.arange(
            t_start_dl, t_stop_dl - winsize_dl + winstep_dl,
            winstep_dl) * pq.ms
    else:
        raise ValueError(
            'the current version only returns left-edge of the window')
    return ts_winpos


@decorate_deprecated_N
def _UE(mat, pattern_hash, method='analytic_TrialByTrial', n_surr=1):
    """
    returns the default results of unitary events analysis
    (Surprise, empirical coincidences and index of where it happened
    in the given mat, n_exp and average rate of neurons)
    """
    rate_avg = _rate_mat_avg_trial(mat)
    n_emp, indices = n_emp_mat_sum_trial(mat, pattern_hash)
    if method == 'surrogate_TrialByTrial':
        dist_exp, n_exp = gen_pval_anal(
            mat, pattern_hash, method, n_surr=n_surr)
        n_exp = np.mean(n_exp)
    elif method == 'analytic_TrialByTrial' or \
            method == 'analytic_TrialAverage':
        dist_exp, n_exp = gen_pval_anal(mat, pattern_hash, method)
    pval = dist_exp(n_emp)
    Js = jointJ(pval)
    return Js, rate_avg, n_exp, n_emp, indices


def jointJ_window_analysis(
        data, binsize, winsize, winstep, pattern_hash,
        method='analytic_TrialByTrial', t_start=None,
        t_stop=None, binary=True, n_surr=100):
    """
    Calculates the joint surprise in a sliding window fashion

    Parameters
    ----------
    data: list
          list of spike trains (neo.SpikeTrain objects) in different trials
                               0-axis --> Trials
                               1-axis --> Neurons
                               2-axis --> Spike times
    binsize: pq.Quantity
        Quantity scalar with dimension time
        size of bins for descritizing spike trains
    winsize: pq.Quantity
        Quantity scalar with dimension time
        size of the window of analysis
    winstep: pq.Quantity
        Quantity scalar with dimension time
        size of the window step
    pattern_hash: list
        list of interested patterns (int) in hash values
        (see hash_from_pattern and inverse_hash_from_pattern functions)
    method: string
         method with which the unitary events whould be computed
         'analytic_TrialByTrial' -- > calculate the expectency
         (analytically) on each trial, then sum over all trials.
         'analytic_TrialAverage' -- > calculate the expectency
         by averaging over trials.
         (cf. Gruen et al. 2003)
         'surrogate_TrialByTrial' -- > calculate the distribution
         of expected coincidences by spike time randomzation in
         each trial and sum over trials.
         Default is 'analytic_trialByTrial'
    t_start: float or pq.Quantity, optional
          The start time to use for the time points.
          If not specified, retrieved from the `t_start`
          attribute of `spiketrain`.
    t_stop: float or pq.Quantity, optional
         The start time to use for the time points.
         If not specified, retrieved from the `t_stop`
         attribute of `spiketrain`.
    n_surr: integer, optional
         number of surrogate to be used
         Default is 100

    Returns
    -------
    result: dictionary
          Js: list of float
              JointSurprise of different given patterns within each window
              shape: different pattern hash --> 0-axis
                  different window --> 1-axis
          indices: list of list of integers
              list of indices of pattern within each window
              shape: different pattern hash --> 0-axis
                  different window --> 1-axis
          n_emp: list of integers
              empirical number of each observed pattern.
              shape: different pattern hash --> 0-axis
                  different window --> 1-axis
          n_exp: list of floats
              expeced number of each pattern.
              shape: different pattern hash --> 0-axis
                  different window --> 1-axis
          rate_avg: list of floats
              average firing rate of each neuron
              shape: different pattern hash --> 0-axis
                  different window --> 1-axis

    """
    if not isinstance(data[0][0], neo.SpikeTrain):
        raise ValueError(
            "structure of the data is not correct: 0-axis should be trials, "
            "1-axis units and 2-axis neo spike trains")

    if t_start is None:
        t_start = data[0][0].t_start.rescale('ms')
    if t_stop is None:
        t_stop = data[0][0].t_stop.rescale('ms')

    # position of all windows (left edges)
    t_winpos = _winpos(t_start, t_stop, winsize, winstep, position='left-edge')
    t_winpos_bintime = _bintime(t_winpos, binsize)

    winsize_bintime = _bintime(winsize, binsize)
    winstep_bintime = _bintime(winstep, binsize)

    if winsize_bintime * binsize != winsize:
        warnings.warn(
            "ratio between winsize and binsize is not integer -- "
            "the actual number for window size is " + str(
                winsize_bintime * binsize))

    if winstep_bintime * binsize != winstep:
        warnings.warn(
            "ratio between winstep and binsize is not integer -- "
            "the actual number for window size is " + str(
                winstep_bintime * binsize))

    num_tr, N = np.shape(data)[:2]

    n_bins = int((t_stop - t_start) / binsize)

    mat_tr_unit_spt = np.zeros((len(data), N, n_bins))
    for tr, sts in enumerate(data):
        sts = list(sts)
        bs = conv.BinnedSpikeTrain(
            sts, t_start=t_start, t_stop=t_stop, binsize=binsize)
        if binary is True:
            mat = bs.to_bool_array()
        else:
            raise ValueError(
                "The method only works on the zero_one matrix at the moment")
        mat_tr_unit_spt[tr] = mat

    num_win = len(t_winpos)
    Js_win, n_exp_win, n_emp_win = (np.zeros(num_win) for _ in range(3))
    rate_avg = np.zeros((num_win, N))
    indices_win = {}
    for i in range(num_tr):
        indices_win['trial' + str(i)] = []

    for i, win_pos in enumerate(t_winpos_bintime):
        mat_win = mat_tr_unit_spt[:, :, win_pos:win_pos + winsize_bintime]
        if method == 'surrogate_TrialByTrial':
            Js_win[i], rate_avg[i], n_exp_win[i], n_emp_win[
                i], indices_lst = _UE(
                mat_win, pattern_hash, method, n_surr=n_surr)
        else:
            Js_win[i], rate_avg[i], n_exp_win[i], n_emp_win[
                i], indices_lst = _UE(mat_win, pattern_hash, method)
        for j in range(num_tr):
            if len(indices_lst[j][0]) > 0:
                indices_win[
                    'trial' + str(j)] = np.append(
                    indices_win['trial' + str(j)], indices_lst[j][0] + win_pos)
    return {'Js': Js_win, 'indices': indices_win, 'n_emp': n_emp_win,
            'n_exp': n_exp_win, 'rate_avg': rate_avg / binsize}
