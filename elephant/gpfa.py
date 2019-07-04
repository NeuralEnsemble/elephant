"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
 [1] for neural trajectory (X) visualization of parallel spike trains (Y).

The INPUT consists of a set of trials (Y), each containing a list of spike
trains (N neurons): The OUTPUT is the projection (X) of the data in space
of pre-chosen dimension x_dim < N.

Under the assumption of a linear relation plus noise between the latent
variable X and the actual data Y (Y = C * X + d + Gauss(0,R)), the projection
corresponds to the conditional probability E[X|Y].

A GAUSSIAN PROCESS (X) of dimension x_dim < N is adopted to extract smooth
neural trajectories. The parameters (C, d, R) are estimated from the data using
FACTOR ANALYSIS technique. GPFA is simply a set of Factor Analyzers (FA),
linked together in the low dimensional space by a Gaussian Process (GP).

The analysis consists of the following steps:

0) bin the data to get a sequence of N dimensional vectors for each time
    bin, and choose the reduced dimension x_dim;

1) run

-  gpfa_engine(seq_train, seq_test, x_dim=8, bin_width=20.0, tau_init=100.0,
                eps_init=1.0E-3, min_var_frac=0.01, em_max_iters=500)

2) expectation maximization for the parameters C, d, R and the time-scale of
 the gaussian process, using all the trials provided as input:

-  params_est, seq_train_cut, ll_cut, iter_time = em(params_init, seq,
    em_max_iters=500, tol=1.0E-8, min_var_frac=0.01, freq_ll=5, verbose=False)

3) projection of single trial in the low dimensional space:

-  seq_train, ll_train = exact_inference_with_ll(seq_train, params_est)


4) orthonormalization of the matrix C and the corresponding subspace;

-  postprocess(ws, kern_sd=[])

References:
[1] Yu MB, Cunningham JP, Santhanam G, Ryu SI, Shenoy K V, Sahani M (2009)
Gaussian-process factor analysis for low-dimensional single-trial analysis of
neural population activity. J Neurophysiol 102:614-635.

:copyright: Copyright 2015-2019 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""


import numpy as np
import neo
import quantities as pq

from elephant.gpfa_src import gpfa_core, gpfa_util


def extract_trajectory(seqs, bin_size=20., x_dim=3, min_var_frac=0.01,
                       em_max_iters=500, verbose=False):
    """
    Prepares data and calls functions for extracting neural trajectories.

    Parameters
    ----------

    seqs : np.ndarray
          list of spike trains in different trials
                                        0-axis --> Trials
                                        1-axis --> Neurons
                                        2-axis --> Spike times
    bin_size : float, optional
        Width of each time bin in ms (magnitude).
        Default is 20 ms.
    x_dim : int, optional
        State dimensionality.
        Default is 3.
    min_var_frac : float, optional
                   fraction of overall data variance for each observed
                   dimension to set as the private variance floor.  This is
                   used to combat Heywood cases, where ML parameter learning
                   returns one or more zero private variances. (default: 0.01)
                   (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_max_iters : int, optional
        Number of EM iterations to run (default: 500).
    verbose : bool, optional
              specifies whether to display status messages (default: False)

    Returns
    -------

    parameter_estimates: dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance.
                Currently, only 'rbf' is supported.
            gamma: ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: ndarray of shape (1, #latent_vars)
                GP noise variances
            d: ndarray of shape (#units, 1)
                observation mean
            C: ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            R: ndarray of shape (#units, #latent_vars)
                observation noise covariance

    seqs_train: np.recarray
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
            * trialId: int
                unique trial identifier
            * T: int
                number of timesteps
            * y: ndarray of shape (#units, #bins)
                neural data
            * xsm: ndarray of shape (#latent_vars, #bins)
                posterior mean of latent variables at each time bin
            * Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                posterior covariance between latent variables at each
                timepoint
            * VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                posterior covariance over time for each latent variable

    fit_info: dict
        Information of the fitting process and the parameters used there:
            * iteration_time: A list containing the runtime for each iteration
                step in the EM algorithm.
            * log_likelihood: float, maximized likelihood obtained in the
                E-step of the EM algorithm.
            * bin_size: int, Width of the bins.
            * cvf: int, number for cross-validation folding
                Default is 0 (no cross-validation).
            * has_spikes_bool: Indicates if a neuron has any spikes across
                trials.
            * method: str, Method name.

    Raises
    ------
    AssertionError
        If `seqs` es empty.

    """
    assert len(seqs) > 0, "Got empty trials."

    # Set cross-validation folds
    num_trials = len(seqs)

    test_mask = np.full(num_trials, False, dtype=bool)
    train_mask = ~test_mask

    tr = np.arange(num_trials, dtype=np.int)
    train_trial_idx = tr[train_mask]
    test_trial_idx = tr[test_mask]
    seqs_train = seqs[train_trial_idx]
    seqs_test = seqs[test_trial_idx]

    # Remove inactive units based on training set
    has_spikes_bool = (np.hstack(seqs_train['y']).mean(1) != 0)

    for seq_train in seqs_train:
        seq_train['y'] = seq_train['y'][has_spikes_bool, :]
    for seq_test in seqs_test:
        seq_test['y'] = seq_test['y'][has_spikes_bool, :]

    # Check if training data covariance is full rank
    y_all = np.hstack(seqs_train['y'])
    y_dim = y_all.shape[0]

    if np.linalg.matrix_rank(np.cov(y_all)) < y_dim:
        errmesg = 'Observation covariance matrix is rank deficient.\n' \
                  'Possible causes: ' \
                  'repeated units, not enough observations.'
        raise ValueError(errmesg)

    if verbose:
        print('Number of training trials: {}'.format(len(seqs_train)))
        print('Number of test trials: {}'.format(len(seqs_test)))
        print('Latent space dimensionality: {}'.format(x_dim))
        print('Observation dimensionality: {}'.format(has_spikes_bool.sum()))

    # The following does the heavy lifting.
    params_est, seqs_train, fit_info = gpfa_core.gpfa_engine(
        seq_train=seqs_train,
        seq_test=seqs_test,
        x_dim=x_dim,
        bin_width=bin_size,
        min_var_frac=min_var_frac,
        em_max_iters=em_max_iters,
        verbose=verbose)

    fit_info['has_spikes_bool'] = has_spikes_bool
    fit_info['min_var_frac'] = min_var_frac
    fit_info['bin_size'] = bin_size

    return params_est, seqs_train, fit_info


def postprocess(params_est, seqs_train, seqs_test=None):
    """
    Orthonormalization and other cleanup.

    Parameters
    ----------

    params_est : dict
        First return value of extract_trajectory().
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance
                Currently, only 'rbf' is supported.
            gamma: ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: ndarray of shape (1, #latent_vars)
                GP noise variances
            d: ndarray of shape (#units, 1)
                observation mean
            C: ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            R: ndarray of shape (#units, #latent_vars)
                observation noise covariance

    seqs_train: np.recarray
        Second return value of extract_trajectory().
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
            * trialId: int
                unique trial identifier
            * T: int
                number of timesteps
            * y: ndarray of shape (#units, #bins)
                neural data
            * xsm: ndarray of shape (#latent_vars, #bins)
                posterior mean of latent variables at each time bin
            * Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                posterior covariance between latent variables at each
                timepoint
            * VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                posterior covariance over time for each latent variable

    seqs_test: np.recarray, optional
        Data structure of test dataset.
        Default is None.

    Returns
    -------

    params_est : dict
        Estimated model parameters, including `Corth`, obtained by
        orthonormalizing the columns of C.
    seqs_train : np.recarray
        Training data structure that contains the new field `xorth`,
        the orthonormalized neural trajectories.
    seqs_test : np.recarray
        Test data structure that contains orthonormalized neural
        trajectories in `xorth`, obtained using `params_est`.
        When no test dataset is given, None is returned.


    Raises
    ------
    ValueError
        If `fit_info['kernSDList'] != kern_sd`.

    """
    C = params_est['C']
    X = np.hstack(seqs_train['xsm'])
    Xorth, Corth, _ = gpfa_util.orthogonalize(X, C)
    seqs_train = gpfa_util.segment_by_trial(seqs_train, Xorth, 'xorth')

    params_est['Corth'] = Corth

    if seqs_test is not None:
        print("Extracting neural trajectories for test data...\n")
        seqs_test = gpfa_core.exact_inference_with_ll(seqs_test,
                                                      params_est)
        X = np.hstack(seqs_test['xsm'])
        Xorth, Corth, _ = gpfa_util.orthogonalize(X, C)
        seqs_test = gpfa_util.segment_by_trial(seqs_test, Xorth, 'xorth')

    return params_est, seqs_train, seqs_test


def gpfa(data, bin_size=20*pq.ms, x_dim=3, em_max_iters=500):
    """
    Prepares data and calls functions for extracting neural trajectories.

    Parameters
    ----------

    data : list
          list of spike trains in different trials
                                        0-axis --> Trials
                                        1-axis --> Neurons
                                        2-axis --> Spike times
    bin_size : quantities.Quantity, optional
        Width of each time bin.
        Default is 20 ms.
    x_dim : int, optional
        State dimensionality.
        Default is 3.
    em_max_iters : int, optional
        Number of EM iterations to run (default: 500).

    Returns
    -------

    parameter_estimates: dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance
                Currently, only 'rbf' is supported.
            gamma: ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: ndarray of shape (1, #latent_vars)
                GP noise variances
            d: ndarray of shape (#units, 1)
                observation mean
            C: ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            R: ndarray of shape (#units, #latent_vars)
                observation noise covariance

    seqs_train: numpy.recarray
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
            * trialId: int
                unique trial identifier
            * T: int
                number of timesteps
            * y: ndarray of shape (#units, #bins)
                neural data
            * xsm: ndarray of shape (#latent_vars, #bins)
                posterior mean of latent variables at each time bin
            * Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                posterior covariance between latent variables at each
                timepoint
            * VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                posterior covariance over time for each latent variable

    seqs_test: numpy.recarray
        Same structure as seqs_train, but contains results of the method
        applied to test dataset.
        When no cross-validation is performed, None is returned.

    fit_info: dict
        Information of the fitting process and the parameters used there:
            * iteration_time: A list containing the runtime for each iteration
                step in the EM algorithm.
            * log_likelihood: float, maximized likelihood obtained in the
                E-step of the EM algorithm.
            * bin_size: int, Width of the bins.
            * cvf: int, number for cross-validation folding
                Default is 0 (no cross-validation).
            * has_spikes_bool: Indicates if a neuron has any spikes across
                trials.
            * method: str, Method name.

    Raises
    ------
    AssertionError
        If `data` is an empty list.
        If `bin_size` if not a `pq.Quantity`.
        If `data[0][1][0]` is not a `neo.SpikeTrain`.

    Examples
    --------
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.gpfa import gpfa
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> data = []
    >>> for trial in range(50):
    >>>     n_channels = 20
    >>>     firing_rates = np.random.randint(low=1, high=100,
    >>>         size=n_channels) * pq.Hz
    >>>     spike_times = [homogeneous_poisson_process(rate=rate)
    >>>         for rate in firing_rates]
    >>>     data.append((trial, spike_times))
    >>> params_est, seqs_train, seqs_test, fit_info = gpfa(
    >>>     data, bin_size=20 * pq.ms, x_dim=8)

    """
    # todo does it makes sense to explicitly pass trial_id?
    assert len(data) > 0, "`data` cannot be empty"
    if not isinstance(bin_size, pq.Quantity):
        raise ValueError("'bin_size' must be of type pq.Quantity")
    assert isinstance(data[0][1][0], neo.SpikeTrain), \
        "structure of the data is not correct: 0-axis should "\
        "be trials, 1-axis neo spike trains "\
        "and 2-axis spike times"

    seqs = gpfa_util.get_seq(data, bin_size)
    params_est, seqs_train, fit_info = extract_trajectory(
        seqs, bin_size=bin_size.rescale('ms').magnitude, x_dim=x_dim,
        em_max_iters=em_max_iters)
    params_est, seqs_train, seqs_test = postprocess(params_est, seqs_train)

    return params_est, seqs_train, seqs_test, fit_info
