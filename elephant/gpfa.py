"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
 [1] for neural trajectory visualization of parallel spike trains.

The input consists of a set of trials (Y), each containing a list of spike
trains (N neurons). The output is the projection (X) of the data in space
of pre-chosen dimension x_dim < N.

Under the assumption of a linear relation between the latent
variable X and the actual data Y in addition to a noise term (i.e.,
Y = C * X + d + Gauss(0,R)), the projection corresponds to the conditional
probability E[X|Y].

A Gaussian process (X) of dimension x_dim < N is adopted to extract smooth
neural trajectories. The parameters (C, d, R) are estimated from the data using
factor analysis technique. GPFA is simply a set of Factor Analyzers (FA),
linked together in the low dimensional space by a Gaussian Process (GP).

Internally, the analysis consists of the following steps:

0) bin the data to get a sequence of N dimensional vectors for each time
   bin (cf., `gpfa_util.get_seq`), and choose the reduced dimension x_dim

1) expectation maximization for the parameters C, d, R and the time-scale of
   the gaussian process, using all the trials provided as input (cf.,
   `gpfa_core.em`)

2) projection of single trial in the low dimensional space (cf.,
   `gpfa_core.exact_inference_with_ll`)

3) orthonormalization of the matrix C and the corresponding subspace:
   (cf., `gpfa_util.orthogonalize`)


There are two principle scenarios of using the GPFA analysis. In the first
scenario, only one single dataset is available. The parameters that describe
the transformation are first extracted from the data, and the orthonormal basis
is constructed. Then the same data is projected into this basis. This analysis
is performed using the `gpfa()` function.

In the second scenario, both a training and a test data set is available. Here,
the parameters are estimated from the training data. In a second step the test
data is projected into the non-orthonormal space obtained from the training
data, and then orthonormalized. From this scenario, it is possible to perform a
cross-validation between training and test data sets. This analysis is
performed by exectuing first `extract_trajectory()` on the training data,
followed by `postprocess()` on the training and test datasets.


References:

The code was ported from the MATLAB code (see INFO.md for more details).

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
    Prepares data, estimates parameters of the latent variables and extracts
    neural trajectories.

    Parameters
    ----------
    seqs : np.recarray
        data structure, whose nth entry (corresponding to the nth experimental
        trial) has fields
            * trialId: unique trial identifier
            * T: (1 x 1) number of timesteps
            * y: (yDim x T) neural data
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
        Contains the embedding of the training data into the latent variable
        space.
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
    ValueError
        If `seqs` is empty.

    """
    if len(seqs) == 0:
        raise ValueError("Got empty trials.")

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
        First return value of extract_trajectory() on the training data set.
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
        Contains the embedding of the training data into the latent variable
        space.
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
        Contains the embedding of the test data into the latent variable space.
        Same data structure as for `seqs_train`.
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

        # Copy additional fields from extract_trajectory from training data to
        # test data
        seqs_test_dup = seqs_train.copy()
        seqs_test_dup["T"] = seqs_test["T"]
        seqs_test_dup["y"] = seqs_test["y"]
        seqs_test_dup["trialId"] = seqs_test["trialId"]

        seqs_test = seqs_test_dup
        seqs_test = gpfa_core.exact_inference_with_ll(seqs_test, params_est)
        X = np.hstack(seqs_test['xsm'])
        Xorth, Corth, _ = gpfa_util.orthogonalize(X, C)
        seqs_test = gpfa_util.segment_by_trial(seqs_test, Xorth, 'xorth')

    return params_est, seqs_train, seqs_test


def gpfa(data, bin_size=20*pq.ms, x_dim=3, em_max_iters=500):
    """
    Prepares data and calls functions for extracting neural trajectories in the
    orthonormal space.

    The function combines calls to `extract_trajectory()` and `postprocess()`
    in a scenario, where no cross-validation of the embedding is performed.

    Parameters
    ----------

    data : list of list of Spiketrain objects
        The outer list corresponds to trials and the inner list corresponds to
        the neurons recorded in that trial, such that data[l][n] is the
        Spiketrain of neuron n in trial l. Note that the number and order of
        Spiketrains objects per trial must be fixed such that data[l][n] and
        data[k][n] refer to the same spike generator for any choice of l,k and
        n.
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
            Corth: ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the orthonormal
                latent variable space
            R: ndarray of shape (#units, #latent_vars)
                observation noise covariance

    seqs_train: numpy.recarray
        Contains the embedding of the data into the latent variable space.
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
            * xorth: ndarray of shape (#latent_vars, #bins)
                trajectory in the orthonormalized space

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
    ValueError
        If `data` is an empty list.
        If `bin_size` if not a `pq.Quantity`.
        If `data[0][1][0]` is not a `neo.SpikeTrain`.

    Examples
    --------
    In the following example, we calculate the neural trajectories of 20
    Poisson spike train generators recorded in 50 trials with randomized
    rates up to 100 Hz.

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
    if len(data) == 0:
        raise ValueError("`data` cannot be empty")
    if not isinstance(bin_size, pq.Quantity):
        raise ValueError("'bin_size' must be of type pq.Quantity")
    if not isinstance(data[0][1][0], neo.SpikeTrain):
        raise ValueError("structure of the data is not correct: 0-axis "
                         "should be trials, 1-axis neo spike trains "
                         "and 2-axis spike times")

    seqs = gpfa_util.get_seq(data, bin_size)
    params_est, seqs_train, fit_info = extract_trajectory(
        seqs, bin_size=bin_size.rescale('ms').magnitude, x_dim=x_dim,
        em_max_iters=em_max_iters)
    params_est, seqs_train, _ = postprocess(params_est, seqs_train)

    return params_est, seqs_train, fit_info
