"""
@ 2009
Byron Yu
byronyu @ stanford.edu
John Cunningham
jcunnin @ stanford.edu

"""

import neo
import quantities as pq

from . import core, util
from elephant.asset import check_quantities

"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
 [1] for visualizing the neural trajectory (X) of parallel spike trains (Y).

The INPUT consists in a set of trials (Y), each containing a list of spike
trains (N neurons): The OUTPUT is the projection (X) of these data in a space
of pre-chosen dimension x_dim < N.

Under the assumption of a linear relation plus noise between the latent
variable X and the actual data Y (Y = C * X + d + Gauss(0,R)), the projection
correspond to the conditional probability E[X|Y].

A GAUSSIAN PROCESS (X) of dimesnion x_dim < N is adopted to extract smooth
neural trajectories. The parameters (C, d, R) are estimated from the data using
FACTOR ANALYSIS tecnique. GPFA is simply a set of Factor Analyzers (FA), linked
togheter in the low dimensional space by a Gaussian Process (GP).

The analysis comprises the following steps:

0) bin the data, to get a sequence of N dimensional vectors, on for each time
    bin;
  and choose the reduced dimension x_dim;

1) call of the functions used:

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
neural population activity. J Neurophysiol 102:614-635
"""


def neural_trajectory(data, method='gpfa', bin_size=20 * pq.ms, x_dim=3,
                      num_folds=0, em_max_iters=500):
    """
    Prepares data and calls functions for extracting neural trajectories.

    Parameters
    ----------

    data : list
          list of spike trains in different trials
                                        0-axis --> Trials
                                        1-axis --> Neurons
                                        2-axis --> Spike times
    method : str
        Method for extracting neural trajectories
        * 'gpfa': Uses the Gaussian Process Factor Analysis method.
        Default is 'gpfa'
    bin_size : quantities.Quantity
        Width of each time bin
        Default is 20 ms
    x_dim : int
        State dimensionality
        Default is 3
    num_folds : int
        Number of cross-validation folds, 0 indicates no cross-validation,
        i.e. train on all trials.
        Default is 0.
        (Cross-validation is not implemented yet)
    em_max_iters : int
        Number of EM iterations to run (default: 500)

    Returns
    -------

    parameter_estimates: dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance
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
        Information of the fitting process and the parameters used there
            * iteration_time: A list containing the runtime for each iteration
                step in the EM algorithm
            * log_likelihood: float, maximized likelihood obtained in the
                E-step of the EM algorithm
            * bin_size: int, Width of the bins
            * cvf: int, number for cross-validation folding
                Default is 0 (no cross-validation)
            * hasSpikesBool: Indicates if a neuron has any spikes across trials
            * method: String, method name

    Raises
    ------
    ValueError
        If `bin_size` if not a `pq.Quantity`.
        If `data[0][1][0]` is not a `neo.SpikeTrain`.

    Examples
    --------
    >>> import numpy as np
    >>> import quantities as pq
    >>> from elephant.gpfa.neural_trajectory import neural_trajectory
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> data = []
    >>> for trial in range(50):
    >>>     n_channels = 20
    >>>     firing_rates = np.random.randint(low=1, high=100,
    >>>         size=n_channels) * pq.Hz
    >>>     spike_times = [homogeneous_poisson_process(rate=rate)
    >>>         for rate in firing_rates]
    >>>     data.append((trial, spike_times))
    >>> params_est, seqs_train, seqs_test, fit_info = neural_trajectory(
    >>>     data, method='gpfa', bin_size=20 * pq.ms, x_dim=8)

    """
    # todo does it makes sense to explicitly pass trial_id?
    check_quantities(bin_size, 'bin_size')
    if not isinstance(data[0][1][0], neo.SpikeTrain):
        raise ValueError("structure of the data is not correct: 0-axis should "
                         "be trials, 1-axis neo spike trains "
                         "and 2-axis spike times")

    seqs = util.get_seq(data, bin_size)
    params_est, seqs_train, fit_info = core.extract_trajectory(
        seqs, method, bin_size.rescale('ms').magnitude, x_dim, num_folds,
        em_max_iters=em_max_iters)
    params_est, seqs_train, seqs_test = core.postprocess(params_est,
                                                         seqs_train, fit_info)

    return params_est, seqs_train, seqs_test, fit_info
