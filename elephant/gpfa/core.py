"""
@ 2009
Byron Yu
byronyu @ stanford.edu
John Cunningham
jcunnin @ stanford.edu

"""

import numpy as np
import quantities as pq

from elephant.gpfa import util
from .gpfa import gpfa_engine, exact_inference_with_ll, two_stage_engine

"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method [1]
for visualizing the neural trajectory (X) of parallel spike trains (Y).

The INPUT consists in a set of trials (Y), each containing a list of spike
trains (N neurons): The OUTPUT is the projection (X) of these data in a space of 
pre-chosen dimension x_dim < N.

Under the assumption of a linear relation plus noise between the latent variable 
X and the actual data Y (Y = C * X + d + Gauss(0,R)), the projection correspond 
to the conditional probability E[X|Y].

A GAUSSIAN PROCESS (X) of dimesnion x_dim < N is adopted to extract smooth neural
trajectories. The parameters (C, d, R) are estimated from the data using FACTOR
ANALYSIS tecnique. GPFA is simply a set of Factor Analyzers (FA), linked togheter
in the low dimensional space by a Gaussian Process (GP).

The analysis comprises the following steps:

0) bin the data, to get a sequence of N dimensional vectors, on for each time bin;
  and choose the reduced dimension x_dim;

1) call of the functions used:

-  gpfa_engine(seq_train, seq_test, x_dim=8, bin_width=20.0, tau_init=100.0,
                eps_init=1.0E-3, min_var_frac=0.01, em_max_iters=500)
                
2) expectation maximization for the parameters C, d, R and the time-scale of the
  gaussian process, using all the trials provided as input:
  
-  params_est, seq_train_cut, ll_cut, iter_time = em(params_init, seq, em_max_iters=500,
                                tol=1.0E-8, min_var_frac=0.01, freq_ll=5, verbose=False)
  
3) projection of single trial in the low dimensional space:

-  seq_train, ll_train = exact_inference_with_ll(seq_train, params_est)


4) orthonormalization of the matrix C and the corresponding subspace;

-  postprocess(ws, kern_sd=[])

References:
[1] Yu MB, Cunningham JP, Santhanam G, Ryu SI, Shenoy K V, Sahani M (2009)
Gaussian-process factor analysis for low-dimensional single-trial analysis of
neural population activity. J Neurophysiol 102:614-635
"""


def extract_trajectory(seqs, method='gpfa', bin_size=20*pq.ms, x_dim=3, num_folds=0):
    """
    Prepares data and calls functions for extracting neural trajectories.

    Parameters
    ----------

    seqs: list containing following structure
        list of spike trains in different trials
            0-axis --> Trials
            1-axis --> Neurons
            2-axis --> Spike times
    method: string,
        Method for extracting neural trajectories
        * 'gpfa': Uses the Gaussian Process Factor Analysis method.
        Default is 'gpfa'
    bin_size:   quantities.Quantity
        Width of each time bin
        Default is 20 ms
    x_dim: int
        State dimensionality 
        Default is 3
    num_folds: int
        Number of cross-validation folds, 0 indicates no cross-validation,
        i.e. train on all trials.
        Default is 0.
        (Cross-validation is not implemented yet.)

    Returns
    -------

    params_est: dict
        Estimated GPFA model parameters
            covType: {'rbf', 'tri', 'logexp'}
                  type of GP covariance
            gamma: ndarray of shape (1, #latent_vars)
                  related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: ndarray of shape (1, #latent_vars)
                GP noise variances
            d: ndarray of shape (#units, 1)
              observation mean
            C: ndarray of shape (#units, #latent_vars)
              mapping between the neuronal data space and the latent variable space
            R: ndarray of shape (#units, #latent_vars)
              observation noise covariance
    seqs_train: Numpy recarray
        A copy of the training data structure, augmented by new fields
            xsm: ndarray of shape (#latent_vars x #bins)
                 posterior mean of latent variables at each time bin
            Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                 posterior covariance between latent variables at each
                 timepoint
            VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                   posterior covariance over time for each latent variable
    fit_info: Dictionary
        Information of the fitting process and the parameters used there
        * iteration_time: A list containing the runtime for each iteration step
        in the EM algorithm
        * log_likelihood: float, maximized likelihood obtained in the E-step of the
        EM algorithm
        * bin_size: int, Width of the bins
        * cvf: int, number for cross-validation folding
            Default is 0 (no cross-validation)
        * hasSpikesBool: Indicates if a neuron has any spikes across trials
        * method: String, method name
    """
    if len(seqs) == 0:
        raise ValueError("Error: No valid trials.")

    # Set cross-validation folds
    num_trials = len(seqs)
    fdiv = np.linspace(0, num_trials, num_folds + 1).astype(np.int) if num_folds > 0 \
        else num_trials

    for cvf in range(0, num_folds+1):
        if cvf == 0:
            print("\n===== Training on all data =====")
        else:
            # TODO: implement cross-validation
            # print("\n===== Cross-validation fold {} of {} =====").\
            #     format(cvf, num_folds)
            raise NotImplementedError

        # Set cross-validataion masks
        test_mask = np.full(num_trials, False, dtype=bool)
        if cvf > 0:
            test_mask[fdiv[cvf]:fdiv[cvf+1]] = True
        train_mask = ~test_mask

        tr = np.arange(num_trials, dtype=np.int)
        if cvf > 0:
            # Randomly reorder trials before partitioning into training and
            # test sets
            np.random.seed(0)
            np.random.shuffle(tr)
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
            errmesg = 'Observation covariance matrix is rank deficient.\n'\
                      'Possible causes: '\
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        print('Number of training trials: {}'.format(len(seqs_train)))
        print('Number of test trials: {}'.format(len(seqs_test)))
        print('Latent space dimensionality: {}'.format(x_dim))
        print('Observation dimensionality: {}'.format(has_spikes_bool.sum()))

        # If doing cross-validation, don't use private noise variance floor.
        if cvf == 0:
            min_var_frac = 0.01
        else:
            min_var_frac = -np.inf

        # The following does the heavy lifting.
        if method == 'gpfa':
            params_est, seqs_train, fit_info = gpfa_engine(seqs_train, seqs_test, x_dim=x_dim,
                                                           bin_width=bin_size, min_var_frac=min_var_frac)
        elif method in ['fa', 'ppca', 'pca']:
            # TODO: implement two_stage_engine()
            params_est, seqs_train, fit_info = two_stage_engine(seqs_train, seqs_test, typ=method,
                                                                xDim=x_dim, binWidth=bin_size)
            raise NotImplementedError

        var_names = ['method', 'cvf', 'has_spikes_bool', 'min_var_frac', 'bin_size']
        for key in var_names:
            fit_info.update({key: locals()[key]})

    return params_est, seqs_train, fit_info


def postprocess(params_est, seqs_train, fit_info, kern_sd=1.0, seqs_test=None):
    """
    Orthonormalization and other cleanup.

    Parameters
    ----------

    params_est, seqs_train, fit_info
        Return variables of extract_trajectory()

    kern_sd: float, optional
        For two-stage methods, this function returns `seqs_train` and
        `params_est` corresponding to `kern_sd`.
        Default is 1.

    secs_test: numpy recarray, optional
        Data structure of test dataset.
        Default is None.

    Returns
    -------

    params_est
        Estimated model parameters, including `Corth` obtained by
        orthonormalizing the columns of C
    seqs_train
        Training data structure containing new field `xorth`,
        the orthonormalized neural trajectories
    seqs_test
        Test data structure containing orthonormalized neural
        trajectories in `xorth`, obtained using `params_est`
        When no test dataset is given, None is returned.

    """
    if hasattr(fit_info, 'kern'):
        if not fit_info['kernSDList']:
            k = 1
        else:
            k = np.where(np.array(fit_info['kernSDList']) == np.array(kern_sd))[0]
            if not k:
                raise ValueError('Selected kernSD not found')
    if fit_info['method'] == 'gpfa':
        C = params_est['C']
        X = np.hstack(seqs_train['xsm'])
        Xorth, Corth, _ = util.orthogonalize(X, C)
        seqs_train = util.segment_by_trial(seqs_train, Xorth, 'xorth')

        params_est['Corth'] = Corth

        if seqs_test is not None:
            print("Extracting neural trajectories for test data...\n")
            seqs_test = exact_inference_with_ll(seqs_test, params_est)
            X = np.hstack(seqs_test['xsm'])
            Xorth, Corth, _ = util.orthogonalize(X, C)
            seqs_test = util.segment_by_trial(seqs_test, Xorth, 'xorth')
    else:
        # TODO: implement postprocessing for the two stage methods
        raise NotImplementedError

    return params_est, seqs_train, seqs_test
