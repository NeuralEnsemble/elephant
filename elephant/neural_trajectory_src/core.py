"""
@ 2009
Byron Yu
byronyu @ stanford.edu
John Cunningham
jcunnin @ stanford.edu

"""

import numpy as np

from elephant.neural_trajectory_src import gpfa_util
from elephant.neural_trajectory_src.gpfa import gpfa_engine, \
    exact_inference_with_ll, two_stage_engine


def extract_trajectory(seqs, method='gpfa', bin_size=20., x_dim=3,
                       num_folds=0, em_max_iters=500):
    """
    Prepares data and calls functions for extracting neural trajectories.

    Parameters
    ----------

    seqs : np.ndarray
          list of spike trains in different trials
                                        0-axis --> Trials
                                        1-axis --> Neurons
                                        2-axis --> Spike times
    method : str, optional
        Method for extracting neural trajectories.
        * 'gpfa': Uses the Gaussian Process Factor Analysis method.
        Default is 'gpfa'.
    bin_size : float, optional
        Width of each time bin in ms (magnitude).
        Default is 20 ms.
    x_dim : int, optional
        State dimensionality.
        Default is 3.
    num_folds : int, optional
        Number of cross-validation folds, 0 indicates no cross-validation,
        i.e. train on all trials.
        Default is 0.
        (Cross-validation is not implemented yet)
    em_max_iters : int, optional
        Number of EM iterations to run (default: 500).

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
    """
    if len(seqs) == 0:
        raise ValueError("Error: No valid trials.")

    # Set cross-validation folds
    num_trials = len(seqs)
    fdiv = np.linspace(0, num_trials, num_folds + 1).astype(np.int) if \
        num_folds > 0 else num_trials

    for cvf in range(0, num_folds + 1):
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
            test_mask[fdiv[cvf]:fdiv[cvf + 1]] = True
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
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
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
            params_est, seqs_train, fit_info = gpfa_engine(
                seq_train=seqs_train,
                seq_test=seqs_test,
                x_dim=x_dim,
                bin_width=bin_size,
                min_var_frac=min_var_frac,
                em_max_iters=em_max_iters)
        elif method in ['fa', 'ppca', 'pca']:
            # TODO: implement two_stage_engine()
            params_est, seqs_train, fit_info = two_stage_engine(
                seqTrain=seqs_train,
                seqTest=seqs_test,
                typ=method,
                xDim=x_dim,
                binWidth=bin_size)
        else:
            raise ValueError("Invalid method: {}".format(method))

        fit_info['method'] = method
        fit_info['cvf'] = cvf
        fit_info['has_spikes_bool'] = has_spikes_bool
        fit_info['min_var_frac'] = min_var_frac
        fit_info['bin_size'] = bin_size

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
    if hasattr(fit_info, 'kern'):
        # FIXME `k` and `kern_sd` are not used
        if not fit_info['kernSDList']:
            k = 1
        else:
            k = np.where(np.array(fit_info['kernSDList']) ==
                         np.array(kern_sd))[0]
            if not k:
                raise ValueError('Selected kernSD not found')
    if fit_info['method'] == 'gpfa':
        C = params_est['C']
        X = np.hstack(seqs_train['xsm'])
        Xorth, Corth, _ = gpfa_util.orthogonalize(X, C)
        seqs_train = gpfa_util.segment_by_trial(seqs_train, Xorth, 'xorth')

        params_est['Corth'] = Corth

        if seqs_test is not None:
            print("Extracting neural trajectories for test data...\n")
            seqs_test = exact_inference_with_ll(seqs_test, params_est)
            X = np.hstack(seqs_test['xsm'])
            Xorth, Corth, _ = gpfa_util.orthogonalize(X, C)
            seqs_test = gpfa_util.segment_by_trial(seqs_test, Xorth, 'xorth')
    else:
        # TODO: implement postprocessing for the two stage methods
        raise NotImplementedError

    return params_est, seqs_train, seqs_test
