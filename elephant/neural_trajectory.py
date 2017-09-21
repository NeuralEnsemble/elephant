"""
@ 2009
Byron Yu
byronyu @ stanford.edu
John Cunningham
jcunnin @ stanford.edu

"""

import numpy as np
import neo

from neural_trajectory_src.gpfa import gpfa_engine, exact_inference_with_ll, \
    two_stage_engine
from neural_trajectory_src import util


def neural_trajectory(data, method='gpfa', bin_size=20, num_folds=0, x_dim=3):
    """
    Prepares data and calls functions for extracting neural trajectories.

    Parameters
    ----------

    data: list containing following structure
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
    num_folds: int
        Number of cross-validation folds, 0 indicates no cross-validation,
        i.e. train on all trials.        
        Default is 0             
    x_dim: int
        State dimensionality 
        Default is 3

    Returns
    -------

    result: Dictionary
        Returns the results of the specified method, with following keys and
        values:
        * iteration_time: A list containing the runtime for each iteration step
        in the EM algorithm
        * log_likelihood: float, maximized likelihood obtained in the E-step of the
        EM algorithm
        * seqTrain: numpy rec array, a copy of the training data structure,
        augmented by new fields
            xsm: ndarray of shape (#latent_vars x #bins)
                  posterior mean of latent variables at each time bin
            Vsm: ndarray of shape (#latent_vars, #latent_vars, #bins)
                 posterior covariance between latent variables at each
                 timepoint
             VsmGP: ndarray of shape (#bins, #bins, #latent_vars)
                    posterior covariance over time for each latent variable
        * bin_size: int, Width of the bins
        * cvf: int, number for cross-validation folding
            Default is 0 (no cross-validation)
        * parameter_estimates: dict, estimated GPFA model parameters
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
        * hasSpikesBool: Indicates if a neuron has any spikes across trials
        * method: String, method name
    """
    if not isinstance(data[0][1], neo.SpikeTrain):
        raise ValueError("structure of the data is not correct: 0-axis should "
                         "be trials, 1-axis neo spike trains "
                         "and 2-axis spike times")
    # Obtain binned spike counts
    seq = util.get_seq(data, bin_size)
    if len(seq) == 0:
        raise ValueError("Error: No valid trials.")
    # Set cross-validation folds
    N = len(seq)
    fdiv = np.linspace(0, N, num_folds + 1).astype(np.int) if num_folds > 0 \
        else N

    result = {}
    for cvf in range(0, num_folds+1):
        if cvf == 0:
            print("\n===== Training on all data =====")
        else:
            # print("\n===== Cross-validation fold {} of {} =====").\
            #     format(cvf, num_folds)
            raise NotImplementedError

        # Set cross-validataion masks
        testMask = np.full(N, False, dtype=bool)
        if cvf > 0:
            testMask[fdiv[cvf]:fdiv[cvf+1]] = True
        trainMask = ~testMask

        tr = np.arange(N, dtype=np.int)
        if cvf > 0:
            # Randomly reorder trials before partitioning into training and
            # test sets
            np.random.seed(0)
            np.random.shuffle(tr)
        trainTrialIdx = tr[trainMask]
        testTrialIdx = tr[testMask]
        seqTrain = seq[trainTrialIdx]
        seqTest = seq[testTrialIdx]

        # Remove inactive units based on training set
        hasSpikesBool = (np.hstack(seqTrain['y']).mean(1) != 0)

        for seq_train in seqTrain:
            seq_train['y'] = seq_train['y'][hasSpikesBool, :]
        for seq_test in seqTest:
            seq_test['y'] = seq_test['y'][hasSpikesBool, :]

        # Check if training data covariance is full rank
        yAll = np.hstack(seqTrain['y'])
        yDim = yAll.shape[0]

        if np.linalg.matrix_rank(np.cov(yAll)) < yDim:
            errmesg = 'Observation covariance matrix is rank deficient.\n'\
                      'Possible causes: '\
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        print('Number of training trials: {}'.format(len(seqTrain)))
        print('Number of test trials: {}'.format(len(seqTest)))
        print('Latent space dimensionality: {}'.format(x_dim))
        print('Observation dimensionality: {}'.format(hasSpikesBool.sum()))

        # If doing cross-validation, don't use private noise variance floor.
        if cvf == 0:
            minVarFrac = 0.01
        else:
            minVarFrac = -np.inf

        # The following does the heavy lifting.
        if method == 'gpfa':
            result = gpfa_engine(seqTrain, seqTest, x_dim=x_dim,
                                 bin_width=bin_size, min_var_frac=minVarFrac)
        elif method in ['fa', 'ppca', 'pca']:
            # TODO
            result = two_stage_engine(seqTrain, seqTest, typ=method, xDim=x_dim,
                                      binWidth=bin_size)
            raise NotImplementedError

        var_names = ['method', 'cvf', 'hasSpikesBool', 'minVarFrac', 'bin_size']
        result.update(dict([(x, locals()[x]) for x in var_names]))

    return result


def postprocess(ws, kern_sd=[]):
    """
    Orthonormalization and other cleanup.

    Parameters
    ----------

    ws
        Workspace variables returned by neuralTraj.py

    kern_sd
        For two-stage methods, this function returns `seqTrain` and
        `parameter_estimates` corresponding to `kern_sd`.
        Default is `ws.kern(1)`


    Returns
    -------

    parameter_estimates
        Estimated model parameters, including `Corth` obtained by
        orthonormalizing the columns of C
    seqTrain
        Training data structure containing new field `xorth`,
        the orthonormalized neural trajectories
    seqTest
        Test data structure containing orthonormalized neural
        trajectories in `xorth`, obtained using `estParams`

    """
    # TODO: assignopts(who, extraopts)
    estParams = []
    seqTrain = []
    seqTest = []
    # check if array is empty
    if not ws:
        raise ValueError("Input argument 'ws' is empty")

    if hasattr(ws, 'kern'):
        # Check if kernSDList is empty
        if not ws['kernSDList']:
            k = 1
        else:
            k = np.where(np.array(ws.kernSDList) == np.array(kern_sd))[0]
            if not k:
                raise ValueError('Selected kernSD not found')
    if ws['method'] == 'gpfa':
        C = ws['estParams']['C']
        X = np.hstack(ws['seqTrain']['xsm'])
        Xorth, Corth, _ = util.orthogonalize(X, C)
        seqTrain = util.segment_by_trial(ws['seqTrain'], Xorth, 'xorth')

        estParams = ws['estParams']
        estParams['Corth'] = Corth

        if 'seqTest' in ws:
            # Check that seqTest is not empty
            if ws['seqTest']:
                print("Extracting neural trajectories for test data...\n")
                ws['seqTest'] = exact_inference_with_ll(ws['seqTest'], estParams)
                X = np.hstack(ws['seqTest']['xsm'])
                Xorth, Corth, _ = util.orthogonalize(X, C)
                seqTest = util.segment_by_trial(ws['seqTest'], Xorth, 'xorth')

    return estParams, seqTrain, seqTest
