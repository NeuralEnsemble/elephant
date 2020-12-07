# -*- coding: utf-8 -*-
"""
GPFA util functions.

:copyright: Copyright 2014-2020 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings

import numpy as np
import quantities as pq
import scipy as sp

from elephant.conversion import BinnedSpikeTrain
from elephant.utils import deprecated_alias


@deprecated_alias(binsize='bin_size')
def get_seqs(data, bin_size, use_sqrt=True):
    """
    Converts the data into a rec array using internally BinnedSpikeTrain.

    Parameters
    ----------
    data : list of list of neo.SpikeTrain
        The outer list corresponds to trials and the inner list corresponds to
        the neurons recorded in that trial, such that data[l][n] is the
        spike train of neuron n in trial l. Note that the number and order of
        neo.SpikeTrains objects per trial must be fixed such that data[l][n]
        and data[k][n] refer to the same spike generator for any choice of l,k
        and n.
    bin_size: quantity.Quantity
        Spike bin width

    use_sqrt: bool
        Boolean specifying whether or not to use square-root transform on
        spike counts (see original paper for motivation).
        Default: True

    Returns
    -------
    seq : np.recarray
        data structure, whose nth entry (corresponding to the nth experimental
        trial) has fields
        T : int
            number of timesteps in the trial
        y : (yDim, T) np.ndarray
            neural data

    Raises
    ------
    ValueError
        if `bin_size` is not a pq.Quantity.
    """
    if not isinstance(bin_size, pq.Quantity):
        raise ValueError("'bin_size' must be of type pq.Quantity")

    seqs = []
    for dat in data:
        sts = dat
        binned_spiketrain = BinnedSpikeTrain(sts, bin_size=bin_size)
        if use_sqrt:
            binned = np.sqrt(binned_spiketrain.to_array())
        else:
            binned = binned_spiketrain.to_array()
        seqs.append(
            (binned_spiketrain.n_bins, binned))
    seqs = np.array(seqs, dtype=[('T', np.int), ('y', 'O')])

    # Remove trials that are shorter than one bin width
    if len(seqs) > 0:
        trials_to_keep = seqs['T'] > 0
        seqs = seqs[trials_to_keep]

    return seqs


def cut_trials(seq_in, seg_length=20):
    """
    Extracts trial segments that are all of the same length.  Uses
    overlapping segments if trial length is not integer multiple
    of segment length.  Ignores trials with length shorter than
    one segment length.

    Parameters
    ----------
    seq_in : np.recarray
        data structure, whose nth entry (corresponding to the nth experimental
        trial) has fields
        T : int
            number of timesteps in trial
        y : (yDim, T) np.ndarray
            neural data

    seg_length : int
        length of segments to extract, in number of timesteps. If infinite,
        entire trials are extracted, i.e., no segmenting.
        Default: 20

    Returns
    -------
    seqOut : np.recarray
        data structure, whose nth entry (corresponding to the nth experimental
        trial) has fields
        T : int
            number of timesteps in segment
        y : (yDim, T) np.ndarray
            neural data

    Raises
    ------
    ValueError
        If `seq_length == 0`.

    """
    if seg_length == 0:
        raise ValueError("At least 1 extracted trial must be returned")
    if np.isinf(seg_length):
        seqOut = seq_in
        return seqOut

    dtype_seqOut = [('segId', np.int), ('T', np.int),
                    ('y', np.object)]
    seqOut_buff = []
    for n, seqIn_n in enumerate(seq_in):
        T = seqIn_n['T']

        # Skip trials that are shorter than segLength
        if T < seg_length:
            warnings.warn(
                'trial corresponding to index {} shorter than one segLength...'
                'skipping'.format(n))
            continue

        numSeg = np.int(np.ceil(float(T) / seg_length))

        # Randomize the sizes of overlaps
        if numSeg == 1:
            cumOL = np.array([0, ])
        else:
            totalOL = (seg_length * numSeg) - T
            probs = np.ones(numSeg - 1, np.float) / (numSeg - 1)
            randOL = np.random.multinomial(totalOL, probs)
            cumOL = np.hstack([0, np.cumsum(randOL)])

        seg = np.empty(numSeg, dtype_seqOut)
        seg['T'] = seg_length

        for s, seg_s in enumerate(seg):
            tStart = seg_length * s - cumOL[s]
            seg_s['y'] = seqIn_n['y'][:, tStart:tStart + seg_length]

        seqOut_buff.append(seg)

    if len(seqOut_buff) > 0:
        seqOut = np.hstack(seqOut_buff)
    else:
        seqOut = np.empty(0, dtype_seqOut)

    return seqOut


def rdiv(a, b):
    """
    Returns the solution to x b = a. Equivalent to MATLAB right matrix
    division: a / b
    """
    return np.linalg.solve(b.T, a.T).T


def logdet(A):
    """
    log(det(A)) where A is positive-definite.
    This is faster and more stable than using log(det(A)).

    Written by Tom Minka
    (c) Microsoft Corporation. All rights reserved.
    """
    U = np.linalg.cholesky(A)
    return 2 * (np.log(np.diag(U))).sum()


def make_k_big(params, n_timesteps):
    """
    Constructs full GP covariance matrix across all state dimensions and
    timesteps.

    Parameters
    ----------
    params : dict
        GPFA model parameters
    n_timesteps : int
        number of timesteps

    Returns
    -------
    K_big : np.ndarray
        GP covariance matrix with dimensions (xDim * T) x (xDim * T).
        The (t1, t2) block is diagonal, has dimensions xDim x xDim, and
        represents the covariance between the state vectors at timesteps t1 and
        t2. K_big is sparse and striped.
    K_big_inv : np.ndarray
        Inverse of K_big
    logdet_K_big : float
        Log determinant of K_big

    Raises
    ------
    ValueError
        If `params['covType'] != 'rbf'`.

    """
    if params['covType'] != 'rbf':
        raise ValueError("Only 'rbf' GP covariance type is supported.")

    xDim = params['C'].shape[1]

    K_big = np.zeros((xDim * n_timesteps, xDim * n_timesteps))
    K_big_inv = np.zeros((xDim * n_timesteps, xDim * n_timesteps))
    Tdif = np.tile(np.arange(0, n_timesteps), (n_timesteps, 1)).T \
        - np.tile(np.arange(0, n_timesteps), (n_timesteps, 1))
    logdet_K_big = 0

    for i in range(xDim):
        K = (1 - params['eps'][i]) * np.exp(-params['gamma'][i] / 2 *
                                            Tdif ** 2) \
            + params['eps'][i] * np.eye(n_timesteps)
        K_big[i::xDim, i::xDim] = K
        # the original MATLAB program uses here a special algorithm, provided
        # in C and MEX, for inversion of Toeplitz matrix:
        # [K_big_inv(idx+i, idx+i), logdet_K] = invToeplitz(K);
        # TODO: use an inversion method optimized for Toeplitz matrix
        # Below is an attempt to use such a method, not leading to a speed-up.
        # # K_big_inv[i::xDim, i::xDim] = sp.linalg.solve_toeplitz((K[:, 0],
        # K[0, :]), np.eye(T))
        K_big_inv[i::xDim, i::xDim] = np.linalg.inv(K)
        logdet_K = logdet(K)

        logdet_K_big = logdet_K_big + logdet_K

    return K_big, K_big_inv, logdet_K_big


def inv_persymm(M, blk_size):
    """
    Inverts a matrix that is block persymmetric.  This function is
    faster than calling inv(M) directly because it only computes the
    top half of inv(M).  The bottom half of inv(M) is made up of
    elements from the top half of inv(M).

    WARNING: If the input matrix M is not block persymmetric, no
    error message will be produced and the output of this function will
    not be meaningful.

    Parameters
    ----------
    M : (blkSize*T, blkSize*T) np.ndarray
        The block persymmetric matrix to be inverted.
        Each block is blkSize x blkSize, arranged in a T x T grid.
    blk_size : int
        Edge length of one block

    Returns
    -------
    invM : (blkSize*T, blkSize*T) np.ndarray
        Inverse of M
    logdet_M : float
        Log determinant of M
    """
    T = int(M.shape[0] / blk_size)
    Thalf = np.int(np.ceil(T / 2.0))
    mkr = blk_size * Thalf

    invA11 = np.linalg.inv(M[:mkr, :mkr])
    invA11 = (invA11 + invA11.T) / 2

    # Multiplication of a sparse matrix by a dense matrix is not supported by
    # SciPy. Making A12 a sparse matrix here  an error later.
    off_diag_sparse = False
    if off_diag_sparse:
        A12 = sp.sparse.csr_matrix(M[:mkr, mkr:])
    else:
        A12 = M[:mkr, mkr:]

    term = invA11.dot(A12)
    F22 = M[mkr:, mkr:] - A12.T.dot(term)

    res12 = rdiv(-term, F22)
    res11 = invA11 - res12.dot(term.T)
    res11 = (res11 + res11.T) / 2

    # Fill in bottom half of invM by picking elements from res11 and res12
    invM = fill_persymm(np.hstack([res11, res12]), blk_size, T)

    logdet_M = -logdet(invA11) + logdet(F22)

    return invM, logdet_M


def fill_persymm(p_in, blk_size, n_blocks, blk_size_vert=None):
    """
     Fills in the bottom half of a block persymmetric matrix, given the
     top half.

     Parameters
     ----------
     p_in :  (xDim*Thalf, xDim*T) np.ndarray
        Top half of block persymmetric matrix, where Thalf = ceil(T/2)
     blk_size : int
        Edge length of one block
     n_blocks : int
        Number of blocks making up a row of Pin
     blk_size_vert : int, optional
        Vertical block edge length if blocks are not square.
        `blk_size` is assumed to be the horizontal block edge length.

     Returns
     -------
     Pout : (xDim*T, xDim*T) np.ndarray
        Full block persymmetric matrix
    """
    if blk_size_vert is None:
        blk_size_vert = blk_size

    Nh = blk_size * n_blocks
    Nv = blk_size_vert * n_blocks
    Thalf = np.int(np.floor(n_blocks / 2.0))
    THalf = np.int(np.ceil(n_blocks / 2.0))

    Pout = np.empty((blk_size_vert * n_blocks, blk_size * n_blocks))
    Pout[:blk_size_vert * THalf, :] = p_in
    for i in range(Thalf):
        for j in range(n_blocks):
            Pout[Nv - (i + 1) * blk_size_vert:Nv - i * blk_size_vert,
                 Nh - (j + 1) * blk_size:Nh - j * blk_size] \
                = p_in[i * blk_size_vert:(i + 1) *
                       blk_size_vert,
                       j * blk_size:(j + 1) * blk_size]

    return Pout


def make_precomp(seqs, xDim):
    """
    Make the precomputation matrices specified by the GPFA algorithm.

    Usage: [precomp] = makePautoSum( seq , xDim )

    Parameters
    ----------
    seqs : np.recarray
        The sequence struct of inferred latents, etc.
    xDim : int
       The dimension of the latent space.

    Returns
    -------
    precomp : np.recarray
        The precomp struct will be updated with the posterior covaraince and
        the other requirements.

    Notes
    -----
    All inputs are named sensibly to those in `learnGPparams`.
    This code probably should not be called from anywhere but there.

    We bother with this method because we
    need this particular matrix sum to be
    as fast as possible.  Thus, no error checking
    is done here as that would add needless computation.
    Instead, the onus is on the caller (which should be
    learnGPparams()) to make sure this is called correctly.

    Finally, see the notes in the GPFA README.
    """

    Tall = seqs['T']
    Tmax = (Tall).max()
    Tdif = np.tile(np.arange(0, Tmax), (Tmax, 1)).T \
        - np.tile(np.arange(0, Tmax), (Tmax, 1))

    # assign some helpful precomp items
    # this is computationally cheap, so we keep a few loops in MATLAB
    # for ease of readability.
    precomp = np.empty(xDim, dtype=[(
        'absDif', np.object), ('difSq', np.object), ('Tall', np.object),
        ('Tu', np.object)])
    for i in range(xDim):
        precomp[i]['absDif'] = np.abs(Tdif)
        precomp[i]['difSq'] = Tdif ** 2
        precomp[i]['Tall'] = Tall
    # find unique numbers of trial lengths
    trial_lengths_num_unique = np.unique(Tall)
    # Loop once for each state dimension (each GP)
    for i in range(xDim):
        precomp_Tu = np.empty(len(trial_lengths_num_unique), dtype=[(
            'nList', np.object), ('T', np.int), ('numTrials', np.int),
            ('PautoSUM', np.object)])
        for j, trial_len_num in enumerate(trial_lengths_num_unique):
            precomp_Tu[j]['nList'] = np.where(Tall == trial_len_num)[0]
            precomp_Tu[j]['T'] = trial_len_num
            precomp_Tu[j]['numTrials'] = len(precomp_Tu[j]['nList'])
            precomp_Tu[j]['PautoSUM'] = np.zeros((trial_len_num,
                                                  trial_len_num))
            precomp[i]['Tu'] = precomp_Tu

    # at this point the basic precomp is built.  The previous steps
    # should be computationally cheap.  We now try to embed the
    # expensive computation in a MEX call, defaulting to MATLAB if
    # this fails.  The expensive computation is filling out PautoSUM,
    # which we initialized previously as zeros.

    ############################################################
    # Fill out PautoSum
    ############################################################
    # Loop once for each state dimension (each GP)
    for i in range(xDim):
        # Loop once for each trial length (each of Tu)
        for j in range(len(trial_lengths_num_unique)):
            # Loop once for each trial (each of nList)
            for n in precomp[i]['Tu'][j]['nList']:
                precomp[i]['Tu'][j]['PautoSUM'] += seqs[n]['VsmGP'][:, :, i] \
                    + np.outer(seqs[n]['latent_variable'][i, :],
                               seqs[n]['latent_variable'][i, :])
    return precomp


def grad_betgam(p, pre_comp, const):
    """
    Gradient computation for GP timescale optimization.
    This function is called by minimize.m.

    Parameters
    ----------
    p : float
        variable with respect to which optimization is performed,
        where :math:`p = log(1 / timescale^2)`
    pre_comp : np.recarray
        structure containing precomputations
    const : dict
        contains hyperparameters

    Returns
    -------
    f : float
        value of objective function E[log P({x},{y})] at p
    df : float
        gradient at p
    """
    Tall = pre_comp['Tall']
    Tmax = Tall.max()

    # temp is Tmax x Tmax
    temp = (1 - const['eps']) * np.exp(-np.exp(p) / 2 * pre_comp['difSq'])
    Kmax = temp + const['eps'] * np.eye(Tmax)
    dKdgamma_max = -0.5 * temp * pre_comp['difSq']

    dEdgamma = 0
    f = 0
    for j in range(len(pre_comp['Tu'])):
        T = pre_comp['Tu'][j]['T']
        Thalf = np.int(np.ceil(T / 2.0))

        Kinv = np.linalg.inv(Kmax[:T, :T])
        logdet_K = logdet(Kmax[:T, :T])

        KinvM = Kinv[:Thalf, :].dot(dKdgamma_max[:T, :T])  # Thalf x T
        KinvMKinv = (KinvM.dot(Kinv)).T  # Thalf x T

        dg_KinvM = np.diag(KinvM)
        tr_KinvM = 2 * dg_KinvM.sum() - np.fmod(T, 2) * dg_KinvM[-1]

        mkr = np.int(np.ceil(0.5 * T ** 2))
        numTrials = pre_comp['Tu'][j]['numTrials']
        PautoSUM = pre_comp['Tu'][j]['PautoSUM']

        pauto_kinv_dot = PautoSUM.ravel('F')[:mkr].dot(
            KinvMKinv.ravel('F')[:mkr])
        pauto_kinv_dot_rest = PautoSUM.ravel('F')[-1:mkr - 1:- 1].dot(
            KinvMKinv.ravel('F')[:(T ** 2 - mkr)])
        dEdgamma = dEdgamma - 0.5 * numTrials * tr_KinvM \
            + 0.5 * pauto_kinv_dot \
            + 0.5 * pauto_kinv_dot_rest

        f = f - 0.5 * numTrials * logdet_K \
            - 0.5 * (PautoSUM * Kinv).sum()

    f = -f
    # exp(p) is needed because we're computing gradients with
    # respect to log(gamma), rather than gamma
    df = -dEdgamma * np.exp(p)

    return f, df


def orthonormalize(x, l):
    """
    Orthonormalize the columns of the loading matrix and apply the
    corresponding linear transform to the latent variables.
    In the following description, yDim and xDim refer to data dimensionality
    and latent dimensionality, respectively.

    Parameters
    ----------
    x :  (xDim, T) np.ndarray
        Latent variables
    l :  (yDim, xDim) np.ndarray
        Loading matrix

    Returns
    -------
    latent_variable_orth : (xDim, T) np.ndarray
        Orthonormalized latent variables
    Lorth : (yDim, xDim) np.ndarray
        Orthonormalized loading matrix
    TT :  (xDim, xDim) np.ndarray
       Linear transform applied to latent variables
    """
    xDim = l.shape[1]
    if xDim == 1:
        TT = np.sqrt(np.dot(l.T, l))
        Lorth = rdiv(l, TT)
        latent_variable_orth = np.dot(TT, x)
    else:
        UU, DD, VV = sp.linalg.svd(l, full_matrices=False)
        # TT is transform matrix
        TT = np.dot(np.diag(DD), VV)

        Lorth = UU
        latent_variable_orth = np.dot(TT, x)
    return latent_variable_orth, Lorth, TT


def segment_by_trial(seqs, x, fn):
    """
    Segment and store data by trial.

    Parameters
    ----------
    seqs : np.recarray
        Data structure that has field T, the number of timesteps
    x : np.ndarray
        Data to be segmented (any dimensionality x total number of timesteps)
    fn : str
        New field name of seq where segments of X are stored

    Returns
    -------
    seqs_new : np.recarray
        Data structure with new field `fn`

    Raises
    ------
    ValueError
        If `seqs['T']) != x.shape[1]`.

    """
    if np.sum(seqs['T']) != x.shape[1]:
        raise ValueError('size of X incorrect.')

    dtype_new = [(i, seqs[i].dtype) for i in seqs.dtype.names]
    dtype_new.append((fn, np.object))
    seqs_new = np.empty(len(seqs), dtype=dtype_new)
    for dtype_name in seqs.dtype.names:
        seqs_new[dtype_name] = seqs[dtype_name]

    ctr = 0
    for n, T in enumerate(seqs['T']):
        seqs_new[n][fn] = x[:, ctr:ctr + T]
        ctr += T

    return seqs_new
