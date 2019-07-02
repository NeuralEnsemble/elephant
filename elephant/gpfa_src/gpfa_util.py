import warnings

import numpy as np
import scipy as sp

from elephant.conversion import BinnedSpikeTrain
from elephant.utils import check_quantities


def get_seq(data, bin_size, use_sqrt=True):
    """
    Converts the data into a rec array using internally BinnedSpikeTrain.

    Parameters
    ----------

    data
        structure whose nth entry (corresponding to the nth
        experimental trial) has fields
            * trialId: unique trial identifier
            * spikes: 0/1 matrix of the raw spiking activity across
                       all neurons. Each row corresponds to a neuron.
                       Each column corresponds to a 1 msec timestep.
    bin_size: quantity.Quantity
        Spike bin width

    use_sqrt: bool
        Logical specifying whether or not to use square-root transform on
        spike counts
        Default is  True

    Returns
    -------

    seq
        data structure, whose nth entry (corresponding to the nth experimental
        trial) has fields
            * trialId: unique trial identifier
            * T: (1 x 1) number of timesteps
            * y: (yDim x T) neural data

    Raises
    ------
    ValueError
        if `bin_size` is not a pq.Quantity.

    """
    check_quantities(bin_size, 'bin_size')

    seq = []
    for dat in data:
        trial_id = dat[0]
        sts = dat[1]
        binned_sts = BinnedSpikeTrain(sts, binsize=bin_size)
        if use_sqrt:
            binned = np.sqrt(binned_sts.to_array())
        else:
            binned = binned_sts.to_array()
        seq.append(
            (trial_id, binned_sts.num_bins, binned))
    seq = np.array(seq,
                   dtype=[('trialId', np.int), ('T', np.int),
                          ('y', 'O')])

    # Remove trials that are shorter than one bin width
    if len(seq) > 0:
        trials_to_keep = seq['T'] > 0
        seq = seq[trials_to_keep]

    return seq


def cut_trials(seq_in, seg_length=20):
    """
    Extracts trial segments that are all of the same length.  Uses
    overlapping segments if trial length is not integer multiple
    of segment length.  Ignores trials with length shorter than
    one segment length.

    Parameters
    ----------

    seq_in
        data structure, whose nth entry (corresponding to
        the nth experimental trial) has fields
            * trialId: unique trial identifier
            * T: (1 x 1) number of timesteps in trial
            * y: (yDim x T) neural data

    seg_length : int
        length of segments to extract, in number of timesteps. If infinite,
        entire trials are extracted, i.e., no segmenting.
        Default is 20


    Returns
    -------

    seqOut
        data structure, whose nth entry (corresponding to
        the nth experimental trial) has fields
            * trialId: identifier of trial from which segment was taken
            * segId: segment identifier within trial
            * T: (1 x 1) number of timesteps in segment
            * y: (yDim x T) neural data
    """
    assert seg_length > 0, "At least 1 extracted trial must be returned"
    if np.isinf(seg_length):
        seqOut = seq_in
        return seqOut

    dtype_seqOut = [('trialId', np.int), ('segId', np.int), ('T', np.int),
                    ('y', np.object)]
    seqOut_buff = []
    for n, seqIn_n in enumerate(seq_in):
        T = seqIn_n['T']

        # Skip trials that are shorter than segLength
        if T < seg_length:
            warnings.warn('trialId {0:4d} shorter than one segLength...'
                          'skipping'.format(seqIn_n['trialId']))
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
        seg['trialId'] = seqIn_n['trialId']
        seg['T'] = seg_length

        for s, seg_s in enumerate(seg):
            tStart = seg_length * s - cumOL[s]

            seg_s['segId'] = s
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

    params
        GPFA model parameters
    n_timesteps : int
        number of timesteps

    Returns
    -------

    K_big
        GP covariance matrix with dimensions (xDim * T) x (xDim * T).
                   The (t1, t2) block is diagonal, has dimensions xDim x xDim,
                   and represents the covariance between the state vectors at
                   timesteps t1 and t2.  K_big is sparse and striped.
    K_big_inv
        Inverse of K_big
    logdet_K_big
        Log determinant of K_big
    """
    xDim = params['C'].shape[1]

    K_big = np.zeros((xDim * n_timesteps, xDim * n_timesteps))
    K_big_inv = np.zeros((xDim * n_timesteps, xDim * n_timesteps))
    Tdif = np.tile(np.arange(0, n_timesteps), (n_timesteps, 1)).T \
        - np.tile(np.arange(0, n_timesteps), (n_timesteps, 1))
    logdet_K_big = 0

    for i in range(xDim):
        if params['covType'] == 'rbf':
            K = (1 - params['eps'][i]) * np.exp(-params['gamma'][i] / 2 *
                                                Tdif ** 2) \
                + params['eps'][i] * np.eye(n_timesteps)
        elif params['covType'] == 'tri':
            K = np.maximum(1 - params['eps'][i] - params['a'][i] *
                           np.abs(Tdif), 0) \
                + params['eps'][i] * np.eye(n_timesteps)
        elif params['covType'] == 'logexp':
            z = params['gamma'] \
                * (1 - params['eps'][i] - params['a'][i] * np.abs(Tdif))
            outUL = (z > 36)
            outLL = (z < -19)
            inLim = ~outUL & ~outLL

            hz = np.full(z.shape, np.nan)
            hz[outUL] = z[outUL]
            hz[outLL] = np.exp(z[outLL])
            hz[inLim] = np.log(1 + np.exp(z[inLim]))

            K = hz / params['gamma'] + params['eps'][i] * np.eye(n_timesteps)

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

    M: numpy.ndarray
        The block persymmetric matrix to be inverted
        ((blkSize*T) x (blkSize*T)).
        Each block is blkSize x blkSize, arranged in a T x T grid.
    blk_size: int
        Edge length of one block

    Returns
    -------
    invM
        Inverse of M ((blkSize*T) x (blkSize*T))
    logdet_M
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

     p_in
        Top half of block persymmetric matrix (xDim*Thalf) x (xDim*T),
        where Thalf = ceil(T/2)
     blk_size : int
        Edge length of one block
     n_blocks : int
        Number of blocks making up a row of Pin
     blk_size_vert : int, optional
        Vertical block edge length if blocks are not square.
        `blk_size` is assumed to be the horizontal block edge length.

     Returns
     -------

     Pout
        Full block persymmetric matrix (xDim*T) x (xDim*T)
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


def make_precomp(seq, xDim):
    """
    Make the precomputation matrices specified by the GPFA algorithm.

    Usage: [precomp] = makePautoSum( seq , xDim )

    Parameters
    ----------

    seq
        The sequence struct of inferred latents, etc.
    xDim : int
       The dimension of the latent space.

    Returns
    -------

    precomp
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

    Tall = seq['T']
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
    Tu = np.unique(Tall)
    # Loop once for each state dimension (each GP)
    for i in range(xDim):
        precomp_Tu = np.empty(len(Tu), dtype=[(
            'nList', np.object), ('T', np.int), ('numTrials', np.int),
            ('PautoSUM', np.object)])
        for j in range(len(Tu)):
            T = Tu[j]
            precomp_Tu[j]['nList'] = np.where(Tall == T)[0]
            precomp_Tu[j]['T'] = T
            precomp_Tu[j]['numTrials'] = len(precomp_Tu[j]['nList'])
            precomp_Tu[j]['PautoSUM'] = np.zeros((T, T))
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
        for j in range(len(Tu)):
            # Loop once for each trial (each of nList)
            for n in precomp[i]['Tu'][j]['nList']:
                precomp[i]['Tu'][j]['PautoSUM'] += seq[n]['VsmGP'][:, :, i] \
                    + np.outer(
                    seq[n]['xsm'][i, :], seq[n]['xsm'][i, :])
    return precomp


def grad_betgam(p, pre_comp, const):
    """
    Gradient computation for GP timescale optimization.
    This function is called by minimize.m.

    Parameters
    ----------

    p
        variable with respect to which optimization is performed,
        where p = log(1 / timescale ^2)
    pre_comp
        structure containing precomputations
    const
        contains hyperparameters

    Returns
    -------

    f           - value of objective function E[log P({x},{y})] at p
    df          - gradient at p
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


def orthogonalize(x, l):
    """

    Orthonormalize the columns of the loading matrix and
    apply the corresponding linear transform to the latent variables.

     yDim: data dimensionality
     xDim: latent dimensionality

    Parameters
    ----------

    x : np.ndarray
        Latent variables (xDim x T)
    l : np.ndarray
        Loading matrix (yDim x xDim)

    Returns
    -------

    Xorth
        Orthonormalized latent variables (xDim x T)
    Lorth
        Orthonormalized loading matrix (yDim x xDim)
    TT
       Linear transform applied to latent variables (xDim x xDim)
    """
    xDim = l.shape[1]
    if xDim == 1:
        TT = np.sqrt(np.dot(l.T, l))
        Lorth = rdiv(l, TT)
        Xorth = np.dot(TT, x)
    else:
        UU, DD, VV = sp.linalg.svd(l, full_matrices=False)
        # TT is transform matrix
        TT = np.dot(np.diag(DD), VV)

        Lorth = UU
        Xorth = np.dot(TT, x)
    return Xorth, Lorth, TT


def segment_by_trial(seq, x, fn):
    """
    Segment and store data by trial.

    Parameters
    ----------

    seq
        Data structure that has field T, the number of timesteps
    x : np.ndarray
        Data to be segmented (any dimensionality x total number of timesteps)
    fn
        New field name of seq where segments of X are stored

    Returns
    -------

    seq_new
        Data structure with new field `fn`

    Raises
    ------
    ValueError
        If `seq['T']) != x.shape[1]`.

    """
    if np.sum(seq['T']) != x.shape[1]:
        raise (ValueError, 'size of X incorrect.')

    dtype_new = [(i, seq[i].dtype) for i in seq.dtype.names]
    dtype_new.append((fn, np.object))
    seq_new = np.empty(len(seq), dtype=dtype_new)
    for dtype_name in seq.dtype.names:
        seq_new[dtype_name] = seq[dtype_name]

    ctr = 0
    for n, T in enumerate(seq['T']):
        seq_new[n][fn] = x[:, ctr:ctr + T]
        ctr += T

    return seq_new
