from collections import Iterable

import numpy as np
import scipy as sp

from elephant.conversion import BinnedSpikeTrain


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

    """
    # TODO: revise the docstring to a Python format

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

    seg_length 
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
            print('Warning: trialId {:4d} '.format(seqIn_n['trialId']) +
                  'shorter than one segLength...skipping')
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


def ldiv(a, b):
    """
    Returns the solution to a x = b. Equivalent to MATLAB left matrix
    division: a \ b
    """
    if a.shape[0] == a.shape[1]:
        return np.linalg.solve(a, b)
    else:
        return np.linalg.lstsq(a, b)


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

    params: GPFA model parameters
    n_timesteps: number of timesteps

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
            K = (1 - params['eps'][i]) * np.exp(-params['gamma'][i] / 2 * Tdif**2) \
                + params['eps'][i] * np.eye(n_timesteps)
        elif params['covType'] == 'tri':
            K = np.maximum(1 - params['eps'][i] - params['a'][i] * np.abs(Tdif), 0) \
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
        # Below is an attempt to use such a method, not leading to a speed-up...
        # # K_big_inv[i::xDim, i::xDim] = sp.linalg.solve_toeplitz((K[:, 0], K[0, :]), np.eye(T))
        K_big_inv[i::xDim, i::xDim] = np.linalg.inv(K)
        logdet_K = logdet(K)

        logdet_K_big = logdet_K_big + logdet_K

    return K_big, K_big_inv, logdet_K_big


def inv_persymm(M, blk_size, off_diag_sparse=False):
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
        The block persymmetric matrix to be inverted ((blkSize*T) x (blkSize*T)).  
        Each block is blkSize x blkSize, arranged in a T x T grid.
    blk_size: int 
        Edge length of one block
    off_diag_sparse: bool 
        Logical that specifies whether off-diagonal blocks are sparse (default: false)

    Returns
    -------
    invM     
        Inverse of M ((blkSize*T) x (blkSize*T))
    logdet_M 
        Log determinant of M
    """
    T = np.int(M.shape[0] / blk_size)
    Thalf = np.int(np.ceil(T / 2.0))
    mkr = blk_size * Thalf

    invA11 = np.linalg.inv(M[:mkr, :mkr])
    invA11 = (invA11 + invA11.T) / 2

    # Multiplication of a sparse matrix by a dense matrix is not supported by
    # SciPy. Making A12 a sparse matrix here raises an error later.
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
     blk_size
        Edge length of one block
     n_blocks
        Number of blocks making up a row of Pin
     blk_size_vert
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
                = p_in[i * blk_size_vert:(i + 1) * blk_size_vert, j * blk_size:(j + 1) * blk_size]

    return Pout


def make_precomp(seq, xDim):
    """
    Make the precomputation matrices specified by the GPFA algorithm.

    Usage: [precomp] = makePautoSum( seq , xDim )

    Parameters
    ----------

    seq
        The sequence struct of inferred latents, etc.
    xDim
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
        'absDif', np.object), ('difSq', np.object), ('Tall', np.object), ('Tu', np.object)])
    for i in range(xDim):
        precomp[i]['absDif'] = np.abs(Tdif)
        precomp[i]['difSq'] = Tdif**2
        precomp[i]['Tall'] = Tall
    # find unique numbers of trial lengths
    Tu = np.unique(Tall)
    # Loop once for each state dimension (each GP)
    for i in range(xDim):
        precomp_Tu = np.empty(len(Tu), dtype=[(
            'nList', np.object), ('T', np.int), ('numTrials', np.int), ('PautoSUM', np.object)])
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
                # precomp[i]['Tu'][j]['PautoSUM'] += seq_lat[n]['VsmGP'][:, :, i] + seq_lat[n]['xsm'][i, :].T.dot(seq_lat[n]['xsm'][i, :])
                precomp[i]['Tu'][j]['PautoSUM'] += seq[n]['VsmGP'][:, :, i] \
                    + np.outer(seq[n]['xsm'][i, :], seq[n]['xsm'][i, :])
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

        mkr = np.int(np.ceil(0.5 * T**2))
        numTrials = pre_comp['Tu'][j]['numTrials']
        PautoSUM = pre_comp['Tu'][j]['PautoSUM']

        dEdgamma = dEdgamma - 0.5 * numTrials * tr_KinvM \
            + 0.5 * PautoSUM.ravel('F')[:mkr].dot(KinvMKinv.ravel('F')[:mkr]) \
            + 0.5 * PautoSUM.ravel('F')[-1:mkr - 1:-
                                        1].dot(KinvMKinv.ravel('F')[:(T**2 - mkr)])

        f = f - 0.5 * numTrials * logdet_K \
            - 0.5 * (PautoSUM * Kinv).sum()

    f = -f
    # exp(p) is needed because we're computing gradients with
    # respect to log(gamma), rather than gamma
    df = -dEdgamma * np.exp(p)

    return f, df


def fastfa(x, z_dim, typ='fa', tol=1.0E-8, cyc=10 ** 8, min_var_frac=0.01,
           verbose=False):
    """
    Factor analysis and probabilistic PCA.

      xDim: data dimensionality
      zDim: latent dimensionality
      N:    number of data points

    Parameters
    ----------

    x
        Data matrix (xDim x N)
    z_dim
        Number of factors
    typ
        'fa' or 'ppca'
        Default is 'fa'
    tol
        Stopping criterion for EM
        Default is 1e-8
    cyc
        Maximum number of EM iterations
        Default is 1e8
    min_var_frac
        Fraction of overall data variance for each observed dimension
        to set as the private variance floor.  This is used to combat
        Heywood cases, where ML parameter learning returns one or more
        zero private variances. (default: 0.01)
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    verbose
        Logical that specifies whether to display status messages
        Default is **False**

    Returns
    -------

    estParams.L
        Factor loadings (xDim x zDim)
    estParams.Ph
        Diagonal of uniqueness matrix (xDim x 1)
    estParams.d
        Data mean (xDim x 1)
    LL
        Log likelihood at each EM iteration

    Notes
    -----

    Code adapted from ffa.m by Zoubin Ghahramani.
    """

    xDim, N = x.shape

    # Initialization of parameters
    cX = np.cov(x, bias=True)
    if np.linalg.matrix_rank(cX) == xDim:
        cX_chol = np.linalg.cholesky(cX).T
        scale = np.exp(2 * (np.log(np.diag(cX_chol))).sum() / xDim)
    else:
        # cX may not be full rank because N < xDim
        print('WARNING in fastfa.py: Data matrix is not full rank.')
        r = np.linalg.matrix_rank(cX)
        e = np.sort(np.linalg.eigvals(cX))[::-1]
        scale = sp.stats.gmean(e[:r])

    L = np.random.randn(xDim, z_dim) * np.sqrt(scale / z_dim)
    Ph = np.diag(cX)
    d = x.mean(axis=1)

    varFloor = min_var_frac * np.diag(cX)

    I = np.eye(z_dim)
    const = -xDim / 2.0 * np.log(2 * np.pi)
    LLi = 0
    LL = []

    for i in range(1, cyc + 1):
        # =======
        # E-step
        # =======
        iPh = np.diag(1.0 / Ph)
        iPhL = iPh.dot(L)

        MM = iPh - rdiv(iPhL, I + L.T.dot(iPhL)).dot(iPhL.T)
        beta = L.T.dot(MM)  # zDim x xDim

        cX_beta = cX.dot(beta.T)  # xDim x zDim
        EZZ = I - beta.dot(L) + beta.dot(cX_beta)

        # Compute log likelihood
        LLold = LLi
        MM_chol = np.linalg.cholesky(MM).T
        ldM = (np.log(np.diag(MM_chol))).sum()
        LLi = N * const + N * ldM - 0.5 * N * (MM * cX).sum()
        if verbose:
            print('EM iteration {:5d} lik {:8.1f} \r'.format(i, LLi),)
        LL.append(LLi)

        # =======
        # M-step
        # =======
        L = rdiv(cX_beta, EZZ)
        Ph = np.diag(cX) - (cX_beta * L).sum(1)

        if typ == 'ppca':
            Ph = Ph.mean() * np.ones(xDim)
        if typ == 'fa':
            # Set minimum private variance
            Ph = np.maximum(varFloor, Ph)

        if i <= 2:
            LLbase = LLi
        elif LLi < LLold:
            print('VIOLATION')
        elif (LLi - LLbase) < (1 + tol) * (LLold - LLbase):
            print('iteration {}'.format(i))
            break

    if verbose:
        print()

    if np.any(Ph == varFloor):
        print('Warning: Private variance floor used'
              'for one or more observed dimensions in FA.')

    estParams = {'L': L, 'Ph': Ph, 'd': d}

    return estParams, LL


def minimize(x, f, length, *args):
    """
    Minimize a differentiable multivariate function.

    Usage: x, f_x, i = minimize(x, f, length, P1, P2, P3, ... )

    where the starting point is given by `x` (D by 1), and the function named in
    the string `f`, must return a function value and a vector of partial
    derivatives of f wrt x, the `length` gives the length of the run: if it is
    positive, it gives the maximum number of line searches, if negative its
    absolute gives the maximum allowed number of function evaluations. You can
    (optionally) give `length` a second component, which will indicate the
    reduction in function value to be expected in the first line-search (defaults
    to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.

    The function returns when either its length is up, or if no further progress
    can be made (ie, we are at a (local) minimum, or so close that due to
    numerical problems, we cannot get any closer). NOTE: If the function
    terminates within a few iterations, it could be an indication that the
    function values and derivatives are not consistent (ie, there may be a bug in
    the implementation of your `f` function). The function returns the found
    solution `x`, a vector of function values `f_x` indicating the progress made
    and `i` the number of iterations (line searches or function evaluations,
    depending on the sign of `length`) used.

    The Polack-Ribiere flavour of conjugate gradients is used to compute search
    directions, and a line search using quadratic and cubic polynomial
    approximations and the Wolfe-Powell stopping criteria is used together with
    the slope ratio method for guessing initial step sizes. Additionally a bunch
    of checks are made to make sure that exploration is taking place and that
    extrapolation will not be unboundedly large.

    Parameters
    ----------

    x: numpy.ndarray
        input matrix (Dx1)
    f: string
        Function name
    length: iterable
        Length of the run

    Returns
    -------
    x: float
        A vector of function values
    f_x: list
        Indicates the progress
    i: int
        Number of iterations


    Notes
    -----
    Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

    Permission is granted for anyone to copy, use, or modify these
    programs and accompanying documents for purposes of research or
    education, provided this copyright notice is retained, and note is
    made of any changes that have been made.

    These programs and documents are distributed without any warranty,
    express or implied.  As the programs were written for research
    purposes only, they have not been tested to the degree that would be
    advisable in any important application.  All use of these programs is
    entirely at the user's own risk.
    """
    INT = 0.1  # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0  # extrapolate maximum 3 times the current step-size
    MAX = 20  # max 20 function evaluations per line search
    RATIO = 10  # maximum allowed slope ratio
    SIG = 0.1
    RHO = SIG / 2
    # SIG and RHO are the constants controlling the Wolfe-
    # Powell conditions. SIG is the maximum allowed absolute ratio between
    # previous and new slopes (derivatives in the search direction),
    # thus setting # SIG to low (positive) values forces higher precision
    # in the line-searches. RHO is the minimum allowed fraction of the
    # expected (from the slope at the initial point in the linesearch).
    # Constants must satisfy 0 < RHO < SIG < 1.
    # Tuning of SIG (depending on the nature of the function to be optimized)
    # may speed up the minimization; it is probably not worth playing much
    # with RHO.

    # The code falls naturally into 3 parts, after the initial line search is
    # started in the direction of steepest descent. 1) we first enter a while
    # loop which uses point 1 (p1) and (p2) to compute an extrapolation (p3),
    # until we have extrapolated far enough (Wolfe-Powell conditions).
    # 2) if necessary, we enter the second loop which takes p2, p3 and p4
    # chooses the subinterval containing a (local) minimum, and interpolates
    # it, unil an acceptable point is found (Wolfe-Powell conditions).
    # Note, that points are always maintained
    # in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction
    # using conjugate gradients (Polack-Ribiere flavour), or revert to
    # steepest if there was a problem in the previous line-search.
    # Return the best value so far, if two consecutive line-searches fail,
    # or whenever we run out of function evaluations or line-searches.
    # During extrapolation, the `f` function may fail either with an error or
    # returning Nan or Inf, and minimize should handle this gracefully.

    # In this Python translation we assume x is a scalar, which is valid in
    # typical use cases (such as GPFA without GP noise learning)
    # TODO: extend the code to be applicable to cases where x is a vector

    if isinstance(length, Iterable):
        red = length[1]
        length = length[0]
    else:
        red = 1
        length = length
    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'

    i = 0  # zero the run length counter
    ls_failed = 0  # no previous line search has failed
    f0, df0 = eval(f)(x, *args)  # get function value and gradient
    f_x = [f0, ]
    i += (length < 0)  # count epochs?!
    # initial search direction (steepest) and slope
    s = -df0
    d0 = -s**2
    x3 = red / (1.0 - d0)  # initial step is red/(|s|+1)

    while i < np.abs(length):  # while not finished
        i += (length > 0)  # count iterations?!

        # make a copy of current values
        X0 = x
        F0 = f0
        dF0 = df0
        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        # keep extrapolating as long as necessary
        while True:
            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0
            while not success and M > 0:
                try:
                    M -= 1
                    i += (length < 0)  # count epochs?!
                    f3, df3 = eval(f)(x + x3 * s, *args)
                    if np.isnan(f3) or np.isinf(f3) or np.isnan(df3) or np.isinf(df3):
                        raise ValueError
                    success = 1
                except:  # catch any error which occured in f
                    x3 = (x2 + x3) / 2  # bisect and try again
            if f3 < F0:
                # keep best values
                X0 = x + x3 * s
                F0 = f3
                dF0 = df3
            # new slope
            d3 = df3 * s
            # are we done extrapolating?
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break
            # move point 2 to point 1
            x1 = x2
            f1 = f2
            d1 = d2
            # move point 3 to point 2
            x2 = x3
            f2 = f3
            d2 = d3
            # make cubic extrapolation
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            # num. error possible, ok!
            x3 = x1 - d1 * (x2 - x1)**2 / \
                (B + np.sqrt(B * B - A * d1 * (x2 - x1)))
            # num prob | wrong sign?
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2 * EXT  # extrapolate maximum amount
            elif x3 > x2 * EXT:  # new point beyond extrapolation limit?
                x3 = x2 * EXT  # extrapolate maximum amount
            # new point too close to previous point?
            elif x3 < x2 + INT * (x2 - x1):
                x3 = x2 + INT * (x2 - x1)
                # end extrapolation

        # keep interpolating
        while (np.abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:  # choose subinterval
                # move point 3 to point 4
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                # move point 3 to point 2
                x2 = x3
                f2 = f3
                d2 = d3
            if f4 > f0:
                # quadratic interpolation
                x3 = x2 - (0.5 * d2 * (x4 - x2)**2) / \
                    (f4 - f2 - d2 * (x4 - x2))
            else:
                # cubic interpolation
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                # num. error possible, ok!
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2)**2) - B) / A
            if np.isnan(x3) or np.isinf(x3):
                # if we had a numerical problem then bisect
                x3 = (x2 + x4) / 2
            # don't accept too close
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))
            f3, df3 = eval(f)(x + x3 * s, *args)
            if f3 < F0:
                # keep best values
                X0 = x + x3 * s
                F0 = f3
                dF0 = df3
            M -= 1
            i += (length < 0)  # count epochs?!
            d3 = df3 * s  # new slope
        # end interpolation

        # if line search succeeded
        if np.abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            # update variables
            x += x3 * s
            f0 = f3
            f_x.append(f0)
            # print '{} {:6d};  Value {:4.6e}'.format(S, i, f0)
            s = (df3 * df3 - df0 * df3) / (df0 * df0) * \
                s - df3  # Polack-Ribiere CG direction
            df0 = df3  # swap derivatives
            d3 = d0
            d0 = df0 * s
            # new slope must be negative
            if d0 > 0:
                # otherwise use steepest direction
                s = -df0
                d0 = -s * s
            # slope ratio but max RATIO
            x3 = x3 * min(RATIO, d3 / (d0 - np.finfo(np.float64).tiny))
            ls_failed = 0  # this line search did not fail
        else:
            # restore best point so far
            x = X0
            f0 = F0
            df0 = dF0
            # line search failed twice in a row
            if ls_failed or i > np.abs(length):
                break  # or we ran out of time, so we give up
            # try steepest
            s = -df0
            d0 = -s * s
            x3 = 1 / (1 - d0)
            ls_failed = 1  # this line search failed

    return x, f_x, i


def orthogonalize(x, l):
    """

    Orthonormalize the columns of the loading matrix and
    apply the corresponding linear transform to the latent variables.

     yDim: data dimensionality
     xDim: latent dimensionality

    Parameters
    ----------

    x 
        Latent variables (xDim x T)
    l
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
    x
        Data to be segmented (any dimensionality x total number of timesteps)
    fn
        New field name of seq where segments of X are stored

    Returns
    -------

    seq_new
        Data structure with new field `fn`
    """
    if np.sum(seq['T']) != x.shape[1]:
        raise(ValueError, 'size of X incorrect.')

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
