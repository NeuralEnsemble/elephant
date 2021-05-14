# -*- coding: utf-8 -*-
"""
GPFA core functionality.

:copyright: Copyright 2014-2020 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import time
import warnings

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.sparse as sparse
from sklearn.decomposition import FactorAnalysis
from tqdm import trange

from . import gpfa_util


def fit(seqs_train, x_dim=3, bin_width=20.0, min_var_frac=0.01, em_tol=1.0E-8,
        em_max_iters=500, tau_init=100.0, eps_init=1.0E-3, freq_ll=5,
        verbose=False):
    """
    Fit the GPFA model with the given training data.

    Parameters
    ----------
    seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields
        T : int
            number of bins
        y : (#units, T) np.ndarray
            neural data
    x_dim : int, optional
        state dimensionality
        Default: 3
    bin_width : float, optional
        spike bin width in msec
        Default: 20.0
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    em_max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tau_init : float, optional
        GP timescale initialization in msec
        Default: 100
    eps_init : float, optional
        GP noise variance initialization
        Default: 1e-3
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations. freq_ll = 1
        means that data likelihood is computed at every iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False

    Returns
    -------
    parameter_estimates : dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'logexp'}
                type of GP covariance
            gamma: np.ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: np.ndarray of shape (1, #latent_vars)
                GP noise variances
            d: np.ndarray of shape (#units, 1)
                observation mean
            C: np.ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            R: np.ndarray of shape (#units, #latent_vars)
                observation noise covariance

    fit_info : dict
        Information of the fitting process and the parameters used there
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
    """
    # For compute efficiency, train on equal-length segments of trials
    seqs_train_cut = gpfa_util.cut_trials(seqs_train)
    if len(seqs_train_cut) == 0:
        warnings.warn('No segments extracted for training. Defaulting to '
                      'segLength=Inf.')
        seqs_train_cut = gpfa_util.cut_trials(seqs_train, seg_length=np.inf)

    # ==================================
    # Initialize state model parameters
    # ==================================
    params_init = dict()
    params_init['covType'] = 'rbf'
    # GP timescale
    # Assume binWidth is the time step size.
    params_init['gamma'] = (bin_width / tau_init) ** 2 * np.ones(x_dim)
    # GP noise variance
    params_init['eps'] = eps_init * np.ones(x_dim)

    # ========================================
    # Initialize observation model parameters
    # ========================================
    print('Initializing parameters using factor analysis...')

    y_all = np.hstack(seqs_train_cut['y'])
    fa = FactorAnalysis(n_components=x_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(y_all, bias=True)))
    fa.fit(y_all.T)
    params_init['d'] = y_all.mean(axis=1)
    params_init['C'] = fa.components_.T
    params_init['R'] = np.diag(fa.noise_variance_)

    # Define parameter constraints
    params_init['notes'] = {
        'learnKernelParams': True,
        'learnGPNoise': False,
        'RforceDiagonal': True,
    }

    # =====================
    # Fit model parameters
    # =====================
    print('\nFitting GPFA model...')

    params_est, seqs_train_cut, ll_cut, iter_time = em(
        params_init, seqs_train_cut, min_var_frac=min_var_frac,
        max_iters=em_max_iters, tol=em_tol, freq_ll=freq_ll, verbose=verbose)

    fit_info = {'iteration_time': iter_time, 'log_likelihoods': ll_cut}

    return params_est, fit_info


def em(params_init, seqs_train, max_iters=500, tol=1.0E-8, min_var_frac=0.01,
       freq_ll=5, verbose=False):
    """
    Fits GPFA model parameters using expectation-maximization (EM) algorithm.

    Parameters
    ----------
    params_init : dict
        GPFA model parameters at which EM algorithm is initialized
        covType : {'rbf', 'tri', 'logexp'}
            type of GP covariance
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by
            'bin_width / sqrt(gamma)'
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the
            latent variable space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance
    seqs_train : np.recarray
        training data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
        T : int
            number of bins
        y : np.ndarray (yDim x T)
            neural data
    max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations.
        freq_ll = 1 means that data likelihood is computed at every
        iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False

    Returns
    -------
    params_est : dict
        GPFA model parameter estimates, returned by EM algorithm (same
        format as params_init)
    seqs_latent : np.recarray
        a copy of the training data structure, augmented with the new
        fields:
        latent_variable : np.ndarray of shape (#latent_vars x #bins)
            posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
            posterior covariance between latent variables at each
            timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
            posterior covariance over time for each latent
            variable
    ll : list
        list of log likelihoods after each EM iteration
    iter_time : list
        lisf of computation times (in seconds) for each EM iteration
    """
    params = params_init
    t = seqs_train['T']
    y_dim, x_dim = params['C'].shape
    lls = []
    ll_old = ll_base = ll = 0.0
    iter_time = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(seqs_train['y'])))
    seqs_latent = None

    # Loop once for each iteration of EM algorithm
    for iter_id in trange(1, max_iters + 1, desc='EM iteration',
                          disable=not verbose):
        if verbose:
            print()
        tic = time.time()
        get_ll = (np.fmod(iter_id, freq_ll) == 0) or (iter_id <= 2)

        # ==== E STEP =====
        if not np.isnan(ll):
            ll_old = ll
        seqs_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                  get_ll=get_ll)
        lls.append(ll)

        # ==== M STEP ====
        sum_p_auto = np.zeros((x_dim, x_dim))
        for seq_latent in seqs_latent:
            sum_p_auto += seq_latent['Vsm'].sum(axis=2) \
                + seq_latent['latent_variable'].dot(
                seq_latent['latent_variable'].T)
        y = np.hstack(seqs_train['y'])
        latent_variable = np.hstack(seqs_latent['latent_variable'])
        sum_yxtrans = y.dot(latent_variable.T)
        sum_xall = latent_variable.sum(axis=1)[:, np.newaxis]
        sum_yall = y.sum(axis=1)[:, np.newaxis]

        # term is (xDim+1) x (xDim+1)
        term = np.vstack([np.hstack([sum_p_auto, sum_xall]),
                          np.hstack([sum_xall.T, t.sum().reshape((1, 1))])])
        # yDim x (xDim+1)
        cd = gpfa_util.rdiv(np.hstack([sum_yxtrans, sum_yall]), term)

        params['C'] = cd[:, :x_dim]
        params['d'] = cd[:, -1]

        # yCent must be based on the new d
        # yCent = bsxfun(@minus, [seq.y], currentParams.d);
        # R = (yCent * yCent' - (yCent * [seq.latent_variable]') * \
        #     currentParams.C') / sum(T);
        c = params['C']
        d = params['d'][:, np.newaxis]
        if params['notes']['RforceDiagonal']:
            sum_yytrans = (y * y).sum(axis=1)[:, np.newaxis]
            yd = sum_yall * d
            term = ((sum_yxtrans - d.dot(sum_xall.T)) * c).sum(axis=1)
            term = term[:, np.newaxis]
            r = d ** 2 + (sum_yytrans - 2 * yd - term) / t.sum()

            # Set minimum private variance
            r = np.maximum(var_floor, r)
            params['R'] = np.diag(r[:, 0])
        else:
            sum_yytrans = y.dot(y.T)
            yd = sum_yall.dot(d.T)
            term = (sum_yxtrans - d.dot(sum_xall.T)).dot(c.T)
            r = d.dot(d.T) + (sum_yytrans - yd - yd.T - term) / t.sum()

            params['R'] = (r + r.T) / 2  # ensure symmetry

        if params['notes']['learnKernelParams']:
            res = learn_gp_params(seqs_latent, params, verbose=verbose)
            params['gamma'] = res['gamma']

        t_end = time.time() - tic
        iter_time.append(t_end)

        # Verify that likelihood is growing monotonically
        if iter_id <= 2:
            ll_base = ll
        elif verbose and ll < ll_old:
            print('\nError: Data likelihood has decreased ',
                  'from {0} to {1}'.format(ll_old, ll))
        elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
            break

    if len(lls) < max_iters:
        print('Fitting has converged after {0} EM iterations.)'.format(
            len(lls)))

    if np.any(np.diag(params['R']) == var_floor):
        warnings.warn('Private variance floor used for one or more observed '
                      'dimensions in GPFA.')

    return params, seqs_latent, lls, iter_time


def exact_inference_with_ll(seqs, params, get_ll=True):
    """
    Extracts latent trajectories from neural data, given GPFA model parameters.

    Parameters
    ----------
    seqs : np.recarray
        Input data structure, whose n-th element (corresponding to the n-th
        experimental trial) has fields:
        y : np.ndarray of shape (#units, #bins)
            neural data
        T : int
            number of bins
    params : dict
        GPFA model parameters whe the following fields:
        C : np.ndarray
            FA factor loadings matrix
        d : np.ndarray
            FA mean vector
        R : np.ndarray
            FA noise covariance matrix
        gamma : np.ndarray
            GP timescale
        eps : np.ndarray
            GP noise variance
    get_ll : bool, optional
          specifies whether to compute data log likelihood (default: True)

    Returns
    -------
    seqs_latent : np.recarray
        a copy of the input data structure, augmented with the new
        fields:
        latent_variable :  (#latent_vars, #bins) np.ndarray
              posterior mean of latent variables at each time bin
        Vsm :  (#latent_vars, #latent_vars, #bins) np.ndarray
              posterior covariance between latent variables at each
              timepoint
        VsmGP :  (#bins, #bins, #latent_vars) np.ndarray
                posterior covariance over time for each latent
                variable
    ll : float
        data log likelihood, np.nan is returned when `get_ll` is set False
    """
    y_dim, x_dim = params['C'].shape

    # copy the contents of the input data structure to output structure
    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names]
    dtype_out.extend([('latent_variable', np.object), ('Vsm', np.object),
                      ('VsmGP', np.object)])
    seqs_latent = np.empty(len(seqs), dtype=dtype_out)
    for dtype_name in seqs.dtype.names:
        seqs_latent[dtype_name] = seqs[dtype_name]

    # Precomputations
    if params['notes']['RforceDiagonal']:
        rinv = np.diag(1.0 / np.diag(params['R']))
        logdet_r = (np.log(np.diag(params['R']))).sum()
    else:
        rinv = linalg.inv(params['R'])
        rinv = (rinv + rinv.T) / 2  # ensure symmetry
        logdet_r = gpfa_util.logdet(params['R'])

    c_rinv = params['C'].T.dot(rinv)
    c_rinv_c = c_rinv.dot(params['C'])

    t_all = seqs_latent['T']
    t_uniq = np.unique(t_all)
    ll = 0.

    # Overview:
    # - Outer loop on each element of Tu.
    # - For each element of Tu, find all trials with that length.
    # - Do inference and LL computation for all those trials together.
    for t in t_uniq:
        k_big, k_big_inv, logdet_k_big = gpfa_util.make_k_big(params, t)
        k_big = sparse.csr_matrix(k_big)

        blah = [c_rinv_c for _ in range(t)]
        c_rinv_c_big = linalg.block_diag(*blah)  # (xDim*T) x (xDim*T)
        minv, logdet_m = gpfa_util.inv_persymm(k_big_inv + c_rinv_c_big, x_dim)

        # Note that posterior covariance does not depend on observations,
        # so can compute once for all trials with same T.
        # xDim x xDim posterior covariance for each timepoint
        vsm = np.full((x_dim, x_dim, t), np.nan)
        idx = np.arange(0, x_dim * t + 1, x_dim)
        for i in range(t):
            vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

        # T x T posterior covariance for each GP
        vsm_gp = np.full((t, t, x_dim), np.nan)
        for i in range(x_dim):
            vsm_gp[:, :, i] = minv[i::x_dim, i::x_dim]

        # Process all trials with length T
        n_list = np.where(t_all == t)[0]
        # dif is yDim x sum(T)
        dif = np.hstack(seqs_latent[n_list]['y']) - params['d'][:, np.newaxis]
        # term1Mat is (xDim*T) x length(nList)
        term1_mat = c_rinv.dot(dif).reshape((x_dim * t, -1), order='F')

        # Compute blkProd = CRinvC_big * invM efficiently
        # blkProd is block persymmetric, so just compute top half
        t_half = np.int(np.ceil(t / 2.0))
        blk_prod = np.zeros((x_dim * t_half, x_dim * t))
        idx = range(0, x_dim * t_half + 1, x_dim)
        for i in range(t_half):
            blk_prod[idx[i]:idx[i + 1], :] = c_rinv_c.dot(
                minv[idx[i]:idx[i + 1], :])
        blk_prod = k_big[:x_dim * t_half, :].dot(
            gpfa_util.fill_persymm(np.eye(x_dim * t_half, x_dim * t) -
                                   blk_prod, x_dim, t))
        # latent_variableMat is (xDim*T) x length(nList)
        latent_variable_mat = gpfa_util.fill_persymm(
            blk_prod, x_dim, t).dot(term1_mat)

        for i, n in enumerate(n_list):
            seqs_latent[n]['latent_variable'] = \
                latent_variable_mat[:, i].reshape((x_dim, t), order='F')
            seqs_latent[n]['Vsm'] = vsm
            seqs_latent[n]['VsmGP'] = vsm_gp

        if get_ll:
            # Compute data likelihood
            val = -t * logdet_r - logdet_k_big - logdet_m \
                  - y_dim * t * np.log(2 * np.pi)
            ll = ll + len(n_list) * val - (rinv.dot(dif) * dif).sum() \
                + (term1_mat.T.dot(minv) * term1_mat.T).sum()

    if get_ll:
        ll /= 2
    else:
        ll = np.nan

    return seqs_latent, ll


def learn_gp_params(seqs_latent, params, verbose=False):
    """Updates parameters of GP state model, given neural trajectories.

    Parameters
    ----------
    seqs_latent : np.recarray
        data structure containing neural trajectories;
    params : dict
        current GP state model parameters, which gives starting point
        for gradient optimization;
    verbose : bool, optional
        specifies whether to display status messages (default: False)

    Returns
    -------
    param_opt : np.ndarray
        updated GP state model parameter

    Raises
    ------
    ValueError
        If `params['covType'] != 'rbf'`.
        If `params['notes']['learnGPNoise']` set to True.

    """
    if params['covType'] != 'rbf':
        raise ValueError("Only 'rbf' GP covariance type is supported.")
    if params['notes']['learnGPNoise']:
        raise ValueError("learnGPNoise is not supported.")
    param_name = 'gamma'

    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]
    precomp = gpfa_util.make_precomp(seqs_latent, x_dim)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        const = {'eps': params['eps'][i]}
        initp = np.log(param_init[i])
        res_opt = optimize.minimize(gpfa_util.grad_betgam, initp,
                                    args=(precomp[i], const),
                                    method='L-BFGS-B', jac=True)
        param_opt['gamma'][i] = np.exp(res_opt.x)

        if verbose:
            print('\n Converged p; xDim:{}, p:{}'.format(i, res_opt.x))

    return param_opt


def orthonormalize(params_est, seqs):
    """
    Orthonormalize the columns of the loading matrix C and apply the
    corresponding linear transform to the latent variables.

    Parameters
    ----------
    params_est : dict
        First return value of extract_trajectory() on the training data set.
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
        covType : {'rbf', 'tri', 'logexp'}
            type of GP covariance
            Currently, only 'rbf' is supported.
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by 'bin_width / sqrt(gamma)'
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the latent variable
            space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance

    seqs : np.recarray
        Contains the embedding of the training data into the latent variable
        space.
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
        T : int
          number of timesteps
        y : np.ndarray of shape (#units, #bins)
          neural data
        latent_variable : np.ndarray of shape (#latent_vars, #bins)
          posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
          posterior covariance between latent variables at each
          timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
          posterior covariance over time for each latent variable

    Returns
    -------
    params_est : dict
        Estimated model parameters, including `Corth`, obtained by
        orthonormalizing the columns of C.
    seqs : np.recarray
        Training data structure that contains the new field
        `latent_variable_orth`, the orthonormalized neural trajectories.
    """
    C = params_est['C']
    X = np.hstack(seqs['latent_variable'])
    latent_variable_orth, Corth, _ = gpfa_util.orthonormalize(X, C)
    seqs = gpfa_util.segment_by_trial(
        seqs, latent_variable_orth, 'latent_variable_orth')

    params_est['Corth'] = Corth

    return Corth, seqs
