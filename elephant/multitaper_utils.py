"""
Copyright (c) 2006-2011, NIPY Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NIPY Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Nitime 0.6 code slightly adapted by D. Mingers, Aug 2016
d.mingers@web.de

CONTENT OF THIS FILE:

Miscellaneous utilities for time series analysis.

"""
from __future__ import print_function
import warnings
import numpy as np
import scipy.linalg as linalg
import scipy.signal as sig
import scipy.fftpack as fftpack
import scipy.signal.signaltools as signaltools

# import matplotlib as mpl
from matplotlib import pyplot as plt


#-----------------------------------------------------------------------------
# Stats utils
#-----------------------------------------------------------------------------

def normalize_coherence(x, dof, copy=True):
    """
    The generally accepted choice to transform coherence measures into
    a more normal distribution

    Parameters
    ----------
    x : ndarray, real
       square-root of magnitude-square coherence measures
    dof : int
       number of degrees of freedom in the multitaper model
    copy : bool
        Copy or return inplace modified x.

    Returns
    -------
    y : ndarray, real
        The transformed array.
    """
    if copy:
        x = x.copy()
    np.arctanh(x, x)
    x *= np.sqrt(dof)
    return x


def normal_coherence_to_unit(y, dof, out=None):
    """
    The inverse transform of the above normalization
    """
    if out is None:
        x = y / np.sqrt(dof)
    else:
        y /= np.sqrt(dof)
        x = y
    np.tanh(x, x)
    return x


def expected_jk_variance(K):
    """Compute the expected value of the jackknife variance estimate
    over K windows below. This expected value formula is based on the
    asymptotic expansion of the trigamma function derived in
    [Thompson_1994]

    Paramters
    ---------

    K : int
      Number of tapers used in the multitaper method

    Returns
    -------

    evar : float
      Expected value of the jackknife variance estimator

    """

    kf = float(K)
    return ((1 / kf) * (kf - 1) / (kf - 0.5) *
            ((kf - 1) / (kf - 2)) ** 2 * (kf - 3) / (kf - 2))


def jackknifed_sdf_variance(yk, eigvals, sides='onesided', adaptive=True):
    r"""
    Returns the variance of the log-sdf estimated through jack-knifing
    a group of independent sdf estimates.

    Parameters
    ----------

    yk : ndarray (K, L)
       The K DFTs of the tapered sequences
    eigvals : ndarray (K,)
       The eigenvalues corresponding to the K DPSS tapers
    sides : str, optional
       Compute the jackknife pseudovalues over as one-sided or
       two-sided spectra
    adpative : bool, optional
       Compute the adaptive weighting for each jackknife pseudovalue

    Returns
    -------

    var : The estimate for log-sdf variance

    Notes
    -----

    The jackknifed mean estimate is distributed about the true mean as
    a Student's t-distribution with (K-1) degrees of freedom, and
    standard error equal to sqrt(var). However, Thompson and Chave [1]
    point out that this variance better describes the sample mean.


    [1] Thomson D J, Chave A D (1991) Advances in Spectrum Analysis and Array
    Processing (Prentice-Hall, Englewood Cliffs, NJ), 1, pp 58-113.
    """
    K = yk.shape[0]

    from nitime.algorithms import mtm_cross_spectrum

    # the samples {S_k} are defined, with or without weights, as
    # S_k = | x_k |**2
    # | x_k |**2 = | y_k * d_k |**2          (with adaptive weights)
    # | x_k |**2 = | y_k * sqrt(eig_k) |**2  (without adaptive weights)

    all_orders = set(range(K))
    jk_sdf = []
    # get the leave-one-out estimates -- ideally, weights are recomputed
    # for each leave-one-out. This is now the case.
    for i in range(K):
        items = list(all_orders.difference([i]))
        spectra_i = np.take(yk, items, axis=0)
        eigs_i = np.take(eigvals, items)
        if adaptive:
            # compute the weights
            weights, _ = adaptive_weights(spectra_i, eigs_i, sides=sides)
        else:
            weights = eigs_i[:, None]
        # this is the leave-one-out estimate of the sdf
        jk_sdf.append(
            mtm_cross_spectrum(
                spectra_i, spectra_i, weights, sides=sides
                )
            )
    # log-transform the leave-one-out estimates and the mean of estimates
    jk_sdf = np.log(jk_sdf)
    # jk_avg should be the mean of the log(jk_sdf(i))
    jk_avg = jk_sdf.mean(axis=0)

    K = float(K)

    jk_var = (jk_sdf - jk_avg)
    np.power(jk_var, 2, jk_var)
    jk_var = jk_var.sum(axis=0)

    # Thompson's recommended factor, eq 18
    # Jackknifing Multitaper Spectrum Estimates
    # IEEE SIGNAL PROCESSING MAGAZINE [20] JULY 2007
    f = (K - 1) ** 2 / K / (K - 0.5)
    jk_var *= f
    return jk_var


def jackknifed_coh_variance(tx, ty, eigvals, adaptive=True):
    """
    Returns the variance of the coherency between x and y, estimated
    through jack-knifing the tapered samples in {tx, ty}.

    Parameters
    ----------

    tx : ndarray, (K, L)
       The K complex spectra of tapered timeseries x
    ty : ndarray, (K, L)
       The K complex spectra of tapered timeseries y
    eigvals : ndarray (K,)
       The eigenvalues associated with the K DPSS tapers

    Returns
    -------

    jk_var : ndarray
       The variance computed in the transformed domain (see
       normalize_coherence)
    """

    K = tx.shape[0]

    # calculate leave-one-out estimates of MSC (magnitude squared coherence)
    jk_coh = []
    # coherence is symmetric (right??)
    sides = 'onesided'
    all_orders = set(range(K))

    import nitime.algorithms as alg

    # get the leave-one-out estimates
    for i in range(K):
        items = list(all_orders.difference([i]))
        tx_i = np.take(tx, items, axis=0)
        ty_i = np.take(ty, items, axis=0)
        eigs_i = np.take(eigvals, items)
        if adaptive:
            wx, _ = adaptive_weights(tx_i, eigs_i, sides=sides)
            wy, _ = adaptive_weights(ty_i, eigs_i, sides=sides)
        else:
            wx = wy = eigs_i[:, None]
        # The CSD
        sxy_i = alg.mtm_cross_spectrum(tx_i, ty_i, (wx, wy), sides=sides)
        # The PSDs
        sxx_i = alg.mtm_cross_spectrum(tx_i, tx_i, wx, sides=sides)
        syy_i = alg.mtm_cross_spectrum(ty_i, ty_i, wy, sides=sides)
        # these are the | c_i | samples
        msc = np.abs(sxy_i)
        msc /= np.sqrt(sxx_i * syy_i)
        jk_coh.append(msc)

    jk_coh = np.array(jk_coh)
    # now normalize the coherence estimates and take the mean
    normalize_coherence(jk_coh, 2 * K - 2, copy=False)  # inplace
    jk_avg = np.mean(jk_coh, axis=0)

    jk_var = (jk_coh - jk_avg)
    np.power(jk_var, 2, jk_var)
    jk_var = jk_var.sum(axis=0)

    # Do/Don't use the alternative scaling here??
    f = float(K - 1) / K

    jk_var *= f

    return jk_var


#-----------------------------------------------------------------------------
# Multitaper utils
#-----------------------------------------------------------------------------
def adaptive_weights(yk, eigvals, sides='onesided', max_iter=150):
    r"""
    Perform an iterative procedure to find the optimal weights for K
    direct spectral estimators of DPSS tapered signals.

    Parameters
    ----------

    yk : ndarray (K, N)
       The K DFTs of the tapered sequences
    eigvals : ndarray, length-K
       The eigenvalues of the DPSS tapers
    sides : str
       Whether to compute weights on a one-sided or two-sided spectrum
    max_iter : int
       Maximum number of iterations for weight computation

    Returns
    -------

    weights, nu

       The weights (array like sdfs), and the
       "equivalent degrees of freedom" (array length-L)

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

    If there are less than 3 tapers, then the adaptive weights are not
    found. The square root of the eigenvalues are returned as weights,
    and the degrees of freedom are 2*K

    """
    from nitime.algorithms import mtm_cross_spectrum
    K = len(eigvals)
    if len(eigvals) < 3:
        print("""
        Warning--not adaptively combining the spectral estimators
        due to a low number of tapers.
        """)
        # we'll hope this is a correct length for L
        N = yk.shape[-1]
        L = N / 2 + 1 if sides == 'onesided' else N
        return (np.multiply.outer(np.sqrt(eigvals), np.ones(L)), 2 * K)
    rt_eig = np.sqrt(eigvals)

    # combine the SDFs in the traditional way in order to estimate
    # the variance of the timeseries
    N = yk.shape[1]
    sdf = mtm_cross_spectrum(yk, yk, eigvals[:, None], sides=sides)
    L = sdf.shape[-1]
    var_est = np.sum(sdf, axis=-1) / N
    bband_sup = (1-eigvals)*var_est

    # The process is to iteratively switch solving for the following
    # two expressions:
    # (1) Adaptive Multitaper SDF:
    # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
    #
    # (2) Weights
    # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
    #
    # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
    # and the expected value of the broadband bias function
    # E{B_k(f)} is replaced by its full-band integration
    # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

    # start with an estimate from incomplete data--the first 2 tapers
    sdf_iter = mtm_cross_spectrum(yk[:2], yk[:2], eigvals[:2, None],
                                  sides=sides)
    err = np.zeros((K, L))
    # for numerical considerations, don't bother doing adaptive
    # weighting after 150 dB down
    min_pwr = sdf_iter.max() * 10 ** (-150/20.)
    default_weights = np.where(sdf_iter < min_pwr)[0]
    adaptiv_weights = np.where(sdf_iter >= min_pwr)[0]

    w_def = rt_eig[:,None] * sdf_iter[default_weights]
    w_def /= eigvals[:, None] * sdf_iter[default_weights] + bband_sup[:,None]

    d_sdfs = np.abs(yk[:,adaptiv_weights])**2
    if L < N:
        d_sdfs *= 2
    sdf_iter = sdf_iter[adaptiv_weights]
    yk = yk[:,adaptiv_weights]
    for n in range(max_iter):
        d_k = rt_eig[:,None] * sdf_iter[None, :]
        d_k /= eigvals[:, None]*sdf_iter[None, :] + bband_sup[:,None]
        # Test for convergence -- this is overly conservative, since
        # iteration only stops when all frequencies have converged.
        # A better approach is to iterate separately for each freq, but
        # that is a nonvectorized algorithm.
        #sdf_iter = mtm_cross_spectrum(yk, yk, d_k, sides=sides)
        sdf_iter = np.sum( d_k**2 * d_sdfs, axis=0 )
        sdf_iter /= np.sum( d_k**2, axis=0 )
        # Compute the cost function from eq 5.4 in Thomson 1982
        cfn = eigvals[:,None] * (sdf_iter[None,:] - d_sdfs)
        cfn /= (eigvals[:,None] * sdf_iter[None,:] + bband_sup[:,None])**2
        cfn = np.sum(cfn, axis=0)
        # there seem to be some pathological freqs sometimes ..
        # this should be a good heuristic
        if np.percentile(cfn**2, 95) < 1e-12:
            break
    else:  # If you have reached maximum number of iterations
        # Issue a warning and return non-converged weights:
        e_s = 'Breaking due to iterative meltdown in '
        e_s += 'nitime.utils.adaptive_weights.'
        warnings.warn(e_s, RuntimeWarning)
    weights = np.zeros( (K,L) )
    weights[:,adaptiv_weights] = d_k
    weights[:,default_weights] = w_def
    nu = 2 * (weights ** 2).sum(axis=-2)
    return weights, nu


#-----------------------------------------------------------------------------
# Eigensystem utils
#-----------------------------------------------------------------------------

# If we can get it, we want the cythonized version
try:
    from _utils import tridisolve

# If that doesn't work, we define it here:
except ImportError:
    def tridisolve(d, e, b, overwrite_b=True):
        """
        Symmetric tridiagonal system solver,
        from Golub and Van Loan, Matrix Computations pg 157

        Parameters
        ----------

        d : ndarray
          main diagonal stored in d[:]
        e : ndarray
          superdiagonal stored in e[:-1]
        b : ndarray
          RHS vector

        Returns
        -------

        x : ndarray
          Solution to Ax = b (if overwrite_b is False). Otherwise solution is
          stored in previous RHS vector b

        """
        N = len(b)
        # work vectors
        dw = d.copy()
        ew = e.copy()
        if overwrite_b:
            x = b
        else:
            x = b.copy()
        for k in range(1, N):
            # e^(k-1) = e(k-1) / d(k-1)
            # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
            t = ew[k - 1]
            ew[k - 1] = t / dw[k - 1]
            dw[k] = dw[k] - t * ew[k - 1]
        for k in range(1, N):
            x[k] = x[k] - ew[k - 1] * x[k - 1]
        x[N - 1] = x[N - 1] / dw[N - 1]
        for k in range(N - 2, -1, -1):
            x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

        if not overwrite_b:
            return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    """Perform an inverse iteration to find the eigenvector corresponding
    to the given eigenvalue in a symmetric tridiagonal system.

    Parameters
    ----------

    d : ndarray
      main diagonal of the tridiagonal system
    e : ndarray
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : ndarray
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates

    Returns
    -------

    e : ndarray
      The converged eigenvector

    """
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0

# #-----------------------------------------------------------------------------
# # Correlation/Covariance utils
# #-----------------------------------------------------------------------------


def remove_bias(x, axis):
    "Subtracts an estimate of the mean from signal x at axis"
    padded_slice = [slice(d) for d in x.shape]
    padded_slice[axis] = np.newaxis
    mn = np.mean(x, axis=axis)
    return x - mn[tuple(padded_slice)]


def crosscov(x, y, axis=-1, all_lags=False, debias=True, normalize=True):
    """Returns the crosscovariance sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters
    ----------

    x : ndarray
    y : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of s_xy
       to be the length of x and y. If False, then the zero lag covariance
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
    debias : {True/False}
       Always removes an estimate of the mean along the axis, unless
       told not to (eg X and Y are known zero-mean)

    Returns
    -------

    cxy : ndarray
       The crosscovariance function

    Notes
    -----

    cross covariance of processes x and y is defined as

    .. math::

    C_{xy}[k]=E\{(X(n+k)-E\{X\})(Y(n)-E\{Y\})^{*}\}

    where X and Y are discrete, stationary (or ergodic) random processes

    Also note that this routine is the workhorse for all auto/cross/cov/corr
    functions.

    """
    if x.shape[axis] != y.shape[axis]:
        raise ValueError(
            'crosscov() only works on same-length sequences for now'
            )
    if debias:
        x = remove_bias(x, axis)
        y = remove_bias(y, axis)
    slicing = [slice(d) for d in x.shape]
    slicing[axis] = slice(None, None, -1)
    cxy = fftconvolve(x, y[tuple(slicing)].conj(), axis=axis, mode='full')
    N = x.shape[axis]
    if normalize:
        cxy /= N
    if all_lags:
        return cxy
    slicing[axis] = slice(N - 1, 2 * N - 1)
    return cxy[tuple(slicing)]


def crosscorr(x, y, **kwargs):
    """
    Returns the crosscorrelation sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters
    ----------

    x : ndarray
    y : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Returns
    -------

    rxy : ndarray
       The crosscorrelation function

    Notes
    -----

    cross correlation is defined as

    .. math::

    R_{xy}[k]=E\{X[n+k]Y^{*}[n]\}

    where X and Y are discrete, stationary (ergodic) random processes
    """
    # just make the same computation as the crosscovariance,
    # but without subtracting the mean
    kwargs['debias'] = False
    rxy = crosscov(x, y, **kwargs)
    return rxy


def autocov(x, **kwargs):
    """Returns the autocovariance of signal s at all lags.

    Parameters
    ----------

    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Returns
    -------

    cxx : ndarray
       The autocovariance function

    Notes
    -----

    Adheres to the definition

    .. math::

    C_{xx}[k]=E\{(X[n+k]-E\{X\})(X[n]-E\{X\})^{*}\}

    where X is a discrete, stationary (ergodic) random process
    """
    # only remove the mean once, if needed
    debias = kwargs.pop('debias', True)
    axis = kwargs.get('axis', -1)
    if debias:
        x = remove_bias(x, axis)
    kwargs['debias'] = False
    return crosscov(x, x, **kwargs)


def autocorr(x, **kwargs):
    """Returns the autocorrelation of signal s at all lags.

    Parameters
    ----------

    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Notes
    -----

    Adheres to the definition

    .. math::

    R_{xx}[k]=E\{X[n+k]X^{*}[n]\}

    where X is a discrete, stationary (ergodic) random process

    """
    # do same computation as autocovariance,
    # but without subtracting the mean
    kwargs['debias'] = False
    return autocov(x, **kwargs)


def fftconvolve(in1, in2, mode="full", axis=None):
    """ Convolve two N-dimensional arrays using FFT. See convolve.

    This is a fix of scipy.signal.fftconvolve, adding an axis argument and
    importing locally the stuff only needed for this function

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))

    if axis is None:
        size = s1 + s2 - 1
        fslice = tuple([slice(0, int(sz)) for sz in size])
    else:
        equal_shapes = s1 == s2
        # allow equal_shapes[axis] to be False
        equal_shapes[axis] = True
        assert equal_shapes.all(), 'Shape mismatch on non-convolving axes'
        size = s1[axis] + s2[axis] - 1
        fslice = [slice(l) for l in s1]
        fslice[axis] = slice(0, int(size))
        fslice = tuple(fslice)

    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))
    if axis is None:
        IN1 = fftpack.fftn(in1, fsize)
        IN1 *= fftpack.fftn(in2, fsize)
        ret = fftpack.ifftn(IN1)[fslice].copy()
    else:
        IN1 = fftpack.fft(in1, fsize, axis=axis)
        IN1 *= fftpack.fft(in2, fsize, axis=axis)
        ret = fftpack.ifft(IN1, axis=axis)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return signaltools._centered(ret, osize)
    elif mode == "valid":
        return signaltools._centered(ret, abs(s2 - s1) + 1)




#----------goodness of fit utilities ----------------------------------------

def akaike_information_criterion(ecov, p, m, Ntotal, corrected=False):

    """

    A measure of the goodness of fit of an auto-regressive model based on the
    model order and the error covariance.

    Parameters
    ----------

    ecov : float array
        The error covariance of the system
    p
        the number of channels
    m : int
        the model order
    Ntotal
        the number of total time-points (across channels)
    corrected : boolean (optional)
        Whether to correct for small sample size

    Returns
    -------

    AIC : float
        The value of the AIC


    Notes
    -----
    This is an implementation of equation (50) in Ding et al. (2006):

    M Ding and Y Chen and S Bressler (2006) Granger Causality: Basic Theory and
    Application to Neuroscience. http://arxiv.org/abs/q-bio/0608035v1


    Correction for small sample size is taken from:
    http://en.wikipedia.org/wiki/Akaike_information_criterion.

    """

    AIC = (2 * (np.log(linalg.det(ecov))) +
           ((2 * (p ** 2) * m) / (Ntotal)))

    if corrected is None:
        return AIC
    else:
        return AIC + (2 * m * (m + 1)) / (Ntotal - m - 1)


def bayesian_information_criterion(ecov, p, m, Ntotal):
    """The Bayesian Information Criterion, also known as the Schwarz criterion
     is a measure of goodness of fit of a statistical model, based on the
     number of model parameters and the likelihood of the model

    Parameters
    ----------
    ecov : float array
        The error covariance of the system

    p : int
        the system size (how many variables).

    m : int
        the model order.

    corrected : boolean (optional)
        Whether to correct for small sample size


    Returns
    -------

    BIC : float
        The value of the BIC
    a
        the resulting autocovariance vector

    Notes
    -----
    This is an implementation of equation (51) in Ding et al. (2006):

    .. math ::

    BIC(m) = 2 log(|\Sigma|) + \frac{2p^2 m log(N_{total})}{N_{total}},

    where $\Sigma$ is the noise covariance matrix. In auto-regressive model
    estimation, this matrix will contain in $\Sigma_{i,j}$ the residual
    variance in estimating time-series $i$ from $j$, $p$ is the dimensionality
    of the data, $m$ is the number of parameters in the model and $N_{total}$
    is the number of time-points.

    M Ding and Y Chen and S Bressler (2006) Granger Causality: Basic Theory and
    Application to Neuroscience. http://arxiv.org/abs/q-bio/0608035v1


    See http://en.wikipedia.org/wiki/Schwarz_criterion

    """

    BIC = (2 * (np.log(linalg.det(ecov))) +
            ((2 * (p ** 2) * m * np.log(Ntotal)) / (Ntotal)))

    return BIC

#-----------------------------------------------------------------------------
# testing utils
#-----------------------------------------------------------------------------


def circle_to_hz(omega, Fsamp):
    """For a frequency grid spaced on the unit circle of an imaginary plane,
    return the corresponding freqency grid in Hz.
    """
    return Fsamp * omega / (2 * np.pi)


def ar_generator(N=512, sigma=1., coefs=None, drop_transients=0, v=None):
    """
    This generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    where v(n) is a stationary stochastic process with zero mean
    and variance = sigma. XXX: confusing variance notation

    Parameters
    ----------

    N : int
      sequence length
    sigma : float
      power of the white noise driving process
    coefs : sequence
      AR coefficients for k = 1, 2, ..., P
    drop_transients : int
      number of initial IIR filter transient terms to drop
    v : ndarray
      custom noise process

    Parameters
    ----------

    N : float
       The number of points in the AR process generated. Default: 512
    sigma : float
       The variance of the noise in the AR process. Default: 1
    coefs : list or array of floats
       The AR model coefficients. Default: [2.7607, -3.8106, 2.6535, -0.9238],
       which is a sequence shown to be well-estimated by an order 8 AR system.
    drop_transients : float
       How many samples to drop from the beginning of the sequence (the
       transient phases of the process), so that the process can be considered
       stationary.
    v : float array
       Optionally, input a specific sequence of noise samples (this over-rides
       the sigma parameter). Default: None

    Returns
    -------

    u : ndarray
       the AR sequence
    v : ndarray
       the unit-variance innovations sequence
    coefs : ndarray
       feedback coefficients from k=1,len(coefs)

    The form of the feedback coefficients is a little different than
    the normal linear constant-coefficient difference equation. Therefore
    the transfer function implemented in this method is

    H(z) = sigma**0.5 / ( 1 - sum_k coefs(k)z**(-k) )    1 <= k <= P

    Examples
    --------

    >>> import nitime.algorithms as alg
    >>> ar_seq, nz, alpha = ar_generator()
    >>> fgrid, hz = alg.freq_response(1.0, a=np.r_[1, -alpha])
    >>> sdf_ar = (hz * hz.conj()).real

    """
    if coefs is None:
        # this sequence is shown to be estimated well by an order 8 AR system
        coefs = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    else:
        coefs = np.asarray(coefs)

    # The number of terms we generate must include the dropped transients, and
    # then at the end we cut those out of the returned array.
    N += drop_transients

    # Typically uses just pass sigma in, but optionally they can provide their
    # own noise vector, case in which we use it
    if v is None:
        v = np.random.normal(size=N)
        v -= v[drop_transients:].mean()

    b = [sigma ** 0.5]
    a = np.r_[1, -coefs]
    u = sig.lfilter(b, a, v)

    # Only return the data after the drop_transients terms
    return u[drop_transients:], v[drop_transients:], coefs

# -----------------------------------------  COPIED from VIZ: Plotting function ----------------------------------------


def plot_spectral_estimate(f, sdf, sdf_ests, limits=None, elabels=()):
    """
    Plot an estimate of a spectral transform against the ground truth.

    Utility file used in building the documentation
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax_limits = (sdf.min() - 2*np.abs(sdf.min()),
                 sdf.max() + 1.25*np.abs(sdf.max()))
    ax.plot(f, sdf, 'c', label='True S(f)')

    if not elabels:
        elabels = ('',) * len(sdf_ests)
    colors = 'bgkmy'
    for e, l, c in zip(sdf_ests, elabels, colors):
        ax.plot(f, e, color=c, linewidth=2, label=l)

    if limits is not None:
        ax.fill_between(f, limits[0], y2=limits[1], color=(1, 0, 0, .3),
                        alpha=0.5)

    ax.set_ylim(ax_limits)
    ax.legend()
    return fig

